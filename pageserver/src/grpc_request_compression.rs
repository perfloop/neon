//! Bounded request decompression for PageService.
//!
//! Tonic 0.13 bounds a compressed gRPC frame before it decompresses it. PageService's
//! `GetPageRequest` contains an untrusted repeated field, so that ordering would allow a small
//! compressed frame to make Tonic materialize an arbitrarily large protobuf message. This service
//! layer handles the two request encodings that PageService supports, caps both their wire and
//! decoded sizes, and passes an identity-framed message to Tonic.

use std::io::Read;
use std::sync::Arc;
use std::task::{Context, Poll};
#[cfg(feature = "testing")]
use std::time::Instant;

use bytes::{Buf, Bytes, BytesMut};
use http_body::Frame;
use http_body_util::StreamBody;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::codegen::Body as HttpBody;
use tonic::{Status, body::Body, codegen::Service};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const GRPC_HEADER_SIZE: usize = 5;
const GRPC_COMPRESSION_FLAG_UNCOMPRESSED: u8 = 0;
const GRPC_COMPRESSION_FLAG_COMPRESSED: u8 = 1;
const GRPC_ENCODING_HEADER: &str = "grpc-encoding";
const GET_PAGES_PATH: &str = "/page_api.PageService/GetPages";
// Tonic's zstd request encoder uses a 2 MiB window. Retain compatibility with those frames but
// reject a wire-valid frame that advertises an arbitrarily large decoder window before libzstd
// allocates it.
const MAX_ZSTD_WINDOW_LOG: u32 = 21;

/// Test-only request metadata that asks the bounded ingress probe to retain a server-side
/// allocation and elapsed-time observation. It is stripped before the generated service sees the
/// request.
#[cfg(feature = "testing")]
const INGRESS_OBSERVATION_HEADER: &str = "neon-test-ingress-observe";

#[derive(Clone, Copy)]
enum Encoding {
    Gzip,
    Zstd,
}

impl Encoding {
    fn from_header(value: &[u8]) -> Option<Self> {
        match value {
            b"gzip" => Some(Self::Gzip),
            b"zstd" => Some(Self::Zstd),
            _ => None,
        }
    }
}

/// Wraps PageService so gzip/zstd request messages are decompressed with an output limit before
/// they reach Tonic's protobuf decoder.
#[derive(Clone)]
pub(crate) struct BoundedGrpcRequestCompression<S> {
    inner: S,
    max_decoded_message_size: usize,
    max_compressed_message_size: usize,
    // GetPages streams can be numerous. Bound active decompression so queued streams retain only
    // their incrementally-filled <=4 KiB wire payload instead of every stream allocating a codec
    // window concurrently.
    decompression_limiter: Arc<Semaphore>,
}

impl<S> BoundedGrpcRequestCompression<S> {
    pub(crate) fn new(
        inner: S,
        max_decoded_message_size: usize,
        max_compressed_message_size: usize,
    ) -> Self {
        Self {
            inner,
            max_decoded_message_size,
            max_compressed_message_size,
            decompression_limiter: new_decompression_limiter(),
        }
    }
}

impl<S: tonic::server::NamedService> tonic::server::NamedService
    for BoundedGrpcRequestCompression<S>
{
    const NAME: &'static str = S::NAME;
}

impl<S, B> Service<http::Request<B>> for BoundedGrpcRequestCompression<S>
where
    S: Service<http::Request<Body>>,
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError> + Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        self.inner.call(transform_request_with_limiter(
            req,
            self.max_decoded_message_size,
            self.max_compressed_message_size,
            self.decompression_limiter.clone(),
        ))
    }
}

fn new_decompression_limiter() -> Arc<Semaphore> {
    // PageService runs on the configured compute-request runtime. This bounds simultaneously
    // active decoder windows to that runtime's worker ceiling; queued streams retain only their
    // bounded wire payloads.
    Arc::new(Semaphore::new(crate::task_mgr::TOKIO_WORKER_THREADS.get()))
}

#[cfg(test)]
fn transform_request<B>(
    req: http::Request<B>,
    max_decoded_message_size: usize,
    max_compressed_message_size: usize,
) -> http::Request<Body>
where
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError> + Send + 'static,
{
    transform_request_with_limiter(
        req,
        max_decoded_message_size,
        max_compressed_message_size,
        new_decompression_limiter(),
    )
}

fn transform_request_with_limiter<B>(
    req: http::Request<B>,
    max_decoded_message_size: usize,
    max_compressed_message_size: usize,
    decompression_limiter: Arc<Semaphore>,
) -> http::Request<Body>
where
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError> + Send + 'static,
{
    // GetPages is the only PageService request with an untrusted repeated field. Keep the
    // transport behavior and message-size policy for the other PageService RPCs unchanged.
    if req.uri().path() != GET_PAGES_PATH {
        return req.map(Body::new);
    }

    let (mut parts, body) = req.into_parts();
    let encoding_header = parts
        .headers
        .get(GRPC_ENCODING_HEADER)
        .map(http::HeaderValue::as_bytes);
    let encoding = encoding_header.and_then(Encoding::from_header);

    // `identity` has the same no-compression meaning as an absent header, so it must still pass
    // through the GetPages size gate. Let Tonic retain its normal Unimplemented behavior for any
    // other unsupported request encoding, including a non-UTF-8 header value.
    let identity = encoding_header.is_none() || encoding_header == Some(b"identity".as_slice());
    if !identity && encoding.is_none() {
        return http::Request::from_parts(parts, Body::new(body));
    }

    #[cfg(feature = "testing")]
    let observation = parts
        .headers
        .remove(INGRESS_OBSERVATION_HEADER)
        .is_some()
        .then(IngressObservation::new);
    #[cfg(not(feature = "testing"))]
    let observation = ();

    #[cfg(feature = "testing")]
    if let Some(observation) = &observation {
        parts.extensions.insert(observation.clone());
    }

    // The wire payload is replaced by a bounded identity frame. The original content length (if
    // present) describes the compressed HTTP body and must not accompany the replacement body.
    parts.headers.remove(http::header::CONTENT_LENGTH);

    // One transformed frame is sufficient to hand the stream to Tonic. Keeping this channel
    // bounded to one also prevents a fast peer from retaining several maximum-size wire frames
    // while the GetPages handler is working on the first request.
    let (tx, rx) = mpsc::channel(1);
    tokio::spawn(forward_frames_with_limiter(
        body,
        tx,
        encoding,
        max_decoded_message_size,
        max_compressed_message_size,
        decompression_limiter,
        observation,
    ));
    let body = Body::new(StreamBody::new(ReceiverStream::new(rx)));
    http::Request::from_parts(parts, body)
}

#[cfg(test)]
async fn forward_frames<B>(
    body: B,
    tx: mpsc::Sender<Result<Frame<Bytes>, Status>>,
    encoding: Option<Encoding>,
    max_decoded_message_size: usize,
    max_compressed_message_size: usize,
    observation: Observation,
) where
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError> + Send + 'static,
{
    forward_frames_with_limiter(
        body,
        tx,
        encoding,
        max_decoded_message_size,
        max_compressed_message_size,
        new_decompression_limiter(),
        observation,
    )
    .await
}

async fn forward_frames_with_limiter<B>(
    body: B,
    tx: mpsc::Sender<Result<Frame<Bytes>, Status>>,
    encoding: Option<Encoding>,
    max_decoded_message_size: usize,
    max_compressed_message_size: usize,
    decompression_limiter: Arc<Semaphore>,
    observation: Observation,
) where
    B: HttpBody<Data = Bytes> + Send + 'static,
    B::Error: Into<BoxError> + Send + 'static,
{
    let mut decoder = GrpcFrameDecoder::new(
        encoding,
        max_decoded_message_size,
        max_compressed_message_size,
        decompression_limiter,
        observation,
    );
    let mut body = std::pin::pin!(body);

    loop {
        // `forward_frames` owns the original HTTP body. If Tonic drops its replacement body before
        // the peer sends another DATA frame, waiting only on `poll_frame` would otherwise retain
        // the detached task and connection indefinitely. Observe the bounded output receiver too.
        let frame = tokio::select! {
            _ = tx.closed() => break,
            frame = futures::future::poll_fn(|cx| body.as_mut().poll_frame(cx)) => frame,
        };
        let result = match frame {
            Some(Ok(frame)) => match frame.into_data() {
                Ok(data) => decoder.push(data, &tx).await,
                Err(frame) => {
                    let trailers = frame
                        .into_trailers()
                        .expect("HTTP body frame was neither data nor trailers");
                    match decoder.finish() {
                        Ok(()) => tx
                            .send(Ok(Frame::trailers(trailers)))
                            .await
                            .map_err(|_| Status::cancelled("gRPC request consumer closed")),
                        Err(status) => Err(status),
                    }
                }
            },
            Some(Err(error)) => Err(Status::internal(format!(
                "gRPC request body error: {}",
                error.into()
            ))),
            None => match decoder.finish() {
                Ok(()) => break,
                Err(status) => Err(status),
            },
        };

        if let Err(status) = result {
            let _ = tx.send(Err(status)).await;
            break;
        }
    }
}

#[cfg(feature = "testing")]
type Observation = Option<IngressObservation>;
#[cfg(not(feature = "testing"))]
type Observation = ();

struct GrpcFrameDecoder {
    encoding: Option<Encoding>,
    max_decoded_message_size: usize,
    max_compressed_message_size: usize,
    decompression_limiter: Arc<Semaphore>,
    header: [u8; GRPC_HEADER_SIZE],
    header_len: usize,
    payload: BytesMut,
    expected_payload_len: Option<usize>,
    #[cfg(feature = "testing")]
    observation: Observation,
}

impl GrpcFrameDecoder {
    fn new(
        encoding: Option<Encoding>,
        max_decoded_message_size: usize,
        max_compressed_message_size: usize,
        decompression_limiter: Arc<Semaphore>,
        observation: Observation,
    ) -> Self {
        #[cfg(not(feature = "testing"))]
        let _ = observation;
        Self {
            encoding,
            max_decoded_message_size,
            max_compressed_message_size,
            decompression_limiter,
            header: [0; GRPC_HEADER_SIZE],
            header_len: 0,
            payload: BytesMut::new(),
            expected_payload_len: None,
            #[cfg(feature = "testing")]
            observation,
        }
    }

    async fn push(
        &mut self,
        mut data: Bytes,
        tx: &mpsc::Sender<Result<Frame<Bytes>, Status>>,
    ) -> Result<(), Status> {
        while data.has_remaining() {
            if self.expected_payload_len.is_none() {
                let needed = GRPC_HEADER_SIZE - self.header_len;
                let take = needed.min(data.remaining());
                self.header[self.header_len..self.header_len + take].copy_from_slice(&data[..take]);
                self.header_len += take;
                data.advance(take);
                if self.header_len != GRPC_HEADER_SIZE {
                    continue;
                }

                let flag = self.header[0];
                if flag != GRPC_COMPRESSION_FLAG_UNCOMPRESSED
                    && flag != GRPC_COMPRESSION_FLAG_COMPRESSED
                {
                    return Err(Status::internal(
                        "gRPC request has an invalid compression flag",
                    ));
                }
                if flag == GRPC_COMPRESSION_FLAG_COMPRESSED && self.encoding.is_none() {
                    return Err(Status::internal(
                        "compressed gRPC request has no supported grpc-encoding",
                    ));
                }

                #[cfg(feature = "testing")]
                if let Some(observation) = &self.observation {
                    // Start before body buffering. Allocation samples below cover every
                    // synchronous ingress operation through identity-frame reconstruction, and
                    // deliberately exclude later asynchronous protobuf handling.
                    observation.begin();
                }

                let len = u32::from_be_bytes(self.header[1..].try_into().unwrap()) as usize;
                let limit = if flag == GRPC_COMPRESSION_FLAG_COMPRESSED {
                    self.max_compressed_message_size
                } else {
                    self.max_decoded_message_size
                };
                if len > limit {
                    return Err(self.limit_status(flag == GRPC_COMPRESSION_FLAG_COMPRESSED));
                }
                // Do not reserve from a peer-controlled header. `BytesMut` grows only as DATA
                // arrives, so a peer that declares an accepted frame and stalls retains no
                // maximum-frame allocation merely from this envelope header.
                self.expected_payload_len = Some(len);
            }

            let expected = self.expected_payload_len.expect("set after gRPC header");
            let take = (expected - self.payload.len()).min(data.remaining());
            #[cfg(feature = "testing")]
            let allocated_before = self.thread_allocated_before();
            self.payload.extend_from_slice(&data[..take]);
            #[cfg(feature = "testing")]
            self.record_thread_allocation_since(allocated_before);
            data.advance(take);
            if self.payload.len() != expected {
                continue;
            }

            let compressed = self.header[0] == GRPC_COMPRESSION_FLAG_COMPRESSED;
            let payload = if compressed {
                let permit = self.acquire_decompression_permit(tx).await?;
                #[cfg(feature = "testing")]
                let allocated_before = self.thread_allocated_before();
                let result = self.decompress();
                #[cfg(feature = "testing")]
                self.record_thread_allocation_since(allocated_before);
                drop(permit);
                let payload = result?;
                if payload.len() > self.max_decoded_message_size {
                    return Err(self.limit_status(true));
                }
                payload
            } else {
                self.payload.split().freeze()
            };
            self.emit(payload, tx).await?;
            self.reset();
        }
        Ok(())
    }

    async fn acquire_decompression_permit(
        &self,
        tx: &mpsc::Sender<Result<Frame<Bytes>, Status>>,
    ) -> Result<OwnedSemaphorePermit, Status> {
        tokio::select! {
            _ = tx.closed() => Err(Status::cancelled("gRPC request consumer closed")),
            permit = self.decompression_limiter.clone().acquire_owned() => permit
                .map_err(|_| Status::internal("PageService decompression limiter closed")),
        }
    }

    fn decompress(&mut self) -> Result<Bytes, Status> {
        let mut output = Vec::with_capacity(self.max_decoded_message_size.min(self.payload.len()));
        let limit = (self.max_decoded_message_size + 1) as u64;
        let result = match self.encoding.expect("compressed flag requires encoding") {
            Encoding::Gzip => {
                // Bound all concatenated gzip members as one gRPC payload rather than letting
                // a later member escape the decoded-output limit.
                let decoder = flate2::read::MultiGzDecoder::new(self.payload.as_ref());
                decoder.take(limit).read_to_end(&mut output)
            }
            Encoding::Zstd => {
                let mut decoder =
                    zstd::stream::read::Decoder::new(self.payload.as_ref()).map_err(|error| {
                        Status::internal(format!("invalid zstd gRPC request: {error}"))
                    })?;
                decoder
                    .window_log_max(MAX_ZSTD_WINDOW_LOG)
                    .map_err(|error| {
                        Status::out_of_range(format!(
                            "zstd gRPC request exceeds PageService decoder window limit: {error}"
                        ))
                    })?;
                decoder.take(limit).read_to_end(&mut output)
            }
        };
        result.map_err(|error| {
            Status::internal(format!("invalid compressed gRPC request: {error}"))
        })?;

        Ok(Bytes::from(output))
    }

    async fn emit(
        &self,
        payload: Bytes,
        tx: &mpsc::Sender<Result<Frame<Bytes>, Status>>,
    ) -> Result<(), Status> {
        #[cfg(feature = "testing")]
        let allocated_before = self.thread_allocated_before();
        let mut frame = BytesMut::with_capacity(GRPC_HEADER_SIZE + payload.len());
        frame.extend_from_slice(&[GRPC_COMPRESSION_FLAG_UNCOMPRESSED]);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(&payload);
        #[cfg(feature = "testing")]
        self.record_thread_allocation_since(allocated_before);
        tx.send(Ok(Frame::data(frame.freeze())))
            .await
            .map_err(|_| Status::cancelled("gRPC request consumer closed"))
    }

    #[cfg(feature = "testing")]
    fn thread_allocated_before(&self) -> Option<u64> {
        self.observation
            .as_ref()
            .and_then(|_| thread_allocated_bytes())
    }

    #[cfg(feature = "testing")]
    fn record_thread_allocation_since(&self, allocated_before: Option<u64>) {
        if let (Some(observation), Some(allocated_before), Some(allocated_after)) = (
            &self.observation,
            allocated_before,
            thread_allocated_bytes(),
        ) {
            // This covers a synchronous portion of the forwarding task. Sampling only on one
            // side of an await keeps a cumulative per-thread counter from being attributed to a
            // task after it migrates between Tokio workers.
            observation
                .record_ingress_thread_allocation(allocated_after.saturating_sub(allocated_before));
        }
    }

    fn limit_status(&self, compressed: bool) -> Status {
        let kind = if compressed { "compressed" } else { "identity" };
        #[cfg(feature = "testing")]
        let mut message = format!(
            "{kind} gRPC request exceeds PageService ingress limit of {} decoded bytes",
            self.max_decoded_message_size
        );
        #[cfg(not(feature = "testing"))]
        let message = format!(
            "{kind} gRPC request exceeds PageService ingress limit of {} decoded bytes",
            self.max_decoded_message_size
        );
        #[cfg(feature = "testing")]
        if let Some(observation) = &self.observation {
            let measurement = observation.finish();
            message.push_str(&format!(
                "; server_ingress_thread_allocated_bytes={}; \
                 server_ingress_thread_allocation_observed={}; server_elapsed_ns={}",
                measurement.ingress_thread_allocated_bytes,
                measurement.ingress_thread_allocation_observed,
                measurement.elapsed_ns,
            ));
        }
        Status::out_of_range(message)
    }

    fn reset(&mut self) {
        self.header_len = 0;
        self.expected_payload_len = None;
        self.payload.clear();
    }

    fn finish(&self) -> Result<(), Status> {
        if self.header_len != 0 || self.expected_payload_len.is_some() {
            return Err(Status::internal("unexpected EOF in gRPC request frame"));
        }
        Ok(())
    }
}

#[cfg(feature = "testing")]
#[derive(Clone)]
pub(crate) struct IngressObservation(std::sync::Arc<std::sync::Mutex<IngressObservationState>>);

#[cfg(feature = "testing")]
#[derive(Default)]
struct IngressObservationState {
    started_at: Option<Instant>,
    ingress_thread_allocated_bytes: u64,
    ingress_thread_allocation_observed: bool,
}

#[cfg(feature = "testing")]
pub(crate) struct IngressMeasurement {
    /// Cumulative allocations made by synchronous portions of the forwarding task from wire-body
    /// buffering through bounded decompression and identity-frame reconstruction. This deliberately
    /// excludes unrelated tasks and later Prost allocations.
    pub(crate) ingress_thread_allocated_bytes: u64,
    pub(crate) ingress_thread_allocation_observed: bool,
    pub(crate) elapsed_ns: u128,
}

#[cfg(feature = "testing")]
impl IngressObservation {
    fn new() -> Self {
        Self(std::sync::Arc::new(std::sync::Mutex::new(
            IngressObservationState::default(),
        )))
    }

    fn begin(&self) {
        let mut state = self.0.lock().expect("ingress observation lock poisoned");
        if state.started_at.is_none() {
            state.started_at = Some(Instant::now());
        }
        // Mark that this test process can sample the same forwarding task even when the envelope
        // is rejected from its header before it needs to grow a payload buffer.
        state.ingress_thread_allocation_observed |= thread_allocated_bytes().is_some();
    }

    fn record_ingress_thread_allocation(&self, bytes: u64) {
        let mut state = self.0.lock().expect("ingress observation lock poisoned");
        state.ingress_thread_allocated_bytes =
            state.ingress_thread_allocated_bytes.saturating_add(bytes);
        state.ingress_thread_allocation_observed = true;
    }

    pub(crate) fn finish(&self) -> IngressMeasurement {
        let state = self.0.lock().expect("ingress observation lock poisoned");
        IngressMeasurement {
            ingress_thread_allocated_bytes: state.ingress_thread_allocated_bytes,
            ingress_thread_allocation_observed: state.ingress_thread_allocation_observed,
            elapsed_ns: state
                .started_at
                .map(|started_at| started_at.elapsed().as_nanos())
                .unwrap_or_default(),
        }
    }

    pub(crate) fn unobserved_measurement() -> IngressMeasurement {
        IngressMeasurement {
            ingress_thread_allocated_bytes: 0,
            ingress_thread_allocation_observed: false,
            elapsed_ns: 0,
        }
    }
}

#[cfg(feature = "testing")]
fn thread_allocated_bytes() -> Option<u64> {
    use tikv_jemalloc_ctl::thread;

    thread::allocatedp::read()
        .ok()
        .map(|allocated| allocated.get())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::io::Write;
    use std::sync::Arc;
    use std::time::Duration;

    use bytes::{Bytes, BytesMut};
    use flate2::{Compression, write::GzEncoder};
    use futures::{StreamExt, stream};
    use http_body::Frame;
    use http_body_util::{BodyExt, Full, StreamBody};
    use tokio::sync::{Semaphore, mpsc};
    use tonic::body::Body;

    use super::{
        GET_PAGES_PATH, GRPC_ENCODING_HEADER, GrpcFrameDecoder, MAX_ZSTD_WINDOW_LOG,
        forward_frames, transform_request,
    };

    fn grpc_frame(flag: u8, payload: &[u8]) -> Bytes {
        let mut frame = BytesMut::with_capacity(5 + payload.len());
        frame.extend_from_slice(&[flag]);
        frame.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        frame.extend_from_slice(payload);
        frame.freeze()
    }

    fn gzip(payload: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    fn zstd_with_window(window_log: u32, payload: &[u8]) -> Vec<u8> {
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).unwrap();
        encoder.window_log(window_log).unwrap();
        encoder.write_all(payload).unwrap();
        encoder.finish().unwrap()
    }

    #[tokio::test]
    async fn identity_header_still_uses_the_get_pages_size_gate() {
        // A valid gRPC envelope with a declared payload larger than the one-byte decoded cap.
        // The layer must reject from the header without waiting for that payload or delegating to
        // Tonic's default four-megabyte decoder limit.
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .header(GRPC_ENCODING_HEADER, "identity")
            .body(Body::new(Full::new(Bytes::from_static(&[0, 0, 0, 0, 2]))))
            .unwrap();
        let response = transform_request(request, 1, 1);
        let error = response
            .into_body()
            .frame()
            .await
            .expect("transformed body should report the oversized frame")
            .expect_err("declared identity frame must exceed the layer cap");
        assert_eq!(error.code(), tonic::Code::OutOfRange);
    }

    #[tokio::test]
    async fn gzip_frames_are_bounded_decompressed_across_body_fragments() {
        // HTTP/2 DATA boundaries need not coincide with gRPC envelopes. Preserve multiple request
        // messages and rewrite each compressed envelope to the identity framing Tonic expects.
        let first = grpc_frame(1, &gzip(b"first"));
        let second = grpc_frame(1, &gzip(b"second"));
        let mut input = BytesMut::new();
        input.extend_from_slice(&first);
        input.extend_from_slice(&second);
        let input = input.freeze();
        let source = StreamBody::new(stream::iter(vec![
            Ok::<_, Infallible>(Frame::data(input.slice(..3))),
            Ok(Frame::data(input.slice(3..9))),
            Ok(Frame::data(input.slice(9..))),
        ]));
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .header(GRPC_ENCODING_HEADER, "gzip")
            .body(source)
            .unwrap();

        let output = transform_request(request, 16, 64)
            .into_body()
            .collect()
            .await
            .expect("bounded gzip transform should succeed")
            .to_bytes();
        let mut expected = BytesMut::new();
        expected.extend_from_slice(&grpc_frame(0, b"first"));
        expected.extend_from_slice(&grpc_frame(0, b"second"));
        assert_eq!(output, expected.freeze());
    }

    #[tokio::test]
    async fn concatenated_gzip_members_share_the_decoded_output_limit() {
        let mut compressed = gzip(b"first");
        compressed.extend(gzip(b"second"));
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .header(GRPC_ENCODING_HEADER, "gzip")
            .body(Body::new(Full::new(grpc_frame(1, &compressed))))
            .unwrap();
        let error = transform_request(request, 6, 64)
            .into_body()
            .frame()
            .await
            .expect("transformed body should report oversized concatenated output")
            .expect_err("later gzip members must not bypass the decoded-output cap");
        assert_eq!(error.code(), tonic::Code::OutOfRange);
    }

    #[tokio::test]
    async fn zstd_window_limit_rejects_before_decoding_payload() {
        // A stream encoder with an unknown source size can advertise a decoder window larger than
        // the message output. The wire payload remains small, so this specifically exercises the
        // decoder-window bound rather than either gRPC length limit.
        let decoded = vec![0; (1_usize << MAX_ZSTD_WINDOW_LOG) + 1];
        let compressed = zstd_with_window(MAX_ZSTD_WINDOW_LOG + 1, &decoded);
        assert!(compressed.len() < 1024);
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .header(GRPC_ENCODING_HEADER, "zstd")
            .body(Body::new(Full::new(grpc_frame(1, &compressed))))
            .unwrap();
        let error = transform_request(request, decoded.len() + 1, 1024)
            .into_body()
            .frame()
            .await
            .expect("transformed body should report the oversized zstd window")
            .expect_err("zstd frame above the decoder-window cap must not pass through");
        assert_eq!(error.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn malformed_compression_flag_reaches_tonic_as_a_stream_error() {
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .body(Body::new(Full::new(grpc_frame(2, b"body"))))
            .unwrap();
        let error = transform_request(request, 16, 64)
            .into_body()
            .frame()
            .await
            .expect("transformed body should report the malformed envelope")
            .expect_err("an invalid compression flag must not reach protobuf decoding");
        assert_eq!(error.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn forwards_trailers_after_transformed_request_data() {
        let mut trailers = http::HeaderMap::new();
        trailers.insert("x-test-trailer", "present".parse().unwrap());
        let source = StreamBody::new(stream::iter(vec![
            Ok::<_, Infallible>(Frame::data(grpc_frame(1, &gzip(b"body")))),
            Ok(Frame::trailers(trailers.clone())),
        ]));
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .header(GRPC_ENCODING_HEADER, "gzip")
            .body(source)
            .unwrap();
        let mut body = transform_request(request, 16, 64).into_body();

        let data = body
            .frame()
            .await
            .expect("transformed data frame")
            .expect("successful transformed data")
            .into_data()
            .expect("data frame");
        assert_eq!(data, grpc_frame(0, b"body"));
        let received_trailers = body
            .frame()
            .await
            .expect("forwarded trailers")
            .expect("successful trailers")
            .into_trailers()
            .expect("trailer frame");
        assert_eq!(received_trailers, trailers);
        assert!(body.frame().await.is_none());
    }

    #[tokio::test]
    async fn forwarding_task_stops_when_tonic_drops_the_replacement_body() {
        // Hold the source body open after its first message. The forwarding task must watch the
        // replacement receiver closing rather than waiting forever for a second DATA frame.
        let source = StreamBody::new(
            stream::once(async { Ok::<_, Infallible>(Frame::data(grpc_frame(0, b"first"))) })
                .chain(stream::pending()),
        );
        let (tx, mut rx) = mpsc::channel(1);
        let forwarding = tokio::spawn(forward_frames(source, tx, None, 16, 64, ()));
        let first = rx
            .recv()
            .await
            .expect("forwarding task should emit the first request message")
            .expect("first message should transform successfully")
            .into_data()
            .expect("first output is data");
        assert_eq!(first, grpc_frame(0, b"first"));
        drop(rx);
        tokio::time::timeout(Duration::from_secs(1), forwarding)
            .await
            .expect("forwarding task must observe receiver cancellation")
            .expect("forwarding task should not panic");
    }

    #[tokio::test]
    async fn compressed_decoder_waits_for_the_shared_limiter() {
        // A compressed frame must not create another decoder window while an earlier stream owns
        // the shared permit. This is the aggregation bound for many concurrent GetPages streams.
        let limiter = Arc::new(Semaphore::new(1));
        let decoder =
            GrpcFrameDecoder::new(Some(super::Encoding::Gzip), 16, 64, limiter.clone(), ());
        let held = limiter.clone().acquire_owned().await.unwrap();
        let (tx, _rx) = mpsc::channel(1);

        assert!(
            tokio::time::timeout(
                Duration::from_millis(10),
                decoder.acquire_decompression_permit(&tx),
            )
            .await
            .is_err()
        );

        drop(held);
        let permit = tokio::time::timeout(
            Duration::from_secs(1),
            decoder.acquire_decompression_permit(&tx),
        )
        .await
        .expect("shared decoder permit should become available")
        .expect("limiter should remain open");
        drop(permit);
    }

    #[tokio::test]
    async fn unsupported_non_utf8_encoding_is_left_for_tonic() {
        // The wrapper only owns gzip/zstd and identity. Preserve even a malformed metadata value
        // so Tonic can return its established unsupported-encoding response.
        let original = grpc_frame(0, b"body");
        let request = http::Request::builder()
            .uri(GET_PAGES_PATH)
            .header(
                GRPC_ENCODING_HEADER,
                http::HeaderValue::from_bytes(b"\xff").unwrap(),
            )
            .body(Body::new(Full::new(original.clone())))
            .unwrap();
        let response = transform_request(request, 1, 1);
        assert_eq!(
            response
                .headers()
                .get(GRPC_ENCODING_HEADER)
                .unwrap()
                .as_bytes(),
            b"\xff"
        );
        assert_eq!(
            response
                .into_body()
                .collect()
                .await
                .expect("unsupported request body is passed through")
                .to_bytes(),
            original
        );
    }
}
