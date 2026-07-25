use anyhow::{Context, bail};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use flate2::Compression;
use flate2::write::GzEncoder;
use futures::StreamExt;
use pageserver_page_api as page_api;
use prost::Message;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::codec::CompressionEncoding;
use tonic::metadata::AsciiMetadataValue;
use tonic::transport::Endpoint;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use utils::shard::{ShardCount, ShardIndex, ShardNumber};

const PAGE_SIZE: usize = 8192;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Bounded,
    Ingress,
    Probe,
    Raw,
    Suffix,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum IngressCompression {
    Identity,
    Gzip,
    Zstd,
}

impl IngressCompression {
    fn name(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Gzip => "gzip",
            Self::Zstd => "zstd",
        }
    }
}

#[derive(Debug, Parser)]
struct Args {
    #[arg(long)]
    endpoint: String,
    #[arg(long)]
    tenant_id: TenantId,
    #[arg(long)]
    timeline_id: TimelineId,
    #[arg(long)]
    read_lsn: Lsn,
    #[arg(long)]
    spcnode: u32,
    #[arg(long)]
    dbnode: u32,
    #[arg(long)]
    relnode: u32,
    #[arg(long, default_value_t = 0)]
    start_block: u32,
    #[arg(long)]
    repeat_block: Option<u32>,
    #[arg(long)]
    suffix_block: Option<u32>,
    #[arg(long, default_value_t = 0)]
    shard_number: u8,
    #[arg(long, default_value_t = 0)]
    shard_count: u8,
    #[arg(long)]
    frame_size: usize,
    #[arg(long, default_value_t = 32)]
    fallback_chunk_size: usize,
    #[arg(long, value_enum, default_value_t = Mode::Raw)]
    mode: Mode,
    #[arg(long, value_enum, default_value_t = IngressCompression::Identity)]
    ingress_compression: IngressCompression,
}

fn blocks(args: &Args) -> anyhow::Result<Vec<u32>> {
    if args.frame_size == 0 || args.fallback_chunk_size == 0 {
        bail!("frame_size and fallback_chunk_size must be nonzero");
    }
    let count = u32::try_from(args.frame_size).context("frame_size does not fit in u32")?;
    let mut blocks = if let Some(block) = args.repeat_block {
        vec![block; args.frame_size]
    } else {
        let end = args
            .start_block
            .checked_add(count)
            .context("requested frame overflows the block-number range")?;
        (args.start_block..end).collect()
    };
    if let Some(block) = args.suffix_block {
        blocks.push(block);
    }
    Ok(blocks)
}

fn request(
    args: &Args,
    rel: page_api::RelTag,
    id: u64,
    blocks: Vec<u32>,
) -> page_api::GetPageRequest {
    page_api::GetPageRequest {
        request_id: page_api::RequestID::new(id),
        request_class: page_api::GetPageClass::Normal,
        read_lsn: page_api::ReadLsn {
            request_lsn: args.read_lsn,
            not_modified_since_lsn: Some(args.read_lsn),
        },
        rel,
        block_numbers: blocks,
    }
}

async fn send<S>(
    tx: &mpsc::Sender<page_api::GetPageRequest>,
    responses: &mut S,
    request: page_api::GetPageRequest,
) -> anyhow::Result<page_api::GetPageResponse>
where
    S: futures::Stream<Item = tonic::Result<page_api::GetPageResponse>> + Unpin,
{
    let id = request.request_id;
    tx.send(request).await.context("send GetPages request")?;
    let response = responses
        .next()
        .await
        .context("GetPages stream ended before its response")?
        .context("receive GetPages response")?;
    if response.request_id != id {
        bail!(
            "GetPages response ID {} did not match request ID {}",
            response.request_id,
            id
        );
    }
    Ok(response)
}

fn status(response: &page_api::GetPageResponse) -> &'static str {
    match response.status_code {
        page_api::GetPageStatusCode::Ok => "ok",
        page_api::GetPageStatusCode::InternalError => "internal_error",
        page_api::GetPageStatusCode::InvalidRequest => "invalid_request",
        _ => "other",
    }
}

fn validate(response: &page_api::GetPageResponse, expected: &[u32]) -> anyhow::Result<()> {
    if response.status_code != page_api::GetPageStatusCode::Ok
        || response.pages.len() != expected.len()
    {
        bail!("GetPages response was not a successful full frame: {response:?}");
    }
    for (&block, page) in expected.iter().zip(&response.pages) {
        if page.block_number != block || page.image.len() != PAGE_SIZE {
            bail!("GetPages response was malformed for block {block}");
        }
    }
    Ok(())
}

fn verify_images(
    response: &page_api::GetPageResponse,
    expected: &[u32],
    references: &BTreeMap<u32, Bytes>,
) -> anyhow::Result<usize> {
    validate(response, expected)?;
    let mut mismatches = 0;
    for (&block, page) in expected.iter().zip(&response.pages) {
        let reference = references
            .get(&block)
            .context("GetPages reference image was missing")?;
        if page.image.as_ref() != reference.as_ref() {
            mismatches += 1;
        }
    }
    if mismatches != 0 {
        bail!("GetPages frame image bytes differed from independent references");
    }
    Ok(mismatches)
}

async fn get_references<S>(
    tx: &mpsc::Sender<page_api::GetPageRequest>,
    responses: &mut S,
    args: &Args,
    rel: page_api::RelTag,
    blocks: &[u32],
    next_request_id: &mut u64,
) -> anyhow::Result<BTreeMap<u32, Bytes>>
where
    S: futures::Stream<Item = tonic::Result<page_api::GetPageResponse>> + Unpin,
{
    let mut references = BTreeMap::new();
    for &block in blocks {
        if references.contains_key(&block) {
            continue;
        }
        let reference = send(
            tx,
            responses,
            request(args, rel, *next_request_id, vec![block]),
        )
        .await?;
        validate(&reference, &[block])?;
        references.insert(block, reference.pages[0].image.clone());
        *next_request_id = next_request_id
            .checked_add(1)
            .context("GetPages reference request ID overflow")?;
    }
    Ok(references)
}

fn emit(
    response: &page_api::GetPageResponse,
    mismatches: usize,
    reference_pages: usize,
    request_elapsed_ns: u128,
) {
    println!(
        "{}",
        json!({
            "status": status(response),
            "reason": response.reason.as_deref(),
            "response_pages": response.pages.len(),
            "request_id_matches": true,
            "image_byte_mismatches": mismatches,
            "reference_pages": reference_pages,
            "request_elapsed_ns": request_elapsed_ns,
        })
    );
}

const POST_DECODE_PREFIX: &str = "injected GetPages post-decode frame: ";

fn tonic_status_name(code: tonic::Code) -> &'static str {
    match code {
        tonic::Code::Ok => "ok",
        tonic::Code::Cancelled => "cancelled",
        tonic::Code::Unknown => "unknown",
        tonic::Code::InvalidArgument => "invalid_argument",
        tonic::Code::DeadlineExceeded => "deadline_exceeded",
        tonic::Code::NotFound => "not_found",
        tonic::Code::AlreadyExists => "already_exists",
        tonic::Code::PermissionDenied => "permission_denied",
        tonic::Code::ResourceExhausted => "resource_exhausted",
        tonic::Code::FailedPrecondition => "failed_precondition",
        tonic::Code::Aborted => "aborted",
        tonic::Code::OutOfRange => "out_of_range",
        tonic::Code::Unimplemented => "unimplemented",
        tonic::Code::Internal => "internal",
        tonic::Code::Unavailable => "unavailable",
        tonic::Code::DataLoss => "data_loss",
        tonic::Code::Unauthenticated => "unauthenticated",
    }
}

fn compressed_payload_bytes(
    request: &page_api::proto::GetPageRequest,
    compression: IngressCompression,
) -> anyhow::Result<usize> {
    let encoded = request.encode_to_vec();
    Ok(match compression {
        IngressCompression::Identity => encoded.len(),
        IngressCompression::Gzip => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(&encoded)?;
            encoder.finish()?.len()
        }
        IngressCompression::Zstd => zstd::stream::encode_all(encoded.as_slice(), 0)?.len(),
    })
}

fn post_decode_allocation(reason: Option<&str>) -> Option<(usize, usize)> {
    let reason = reason?.strip_prefix(POST_DECODE_PREFIX)?;
    let (blocks, capacity_bytes) = reason.split_once(", capacity_bytes=")?;
    Some((
        blocks.strip_prefix("blocks=")?.parse().ok()?,
        capacity_bytes.parse().ok()?,
    ))
}

async fn run_ingress(args: &Args, rel: page_api::RelTag, blocks: Vec<u32>) -> anyhow::Result<()> {
    let frame_size = blocks.len();
    let request = page_api::proto::GetPageRequest::from(request(args, rel, 1, blocks));
    let uncompressed_proto_bytes = request.encoded_len();
    let compressed_payload_bytes = compressed_payload_bytes(&request, args.ingress_compression)?;
    let tenant_id: AsciiMetadataValue = args.tenant_id.to_string().try_into()?;
    let timeline_id: AsciiMetadataValue = args.timeline_id.to_string().try_into()?;
    let shard_id: AsciiMetadataValue =
        ShardIndex::new(ShardNumber(args.shard_number), ShardCount(args.shard_count))
            .to_string()
            .try_into()?;
    let channel = Endpoint::from_shared(args.endpoint.clone())
        .context("invalid GetPages endpoint")?
        .connect()
        .await
        .context("connect raw GetPages client")?;
    let mut client = page_api::proto::PageServiceClient::with_interceptor(
        channel,
        move |mut request: tonic::Request<()>| {
            let metadata = request.metadata_mut();
            metadata.insert("neon-tenant-id", tenant_id.clone());
            metadata.insert("neon-timeline-id", timeline_id.clone());
            metadata.insert("neon-shard-id", shard_id.clone());
            Ok(request)
        },
    );
    match args.ingress_compression {
        IngressCompression::Identity => {}
        IngressCompression::Gzip => {
            client = client.send_compressed(CompressionEncoding::Gzip);
        }
        IngressCompression::Zstd => {
            client = client.send_compressed(CompressionEncoding::Zstd);
        }
    }

    let (tx, rx) = mpsc::channel(1);
    let started = Instant::now();
    let mut response_status = None;
    let mut response_reason = None;
    let mut response_pages = None;
    let mut transport_status = None;
    let mut transport_message = None;
    let mut post_decode_blocks = None;
    let mut decoded_block_numbers_capacity_bytes = None;
    match client.get_pages(ReceiverStream::new(rx)).await {
        Ok(response) => {
            tx.send(request)
                .await
                .context("send raw GetPages request")?;
            match response.into_inner().next().await {
                Some(Ok(response)) => {
                    let response: page_api::GetPageResponse = response
                        .try_into()
                        .context("decode raw GetPages response")?;
                    response_status = Some(status(&response));
                    response_reason = response.reason.clone();
                    response_pages = Some(response.pages.len());
                    if let Some((blocks, capacity_bytes)) =
                        post_decode_allocation(response.reason.as_deref())
                    {
                        post_decode_blocks = Some(blocks);
                        decoded_block_numbers_capacity_bytes = Some(capacity_bytes);
                    }
                }
                Some(Err(status)) => {
                    transport_status = Some(tonic_status_name(status.code()));
                    transport_message = Some(status.message().to_owned());
                }
                None => {
                    transport_status = Some("stream_ended");
                }
            }
        }
        Err(status) => {
            transport_status = Some(tonic_status_name(status.code()));
            transport_message = Some(status.message().to_owned());
        }
    }

    println!(
        "{}",
        json!({
            "compression": args.ingress_compression.name(),
            "frame_size": frame_size,
            "uncompressed_proto_bytes": uncompressed_proto_bytes,
            "compressed_payload_bytes": compressed_payload_bytes,
            "grpc_frame_bytes": compressed_payload_bytes + 5,
            "request_elapsed_ns": started.elapsed().as_nanos(),
            "response_status": response_status,
            "response_reason": response_reason,
            "response_pages": response_pages,
            "transport_status": transport_status,
            "transport_message": transport_message,
            "post_decode_reached": post_decode_blocks.is_some(),
            "post_decode_blocks": post_decode_blocks,
            "decoded_block_numbers_capacity_bytes": decoded_block_numbers_capacity_bytes,
        })
    );
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let expected = blocks(&args)?;
    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };
    if matches!(args.mode, Mode::Ingress) {
        return run_ingress(&args, rel, expected).await;
    }

    let mut client = page_api::Client::connect(
        args.endpoint.clone(),
        args.tenant_id,
        args.timeline_id,
        ShardIndex::new(ShardNumber(args.shard_number), ShardCount(args.shard_count)),
        None,
        None,
    )
    .await
    .context("connect GetPages client")?;
    let (tx, rx) = mpsc::channel(1);
    let mut responses = client
        .get_pages(ReceiverStream::new(rx))
        .await
        .context("open GetPages stream")?;

    match args.mode {
        Mode::Ingress => unreachable!("ingress mode returns before opening the typed client"),
        Mode::Bounded => {
            if expected.len() <= args.fallback_chunk_size {
                bail!("bounded mode needs an over-cap frame");
            }
            let normal = expected[..args.fallback_chunk_size].to_vec();
            let mut next_request_id = 1_u64;
            let references = get_references(
                &tx,
                &mut responses,
                &args,
                rel,
                &normal,
                &mut next_request_id,
            )
            .await?;
            let oversized = send(
                &tx,
                &mut responses,
                request(&args, rel, next_request_id, expected.clone()),
            )
            .await?;
            next_request_id = next_request_id
                .checked_add(1)
                .context("GetPages bounded request ID overflow")?;
            if oversized.status_code != page_api::GetPageStatusCode::Ok
                && !oversized.pages.is_empty()
            {
                bail!("non-OK GetPages response contained pages: {oversized:?}");
            }
            let following = send(
                &tx,
                &mut responses,
                request(&args, rel, next_request_id, normal.clone()),
            )
            .await?;
            let mismatches = verify_images(&following, &normal, &references)?;
            println!(
                "{}",
                json!({
                    "oversized_status": status(&oversized),
                    "oversized_reason": oversized.reason,
                    "oversized_pages": oversized.pages.len(),
                    "following_pages": following.pages.len(),
                    "following_image_byte_mismatches": mismatches,
                    "reference_pages": references.len(),
                })
            );
        }
        Mode::Probe => {
            let started = Instant::now();
            let response = send(
                &tx,
                &mut responses,
                request(&args, rel, 1, expected.clone()),
            )
            .await?;
            let request_elapsed_ns = started.elapsed().as_nanos();
            if response.status_code == page_api::GetPageStatusCode::Ok {
                validate(&response, &expected)?;
            }
            emit(&response, 0, 0, request_elapsed_ns);
        }
        Mode::Raw => {
            // Obtain one independent single-page image for each distinct requested block before
            // issuing the public frame. The resulting map also checks every repeated entry.
            let mut next_request_id = 1_u64;
            let references = get_references(
                &tx,
                &mut responses,
                &args,
                rel,
                &expected,
                &mut next_request_id,
            )
            .await?;
            let started = Instant::now();
            let response = send(
                &tx,
                &mut responses,
                request(&args, rel, next_request_id, expected.clone()),
            )
            .await?;
            let request_elapsed_ns = started.elapsed().as_nanos();
            let mismatches = if response.status_code == page_api::GetPageStatusCode::Ok {
                verify_images(&response, &expected, &references)?
            } else {
                0
            };
            emit(&response, mismatches, references.len(), request_elapsed_ns);
        }
        Mode::Suffix => {
            let suffix = args
                .suffix_block
                .context("suffix mode needs --suffix-block")?;
            let prefix = expected
                .strip_suffix(&[suffix])
                .context("suffix mode must end with --suffix-block")?
                .to_vec();
            if prefix.is_empty() {
                bail!("suffix mode needs a nonempty local prefix");
            }
            let mut next_request_id = 1_u64;
            let references = get_references(
                &tx,
                &mut responses,
                &args,
                rel,
                &prefix,
                &mut next_request_id,
            )
            .await?;
            let prefixed = send(
                &tx,
                &mut responses,
                request(&args, rel, next_request_id, expected.clone()),
            )
            .await?;
            next_request_id = next_request_id
                .checked_add(1)
                .context("GetPages suffix request ID overflow")?;
            if prefixed.status_code != page_api::GetPageStatusCode::Ok && !prefixed.pages.is_empty()
            {
                bail!("non-OK GetPages response contained pages: {prefixed:?}");
            }
            let following = send(
                &tx,
                &mut responses,
                request(&args, rel, next_request_id, prefix.clone()),
            )
            .await?;
            let mismatches = verify_images(&following, &prefix, &references)?;
            println!(
                "{}",
                json!({
                    "prefixed_status": status(&prefixed),
                    "prefixed_reason": prefixed.reason,
                    "prefixed_pages": prefixed.pages.len(),
                    "following_pages": following.pages.len(),
                    "following_image_byte_mismatches": mismatches,
                    "reference_pages": references.len(),
                })
            );
        }
    }

    Ok(())
}
