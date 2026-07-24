//! Exercises the public GetPages precondition for a late-chunk error probe.
//!
//! A cache-warm 128-block preflight establishes that every requested block is
//! valid before the companion integration test enables a final-internal-chunk
//! failpoint. The preflight retries the baseline's existing oversized request
//! in 32-block frames, while a bounded-frame implementation completes directly.

use std::time::Instant;

use anyhow::{Context, bail};
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use pageserver_page_api as page_api;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use utils::shard::ShardIndex;

const PAGE_SIZE: usize = 8192;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Preflight,
    Classify,
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
    #[arg(long)]
    frame_cap: usize,
    #[arg(long)]
    fallback_chunk_size: usize,
    #[arg(long, value_enum)]
    mode: Mode,
}

fn request(
    args: &Args,
    rel: page_api::RelTag,
    request_id: u64,
    block_numbers: Vec<u32>,
) -> page_api::GetPageRequest {
    page_api::GetPageRequest {
        request_id: page_api::RequestID::new(request_id),
        request_class: page_api::GetPageClass::Normal,
        read_lsn: page_api::ReadLsn {
            request_lsn: args.read_lsn,
            not_modified_since_lsn: Some(args.read_lsn),
        },
        rel,
        block_numbers,
    }
}

async fn send_and_receive<S>(
    request_tx: &mpsc::Sender<page_api::GetPageRequest>,
    responses: &mut S,
    request: page_api::GetPageRequest,
) -> anyhow::Result<page_api::GetPageResponse>
where
    S: futures::Stream<Item = tonic::Result<page_api::GetPageResponse>> + Unpin,
{
    let request_id = request.request_id;
    request_tx
        .send(request)
        .await
        .context("send GetPages request")?;
    let response = responses
        .next()
        .await
        .context("GetPages stream ended before its response")?
        .context("receive GetPages response")?;
    if response.request_id != request_id {
        bail!(
            "GetPages response ID {} did not match request ID {}",
            response.request_id,
            request_id
        );
    }
    Ok(response)
}

fn validate_success(
    response: &page_api::GetPageResponse,
    expected_blocks: &[u32],
) -> anyhow::Result<()> {
    if response.status_code != page_api::GetPageStatusCode::Ok
        || response.pages.len() != expected_blocks.len()
    {
        bail!("GetPages frame did not return the expected successful response: {response:?}");
    }
    for (&expected_block, page) in expected_blocks.iter().zip(&response.pages) {
        if page.block_number != expected_block || page.image.len() != PAGE_SIZE {
            bail!("GetPages frame returned a malformed page for block {expected_block}");
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.frame_cap == 0 || args.fallback_chunk_size == 0 {
        bail!("frame_cap and fallback_chunk_size must be nonzero");
    }
    let frame_cap = u32::try_from(args.frame_cap).context("frame_cap does not fit in u32")?;

    let mut client = page_api::Client::connect(
        args.endpoint.clone(),
        args.tenant_id,
        args.timeline_id,
        ShardIndex::unsharded(),
        None,
        None,
    )
    .await
    .context("connect GetPages gRPC client")?;
    let (request_tx, request_rx) = mpsc::channel(1);
    let mut responses = client
        .get_pages(ReceiverStream::new(request_rx))
        .await
        .context("open GetPages stream")?;
    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };
    let expected_blocks: Vec<u32> = (0..frame_cap).collect();

    match args.mode {
        Mode::Preflight => {
            let started = Instant::now();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, 1, expected_blocks.clone()),
            )
            .await?;
            let (outcome, request_messages, oversized_fallbacks) = if response.status_code
                == page_api::GetPageStatusCode::Ok
            {
                validate_success(&response, &expected_blocks)?;
                ("direct_completed", 1, 0)
            } else if response.status_code == page_api::GetPageStatusCode::InternalError
                && response.reason.as_deref() == Some("Read error")
                && response.pages.is_empty()
            {
                for (index, chunk) in expected_blocks.chunks(args.fallback_chunk_size).enumerate() {
                    let chunk_response = send_and_receive(
                        &request_tx,
                        &mut responses,
                        request(
                            &args,
                            rel,
                            u64::try_from(index + 2)
                                .context("too many fallback GetPages requests")?,
                            chunk.to_vec(),
                        ),
                    )
                    .await?;
                    validate_success(&chunk_response, chunk)?;
                }
                (
                    "legacy_fallback_completed",
                    1 + expected_blocks.chunks(args.fallback_chunk_size).count(),
                    1,
                )
            } else {
                bail!("preflight GetPages frame unexpectedly failed: {response:?}");
            };
            println!(
                "{}",
                json!({
                    "mode": "preflight",
                    "outcome": outcome,
                    "elapsed_ns": started.elapsed().as_nanos(),
                    "returned_pages": expected_blocks.len(),
                    "request_messages": request_messages,
                    "oversized_fallbacks": oversized_fallbacks,
                })
            );
        }
        Mode::Classify => {
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, 1, expected_blocks),
            )
            .await?;
            if !response.pages.is_empty() {
                bail!(
                    "late-chunk error response exposed {} partial pages",
                    response.pages.len()
                );
            }
            let outcome = match (response.status_code, response.reason.as_deref()) {
                // The baseline reaches this error before it can have four chunks.
                (page_api::GetPageStatusCode::InternalError, Some("Read error")) => {
                    "legacy_internal"
                }
                (
                    page_api::GetPageStatusCode::InternalError,
                    Some("injected final internal GetPages batch error"),
                ) => "final_chunk_injected_error",
                _ => bail!(
                    "late-chunk request returned an unexpected status or reason: {response:?}"
                ),
            };
            println!(
                "{}",
                json!({
                    "mode": "classify",
                    "outcome": outcome,
                    "response_pages": response.pages.len(),
                    "request_id_matches": true,
                    "non_ok_response": response.status_code != page_api::GetPageStatusCode::Ok,
                })
            );
        }
    }

    Ok(())
}
