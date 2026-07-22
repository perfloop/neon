//! Verifies one fixed-size gRPC GetPages frame against a real Pageserver.
//!
//! This is intentionally a narrow integration-test helper. The Python test owns
//! environment setup and metric collection; this binary owns the wire request
//! and validates the response's request ID, page order, and page images.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{Context, bail};
use clap::Parser;
use futures::StreamExt;
use pageserver_page_api as page_api;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use utils::shard::ShardIndex;

const PAGE_SIZE: usize = 8192;

#[derive(Debug, Parser)]
struct Args {
    /// gRPC endpoint, for example http://127.0.0.1:64000.
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
    frame_size: usize,

    /// A client-side fallback used only when the server returns its known
    /// oversized-vectored-read error. It makes the baseline testable while
    /// still requiring the server to return all requested pages.
    #[arg(long, default_value_t = 32)]
    fallback_chunk_size: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.frame_size == 0 {
        bail!("frame_size must be nonzero");
    }
    if args.fallback_chunk_size == 0 {
        bail!("fallback_chunk_size must be nonzero");
    }

    let last_block = args
        .start_block
        .checked_add(
            u32::try_from(args.frame_size - 1).context("frame_size does not fit in a u32")?,
        )
        .context("requested frame overflows the block-number range")?;
    let requested_blocks: Vec<u32> = (args.start_block..=last_block).collect();

    let mut client = page_api::Client::connect(
        args.endpoint,
        args.tenant_id,
        args.timeline_id,
        ShardIndex::unsharded(),
        None,
        None,
    )
    .await
    .context("connect gRPC GetPages client")?;
    let (request_tx, request_rx) = mpsc::channel(1);
    let mut responses = client
        .get_pages(ReceiverStream::new(request_rx))
        .await
        .context("open gRPC GetPages stream")?;

    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };
    let started = Instant::now();
    let mut pending = VecDeque::from([requested_blocks]);
    let mut returned_blocks = Vec::with_capacity(args.frame_size);
    let mut attempts = 0usize;
    let mut oversized_fallbacks = 0usize;

    while let Some(block_numbers) = pending.pop_front() {
        let request_id = page_api::RequestID::new(
            u64::try_from(attempts + 1).context("too many gRPC requests")?,
        );
        attempts += 1;
        request_tx
            .send(page_api::GetPageRequest {
                request_id,
                request_class: page_api::GetPageClass::Normal,
                read_lsn: page_api::ReadLsn {
                    request_lsn: args.read_lsn,
                    not_modified_since_lsn: Some(args.read_lsn),
                },
                rel,
                block_numbers: block_numbers.clone(),
            })
            .await
            .context("send gRPC GetPages request")?;

        let response = responses
            .next()
            .await
            .context("GetPages stream ended before its response")?
            .context("receive gRPC GetPages response")?;
        if response.request_id != request_id {
            bail!(
                "GetPages response ID {} did not match request ID {}",
                response.request_id,
                request_id
            );
        }

        if response.status_code != page_api::GetPageStatusCode::Ok {
            let reason = response.reason.unwrap_or_default();
            if reason.contains("batching oversized")
                && block_numbers.len() > args.fallback_chunk_size
            {
                oversized_fallbacks += 1;
                pending.extend(
                    block_numbers
                        .chunks(args.fallback_chunk_size)
                        .map(ToOwned::to_owned),
                );
                continue;
            }
            bail!(
                "GetPages request for {} blocks failed with {}: {reason}",
                block_numbers.len(),
                response.status_code
            );
        }

        let response_blocks: Vec<u32> = response
            .pages
            .iter()
            .map(|page| page.block_number)
            .collect();
        if response_blocks != block_numbers {
            bail!(
                "GetPages response block order did not match its request: expected {block_numbers:?}, got {response_blocks:?}"
            );
        }
        for page in &response.pages {
            if page.image.len() != PAGE_SIZE {
                bail!(
                    "GetPages response for block {} had {} bytes, expected {PAGE_SIZE}",
                    page.block_number,
                    page.image.len()
                );
            }
        }
        returned_blocks.extend(response_blocks);
    }

    let expected_blocks: Vec<u32> = (args.start_block..=last_block).collect();
    if returned_blocks != expected_blocks {
        bail!(
            "GetPages frame response order did not match its original request: expected {expected_blocks:?}, got {returned_blocks:?}"
        );
    }

    println!(
        "{}",
        json!({
            "requested_pages": args.frame_size,
            "returned_pages": returned_blocks.len(),
            "grpc_request_messages": attempts,
            "oversized_fallbacks": oversized_fallbacks,
            "elapsed_ns": started.elapsed().as_nanos(),
        })
    );
    Ok(())
}
