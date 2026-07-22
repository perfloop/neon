//! Measures stale-parent GetPages frame costs through the public gRPC API.
//!
//! The companion integration test first verifies the at-cap response against
//! independent one-page reads, outside its timed counter brackets. It then uses
//! this helper to issue exactly one 128- or 129-block wire frame through a
//! removed parent shard, so timer and vectored-read deltas describe only that
//! frame rather than the reference reads.

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
    VerifyAtCap,
    AtCap,
    OverCap,
}

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

    /// The aggregate frame bound: four configured vectored-read chunks.
    #[arg(long)]
    frame_cap: usize,

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
        .context("send stale-parent GetPages request")?;
    let response = responses
        .next()
        .await
        .context("GetPages stream ended before its response")?
        .context("receive stale-parent GetPages response")?;
    if response.request_id != request_id {
        bail!(
            "GetPages response ID {} did not match request ID {}",
            response.request_id,
            request_id
        );
    }
    Ok(response)
}

fn validate_page_sequence(
    response: &page_api::GetPageResponse,
    expected_blocks: &[u32],
) -> anyhow::Result<()> {
    if response.status_code != page_api::GetPageStatusCode::Ok
        || response.pages.len() != expected_blocks.len()
    {
        bail!("GetPages frame did not return the expected successful cardinality: {response:?}");
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
    if args.frame_cap == 0 {
        bail!("frame_cap must be nonzero");
    }
    let frame_cap_u32 = u32::try_from(args.frame_cap).context("frame_cap does not fit in u32")?;
    let over_cap = args
        .frame_cap
        .checked_add(1)
        .context("frame_cap overflows the block-number range")?;
    let over_cap_u32 = u32::try_from(over_cap).context("frame_cap + 1 does not fit in u32")?;

    // The test sends to the removed unsharded parent after a completed split.
    // That makes every request below exercise maybe_split_get_page's reroute.
    let mut client = page_api::Client::connect(
        args.endpoint.clone(),
        args.tenant_id,
        args.timeline_id,
        ShardIndex::unsharded(),
        None,
        None,
    )
    .await
    .context("connect stale-parent gRPC GetPages client")?;
    let (request_tx, request_rx) = mpsc::channel(1);
    let mut responses = client
        .get_pages(ReceiverStream::new(request_rx))
        .await
        .context("open stale-parent GetPages stream")?;
    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };

    match args.mode {
        Mode::VerifyAtCap => {
            let mut references = Vec::with_capacity(args.frame_cap);
            for block_number in 0..frame_cap_u32 {
                let response = send_and_receive(
                    &request_tx,
                    &mut responses,
                    request(&args, rel, u64::from(block_number) + 1, vec![block_number]),
                )
                .await?;
                validate_page_sequence(&response, &[block_number])?;
                references.push((block_number, response.pages[0].image.clone()));
            }

            let expected_blocks: Vec<u32> = (0..frame_cap_u32).collect();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(
                    &args,
                    rel,
                    u64::from(frame_cap_u32) + 1,
                    expected_blocks.clone(),
                ),
            )
            .await?;
            validate_page_sequence(&response, &expected_blocks)?;
            for ((expected_block, expected_image), page) in references.iter().zip(&response.pages) {
                if page.block_number != *expected_block || page.image != *expected_image {
                    bail!(
                        "at-cap stale-parent frame did not preserve the image for block {expected_block}"
                    );
                }
            }
            println!(
                "{}",
                json!({
                    "mode": "verify_at_cap",
                    "outcome": "completed",
                    "image_byte_mismatches": 0,
                })
            );
        }
        Mode::AtCap => {
            let expected_blocks: Vec<u32> = (0..frame_cap_u32).collect();
            let started = Instant::now();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, 1, expected_blocks.clone()),
            )
            .await?;
            let elapsed_ns = started.elapsed().as_nanos();
            validate_page_sequence(&response, &expected_blocks)?;
            println!(
                "{}",
                json!({
                    "mode": "at_cap",
                    "outcome": "completed",
                    "response_pages": response.pages.len(),
                    "elapsed_ns": elapsed_ns,
                })
            );
        }
        Mode::OverCap => {
            let expected_blocks: Vec<u32> = (0..over_cap_u32).collect();
            let started = Instant::now();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, 1, expected_blocks.clone()),
            )
            .await?;
            let elapsed_ns = started.elapsed().as_nanos();
            let outcome = match response.status_code {
                page_api::GetPageStatusCode::InvalidRequest => {
                    let expected_reason = format!(
                        "GetPages request has {over_cap} blocks, limit is {}",
                        args.frame_cap
                    );
                    if response.reason.as_deref() != Some(expected_reason.as_str())
                        || !response.pages.is_empty()
                    {
                        bail!(
                            "over-cap stale-parent frame did not return the expected empty InvalidRequest response: {response:?}"
                        );
                    }
                    "invalid_request"
                }
                page_api::GetPageStatusCode::Ok => {
                    // The pre-change implementation admits each locally in-cap
                    // child partition. Keep this baseline branch distinct from
                    // the candidate's aggregate InvalidRequest response.
                    validate_page_sequence(&response, &expected_blocks)?;
                    "legacy_accepted"
                }
                _ => bail!("over-cap stale-parent frame unexpectedly failed: {response:?}"),
            };
            println!(
                "{}",
                json!({
                    "mode": "over_cap",
                    "outcome": outcome,
                    "response_pages": response.pages.len(),
                    "elapsed_ns": elapsed_ns,
                })
            );
        }
    }

    Ok(())
}
