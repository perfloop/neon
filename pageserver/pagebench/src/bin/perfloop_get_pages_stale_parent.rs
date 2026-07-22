//! Exercises GetPages through a removed parent shard after a real shard split.
//!
//! The integration test owns shard creation, split completion, and metric collection.
//! This helper sends public gRPC requests with the former parent shard metadata,
//! then verifies the response order and image bytes against independent single-page
//! requests at the same LSN.

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
    Ok(response)
}

fn assert_legacy_internal(
    response: &page_api::GetPageResponse,
    frame_size: usize,
) -> anyhow::Result<()> {
    if response.status_code != page_api::GetPageStatusCode::InternalError
        || response.reason.as_deref() != Some("Read error")
        || !response.pages.is_empty()
    {
        bail!(
            "GetPages request for {frame_size} blocks did not return the expected legacy empty InternalError response: {response:?}"
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.frame_cap == 0 {
        bail!("frame_cap must be nonzero");
    }
    let over_cap = args
        .frame_cap
        .checked_add(1)
        .context("frame_cap overflows the block-number range")?;
    let frame_cap_u32 = u32::try_from(args.frame_cap).context("frame_cap does not fit in u32")?;
    let over_cap_u32 = u32::try_from(over_cap).context("frame_cap + 1 does not fit in u32")?;

    // The test intentionally directs requests to the pre-split parent shard. A
    // completed split has removed that shard, so this takes maybe_split_get_page.
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
        .context("open stale-parent gRPC GetPages stream")?;
    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };

    match args.mode {
        Mode::AtCap => {
            let mut references = Vec::with_capacity(args.frame_cap);
            for block_number in 0..frame_cap_u32 {
                let response = send_and_receive(
                    &request_tx,
                    &mut responses,
                    request(&args, rel, u64::from(block_number) + 1, vec![block_number]),
                )
                .await?;
                if response.status_code != page_api::GetPageStatusCode::Ok
                    || response.pages.len() != 1
                {
                    bail!(
                        "stale-parent reference request for block {block_number} did not return one page: {response:?}"
                    );
                }
                let page = &response.pages[0];
                if page.block_number != block_number || page.image.len() != PAGE_SIZE {
                    bail!(
                        "stale-parent reference for block {block_number} was malformed: {page:?}"
                    );
                }
                references.push((block_number, page.image.clone()));
            }

            let block_numbers: Vec<u32> = (0..frame_cap_u32).collect();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, u64::from(frame_cap_u32) + 1, block_numbers),
            )
            .await?;
            let direct_completion = match response.status_code {
                page_api::GetPageStatusCode::Ok => {
                    if response.pages.len() != references.len() {
                        bail!(
                            "at-cap stale-parent frame returned {} pages, expected {}",
                            response.pages.len(),
                            references.len()
                        );
                    }
                    for ((expected_block, expected_image), page) in
                        references.iter().zip(&response.pages)
                    {
                        if page.block_number != *expected_block || page.image != *expected_image {
                            bail!(
                                "at-cap stale-parent frame did not preserve image for block {expected_block}"
                            );
                        }
                    }
                    1
                }
                page_api::GetPageStatusCode::InternalError => {
                    // The pre-change implementation sends each child request over
                    // the vectored-key cap. It is accepted only as the baseline's
                    // known legacy response; an InvalidRequest would reveal an
                    // off-by-one frame limit in the candidate.
                    assert_legacy_internal(&response, args.frame_cap)?;
                    0
                }
                _ => bail!("at-cap stale-parent frame unexpectedly failed: {response:?}"),
            };
            println!(
                "{}",
                json!({
                    "mode": "at_cap",
                    "direct_completion": direct_completion,
                    "image_byte_mismatches": 0,
                })
            );
        }
        Mode::OverCap => {
            let block_numbers: Vec<u32> = (0..over_cap_u32).collect();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, 1, block_numbers),
            )
            .await?;
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
                page_api::GetPageStatusCode::InternalError => {
                    // Before aggregate admission, the legacy implementation reaches
                    // child reads and fails only after each over-cap child request
                    // has performed its per-page setup.
                    assert_legacy_internal(&response, over_cap)?;
                    "legacy_internal"
                }
                _ => bail!(
                    "over-cap stale-parent frame unexpectedly completed or failed differently: {response:?}"
                ),
            };
            println!(
                "{}",
                json!({
                    "mode": "over_cap",
                    "outcome": outcome,
                    "response_pages": response.pages.len(),
                })
            );
        }
    }

    Ok(())
}
