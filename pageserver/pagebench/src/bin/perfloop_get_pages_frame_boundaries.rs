//! Validates chunked GetPages response boundaries through the public gRPC API.
//!
//! The companion integration test runs this helper both before and after a real
//! shard split. It compares successful multi-chunk responses with independent
//! one-page reads at the same LSN, and classifies the historical responses that
//! a baseline returns before aggregate frame admission exists.

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
    Direct,
    StaleAtCap,
    StaleOverCap,
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

    /// Required for direct mode; ignored for stale-frame modes.
    #[arg(long)]
    frame_size: Option<usize>,

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

fn validate_frame(
    response: &page_api::GetPageResponse,
    references: &[(u32, bytes::Bytes)],
) -> anyhow::Result<()> {
    if response.status_code != page_api::GetPageStatusCode::Ok
        || response.pages.len() != references.len()
    {
        bail!("GetPages frame did not return the expected successful cardinality: {response:?}");
    }
    for ((expected_block, expected_image), page) in references.iter().zip(&response.pages) {
        if page.block_number != *expected_block || page.image != *expected_image {
            bail!("GetPages frame did not preserve the exact image for block {expected_block}");
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
    let over_cap = args
        .frame_cap
        .checked_add(1)
        .context("frame_cap overflows the block-number range")?;
    let over_cap_u32 = u32::try_from(over_cap).context("frame_cap + 1 does not fit in u32")?;

    // The test selects an unsharded identity both before the split and after
    // its parent is removed. After the split this necessarily reaches the
    // stale-parent reroute path in maybe_split_get_page.
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

    match args.mode {
        Mode::Direct | Mode::StaleAtCap => {
            let frame_size = match args.mode {
                Mode::Direct => args
                    .frame_size
                    .context("direct mode requires --frame-size")?,
                Mode::StaleAtCap => args.frame_cap,
                Mode::StaleOverCap => unreachable!(),
            };
            let frame_size_u32 =
                u32::try_from(frame_size).context("frame_size does not fit in u32")?;
            let mut references = Vec::with_capacity(frame_size);
            for block_number in 0..frame_size_u32 {
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
                        "reference request for block {block_number} did not return one page: {response:?}"
                    );
                }
                let page = &response.pages[0];
                if page.block_number != block_number || page.image.len() != PAGE_SIZE {
                    bail!("reference for block {block_number} was malformed: {page:?}");
                }
                references.push((block_number, page.image.clone()));
            }

            let block_numbers: Vec<u32> = (0..frame_size_u32).collect();
            let response = send_and_receive(
                &request_tx,
                &mut responses,
                request(&args, rel, u64::from(frame_size_u32) + 1, block_numbers),
            )
            .await?;
            let outcome = match response.status_code {
                page_api::GetPageStatusCode::Ok => {
                    validate_frame(&response, &references)?;
                    "completed"
                }
                page_api::GetPageStatusCode::InternalError if matches!(args.mode, Mode::Direct) => {
                    // The direct baseline reaches the low-level vectored cap. A
                    // candidate must instead complete and prove byte equivalence.
                    assert_legacy_internal(&response, frame_size)?;
                    "legacy_internal"
                }
                _ => bail!("GetPages frame unexpectedly failed: {response:?}"),
            };
            println!(
                "{}",
                json!({
                    "mode": match args.mode {
                        Mode::Direct => "direct",
                        Mode::StaleAtCap => "stale_at_cap",
                        Mode::StaleOverCap => unreachable!(),
                    },
                    "outcome": outcome,
                    "image_byte_mismatches": 0,
                })
            );
        }
        Mode::StaleOverCap => {
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
                page_api::GetPageStatusCode::Ok => {
                    // With each child request at or below the low-level cap, the
                    // baseline demonstrates precisely why a per-child admission
                    // check is insufficient. The fixed candidate returns the
                    // InvalidRequest branch above.
                    if response.pages.len() != over_cap {
                        bail!(
                            "legacy stale-parent frame returned {} pages, expected {over_cap}",
                            response.pages.len()
                        );
                    }
                    "legacy_accepted"
                }
                _ => bail!("over-cap stale-parent frame unexpectedly failed: {response:?}"),
            };
            println!(
                "{}",
                json!({
                    "mode": "stale_over_cap",
                    "outcome": outcome,
                    "response_pages": response.pages.len(),
                })
            );
        }
    }

    Ok(())
}
