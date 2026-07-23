//! Verifies that a GetPages error in the final internal chunk is all-or-empty.

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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.frame_cap == 0 {
        bail!("frame_cap must be nonzero");
    }
    let frame_cap = u32::try_from(args.frame_cap).context("frame_cap does not fit in u32")?;

    let mut client = page_api::Client::connect(
        args.endpoint,
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

    let block_numbers: Vec<u32> = (0..frame_cap).collect();
    let request_id = page_api::RequestID::new(1);
    request_tx
        .send(page_api::GetPageRequest {
            request_id,
            request_class: page_api::GetPageClass::Normal,
            read_lsn: page_api::ReadLsn {
                request_lsn: args.read_lsn,
                not_modified_since_lsn: Some(args.read_lsn),
            },
            rel: page_api::RelTag {
                spcnode: args.spcnode,
                dbnode: args.dbnode,
                relnode: args.relnode,
                forknum: 0,
            },
            block_numbers,
        })
        .await
        .context("send GetPages request")?;
    let response = responses
        .next()
        .await
        .context("GetPages stream ended before its response")?
        .context("receive GetPages response")?;
    if response.request_id != request_id || !response.pages.is_empty() {
        bail!("late-chunk error response did not preserve ID and empty pages");
    }

    let outcome = match (response.status_code, response.reason.as_deref()) {
        // The failpoint runs after all four chunks have accumulated results.
        // Accepting only this response proves the stream wrapper discarded that
        // successful prefix rather than failing before it reached the fourth
        // internal batch.
        (
            page_api::GetPageStatusCode::InternalError,
            Some("injected final internal GetPages batch error"),
        ) => "final_chunk_injected_error",
        _ => bail!("late-chunk request did not reach the injected final-batch error"),
    };
    println!(
        "{}",
        json!({
            "outcome": outcome,
            "response_pages": response.pages.len(),
        })
    );
    Ok(())
}
