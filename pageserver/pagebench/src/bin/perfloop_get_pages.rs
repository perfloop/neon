use anyhow::{bail, Context};
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use pageserver_page_api as page_api;
use serde_json::json;
use std::collections::VecDeque;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use utils::shard::ShardIndex;

const PAGE_SIZE: usize = 8192;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Frame,
    Mixed,
    Raw,
    Verify,
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
    frame_size: usize,
    #[arg(long, default_value_t = 32)]
    fallback_chunk_size: usize,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    #[arg(long, value_enum, default_value_t = Mode::Frame)]
    mode: Mode,
}

fn blocks(args: &Args) -> anyhow::Result<Vec<u32>> {
    if args.frame_size == 0 || args.fallback_chunk_size == 0 {
        bail!("frame_size and fallback_chunk_size must be nonzero");
    }
    let count = u32::try_from(args.frame_size).context("frame_size does not fit in u32")?;
    let end = args
        .start_block
        .checked_add(count)
        .context("requested frame overflows the block-number range")?;
    Ok((args.start_block..end).collect())
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

fn emit(response: &page_api::GetPageResponse, mismatches: usize) {
    println!(
        "{}",
        json!({
            "status": status(response),
            "reason": response.reason.as_deref(),
            "response_pages": response.pages.len(),
            "request_id_matches": true,
            "image_byte_mismatches": mismatches,
        })
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let expected = blocks(&args)?;
    let mut client = page_api::Client::connect(
        args.endpoint.clone(),
        args.tenant_id,
        args.timeline_id,
        ShardIndex::unsharded(),
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
    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };

    match args.mode {
        Mode::Raw => {
            let response = send(
                &tx,
                &mut responses,
                request(&args, rel, 1, expected.clone()),
            )
            .await?;
            if response.status_code == page_api::GetPageStatusCode::Ok {
                validate(&response, &expected)?;
            }
            emit(&response, 0);
        }
        Mode::Verify => {
            let mut references = Vec::with_capacity(expected.len());
            for (index, &block) in expected.iter().enumerate() {
                let response = send(
                    &tx,
                    &mut responses,
                    request(&args, rel, u64::try_from(index + 1)?, vec![block]),
                )
                .await?;
                validate(&response, &[block])?;
                references.push(response.pages[0].image.clone());
            }
            let response = send(
                &tx,
                &mut responses,
                request(
                    &args,
                    rel,
                    u64::try_from(expected.len() + 1)?,
                    expected.clone(),
                ),
            )
            .await?;
            let mut mismatches = 0;
            if response.status_code == page_api::GetPageStatusCode::Ok {
                validate(&response, &expected)?;
                mismatches = response
                    .pages
                    .iter()
                    .zip(&references)
                    .filter(|pair| {
                        let (page, reference) = *pair;
                        page.image.as_ref() != reference.as_ref()
                    })
                    .count();
            }
            emit(&response, mismatches);
        }
        Mode::Mixed => {
            if args.repetitions == 0 || expected.len() <= args.fallback_chunk_size {
                bail!("mixed mode needs a nonempty over-cap frame and positive repetitions");
            }
            let normal = expected[..args.fallback_chunk_size].to_vec();
            let mut normal_pages = 0usize;
            let mut wide_pages = 0usize;
            let mut wide_read_errors = 0usize;
            for pair in 0..args.repetitions {
                let normal_id = u64::try_from(
                    pair.checked_mul(2)
                        .and_then(|id| id.checked_add(1))
                        .context("request ID overflow")?,
                )?;
                let normal_response = send(
                    &tx,
                    &mut responses,
                    request(&args, rel, normal_id, normal.clone()),
                )
                .await?;
                validate(&normal_response, &normal)?;
                normal_pages += normal_response.pages.len();

                let wide_id = normal_id.checked_add(1).context("request ID overflow")?;
                let wide_response = send(
                    &tx,
                    &mut responses,
                    request(&args, rel, wide_id, expected.clone()),
                )
                .await?;
                if wide_response.status_code == page_api::GetPageStatusCode::Ok {
                    validate(&wide_response, &expected)?;
                    wide_pages += wide_response.pages.len();
                } else if wide_response.status_code == page_api::GetPageStatusCode::InternalError
                    && wide_response.reason.as_deref() == Some("Read error")
                    && wide_response.pages.is_empty()
                {
                    wide_read_errors += 1;
                } else {
                    bail!("unexpected mixed-mode wide response: {wide_response:?}");
                }
            }
            println!(
                "{}",
                json!({
                    "normal_pages": normal_pages,
                    "wide_pages": wide_pages,
                    "wide_read_errors": wide_read_errors,
                })
            );
        }
        Mode::Frame => {
            let mut pending = VecDeque::from([expected.clone()]);
            let mut returned = Vec::with_capacity(expected.len());
            let mut attempts = 0usize;
            let mut fallbacks = 0usize;
            while let Some(frame) = pending.pop_front() {
                attempts += 1;
                let response = send(
                    &tx,
                    &mut responses,
                    request(&args, rel, u64::try_from(attempts)?, frame.clone()),
                )
                .await?;
                if response.status_code == page_api::GetPageStatusCode::InternalError
                    && response.reason.as_deref() == Some("Read error")
                    && frame.len() > args.fallback_chunk_size
                {
                    fallbacks += 1;
                    pending.extend(
                        frame
                            .chunks(args.fallback_chunk_size)
                            .map(ToOwned::to_owned),
                    );
                    continue;
                }
                validate(&response, &frame)?;
                returned.extend(response.pages.iter().map(|page| page.block_number));
            }
            if returned != expected {
                bail!("GetPages fallback response did not preserve original frame order");
            }
            println!(
                "{}",
                json!({
                    "requested_pages": expected.len(),
                    "returned_pages": returned.len(),
                    "grpc_request_messages": attempts,
                    "oversized_fallbacks": fallbacks,
                })
            );
        }
    }
    Ok(())
}
