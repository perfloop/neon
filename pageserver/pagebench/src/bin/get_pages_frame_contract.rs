use anyhow::{bail, Context};
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use pageserver_page_api as page_api;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use utils::shard::{ShardCount, ShardIndex, ShardNumber};

const PAGE_SIZE: usize = 8192;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Bounded,
    Raw,
    Suffix,
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
    let rel = page_api::RelTag {
        spcnode: args.spcnode,
        dbnode: args.dbnode,
        relnode: args.relnode,
        forknum: 0,
    };

    match args.mode {
        Mode::Bounded => {
            if expected.len() <= args.fallback_chunk_size {
                bail!("bounded mode needs an over-cap frame");
            }
            let oversized = send(
                &tx,
                &mut responses,
                request(&args, rel, 1, expected.clone()),
            )
            .await?;
            if oversized.status_code != page_api::GetPageStatusCode::Ok
                && !oversized.pages.is_empty()
            {
                bail!("non-OK GetPages response contained pages: {oversized:?}");
            }
            let normal = expected[..args.fallback_chunk_size].to_vec();
            let following =
                send(&tx, &mut responses, request(&args, rel, 2, normal.clone())).await?;
            validate(&following, &normal)?;
            println!(
                "{}",
                json!({
                    "oversized_status": status(&oversized),
                    "oversized_reason": oversized.reason,
                    "oversized_pages": oversized.pages.len(),
                    "following_pages": following.pages.len(),
                })
            );
        }
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
            let prefixed = send(
                &tx,
                &mut responses,
                request(&args, rel, 1, expected.clone()),
            )
            .await?;
            if prefixed.status_code != page_api::GetPageStatusCode::Ok && !prefixed.pages.is_empty()
            {
                bail!("non-OK GetPages response contained pages: {prefixed:?}");
            }
            let following =
                send(&tx, &mut responses, request(&args, rel, 2, prefix.clone())).await?;
            validate(&following, &prefix)?;
            println!(
                "{}",
                json!({
                    "prefixed_status": status(&prefixed),
                    "prefixed_reason": prefixed.reason,
                    "prefixed_pages": prefixed.pages.len(),
                    "following_pages": following.pages.len(),
                })
            );
        }
    }

    Ok(())
}
