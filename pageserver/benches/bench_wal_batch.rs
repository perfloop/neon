//! Benchmark the pageserver boundary that merges per-record serialized WAL batches.
//!
//! The setup creates one serialized image per logical WAL record. The measured operation
//! merges those record-local batches and writes the resulting batch through InMemoryLayer,
//! matching the ownership and buffered-write boundary used by WAL ingestion.

use std::env;

use bytes::Bytes;
use camino::Utf8PathBuf;
use criterion::{BatchSize, Criterion, Throughput, black_box, criterion_group, criterion_main};
use pageserver::config::PageServerConf;
use pageserver::context::{DownloadBehavior, RequestContext};
use pageserver::page_cache;
use pageserver::task_mgr::TaskKind;
use pageserver::tenant::storage_layer::InMemoryLayer;
use pageserver::virtual_file;
use pageserver_api::key::Key;
use pageserver_api::models::virtual_file::{IoEngineKind, IoMode};
use pageserver_api::shard::TenantShardId;
use tokio_util::sync::CancellationToken;
use utils::bin_ser::BeSer;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use wal_decoder::models::value::Value;
use wal_decoder::serialized_batch::SerializedValueBatch;

const IMAGE_BYTES: usize = 8 * 1024;
const MERGED_RECORDS: usize = 1024;

struct BenchInput {
    layer: InMemoryLayer,
    batches: Vec<SerializedValueBatch>,
    ctx: RequestContext,
    expected_bytes: usize,
    _gate: utils::sync::gate::Gate,
    _cancel: CancellationToken,
}

fn serialized_image_bytes() -> usize {
    Value::Image(Bytes::from(vec![0; IMAGE_BYTES]))
        .serialized_size()
        .unwrap() as usize
}

async fn make_input(
    conf: &'static PageServerConf,
    record_count: usize,
    payload_seed: u8,
) -> BenchInput {
    let tenant_id = TenantId::generate();
    let tenant_shard_id = TenantShardId::unsharded(tenant_id);
    let timeline_id = TimelineId::generate();
    tokio::fs::create_dir_all(conf.timeline_path(&tenant_shard_id, &timeline_id))
        .await
        .unwrap();

    let ctx = RequestContext::new(
        TaskKind::WalReceiverConnectionHandler,
        DownloadBehavior::Error,
    )
    .with_scope_debug_tools();
    let gate = utils::sync::gate::Gate::default();
    let cancel = CancellationToken::new();
    let layer = InMemoryLayer::create(
        conf,
        timeline_id,
        tenant_shard_id,
        Lsn(0x10),
        &gate,
        &cancel,
        &ctx,
    )
    .await
    .unwrap();

    let mut batches = Vec::with_capacity(record_count);
    for record in 0..record_count {
        let mut key = Key::from_i128(0);
        key.field6 = record as u32;

        // Every setup input has distinguishable contents, so the writer must consume
        // the serialized data rather than benefiting from a constant value.
        let fill = payload_seed.wrapping_add(record as u8);
        let value = Value::Image(Bytes::from(vec![fill; IMAGE_BYTES]));
        let serialized_size = value.serialized_size().unwrap() as usize;
        batches.push(SerializedValueBatch::from_values(vec![(
            key.to_compact(),
            Lsn(0x10 + record as u64),
            serialized_size,
            value,
        )]));
    }

    BenchInput {
        layer,
        batches,
        ctx,
        expected_bytes: serialized_image_bytes() * record_count,
        _gate: gate,
        _cancel: cancel,
    }
}

async fn merge_and_write(mut input: BenchInput) {
    let mut batches = std::mem::take(&mut input.batches).into_iter();
    let mut merged = batches
        .next()
        .expect("benchmark requires at least one record");
    for batch in batches {
        merged.extend(batch);
    }

    assert_eq!(merged.buffer_size(), input.expected_bytes);
    input.layer.put_batch(merged, &input.ctx).await.unwrap();

    // The write has observable file and index updates. Keep the byte count live as
    // well so Criterion cannot remove the completed operation. Retaining `input`
    // to the end also keeps its gate and cancellation token alive until the layer
    // has been dropped.
    black_box(input.expected_bytes);
}

fn bench_wal_batch(c: &mut Criterion) {
    let temp_dir_parent: Utf8PathBuf = env::current_dir().unwrap().try_into().unwrap();
    let temp_dir = camino_tempfile::tempdir_in(temp_dir_parent).unwrap();
    let conf: &'static PageServerConf = Box::leak(Box::new(PageServerConf::dummy_conf(
        temp_dir.path().to_path_buf(),
    )));

    // The measured operation uses the buffered-writer copy path; pin the portable
    // engine so this copy-focused benchmark also runs where io_uring is unavailable.
    virtual_file::init(
        16_384,
        IoEngineKind::StdFs,
        IoMode::Buffered,
        virtual_file::SyncMode::Sync,
    );
    page_cache::init(conf.page_cache_size);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("wal-batch-merge-write");
    group.sample_size(10);

    for (name, record_count) in [
        ("records_1024_images_8192", MERGED_RECORDS),
        ("records_1_images_8192", 1),
    ] {
        group.throughput(Throughput::Bytes(
            (serialized_image_bytes() * record_count) as u64,
        ));
        group.bench_function(name, |b| {
            let mut next_seed = 0u8;
            b.iter_batched(
                || {
                    next_seed = next_seed.wrapping_add(1);
                    runtime.block_on(make_input(conf, record_count, next_seed))
                },
                |input| runtime.block_on(merge_and_write(input)),
                BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_wal_batch);
criterion_main!(benches);
