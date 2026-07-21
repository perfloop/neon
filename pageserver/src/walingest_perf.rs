//! Focused proof surface for the decoded WAL merge boundary.
//!
//! This is deliberately a single owner-local unit test rather than a benchmark framework. It
//! consumes the checked-in safekeeper fixture through the normal decoder and then exercises the
//! real WAL ingest/commit path without freezing or renaming the in-memory layer.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use async_compression::tokio::bufread::ZstdDecoder;
use bytes::Bytes;
use pageserver_api::key::Key;
use pageserver_api::models::TimelineState;
use pageserver_api::models::virtual_file::IoMode;
use pageserver_api::reltag::{BlockNumber, RelTag};
use pageserver_api::shard::ShardIdentity;
use postgres_ffi::waldecoder::WalStreamDecoder;
use postgres_ffi::{PgMajorVersion, WAL_SEGMENT_SIZE};
use tracing::Instrument;
use utils::bin_ser::BeSer;
use utils::lsn::Lsn;
use wal_decoder::models::value::Value;
use wal_decoder::models::{FlushUncommittedRecords, InterpretedWalRecord};
use wal_decoder::serialized_batch::ValueMeta;

use super::*;
use crate::pgdatadir_mapping::{DatadirModification, Version};
use crate::tenant::harness::{TIMELINE_ID, TenantHarness};
use crate::tenant::storage_layer::IoConcurrency;

const FIXTURE_START_LSN: Lsn = Lsn(0x14AEC08);
const INITDB_LSN: Lsn = Lsn(0x10);
const FIXTURE_PG_VERSION: PgMajorVersion = PgMajorVersion::PG15;

#[derive(Clone)]
struct ImageAtLsn {
    key: Key,
    relation: RelTag,
    block: BlockNumber,
    lsn: Lsn,
    image: Bytes,
}

struct FixtureRecord {
    record: InterpretedWalRecord,
    images: Vec<ImageAtLsn>,
    has_data: bool,
}

async fn fixture_records() -> Result<Vec<FixtureRecord>> {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test_data/sk_wal_segment_from_pgbench/000000010000000000000001.zst");
    let file = tokio::fs::File::open(&fixture_path)
        .await
        .with_context(|| format!("open WAL fixture at {}", fixture_path.display()))?;
    let decoder = ZstdDecoder::new(tokio::io::BufReader::new(file));
    let mut reader = tokio::io::BufReader::new(decoder);
    let mut bytes = Vec::new();
    tokio::io::copy_buf(&mut reader, &mut bytes)
        .await
        .context("decompress checked-in WAL fixture")?;

    let shard = ShardIdentity::unsharded();
    let mut decoder = WalStreamDecoder::new(FIXTURE_START_LSN, FIXTURE_PG_VERSION);
    let offset = FIXTURE_START_LSN.segment_offset(WAL_SEGMENT_SIZE);
    let mut records = Vec::new();

    for chunk in bytes[offset..].chunks(50) {
        decoder.feed_bytes(chunk);
        while let Some((lsn, raw_record)) = decoder.poll_decode()? {
            let record = InterpretedWalRecord::from_bytes_filtered(
                raw_record,
                &[shard],
                lsn,
                FIXTURE_PG_VERSION,
            )?
            .remove(&shard)
            .context("fixture record was not decoded for the unsharded timeline")?;
            let images = images_in(&record)?;
            let has_data = record.batch.has_data();
            records.push(FixtureRecord {
                record,
                images,
                has_data,
            });
        }
    }

    anyhow::ensure!(!records.is_empty(), "fixture decoded no WAL records");
    for pair in records.windows(2) {
        anyhow::ensure!(
            pair[0].record.next_record_lsn < pair[1].record.next_record_lsn,
            "fixture records are not in strictly increasing LSN order"
        );
    }
    Ok(records)
}

fn images_in(record: &InterpretedWalRecord) -> Result<Vec<ImageAtLsn>> {
    let mut images = Vec::new();
    for meta in &record.batch.metadata {
        let ValueMeta::Serialized(serialized) = meta else {
            continue;
        };
        let key = Key::from_compact(serialized.key);
        let Ok((relation, block)) = key.to_rel_block() else {
            continue;
        };
        let start = serialized.batch_offset as usize;
        let end = start
            .checked_add(serialized.len)
            .context("serialized value offset overflow")?;
        let raw_value = record
            .batch
            .raw
            .get(start..end)
            .context("serialized value metadata points outside its decoded batch")?;
        if let Value::Image(image) = Value::des(raw_value)? {
            images.push(ImageAtLsn {
                key,
                relation,
                block,
                lsn: serialized.lsn,
                image,
            });
        }
    }
    Ok(images)
}

fn same_key_image_pair(images: &[ImageAtLsn]) -> Option<(ImageAtLsn, ImageAtLsn)> {
    let mut previous = HashMap::new();
    for image in images {
        if let Some(old) = previous.insert(image.key, image.clone()) {
            if old.lsn < image.lsn {
                return Some((old, image.clone()));
            }
        }
    }
    None
}

fn emit_proof(metric: &str, value: impl std::fmt::Display) {
    println!(r#"{{"metric":"{metric}","value":{value}}}"#);
}

#[tokio::test(flavor = "current_thread")]
async fn wal_ingest_merge_perf() -> Result<()> {
    // Unit tests normally choose io_uring on Linux. The proof uses the supported StdFs test
    // engine selected by its command and keeps the production buffered-writer boundary intact.
    crate::virtual_file::set_io_mode(IoMode::Buffered);

    let records = fixture_records().await?;
    let images_by_record = records
        .iter()
        .map(|fixture_record| fixture_record.images.clone())
        .collect::<Vec<_>>();

    let harness = TenantHarness::create("wal_ingest_merge_perf").await?;
    let span = harness
        .span()
        .in_scope(|| tracing::info_span!("wal_ingest_merge_perf", timeline_id = %TIMELINE_ID));
    let (tenant, ctx) = harness.load().await;
    let (uninitialized, ctx) = tenant
        .create_empty_timeline(TIMELINE_ID, INITDB_LSN, FIXTURE_PG_VERSION, &ctx)
        .await?;
    let tline = uninitialized.raw_timeline()?;
    tline.set_state(TimelineState::Active);

    // This is the minimum valid timeline setup for WalIngest::new. It uses one normal
    // DatadirModification commit and deliberately avoids the freeze/rename creation path.
    let mut initialization = tline.begin_modification(INITDB_LSN);
    initialization.init_empty()?;
    initialization.put_checkpoint(postgres_ffi::dispatch_pgversion!(
        tline.pg_version,
        pgv::ZERO_CHECKPOINT.clone()
    ))?;
    initialization.commit(&ctx).await?;

    let mut walingest = WalIngest::new(tline.as_ref(), INITDB_LSN, &ctx)
        .instrument(span.clone())
        .await?;
    let mut modification = tline.begin_modification(Lsn(0));

    let started_at = Instant::now();
    let mut used_records = 0usize;
    let mut data_records = 0usize;
    let mut last_record_lsn = None;
    for fixture_record in records {
        let FixtureRecord {
            record, has_data, ..
        } = fixture_record;
        anyhow::ensure!(
            matches!(record.flush_uncommitted, FlushUncommittedRecords::No),
            "fixture requires an early commit before the pending-byte boundary"
        );
        if has_data {
            data_records += 1;
        }
        last_record_lsn = Some(record.next_record_lsn);
        walingest
            .ingest_record(record, &mut modification, &ctx)
            .instrument(span.clone())
            .await?;
        used_records += 1;

        // Match the receiver's bounded steady-state batch boundary: one commit occurs only after
        // the existing 8 MiB pending-data limit has been crossed.
        if modification.approx_pending_bytes() > DatadirModification::MAX_PENDING_BYTES {
            break;
        }
    }
    let pending_bytes = modification.approx_pending_bytes();
    let intermediate_copy_bytes = modification.ingest_batch_copy_bytes();
    modification.commit(&ctx).await?;
    let elapsed = started_at.elapsed();

    anyhow::ensure!(
        pending_bytes > DatadirModification::MAX_PENDING_BYTES,
        "fixture did not cross the WAL receiver pending-byte boundary"
    );
    anyhow::ensure!(
        data_records >= 2,
        "fixture did not exercise a multi-record data merge"
    );
    let last_record_lsn = last_record_lsn.context("fixture did not supply a record")?;
    assert_eq!(tline.get_last_record_lsn(), last_record_lsn);

    let selected_images = images_by_record
        .into_iter()
        .take(used_records)
        .flatten()
        .collect::<Vec<_>>();
    let first_image = selected_images
        .first()
        .cloned()
        .context("fixture segment contains no decoded page image")?;
    let last_image = selected_images
        .last()
        .cloned()
        .context("fixture segment contains no decoded page image")?;
    let (old_image, new_image) = same_key_image_pair(&selected_images)
        .context("fixture segment contains no same-key image versions")?;

    // Semantic reads traverse InMemoryLayer's index and EphemeralFile, so these assertions check
    // written bytes and metadata-derived offsets without exposing Timeline.layers for the test.
    let io_concurrency = IoConcurrency::spawn_for_test();
    for expected in [&first_image, &old_image, &new_image, &last_image] {
        let actual = tline
            .get_rel_page_at_lsn(
                expected.relation,
                expected.block,
                Version::at(expected.lsn),
                &ctx,
                io_concurrency.clone(),
            )
            .instrument(span.clone())
            .await?;
        assert_eq!(
            actual, expected.image,
            "unexpected value at {}",
            expected.lsn
        );
    }
    assert!(old_image.lsn < new_image.lsn);

    if std::env::var_os("PERFLOOP_PROOF").is_some() {
        emit_proof("merge_write_ns/op", elapsed.as_nanos());
        // This is the source-batch-to-contiguous-aggregate copy only. The writer's final copy is
        // intentionally not included because it remains necessary for the on-disk representation.
        emit_proof("merge_intermediate_copy_bytes/op", intermediate_copy_bytes);
    }

    // The raw timeline intentionally never reaches the freeze/rename creation path. Shut its
    // owner down before UninitializedTimeline removes the test directory and its writer task.
    uninitialized.abort().await;
    Ok(())
}
