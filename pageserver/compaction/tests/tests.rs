use once_cell::sync::OnceCell;
use pageserver_compaction::interface::CompactionLayer;
use pageserver_compaction::simulator::MockTimeline;
use utils::logging;

static LOG_HANDLE: OnceCell<()> = OnceCell::new();

pub(crate) fn setup_logging() {
    LOG_HANDLE.get_or_init(|| {
        logging::init(
            logging::LogFormat::Test,
            logging::TracingErrorLayerEnablement::EnableWithRustLogFilter,
            logging::Output::Stdout,
        )
        .expect("Failed to init test logging");
    });
}

/// Test the extreme case that there are so many updates for a single key that
/// even if we produce an extremely narrow delta layer, spanning just that one
/// key, we still too many records to fit in the target file size. We need to
/// split in the LSN dimension too in that case.
#[tokio::test]
async fn test_many_updates_for_single_key() {
    setup_logging();
    let mut executor = MockTimeline::new();
    executor.target_file_size = 1_000_000; // 1 MB

    // Ingest 10 MB of updates to a single key.
    for _ in 1..1000 {
        executor.ingest_uniform(100, 10, &(0..100_000)).unwrap();
        executor.ingest_uniform(1000, 10, &(0..1)).unwrap();
        executor.compact().await.unwrap();
    }

    // Check that all the layers are smaller than the target size (with some slop)
    for l in executor.live_layers.iter() {
        println!("layer {}: {}", l.short_id(), l.file_size());
    }
    for l in executor.live_layers.iter() {
        assert!(l.file_size() < executor.target_file_size * 2);
        // Sanity check that none of the delta layers are empty either.
        if l.is_delta() {
            assert!(l.file_size() > 0);
        }
    }
}

/// Retained duplicate values can cross a size boundary while sharing an LSN.
/// The executor rejects zero-width ranges, so this exercises the planner's
/// requirement to keep every emitted delta range nonempty.
#[tokio::test]
async fn test_same_lsn_values_do_not_create_empty_delta_ranges() {
    setup_logging();
    let mut executor = MockTimeline::new();
    executor.target_file_size = 100;
    // A fanout of one forces this one L0 tier through retile_deltas.
    executor.set_tiers_per_level(1);

    executor.ingest_duplicate_records(0, 60, 8);
    // Advance the LSN range after the same-LSN records so the compacted layer
    // has a valid upper bound.
    executor.ingest_record(0, 1);
    executor.flush_l0();

    executor.compact().await.unwrap();

    let output_ranges = executor.active_delta_layer_ranges();
    assert!(!output_ranges.is_empty());
    assert!(
        output_ranges.iter().all(|(key_range, lsn_range)| {
            key_range == &(0..1) && lsn_range.start < lsn_range.end
        })
    );
    let expected_record_count = 9;
    assert_eq!(executor.active_delta_record_count(), expected_record_count);

    // A later compaction sees the published output state. It must neither emit
    // a colliding descriptor nor lose the retained duplicate records.
    executor.compact().await.unwrap();
    assert_eq!(executor.active_delta_layer_ranges(), output_ranges);
    assert_eq!(executor.active_delta_record_count(), expected_record_count);
}

/// Fanout one promotes every L0 tier, but an upper tier with depth one must not
/// be rewritten. The mock executor rejects descriptor collisions just as layer
/// publication does, making a repeated compaction an end-to-end lifecycle test.
#[tokio::test]
async fn test_fanout_one_leaves_singleton_upper_tier_unchanged() {
    setup_logging();
    let mut executor = MockTimeline::new();
    executor.target_file_size = 100;
    executor.set_tiers_per_level(1);

    for _ in 0..2 {
        for key in 0..4 {
            executor.ingest_record(key, 25);
        }
        executor.flush_l0();
    }
    executor.compact().await.unwrap();

    let upper_tier = executor.active_delta_layer_ranges();
    assert!(upper_tier.iter().all(|(key_range, lsn_range)| {
        key_range.end - key_range.start > 1
            && lsn_range.end.0 - lsn_range.start.0 > executor.target_file_size
    }));

    for _ in 0..3 {
        executor.compact().await.unwrap();
        assert_eq!(executor.active_delta_layer_ranges(), upper_tier);
    }
}

#[tokio::test]
async fn test_simple_updates() {
    setup_logging();
    let mut executor = MockTimeline::new();
    executor.target_file_size = 500_000; // 500 KB

    // Ingest some traffic.
    for _ in 1..400 {
        executor.ingest_uniform(100, 500, &(0..100_000)).unwrap();
    }

    for l in executor.live_layers.iter() {
        println!("layer {}: {}", l.short_id(), l.file_size());
    }

    println!("Running compaction...");
    executor.compact().await.unwrap();

    for l in executor.live_layers.iter() {
        println!("layer {}: {}", l.short_id(), l.file_size());
    }
}
