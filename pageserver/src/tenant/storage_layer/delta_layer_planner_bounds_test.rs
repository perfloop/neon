use super::*;
use crate::context::DownloadBehavior;
use crate::task_mgr::TaskKind;
use crate::tenant::disk_btree::tests::TestDisk;

#[tokio::test]
async fn test_delta_layer_index_traversal_fragmented_range_bounds() {
    let base_key = Key {
        field1: 0,
        field2: 1663,
        field3: 12972,
        field4: 16396,
        field5: 0,
        field6: 246080,
    };
    let alignment = crate::virtual_file::get_io_buffer_alignment() as u64;
    let indexed_entries = [
        (base_key, 0),
        // These entries are outside the selected ranges. Their positions are
        // the required range-end boundaries for the preceding keys.
        (base_key.add(1), 2 * alignment),
        (base_key.add(2), 4 * alignment),
        (base_key.add(3), 6 * alignment),
        // The final selected key has no following index entry, so it must be
        // flushed with data_end_offset instead.
        (base_key.add(4), 8 * alignment),
    ];

    let mut disk = TestDisk::default();
    let mut writer = DiskBtreeBuilder::<_, DELTA_KEY_SIZE>::new(&mut disk);
    for (key, pos) in indexed_entries {
        let index_key = DeltaKey::from_key_lsn(&key, Lsn(100));
        writer
            .append(&index_key.0, BlobRef::new(pos, false).0)
            .expect("in-memory disk append should never fail");
    }

    let (root_offset, _writer) = writer
        .finish()
        .expect("in-memory disk finish should never fail");
    let reader = DiskBtreeReader::<_, DELTA_KEY_SIZE>::new(0, root_offset, disk);
    let keyspace = KeySpace {
        ranges: vec![
            base_key..base_key.next(),
            base_key.add(2)..base_key.add(2).next(),
            base_key.add(4)..base_key.add(4).next(),
        ],
    };
    let data_end_offset = 10 * alignment;
    let ctx = RequestContext::new(TaskKind::UnitTest, DownloadBehavior::Error);

    let reads = DeltaLayerInner::plan_reads(
        &keyspace,
        Lsn(90)..Lsn(110),
        data_end_offset,
        reader,
        VectoredReadPlanner::new(alignment as usize),
        &ctx,
    )
    .await
    .expect("read planning should not fail");

    let bounds: Vec<_> = reads.iter().map(|read| (read.start, read.end)).collect();
    assert_eq!(
        bounds,
        vec![
            (0, 2 * alignment),
            (4 * alignment, 6 * alignment),
            (8 * alignment, data_end_offset),
        ]
    );
    let actual: Vec<_> = reads
        .iter()
        .flat_map(|read| read.blobs_at.as_slice())
        .map(|(at, blob)| (*at, blob.key, blob.lsn, blob.will_init))
        .collect();
    assert_eq!(
        actual,
        vec![
            (0, base_key, Lsn(100), false),
            (4 * alignment, base_key.add(2), Lsn(100), false),
            (8 * alignment, base_key.add(4), Lsn(100), false),
        ]
    );
}
