use std::env;
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use pageserver::config::PageServerConf;
use pageserver::context::{DownloadBehavior, RequestContext};
use pageserver::page_cache;
use pageserver::task_mgr::TaskKind;
use pageserver::tenant::block_io::{BlockBuf, BlockReader, BlockWriter, FileBlockReader};
use pageserver::tenant::disk_btree::{DiskBtreeBuilder, DiskBtreeReader, benchmark::BenchmarkDisk};
use pageserver::tenant::storage_layer::delta_layer::{BlobRef, DELTA_KEY_SIZE, DeltaLayerInner};
use pageserver::tenant::vectored_blob_io::VectoredRead;
use pageserver::virtual_file::{self, VirtualFile};
use pageserver_api::key::{KEY_SIZE, Key, rel_block_to_key};
use pageserver_api::keyspace::KeySpace;
use pageserver_api::reltag::RelTag;
use utils::lsn::Lsn;

const INDEXED_KEY_COUNT: u32 = 8192;
const MAX_READ_SIZE: usize = 128 * 1024;
// This is the ordered singleton layout in
// `test_delta_layer_index_traversal_fragmented_ranges`, shifted into the
// large fixture so its prefixes remain within one B-tree leaf.
const FRAGMENTED_RANGE_TEST_OFFSETS: [u32; 4] = [0, 3, 6, 9];
const BENCHMARK_RELATION: RelTag = RelTag {
    forknum: 0,
    spcnode: 1663,
    dbnode: 12972,
    relnode: 16396,
};

fn relation_key(block: u32) -> Key {
    rel_block_to_key(BENCHMARK_RELATION, block)
}

struct Fixture {
    reader: DiskBtreeReader<FileBlockReader<'static>, DELTA_KEY_SIZE>,
    counting_reader: DiskBtreeReader<BenchmarkDisk, DELTA_KEY_SIZE>,
    counting_disk: BenchmarkDisk,
    keyspace: KeySpace,
    lsn_range: Range<Lsn>,
    data_end_offset: u64,
    expected: Vec<(Key, Lsn, bool)>,
}

impl Fixture {
    async fn plan<Reader>(
        &self,
        reader: DiskBtreeReader<Reader, DELTA_KEY_SIZE>,
    ) -> Vec<VectoredRead>
    where
        Reader: BlockReader + Clone,
    {
        let ctx = RequestContext::new(TaskKind::DebugTool, DownloadBehavior::Error);
        DeltaLayerInner::plan_reads_for_benchmark(
            &self.keyspace,
            self.lsn_range.clone(),
            self.data_end_offset,
            reader,
            MAX_READ_SIZE,
            &ctx,
        )
        .await
        .expect("synthetic delta-layer index must be readable")
    }

    fn validate(&self, reads: &[VectoredRead]) {
        let actual: Vec<_> = reads
            .iter()
            .flat_map(|read| read.blobs_at.as_slice())
            .map(|(_, blob)| (blob.key, blob.lsn, blob.will_init))
            .collect();
        assert_eq!(actual, self.expected);
    }

    async fn measured_plan(&self) -> Vec<VectoredRead> {
        self.plan(self.reader.clone()).await
    }

    async fn counted_plan(&self) -> Vec<VectoredRead> {
        self.plan(self.counting_reader.clone()).await
    }
}

fn encode_delta_key(key: Key, lsn: Lsn) -> [u8; DELTA_KEY_SIZE] {
    let mut encoded = [0_u8; DELTA_KEY_SIZE];
    key.write_to_byte_slice(&mut encoded[..KEY_SIZE]);
    encoded[KEY_SIZE..].copy_from_slice(&lsn.0.to_be_bytes());
    encoded
}

fn build_index<Writer>(writer: Writer) -> (u32, u64, Writer)
where
    Writer: BlockWriter,
{
    let mut builder = DiskBtreeBuilder::<_, DELTA_KEY_SIZE>::new(writer);
    let mut offset = 0_u64;

    for key_offset in 0..INDEXED_KEY_COUNT {
        let key = relation_key(key_offset);
        for (lsn, will_init) in [(Lsn(80), false), (Lsn(100), true), (Lsn(120), false)] {
            let index_key = encode_delta_key(key, lsn);
            builder
                .append(&index_key, BlobRef::new(offset, will_init).0)
                .expect("synthetic index keys must stay ordered");
            offset += 4096;
        }
    }

    let (root_block, writer) = builder.finish().expect("synthetic index must finish");
    (root_block, offset, writer)
}

fn build_fixture(
    runtime: &tokio::runtime::Runtime,
    directory: &camino::Utf8Path,
    range_count: u32,
) -> Fixture {
    let mut counting_disk = BenchmarkDisk::default();
    let (counting_root_block, data_end_offset, _) = build_index(&mut counting_disk);

    let (file_root_block, file_data_end_offset, blocks) = build_index(BlockBuf::new());
    assert_eq!(file_root_block, counting_root_block);
    assert_eq!(file_data_end_offset, data_end_offset);

    let index_path = directory.join(format!("delta-index-{range_count}"));
    let mut index_file = File::create(&index_path).expect("must create synthetic index file");
    for block in blocks.blocks {
        index_file
            .write_all(&block)
            .expect("must write synthetic index block");
    }
    index_file
        .sync_all()
        .expect("must persist synthetic index file");

    let ctx = RequestContext::new(TaskKind::DebugTool, DownloadBehavior::Error);
    let file = runtime
        .block_on(VirtualFile::open_v2(&index_path, &ctx))
        .expect("must open synthetic index file through VirtualFile");
    let file = Box::leak(Box::new(file));
    let reader = DiskBtreeReader::new(
        0,
        file_root_block,
        FileBlockReader::new(file, page_cache::next_file_id()),
    );
    let counting_reader = DiskBtreeReader::new(0, counting_root_block, counting_disk.clone());

    // Replay prefixes of the sealed fragmented-range test's ordered
    // singleton KeySpace layout on the existing 8,192-key fixture. These are
    // valid relation-block keys, which is the page-service get-page input
    // class. All four selected keys stay in one leaf, while 2/3/4 retain the
    // ordinary caller parameter: the number of logical ranges.
    let selected_keys: Vec<_> = FRAGMENTED_RANGE_TEST_OFFSETS[..range_count as usize]
        .iter()
        .map(|offset| relation_key(1024 + offset))
        .collect();
    let keyspace = KeySpace {
        ranges: selected_keys.iter().map(|key| *key..key.next()).collect(),
    };

    Fixture {
        reader,
        counting_reader,
        counting_disk,
        keyspace,
        lsn_range: Lsn(90)..Lsn(130),
        data_end_offset,
        expected: selected_keys
            .into_iter()
            .flat_map(|key| [(key, Lsn(100), true), (key, Lsn(120), false)])
            .collect(),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let temp_dir_parent: camino::Utf8PathBuf = env::current_dir().unwrap().try_into().unwrap();
    let temp_dir =
        camino_tempfile::tempdir_in(temp_dir_parent).expect("must create benchmark directory");
    let conf = PageServerConf::dummy_conf(temp_dir.path().to_path_buf());
    // This fixture warms index pages before sampling, so its timed path is
    // FileBlockReader plus the Pageserver page cache. Allow StdFs when this
    // host cannot create the preferred io_uring engine for that setup-only I/O.
    let io_engine = virtual_file::io_engine_feature_test()
        .expect("must detect a VirtualFile I/O engine")
        .into();
    virtual_file::init(
        16_384,
        io_engine,
        conf.virtual_file_io_mode,
        virtual_file::SyncMode::Sync,
    );
    page_cache::init(conf.page_cache_size);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime must build");
    let mut group = c.benchmark_group("delta_layer_plan_reads_threshold_file");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(300));

    for (name, range_count) in [
        ("nearby_two_1", 2),
        ("nearby_three_1", 3),
        ("nearby_four_1", 4),
    ] {
        let fixture = build_fixture(&runtime, temp_dir.path(), range_count);

        // Warm the actual FileBlockReader path before Criterion measures it.
        let warmup = runtime.block_on(fixture.measured_plan());
        fixture.validate(&warmup);

        // Count the same immutable index traversal in setup only. BenchmarkDisk
        // makes this structural control observable without entering the timed
        // FileBlockReader/page-cache path.
        fixture.counting_disk.reset_read_count();
        let counted = runtime.block_on(fixture.counted_plan());
        fixture.validate(&counted);
        let index_node_reads = fixture.counting_disk.read_count();
        assert!(index_node_reads > 0, "fixture must visit index nodes");
        eprintln!("PERFLOOP_INDEX_NODE_READS_{name}={index_node_reads}");

        group.throughput(Throughput::Elements(index_node_reads as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(runtime.block_on(fixture.measured_plan())));
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
