use std::ops::Range;
use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use pageserver::context::{DownloadBehavior, RequestContext};
use pageserver::task_mgr::TaskKind;
use pageserver::tenant::disk_btree::{DiskBtreeBuilder, DiskBtreeReader, benchmark::BenchmarkDisk};
use pageserver::tenant::storage_layer::delta_layer::{BlobRef, DELTA_KEY_SIZE, DeltaLayerInner};
use pageserver::tenant::vectored_blob_io::VectoredRead;
use pageserver_api::key::{KEY_SIZE, Key};
use pageserver_api::keyspace::KeySpace;
use utils::lsn::Lsn;

const INDEXED_KEY_COUNT: u32 = 8192;
const RANGE_COUNT: u32 = 32;
const MAX_READ_SIZE: usize = 128 * 1024;

#[derive(Clone, Copy)]
enum RangeLayout {
    Nearby,
    Sparse,
}

struct Fixture {
    reader: DiskBtreeReader<BenchmarkDisk, DELTA_KEY_SIZE>,
    counting_reader: DiskBtreeReader<BenchmarkDisk, DELTA_KEY_SIZE>,
    counting_disk: BenchmarkDisk,
    keyspace: KeySpace,
    lsn_range: Range<Lsn>,
    data_end_offset: u64,
    expected: Vec<(Key, Lsn, bool)>,
}

impl Fixture {
    async fn plan(
        &self,
        reader: DiskBtreeReader<BenchmarkDisk, DELTA_KEY_SIZE>,
    ) -> Vec<VectoredRead> {
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

fn selected_key_offsets(layout: RangeLayout) -> Vec<u32> {
    match layout {
        // A fragmented GET_VECTORED batch over nearby relation pages: spans
        // multiple leaves, but adjacent ranges can share the root/prefix walk.
        RangeLayout::Nearby => (0..RANGE_COUNT).map(|i| 1024 + i * 64).collect(),
        // The concrete regression risk of a shared walk: large gaps need a
        // fresh seek rather than scanning unrelated index leaves.
        RangeLayout::Sparse => (0..RANGE_COUNT).map(|i| 128 + i * 256).collect(),
    }
}

fn build_fixture(layout: RangeLayout) -> Fixture {
    let mut counting_disk = BenchmarkDisk::default();
    let mut builder = DiskBtreeBuilder::<_, DELTA_KEY_SIZE>::new(&mut counting_disk);
    let mut offset = 0_u64;

    for key_offset in 0..INDEXED_KEY_COUNT {
        let key = Key::MIN.add(key_offset);
        for (lsn, will_init) in [(Lsn(80), false), (Lsn(100), true), (Lsn(120), false)] {
            let index_key = encode_delta_key(key, lsn);
            builder
                .append(&index_key, BlobRef::new(offset, will_init).0)
                .expect("synthetic index keys must stay ordered");
            offset += 4096;
        }
    }

    let (root_block, _) = builder.finish().expect("synthetic index must finish");
    let counting_reader = DiskBtreeReader::new(0, root_block, counting_disk.clone());
    let reader = DiskBtreeReader::new(0, root_block, counting_disk.without_read_counter());
    let selected = selected_key_offsets(layout);
    let keyspace = KeySpace {
        ranges: selected
            .iter()
            .map(|key_offset| {
                let key = Key::MIN.add(*key_offset);
                key..key.next()
            })
            .collect(),
    };
    let expected = selected
        .into_iter()
        .flat_map(|key_offset| {
            let key = Key::MIN.add(key_offset);
            [(key, Lsn(100), true), (key, Lsn(120), false)]
        })
        .collect();

    Fixture {
        reader,
        counting_reader,
        counting_disk,
        keyspace,
        lsn_range: Lsn(90)..Lsn(130),
        data_end_offset: offset,
        expected,
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime must build");
    let mut group = c.benchmark_group("delta_layer_plan_reads");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(300));

    for (name, layout) in [
        ("nearby_fragmented_32", RangeLayout::Nearby),
        ("sparse_fragmented_32", RangeLayout::Sparse),
    ] {
        let fixture = build_fixture(layout);

        fixture.counting_disk.reset_read_count();
        let dry_run = runtime.block_on(fixture.counted_plan());
        fixture.validate(&dry_run);
        let index_node_reads = fixture.counting_disk.read_count();
        assert!(index_node_reads > 0, "fixture must visit index nodes");

        // `Elements` is the deterministic number of B-tree nodes read by one
        // validated dry run. cargo-criterion emits it alongside the time sample.
        group.throughput(Throughput::Elements(index_node_reads as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(runtime.block_on(fixture.measured_plan())));
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
