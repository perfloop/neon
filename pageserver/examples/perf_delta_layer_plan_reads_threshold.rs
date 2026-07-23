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
const MAX_READ_SIZE: usize = 128 * 1024;

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

fn build_fixture(range_count: u32) -> Fixture {
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
    // All selected keys stay in one leaf. The 2/3/4 range counts exercise
    // the two-range destination and both sides of cache activation.
    let selected_keys: Vec<_> = (0..range_count)
        .map(|index| Key::MIN.add(1024 + index * 2))
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
        data_end_offset: offset,
        expected: selected_keys
            .into_iter()
            .flat_map(|key| [(key, Lsn(100), true), (key, Lsn(120), false)])
            .collect(),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime must build");
    let mut group = c.benchmark_group("delta_layer_plan_reads_threshold");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(300));

    for (name, range_count) in [
        ("nearby_two_1", 2),
        ("nearby_three_1", 3),
        ("nearby_four_1", 4),
    ] {
        let fixture = build_fixture(range_count);

        fixture.counting_disk.reset_read_count();
        let dry_run = runtime.block_on(fixture.counted_plan());
        fixture.validate(&dry_run);
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
