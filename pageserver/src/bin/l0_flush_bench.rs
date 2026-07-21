//! One-sample frozen-layer flush probe used by the performance proof.
//!
//! The binary deliberately keeps setup out of the measured interval: it builds
//! a 256 MiB frozen layer, establishes and verifies the requested source-file
//! page-cache state, then times exactly one `write_to_disk` call.

use std::fs::File;
use std::io::Read;
use std::num::NonZeroUsize;
use std::os::fd::AsRawFd;
use std::ptr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use bytes::Bytes;
use camino::{Utf8Path, Utf8PathBuf};
use pageserver::config::PageServerConf;
use pageserver::context::{DownloadBehavior, RequestContext};
use pageserver::l0_flush::{L0FlushConfig, L0FlushGlobalState};
use pageserver::page_cache;
use pageserver::task_mgr::TaskKind;
use pageserver::tenant::storage_layer::InMemoryLayer;
use pageserver::virtual_file;
use pageserver::virtual_file::api::{IoEngineKind, IoMode};
use pageserver_api::key::Key;
use pageserver_api::shard::TenantShardId;
use tokio_util::sync::CancellationToken;
use utils::bin_ser::BeSer;
use utils::id::{TenantId, TimelineId};
use utils::lsn::Lsn;
use utils::sync::gate::Gate;
use wal_decoder::models::value::Value;
use wal_decoder::serialized_batch::SerializedValueBatch;

const DEFAULT_BYTES: usize = 256 * 1024 * 1024;
const VALUE_BYTES: usize = 8 * 1024;
const WARM_MIN_RESIDENT_PPM: u64 = 950_000;
const COLD_MAX_RESIDENT_PPM: u64 = 50_000;

#[derive(Clone, Copy, Debug)]
enum CacheMode {
    Warm,
    Cold,
}

impl CacheMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "warm" => Ok(Self::Warm),
            "cold" => Ok(Self::Cold),
            _ => bail!("unknown cache mode {value:?}; expected warm or cold"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    engine: IoEngineKind,
    cache: CacheMode,
    bytes: usize,
}

fn parse_engine(value: &str) -> Result<IoEngineKind> {
    match value {
        "std-fs" => Ok(IoEngineKind::StdFs),
        #[cfg(target_os = "linux")]
        "tokio-epoll-uring" => Ok(IoEngineKind::TokioEpollUring),
        #[cfg(not(target_os = "linux"))]
        "tokio-epoll-uring" => bail!("tokio-epoll-uring is only supported on Linux"),
        _ => bail!("unknown I/O engine {value:?}"),
    }
}

fn parse_config() -> Result<Config> {
    let mut engine = None;
    let mut cache = None;
    let mut bytes = DEFAULT_BYTES;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--engine" => {
                let value = args.next().context("--engine requires a value")?;
                engine = Some(parse_engine(&value)?);
            }
            "--cache" => {
                let value = args.next().context("--cache requires a value")?;
                cache = Some(CacheMode::parse(&value)?);
            }
            "--bytes" => {
                let value = args.next().context("--bytes requires a value")?;
                bytes = value.parse().context("--bytes must be an integer")?;
            }
            "--help" | "-h" => {
                println!(
                    "usage: l0_flush_bench --engine <std-fs|tokio-epoll-uring> --cache <warm|cold> [--bytes N]"
                );
                std::process::exit(0);
            }
            _ => bail!("unknown argument {arg:?}"),
        }
    }

    ensure!(bytes >= VALUE_BYTES, "--bytes must hold at least one value");
    Ok(Config {
        engine: engine.context("--engine is required")?,
        cache: cache.context("--cache is required")?,
        bytes,
    })
}

#[derive(Clone, Copy, Debug)]
struct CacheState {
    resident_pages: u64,
    total_pages: u64,
}

impl CacheState {
    fn resident_ppm(self) -> u64 {
        self.resident_pages
            .saturating_mul(1_000_000)
            .checked_div(self.total_pages)
            .unwrap_or(0)
    }
}

struct PreparedLayer {
    layer: Arc<InMemoryLayer>,
    gate: Gate,
    cancel: CancellationToken,
    source_path: Utf8PathBuf,
}

async fn prepare_layer(
    conf: &'static PageServerConf,
    target_bytes: usize,
) -> Result<PreparedLayer> {
    let mut lsn = Lsn(1000);
    let mut key = Key::from_i128(0);
    let timeline_id = TimelineId::generate();
    let tenant_shard_id = TenantShardId::unsharded(TenantId::generate());
    tokio::fs::create_dir_all(conf.timeline_path(&tenant_shard_id, &timeline_id)).await?;

    let ctx =
        RequestContext::new(TaskKind::DebugTool, DownloadBehavior::Error).with_scope_debug_tools();
    let gate = Gate::default();
    let cancel = CancellationToken::new();
    let layer = Arc::new(
        InMemoryLayer::create(
            conf,
            timeline_id,
            tenant_shard_id,
            lsn,
            &gate,
            &cancel,
            &ctx,
        )
        .await?,
    );

    // Build a runtime-derived non-zero value so the frozen source and the
    // output layer both carry real data, rather than a compiler-foldable zero
    // fixture.
    let value = Value::Image(Bytes::from(
        (0..VALUE_BYTES)
            .map(|offset| ((offset as u32).wrapping_mul(17).wrapping_add(23) & 0xff) as u8)
            .collect::<Vec<_>>(),
    ));
    let serialized_size = usize::try_from(value.serialized_size()?)?;
    let value_count = target_bytes.div_ceil(serialized_size);
    let mut batch = Vec::with_capacity(16);

    for value_number in 0..value_count {
        ensure!(
            value_number <= u32::MAX as usize,
            "too many benchmark values"
        );
        lsn += u64::try_from(serialized_size)?;
        key.field6 = value_number as u32;
        batch.push((key.to_compact(), lsn, serialized_size, value.clone()));
        if batch.len() == 16 {
            layer
                .put_batch(
                    SerializedValueBatch::from_values(std::mem::take(&mut batch)),
                    &ctx,
                )
                .await?;
        }
    }
    if !batch.is_empty() {
        layer
            .put_batch(SerializedValueBatch::from_values(batch), &ctx)
            .await?;
    }
    layer.freeze(lsn + 1).await;

    let source_path = find_ephemeral_file(&conf.timeline_path(&tenant_shard_id, &timeline_id))?;
    Ok(PreparedLayer {
        layer,
        gate,
        cancel,
        source_path,
    })
}

fn find_ephemeral_file(timeline_path: &Utf8Path) -> Result<Utf8PathBuf> {
    let mut files = std::fs::read_dir(timeline_path.as_std_path())?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("ephemeral-"))
        });
    let source = files
        .next()
        .context("frozen layer did not create an ephemeral source file")?;
    ensure!(
        files.next().is_none(),
        "benchmark expected exactly one ephemeral source file"
    );
    Utf8PathBuf::from_path_buf(source).map_err(|path| anyhow!("non-UTF-8 source path: {path:?}"))
}

fn cache_state(path: &Utf8Path) -> Result<CacheState> {
    let file = File::open(path.as_std_path())?;
    let len = usize::try_from(file.metadata()?.len())?;
    ensure!(len > 0, "source file is empty");
    let page_size = usize::try_from({
        // SAFETY: `sysconf` does not dereference a user-provided pointer for
        // `_SC_PAGESIZE`.
        unsafe { nix::libc::sysconf(nix::libc::_SC_PAGESIZE) }
    })?;
    ensure!(page_size > 0, "sysconf returned an invalid page size");
    let page_count = len.div_ceil(page_size);
    let mut resident = vec![0_u8; page_count];
    // SAFETY: the file descriptor remains open for the mapping, `len` is
    // non-zero, and offset zero is page aligned.
    let mapping = unsafe {
        nix::libc::mmap(
            ptr::null_mut(),
            len,
            nix::libc::PROT_NONE,
            nix::libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if mapping == nix::libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error()).context("mmap source file for mincore");
    }
    // SAFETY: `mapping` is a live mapping covering `len` bytes and `resident`
    // has one byte for each page that `mincore` may report.
    let mincore_result = unsafe { nix::libc::mincore(mapping, len, resident.as_mut_ptr()) };
    // SAFETY: `mapping` and `len` are exactly the values returned to this
    // function by `mmap` above.
    let unmap_result = unsafe { nix::libc::munmap(mapping, len) };
    if mincore_result != 0 {
        return Err(std::io::Error::last_os_error()).context("mincore source file");
    }
    if unmap_result != 0 {
        return Err(std::io::Error::last_os_error()).context("munmap source file");
    }

    Ok(CacheState {
        resident_pages: u64::try_from(resident.iter().filter(|page| **page & 1 != 0).count())?,
        total_pages: u64::try_from(page_count)?,
    })
}

fn warm_source(path: &Utf8Path) -> Result<CacheState> {
    let mut file = File::open(path.as_std_path())?;
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut checksum = 0_u64;
    loop {
        let nread = file.read(&mut buffer)?;
        if nread == 0 {
            break;
        }
        for byte in &buffer[..nread] {
            checksum = checksum.rotate_left(5) ^ u64::from(*byte);
        }
    }
    std::hint::black_box(checksum);
    let state = cache_state(path)?;
    ensure!(
        state.resident_ppm() >= WARM_MIN_RESIDENT_PPM,
        "warm source cache residency too low: {state:?}"
    );
    Ok(state)
}

fn cold_source(path: &Utf8Path) -> Result<CacheState> {
    let file = File::open(path.as_std_path())?;
    file.sync_all()?;
    for _ in 0..2 {
        // POSIX_FADV_DONTNEED invalidates clean file-cache pages for this
        // source file without requiring privileged global cache eviction.
        // SAFETY: the descriptor is open, the zero offset/length request
        // means the whole file, and the advice constant is valid on Linux.
        let result = unsafe {
            nix::libc::posix_fadvise(file.as_raw_fd(), 0, 0, nix::libc::POSIX_FADV_DONTNEED)
        };
        if result != 0 {
            return Err(std::io::Error::from_raw_os_error(result))
                .context("posix_fadvise DONTNEED");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let state = cache_state(path)?;
    ensure!(
        state.resident_ppm() <= COLD_MAX_RESIDENT_PPM,
        "cold source cache residency too high: {state:?}"
    );
    Ok(state)
}

fn establish_cache_regime(path: &Utf8Path, cache: CacheMode) -> Result<CacheState> {
    match cache {
        CacheMode::Warm => warm_source(path),
        CacheMode::Cold => cold_source(path),
    }
}

struct Measurement {
    flush_ns: u64,
    source_cache_resident_ppm: u64,
    source_cache_pages: u64,
    load_to_io_buf_zero_init_bytes: u64,
    load_to_io_buf_bytes: u64,
    load_to_io_buf_wall_ns: u64,
    load_to_io_buf_cpu_ns: u64,
    delta_layer_bytes: u64,
}

async fn measure(conf: &'static PageServerConf, config: Config) -> Result<Measurement> {
    let prepared = prepare_layer(conf, config.bytes).await?;
    let cache_state = establish_cache_regime(&prepared.source_path, config.cache)?;
    let ctx =
        RequestContext::new(TaskKind::DebugTool, DownloadBehavior::Error).with_scope_debug_tools();
    let l0_flush_state = L0FlushGlobalState::new(L0FlushConfig::Direct {
        max_concurrency: NonZeroUsize::new(1).unwrap(),
    });

    pageserver::benchmarking::reset();
    let started = Instant::now();
    let (desc, output_path) = prepared
        .layer
        .write_to_disk(
            &ctx,
            None,
            l0_flush_state.inner(),
            &prepared.gate,
            prepared.cancel.clone(),
        )
        .await?
        .context("frozen layer unexpectedly had no values")?;
    let flush_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let output_bytes = std::fs::metadata(output_path.as_std_path())?.len();
    ensure!(output_bytes > 0, "flush did not produce a delta layer");
    std::hint::black_box((desc, output_bytes));
    tokio::fs::remove_file(&output_path).await?;

    let metrics = pageserver::benchmarking::snapshot();
    ensure!(
        metrics.load_calls == 1,
        "expected one full snapshot load, observed {}",
        metrics.load_calls
    );
    ensure!(
        metrics.load_bytes >= u64::try_from(config.bytes)?,
        "snapshot was smaller than requested"
    );
    #[cfg(target_os = "linux")]
    ensure!(
        metrics.load_cpu_ns > 0,
        "CLOCK_THREAD_CPUTIME_ID did not capture loader CPU time"
    );

    Ok(Measurement {
        flush_ns,
        source_cache_resident_ppm: cache_state.resident_ppm(),
        source_cache_pages: cache_state.total_pages,
        load_to_io_buf_zero_init_bytes: metrics.zeroed_write_bytes,
        load_to_io_buf_bytes: metrics.load_bytes,
        load_to_io_buf_wall_ns: metrics.load_wall_ns,
        load_to_io_buf_cpu_ns: metrics.load_cpu_ns,
        delta_layer_bytes: output_bytes,
    })
}

fn emit(metric: &str, value: u64) {
    println!(
        "{}",
        serde_json::json!({ "metric": metric, "value": value })
    );
}

fn main() -> Result<()> {
    let config = parse_config()?;
    let temp_dir = camino_tempfile::tempdir_in(std::env::current_dir()?)?;
    let conf: &'static PageServerConf = Box::leak(Box::new(PageServerConf::dummy_conf(
        temp_dir.path().to_path_buf(),
    )));

    virtual_file::init(
        128,
        config.engine,
        IoMode::Buffered,
        virtual_file::SyncMode::Sync,
    );
    page_cache::init(conf.page_cache_size);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let measurement = runtime.block_on(measure(conf, config))?;

    emit("l0_flush_ns", measurement.flush_ns);
    emit(
        "source_cache_resident_ppm",
        measurement.source_cache_resident_ppm,
    );
    emit("source_cache_pages", measurement.source_cache_pages);
    emit(
        "load_to_io_buf_zero_init_bytes",
        measurement.load_to_io_buf_zero_init_bytes,
    );
    emit("load_to_io_buf_bytes", measurement.load_to_io_buf_bytes);
    emit("load_to_io_buf_wall_ns", measurement.load_to_io_buf_wall_ns);
    emit("load_to_io_buf_cpu_ns", measurement.load_to_io_buf_cpu_ns);
    emit("delta_layer_bytes", measurement.delta_layer_bytes);
    Ok(())
}
