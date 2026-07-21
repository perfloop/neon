//! Benchmark-only counters for keeping performance probes attributable to the
//! operation they measure.
//!
//! This module is compiled only with the `benchmarking` feature. It is not part
//! of a production pageserver build.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ZEROED_WRITE_BYTES: AtomicU64 = AtomicU64::new(0);
static LOAD_CALLS: AtomicU64 = AtomicU64::new(0);
static LOAD_BYTES: AtomicU64 = AtomicU64::new(0);
static LOAD_WALL_NS: AtomicU64 = AtomicU64::new(0);
static LOAD_CPU_NS: AtomicU64 = AtomicU64::new(0);

/// Metrics collected only while a benchmark has explicitly enabled this
/// feature.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    pub zeroed_write_bytes: u64,
    pub load_calls: u64,
    pub load_bytes: u64,
    pub load_wall_ns: u64,
    pub load_cpu_ns: u64,
}

/// Reset all benchmark counters before one measured operation.
pub fn reset() {
    ZEROED_WRITE_BYTES.store(0, Ordering::Relaxed);
    LOAD_CALLS.store(0, Ordering::Relaxed);
    LOAD_BYTES.store(0, Ordering::Relaxed);
    LOAD_WALL_NS.store(0, Ordering::Relaxed);
    LOAD_CPU_NS.store(0, Ordering::Relaxed);
}

/// Return the counters accumulated since the most recent [`reset`].
pub fn snapshot() -> Snapshot {
    Snapshot {
        zeroed_write_bytes: ZEROED_WRITE_BYTES.load(Ordering::Relaxed),
        load_calls: LOAD_CALLS.load(Ordering::Relaxed),
        load_bytes: LOAD_BYTES.load(Ordering::Relaxed),
        load_wall_ns: LOAD_WALL_NS.load(Ordering::Relaxed),
        load_cpu_ns: LOAD_CPU_NS.load(Ordering::Relaxed),
    }
}

/// Record bytes passed to the zeroing `write_bytes` call in `SliceMutExt`.
pub fn record_zeroed_write_bytes(bytes: usize) {
    ZEROED_WRITE_BYTES.fetch_add(u64::try_from(bytes).unwrap(), Ordering::Relaxed);
}

/// Time one successful `load_to_io_buf` invocation.
pub struct LoadTimer {
    wall_started: Instant,
    cpu_started_ns: Option<u64>,
}

impl LoadTimer {
    pub fn start() -> Self {
        Self {
            wall_started: Instant::now(),
            cpu_started_ns: current_thread_cpu_ns(),
        }
    }

    pub fn finish(self, loaded_bytes: usize) {
        LOAD_CALLS.fetch_add(1, Ordering::Relaxed);
        LOAD_BYTES.fetch_add(u64::try_from(loaded_bytes).unwrap(), Ordering::Relaxed);
        LOAD_WALL_NS.fetch_add(
            u64::try_from(self.wall_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        if let (Some(started), Some(finished)) = (self.cpu_started_ns, current_thread_cpu_ns()) {
            LOAD_CPU_NS.fetch_add(finished.saturating_sub(started), Ordering::Relaxed);
        }
    }
}

#[cfg(target_os = "linux")]
fn current_thread_cpu_ns() -> Option<u64> {
    let mut ts = std::mem::MaybeUninit::<nix::libc::timespec>::uninit();
    // SAFETY: `ts` points to writable storage for one `timespec`, and the
    // clock id asks the kernel for CPU time of this thread only.
    if unsafe { nix::libc::clock_gettime(nix::libc::CLOCK_THREAD_CPUTIME_ID, ts.as_mut_ptr()) } != 0
    {
        return None;
    }
    // SAFETY: a zero return from `clock_gettime` initializes the supplied
    // `timespec` completely.
    let ts = unsafe { ts.assume_init() };
    let seconds = u64::try_from(ts.tv_sec).ok()?;
    let nanoseconds = u64::try_from(ts.tv_nsec).ok()?;
    seconds.checked_mul(1_000_000_000)?.checked_add(nanoseconds)
}

#[cfg(not(target_os = "linux"))]
fn current_thread_cpu_ns() -> Option<u64> {
    None
}
