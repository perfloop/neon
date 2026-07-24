#!/usr/bin/env bash
# Build the isolated local artifacts required by the Perfloop GetPages probes.
# Dependency caches are shared under /workspace/deps; all source-derived output
# remains in this worktree.
set -uo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

log_dir="$PWD/test_output"
mkdir -p "$log_dir" /workspace/deps
log="$log_dir/perfloop-get-pages-build.$$.log"

(
    set -euo pipefail

    export PIP_CACHE_DIR=/workspace/deps/pip
    export POETRY_CACHE_DIR=/workspace/deps/pip/poetry
    export POETRY_VIRTUALENVS_IN_PROJECT=true
    export CARGO_TARGET_DIR="$PWD/target"

    git submodule update --init --recursive --depth 1 --jobs 8
    flock /workspace/deps/perfloop-poetry.lock bash -euo pipefail -c '
        if [[ ! -x /workspace/deps/perfloop-poetry/bin/poetry ]]; then
            python3 -m venv /workspace/deps/perfloop-poetry
            /workspace/deps/perfloop-poetry/bin/python -m pip install --upgrade poetry
        fi
    '

    PATH=/workspace/deps/perfloop-poetry/bin:$PATH ./scripts/pysync

    BUILD_TYPE=debug make -j"$(nproc)" CARGO_BUILD_FLAGS='--locked --features testing,rest_broker' \
        postgres-headers-install-v14 postgres-headers-install-v15 postgres-headers-install-v17
    # pgxn/neon links this static library from NEON_CARGO_ARTIFACT_TARGET_DIR,
    # while its nested make invokes cargo from a subdirectory.
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p communicator --features testing,rest_broker
    BUILD_TYPE=debug make -j"$(nproc)" CARGO_BUILD_FLAGS='--locked --features testing,rest_broker' \
        NEON_CARGO_ARTIFACT_TARGET_DIR="$CARGO_TARGET_DIR/debug" neon-pg-ext-v16

    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p pageserver --bin pageserver --features testing
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p safekeeper --bin safekeeper --features testing
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p storage_controller --bin storage_controller --features testing
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p control_plane --bin neon_local
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p compute_tools --bin compute_ctl --features testing
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p storage_broker --bin storage_broker
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p endpoint_storage --bin endpoint_storage
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
        cargo build --locked -p pagebench --bin perfloop_get_pages_frame \
        --bin perfloop_get_pages_frame_boundaries --bin perfloop_get_pages_late_chunk_preflight

    test -x "$CARGO_TARGET_DIR/debug/pageserver"
    test -x "$CARGO_TARGET_DIR/debug/safekeeper"
    test -x "$CARGO_TARGET_DIR/debug/storage_controller"
    test -x "$CARGO_TARGET_DIR/debug/neon_local"
    test -x "$CARGO_TARGET_DIR/debug/compute_ctl"
    test -x "$CARGO_TARGET_DIR/debug/storage_broker"
    test -x "$CARGO_TARGET_DIR/debug/endpoint_storage"
    test -x "$CARGO_TARGET_DIR/debug/perfloop_get_pages_frame"
    test -x "$CARGO_TARGET_DIR/debug/perfloop_get_pages_frame_boundaries"
    test -x "$CARGO_TARGET_DIR/debug/perfloop_get_pages_late_chunk_preflight"
    test -f pg_install/v16/lib/postgresql/neon.so
) >"$log" 2>&1
status=$?

if [[ "$status" -ne 0 ]]; then
    tail -n 200 "$log"
    exit "$status"
fi

rm -f "$log"
printf 'perfloop_get_pages_build completed\n'
