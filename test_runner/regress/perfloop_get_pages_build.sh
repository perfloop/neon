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

    git submodule update --init --recursive --depth 1 --jobs 8
    cargo_target_dir=$(cargo metadata --no-deps --format-version=1 | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')
    # The sandbox's Cargo launcher owns a worktree-keyed target directory.
    # Tests expect NEON_BIN under ./target, so maintain only a local symlink.
    if [[ -e target || -L target ]]; then
        if [[ "$(readlink -f target)" != "$cargo_target_dir" ]]; then
            rm -rf target
            ln -s "$cargo_target_dir" target
        fi
    else
        ln -s "$cargo_target_dir" target
    fi
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
        NEON_CARGO_ARTIFACT_TARGET_DIR="$cargo_target_dir/debug" neon-pg-ext-v16

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

    test -x "$cargo_target_dir/debug/pageserver"
    test -x "$cargo_target_dir/debug/safekeeper"
    test -x "$cargo_target_dir/debug/storage_controller"
    test -x "$cargo_target_dir/debug/neon_local"
    test -x "$cargo_target_dir/debug/compute_ctl"
    test -x "$cargo_target_dir/debug/storage_broker"
    test -x "$cargo_target_dir/debug/endpoint_storage"
    test -x "$cargo_target_dir/debug/perfloop_get_pages_frame"
    test -x "$cargo_target_dir/debug/perfloop_get_pages_frame_boundaries"
    test -x "$cargo_target_dir/debug/perfloop_get_pages_late_chunk_preflight"
    test -f pg_install/v16/lib/postgresql/neon.so
) >"$log" 2>&1
status=$?

if [[ "$status" -ne 0 ]]; then
    tail -n 200 "$log"
    exit "$status"
fi

rm -f "$log"
printf 'perfloop_get_pages_build completed\n'
