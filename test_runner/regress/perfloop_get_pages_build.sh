#!/usr/bin/env bash
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p test_output /workspace/deps
log="$PWD/test_output/perfloop-get-pages-build.$$.log"
(
    set -euo pipefail
    export PIP_CACHE_DIR=/workspace/deps/pip POETRY_CACHE_DIR=/workspace/deps/pip/poetry
    export POETRY_VIRTUALENVS_IN_PROJECT=true
    git submodule update --init --recursive --depth 1 --jobs 8
    target=$(cargo metadata --no-deps --format-version=1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
    if [[ -e target || -L target ]]; then
        [[ "$(readlink -f target)" == "$target" ]] || { rm -rf target; ln -s "$target" target; }
    else
        ln -s "$target" target
    fi
    flock /workspace/deps/perfloop-poetry.lock bash -euo pipefail -c '
        if [[ ! -x /workspace/deps/perfloop-poetry/bin/poetry ]]; then
            python3 -m venv /workspace/deps/perfloop-poetry
            /workspace/deps/perfloop-poetry/bin/python -m pip install --upgrade poetry
        fi
    '
    PATH=/workspace/deps/perfloop-poetry/bin:$PATH ./scripts/pysync
    BUILD_TYPE=debug make -j"$(nproc)" CARGO_BUILD_FLAGS='--locked --features testing,rest_broker' postgres-headers-install-v14 postgres-headers-install-v15 postgres-headers-install-v17
    CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 cargo build --locked -p communicator --features testing,rest_broker
    BUILD_TYPE=debug make -j"$(nproc)" CARGO_BUILD_FLAGS='--locked --features testing,rest_broker' NEON_CARGO_ARTIFACT_TARGET_DIR="$target/debug" neon-pg-ext-v16
    build() { CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 cargo build --locked "$@"; }
    build -p pageserver --bin pageserver --features testing
    build -p safekeeper --bin safekeeper --features testing
    build -p storage_controller --bin storage_controller --features testing
    build -p control_plane --bin neon_local
    build -p compute_tools --bin compute_ctl --features testing
    build -p storage_broker --bin storage_broker
    build -p endpoint_storage --bin endpoint_storage
    build -p pagebench --bin perfloop_get_pages
    for binary in pageserver safekeeper storage_controller neon_local compute_ctl storage_broker endpoint_storage perfloop_get_pages; do
        test -x "$target/debug/$binary"
    done
    test -f pg_install/v16/lib/postgresql/neon.so
    if [[ -n "${PERFLOOP_BENCH_BIN:-}" ]]; then
        mkdir -p "$(dirname "$PERFLOOP_BENCH_BIN")"
        cat >"$PERFLOOP_BENCH_BIN" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
: "${TEST_OUTPUT:?TEST_OUTPUT must name a worktree-local runtime directory}"
log="$TEST_OUTPUT/perfloop-get-pages-benchmark.log"
if PERFLOOP_GET_PAGES_BIN="$PWD/target/debug/perfloop_get_pages" ./scripts/pytest -q -s "${PERFLOOP_GET_PAGES_TEST:?PERFLOOP_GET_PAGES_TEST must name one test}" >"$log" 2>&1; then :; else status=$?; cat "$log" >&2; exit "$status"; fi
python3 - "$log" <<'JSON'
import json, sys
emitted = 0
for line in open(sys.argv[1], encoding="utf-8"):
    try: sample = json.loads(line)
    except json.JSONDecodeError: continue
    if set(sample) == {"metric", "value"} and isinstance(sample["metric"], str):
        print(json.dumps(sample, separators=(",", ":")))
        emitted += 1
if not emitted: raise SystemExit("native GetPages test emitted no proof JSONL metrics")
JSON
RUNNER
        chmod +x "$PERFLOOP_BENCH_BIN"
    fi
) >"$log" 2>&1
status=$?
if [[ "$status" -ne 0 ]]; then tail -n 200 "$log"; exit "$status"; fi
rm -f "$log"
printf 'perfloop_get_pages_build completed\n'
