#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
./test_runner/regress/perfloop_get_pages_build.sh
CARGO_BUILD_JOBS="$(nproc)" CARGO_TERM_PROGRESS_WHEN=never CI=1 \
    cargo build --locked -p pagebench --bin perfloop_get_pages_preflight
printf 'perfloop_get_pages_preflight_build completed\n'
