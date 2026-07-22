from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

from fixtures.log_helper import log
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


BENCH_BIN_ENV = "PERFLOOP_LATE_CHUNK_ERROR_BENCH_BIN"
MAX_GET_VECTORED_KEYS = 32
MAX_GET_PAGE_FRAME_CHUNKS = 4
FRAME_CAP = MAX_GET_VECTORED_KEYS * MAX_GET_PAGE_FRAME_CHUNKS


def run_late_chunk_error(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    frame_bench = Path(
        os.environ.get(BENCH_BIN_ENV, str(neon_binpath / "perfloop_get_pages_late_chunk_error"))
    )
    assert frame_bench.is_file(), f"late-chunk GetPages helper not found at {frame_bench}"
    command = [
        str(frame_bench),
        "--endpoint",
        f"http://localhost:{env.pageserver.service_port.grpc}",
        "--tenant-id",
        str(env.initial_tenant),
        "--timeline-id",
        str(env.initial_timeline),
        "--read-lsn",
        str(read_lsn),
        "--spcnode",
        str(spcnode),
        "--dbnode",
        str(dbnode),
        "--relnode",
        str(relnode),
        "--frame-cap",
        str(FRAME_CAP),
    ]
    basepath = pg_bin.run_capture(command, with_command_header=False)
    result = json.loads(Path(basepath + ".stdout").read_text())
    assert isinstance(result, dict), f"expected an object from {' '.join(command)}, got {result!r}"
    return result


def test_grpc_get_pages_late_chunk_error_is_empty(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    neon_env_builder.pageserver_config_override = (
        f"max_get_vectored_keys={MAX_GET_VECTORED_KEYS}"
    )
    env = neon_env_builder.init_start()
    endpoint = env.endpoints.create_start("main")
    endpoint.safe_psql(
        "CREATE TABLE perfloop_grpc_get_pages_late_error "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_late_error "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_late_error'::regclass"
    )[0]
    assert relation_blocks >= FRAME_CAP
    read_lsn = wait_for_last_flush_lsn(
        env,
        endpoint,
        env.initial_tenant,
        env.initial_timeline,
    )
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal:.*")
    env.pageserver.http_client().configure_failpoints(
        ("ps::grpc-get-pages-final-chunk", "return")
    )

    # All four 32-page chunks are valid. The test-only failpoint runs after the
    # fourth chunk has accumulated results, so the response must discard every
    # prior chunk rather than expose a successful prefix.
    started = perf_counter_ns()
    result = run_late_chunk_error(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        (int(dbnode), int(spcnode), int(relnode)),
    )
    elapsed_ns = perf_counter_ns() - started
    assert result["response_pages"] == 0
    assert result["outcome"] in ("legacy_internal", "final_chunk_injected_error")

    print(json.dumps({"metric": "late_chunk_error_elapsed_ns", "value": elapsed_ns}))
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_empty_pages",
                "value": int(result["response_pages"] == 0),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_reached_final_chunk",
                "value": int(result["outcome"] == "final_chunk_injected_error"),
            }
        )
    )
    print("verified_grpc_get_pages_late_chunk_error response=empty")
    log.info("verified late-chunk GetPages error outcome=%s", result["outcome"])
