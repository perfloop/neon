from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fixtures.log_helper import log
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


BENCH_BIN_ENV = "PERFLOOP_LATE_CHUNK_PREFLIGHT_BENCH_BIN"
MAX_GET_VECTORED_KEYS = 32
MAX_GET_PAGE_FRAME_CHUNKS = 4
FRAME_CAP = MAX_GET_VECTORED_KEYS * MAX_GET_PAGE_FRAME_CHUNKS


def run_late_chunk_preflight(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
    mode: str,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    frame_bench = Path(
        os.environ.get(
            BENCH_BIN_ENV,
            str(neon_binpath / "perfloop_get_pages_late_chunk_preflight"),
        )
    )
    assert frame_bench.is_file(), f"late-chunk preflight helper not found at {frame_bench}"
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
        "--fallback-chunk-size",
        str(MAX_GET_VECTORED_KEYS),
        "--mode",
        mode,
    ]
    basepath = pg_bin.run_capture(command, with_command_header=False)
    result = json.loads(Path(basepath + ".stdout").read_text())
    assert isinstance(result, dict), f"expected an object from {' '.join(command)}, got {result!r}"
    return result


def test_grpc_get_pages_late_chunk_preflight_and_error_boundary(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={MAX_GET_VECTORED_KEYS}"
    env = neon_env_builder.init_start()
    endpoint = env.endpoints.create_start("main")
    endpoint.safe_psql(
        "CREATE TABLE perfloop_grpc_get_pages_late_preflight "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_late_preflight "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_late_preflight'::regclass"
    )[0]
    assert relation_blocks >= FRAME_CAP
    read_lsn = wait_for_last_flush_lsn(
        env,
        endpoint,
        env.initial_tenant,
        env.initial_timeline,
    )
    relation = (int(dbnode), int(spcnode), int(relnode))

    # A warmup and the measured preflight both establish that all 128 public
    # requests are valid before the error is injected. On the pre-image the
    # helper retries the rejected frame in four in-limit requests; the bounded
    # implementation completes the same frame directly.
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal:.*")
    warmup = run_late_chunk_preflight(pg_bin, neon_binpath, env, read_lsn, relation, "preflight")
    assert warmup["returned_pages"] == FRAME_CAP
    assert warmup["outcome"] in {"direct_completed", "legacy_fallback_completed"}

    preflight = run_late_chunk_preflight(pg_bin, neon_binpath, env, read_lsn, relation, "preflight")
    assert preflight["returned_pages"] == FRAME_CAP
    assert preflight["outcome"] in {"direct_completed", "legacy_fallback_completed"}
    assert preflight["request_messages"] >= 1

    # The test-only hook is after the fourth candidate chunk. The pre-image
    # lacks that hook and classifies its prior oversized-frame failure instead;
    # the recorded candidate metric below must be one, and the candidate-local
    # regression test retains the strict final-chunk-only assertion.
    env.pageserver.http_client().configure_failpoints(("ps::grpc-get-pages-final-chunk", "return"))
    failure = run_late_chunk_preflight(pg_bin, neon_binpath, env, read_lsn, relation, "classify")
    assert failure["response_pages"] == 0
    assert failure["request_id_matches"]
    assert failure["non_ok_response"]
    assert failure["outcome"] in {"legacy_internal", "final_chunk_injected_error"}

    print(
        json.dumps(
            {"metric": "late_chunk_error_preflight_elapsed_ns", "value": preflight["elapsed_ns"]}
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_preflight_returned_pages",
                "value": preflight["returned_pages"],
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_preflight_direct_completion",
                "value": int(preflight["outcome"] == "direct_completed"),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_reached_final_chunk",
                "value": int(failure["outcome"] == "final_chunk_injected_error"),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_empty_pages",
                "value": int(failure["response_pages"] == 0),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_request_id_matches",
                "value": int(failure["request_id_matches"]),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "late_chunk_error_non_ok_response",
                "value": int(failure["non_ok_response"]),
            }
        )
    )
    print("verified_grpc_get_pages_late_chunk_preflight response=empty")
    log.info(
        "verified late-chunk preflight=%s failure=%s",
        preflight["outcome"],
        failure["outcome"],
    )
