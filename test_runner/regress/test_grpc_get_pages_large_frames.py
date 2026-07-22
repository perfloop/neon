from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fixtures.log_helper import log
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.common_types import Lsn
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


FRAME_SIZES_ENV = "PERFLOOP_GRPC_GET_PAGE_FRAME_SIZES"
SMGR_STARTED_METRIC = "pageserver_smgr_query_started_count_total"
GET_VECTORED_COUNT_METRIC = "pageserver_get_vectored_seconds_count"


def frame_sizes() -> list[int]:
    raw_sizes = os.environ.get(FRAME_SIZES_ENV, "33,100")
    sizes = [int(size) for size in raw_sizes.split(",") if size]
    assert sizes, f"{FRAME_SIZES_ENV} must name at least one frame size"
    assert all(size > 0 for size in sizes), f"{FRAME_SIZES_ENV} must contain only positive sizes"
    return sizes


def metric_value(metric_name: str, filters: dict[str, str], env: NeonEnv) -> float:
    value = env.pageserver.http_client().get_metric_value(metric_name, filters)
    return 0.0 if value is None else value


def run_get_pages_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: Lsn,
    relation: tuple[int, int, int],
    frame_size: int,
) -> dict[str, int]:
    dbnode, spcnode, relnode = relation
    command = [
        str(neon_binpath / "perfloop_get_pages_frame"),
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
        "--frame-size",
        str(frame_size),
    ]
    basepath = pg_bin.run_capture(command, with_command_header=False)
    stdout_path = Path(basepath + ".stdout")
    result = json.loads(stdout_path.read_text())
    assert isinstance(result, dict), f"expected an object from {' '.join(command)}, got {result!r}"
    return {key: int(value) for key, value in result.items()}


@pytest.mark.timeout(120)
def test_grpc_get_pages_large_frames(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    sizes = frame_sizes()
    env = neon_env_builder.init_start()
    endpoint = env.endpoints.create_start("main")

    endpoint.safe_psql(
        "CREATE TABLE perfloop_grpc_get_pages_frame "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_frame "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_frame'::regclass"
    )[0]
    assert relation_blocks >= max(sizes), (
        f"test relation has only {relation_blocks} blocks, need at least {max(sizes)}"
    )
    read_lsn = wait_for_last_flush_lsn(
        env,
        endpoint,
        env.initial_tenant,
        env.initial_timeline,
    )

    # The unfixed server emits this expected per-request error before the helper retries with
    # in-limit frames. The helper only accepts this precise error and verifies every returned page.
    env.pageserver.allowed_errors.append(r".*batching oversized.*")
    smgr_filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    get_vectored_filters = {"task_kind": "PageRequestHandler"}
    relation = (int(dbnode), int(spcnode), int(relnode))

    for frame_size in sizes:
        smgr_before = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
        vectored_before = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
        result = run_get_pages_frame(
            pg_bin,
            neon_binpath,
            env,
            read_lsn,
            relation,
            frame_size,
        )
        smgr_after = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
        vectored_after = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)

        timer_starts = int(smgr_after - smgr_before)
        vectored_calls = int(vectored_after - vectored_before)
        assert result["requested_pages"] == frame_size
        assert result["returned_pages"] == frame_size
        assert timer_starts >= frame_size
        assert vectored_calls >= 1

        # Each line is a single native-integration-test sample. The Perfloop controller repeats
        # this test invocation and compares the server-attributed counters across revisions.
        print(
            json.dumps(
                {
                    "metric": "server_get_page_timer_starts_per_completed_frame",
                    "value": timer_starts,
                }
            )
        )
        print(
            json.dumps(
                {
                    "metric": "server_get_vectored_calls_per_completed_frame",
                    "value": vectored_calls,
                }
            )
        )
        print(
            json.dumps(
                {
                    "metric": "grpc_request_messages_per_completed_frame",
                    "value": result["grpc_request_messages"],
                }
            )
        )
        print(
            json.dumps(
                {
                    "metric": "grpc_oversized_fallbacks_per_completed_frame",
                    "value": result["oversized_fallbacks"],
                }
            )
        )
        print(
            json.dumps(
                {
                    "metric": "grpc_pages_returned_per_completed_frame",
                    "value": result["returned_pages"],
                }
            )
        )
        print(
            json.dumps(
                {
                    "metric": "grpc_frame_elapsed_ns",
                    "value": result["elapsed_ns"],
                }
            )
        )
        print(
            f"verified_grpc_get_pages_frame size={frame_size} pages={result['returned_pages']} "
            "order=preserved"
        )
        log.info(
            "verified gRPC GetPages frame size=%s timer_starts=%s vectored_calls=%s "
            "request_messages=%s oversized_fallbacks=%s",
            frame_size,
            timer_starts,
            vectored_calls,
            result["grpc_request_messages"],
            result["oversized_fallbacks"],
        )
