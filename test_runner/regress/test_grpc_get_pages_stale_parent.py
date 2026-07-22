from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

from fixtures.common_types import TenantShardId
from fixtures.log_helper import log
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


BENCH_BIN_ENV = "PERFLOOP_STALE_PARENT_BENCH_BIN"
SMGR_STARTED_METRIC = "pageserver_smgr_query_started_count_total"
GET_VECTORED_COUNT_METRIC = "pageserver_get_vectored_seconds_count"
MAX_GET_VECTORED_KEYS = 32
MAX_GET_PAGE_FRAME_CHUNKS = 4
FRAME_CAP = MAX_GET_VECTORED_KEYS * MAX_GET_PAGE_FRAME_CHUNKS


def metric_value(metric_name: str, filters: dict[str, str], env: NeonEnv) -> float:
    # A stale-parent frame is rerouted to two child shards on the same Pageserver;
    # sum their labeled counters to obtain work for the one incoming wire frame.
    value = env.pageserver.http_client().get_metric_value(
        metric_name,
        filters,
        aggregate="sum",
    )
    return 0.0 if value is None else value


def run_stale_parent_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
    mode: str,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    frame_bench = Path(
        os.environ.get(BENCH_BIN_ENV, str(neon_binpath / "perfloop_get_pages_stale_parent"))
    )
    assert frame_bench.is_file(), f"stale-parent GetPages helper not found at {frame_bench}"
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
        "--mode",
        mode,
    ]
    basepath = pg_bin.run_capture(command, with_command_header=False)
    stdout_path = Path(basepath + ".stdout")
    result = json.loads(stdout_path.read_text())
    assert isinstance(result, dict), f"expected an object from {' '.join(command)}, got {result!r}"
    return result


def test_grpc_get_pages_stale_parent_frame_admission(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    # A one-page stripe makes consecutive relation blocks span both children.
    # Pin the current production cap so the test exercises exactly the four-chunk
    # aggregate contract rather than an ambient configuration default.
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = (
        f"max_get_vectored_keys={MAX_GET_VECTORED_KEYS}"
    )
    env = neon_env_builder.init_start(
        initial_tenant_shard_count=1,
        initial_tenant_shard_stripe_size=1,
    )
    endpoint = env.endpoints.create_start("main")

    endpoint.safe_psql(
        "CREATE TABLE perfloop_grpc_get_pages_stale_parent "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_stale_parent "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_stale_parent'::regclass"
    )[0]
    assert relation_blocks >= FRAME_CAP + 1, (
        f"test relation has only {relation_blocks} blocks, need at least {FRAME_CAP + 1}"
    )
    read_lsn = wait_for_last_flush_lsn(
        env,
        endpoint,
        env.initial_tenant,
        env.initial_timeline,
    )

    # Completing the split removes the parent. The helper still sends the old
    # unsharded parent identity, so every request below uses the stale-parent
    # reroute path rather than the normal direct GetPages path.
    env.storage_controller.tenant_shard_split(env.initial_tenant, shard_count=2)
    parent_shard = TenantShardId(env.initial_tenant, 0, 1)
    assert not env.pageserver.tenant_dir(parent_shard).exists()

    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )
    smgr_filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    get_vectored_filters = {"task_kind": "PageRequestHandler"}
    relation = (int(dbnode), int(spcnode), int(relnode))

    # This verifies each image byte against an independently fetched one-page
    # result and checks response order for a frame that is exactly four chunks.
    # The legacy baseline may still fail the combined frame after its child
    # batches exceed the low-level vectored limit; the candidate must complete it.
    at_cap = run_stale_parent_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "at-cap",
    )
    assert set(at_cap) == {
        "mode",
        "direct_completion",
        "image_byte_mismatches",
    }
    assert at_cap["mode"] == "at_cap"
    assert at_cap["direct_completion"] in (0, 1)
    assert at_cap["image_byte_mismatches"] == 0

    # Measure only the over-cap request. Before the fix the stale-parent path
    # performs child per-page setup before the low-level vectored cap rejects it
    # with the legacy InternalError. The repaired path must reject the aggregate
    # 129-block frame before splitting or timers. The low-level cap rejects before
    # its vectored-read counter starts, so both legal outcomes have zero completed
    # vectored reads; an accepted response is rejected by the helper, catching the
    # per-child-only limit bug.
    smgr_before = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_before = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
    over_cap_started = perf_counter_ns()
    over_cap = run_stale_parent_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "over-cap",
    )
    over_cap_elapsed_ns = perf_counter_ns() - over_cap_started
    smgr_after = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_after = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)

    timer_starts = int(smgr_after - smgr_before)
    vectored_calls = int(vectored_after - vectored_before)
    assert over_cap["response_pages"] == 0
    assert over_cap["outcome"] in ("legacy_internal", "invalid_request")
    assert vectored_calls == 0
    if over_cap["outcome"] == "invalid_request":
        assert timer_starts == 0
    else:
        assert timer_starts > 0

    print(
        json.dumps(
            {"metric": "stale_parent_over_cap_elapsed_ns", "value": over_cap_elapsed_ns}
        )
    )
    print(json.dumps({"metric": "stale_parent_over_cap_timer_starts", "value": timer_starts}))
    print(json.dumps({"metric": "stale_parent_over_cap_vectored_calls", "value": vectored_calls}))
    print(
        json.dumps(
            {
                "metric": "stale_parent_at_cap_direct_completion",
                "value": at_cap["direct_completion"],
            }
        )
    )
    print(json.dumps({"metric": "stale_parent_image_byte_mismatches", "value": 0}))
    print(
        "verified_stale_parent_get_pages "
        f"frame_cap={FRAME_CAP} at_cap_images=verified over_cap_response=empty"
    )
    log.info(
        "verified stale-parent GetPages frame_cap=%s at_cap_direct=%s over_cap=%s "
        "timer_starts=%s vectored_calls=%s",
        FRAME_CAP,
        at_cap["direct_completion"],
        over_cap["outcome"],
        timer_starts,
        vectored_calls,
    )
