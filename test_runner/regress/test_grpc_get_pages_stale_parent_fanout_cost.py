from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fixtures.common_types import TenantShardId
from fixtures.log_helper import log
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


BENCH_BIN_ENV = "PERFLOOP_STALE_PARENT_FANOUT_COST_BENCH_BIN"
SMGR_STARTED_METRIC = "pageserver_smgr_query_started_count_total"
GET_VECTORED_COUNT_METRIC = "pageserver_get_vectored_seconds_count"
MAX_GET_VECTORED_KEYS = 32
MAX_GET_PAGE_FRAME_CHUNKS = 4
FRAME_CAP = MAX_GET_VECTORED_KEYS * MAX_GET_PAGE_FRAME_CHUNKS
CHILD_SHARD_COUNT = 8


def metric_value(metric_name: str, filters: dict[str, str], env: NeonEnv) -> float:
    value = env.pageserver.http_client().get_metric_value(
        metric_name,
        filters,
        aggregate="sum",
    )
    return 0.0 if value is None else value


def run_fanout_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
    mode: str,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    helper = Path(
        os.environ.get(
            BENCH_BIN_ENV,
            str(neon_binpath / "perfloop_get_pages_stale_parent_fanout_cost"),
        )
    )
    assert helper.is_file(), f"stale-parent fanout helper not found at {helper}"
    command = [
        str(helper),
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
    result = json.loads(Path(basepath + ".stdout").read_text())
    assert isinstance(result, dict), f"expected an object from {' '.join(command)}, got {result!r}"
    return result


def sample_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
    mode: str,
    smgr_filters: dict[str, str],
    get_vectored_filters: dict[str, str],
) -> tuple[dict[str, Any], int, int]:
    # This helper sends exactly one frame and does not make its one-page image
    # references. The counter brackets therefore cover only the sampled frame.
    smgr_before = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_before = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
    result = run_fanout_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        mode,
    )
    smgr_after = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_after = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
    return result, int(smgr_after - smgr_before), int(vectored_after - vectored_before)


def test_grpc_get_pages_stale_parent_fanout_cost(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    # One-page stripes spread consecutive blocks across all eight children. The
    # 129-block frame keeps every child partition below 32 blocks, so only the
    # aggregate admission before GetPageSplitter can reject it.
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={MAX_GET_VECTORED_KEYS}"
    env = neon_env_builder.init_start(
        initial_tenant_shard_count=1,
        initial_tenant_shard_stripe_size=1,
    )
    endpoint = env.endpoints.create_start("main")
    endpoint.safe_psql(
        "CREATE TABLE perfloop_grpc_get_pages_fanout_cost "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_fanout_cost "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_fanout_cost'::regclass"
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

    env.storage_controller.tenant_shard_split(
        env.initial_tenant,
        shard_count=CHILD_SHARD_COUNT,
    )
    parent_shard = TenantShardId(env.initial_tenant, 0, 1)
    assert not env.pageserver.tenant_dir(parent_shard).exists()
    child_locations = env.storage_controller.locate(env.initial_tenant)
    assert len(child_locations) == CHILD_SHARD_COUNT
    assert {int(location["node_id"]) for location in child_locations} == {env.pageserver.id}
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with InvalidArgument:.*"
    )

    relation = (int(dbnode), int(spcnode), int(relnode))
    # Establish exact at-cap image/order equivalence outside the cost samples.
    # The later counter brackets must contain no one-page reference reads.
    verify_at_cap = run_fanout_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "verify-at-cap",
    )
    assert verify_at_cap == {
        "mode": "verify_at_cap",
        "outcome": "completed",
        "image_byte_mismatches": 0,
    }

    smgr_filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    get_vectored_filters = {"task_kind": "PageRequestHandler"}
    at_cap, at_cap_timer_starts, at_cap_vectored_calls = sample_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "at-cap",
        smgr_filters,
        get_vectored_filters,
    )
    assert at_cap["mode"] == "at_cap"
    assert at_cap["outcome"] == "completed"
    assert at_cap["response_pages"] == FRAME_CAP
    assert at_cap_timer_starts == FRAME_CAP
    assert at_cap_vectored_calls == CHILD_SHARD_COUNT

    over_cap, over_cap_timer_starts, over_cap_vectored_calls = sample_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "over-cap",
        smgr_filters,
        get_vectored_filters,
    )
    assert over_cap["mode"] == "over_cap"
    assert over_cap["outcome"] in ("legacy_accepted", "invalid_request")
    if over_cap["outcome"] == "invalid_request":
        assert over_cap["response_pages"] == 0
        assert over_cap_timer_starts == 0
        assert over_cap_vectored_calls == 0
    else:
        assert over_cap["response_pages"] == FRAME_CAP + 1
        assert over_cap_timer_starts == FRAME_CAP + 1
        assert over_cap_vectored_calls == CHILD_SHARD_COUNT

    at_cap_elapsed_ns = int(at_cap["elapsed_ns"])
    over_cap_elapsed_ns = int(over_cap["elapsed_ns"])
    assert at_cap_elapsed_ns > 0
    assert over_cap_elapsed_ns > 0
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_elapsed_ns",
                "value": at_cap_elapsed_ns,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_timer_starts",
                "value": at_cap_timer_starts,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_vectored_calls",
                "value": at_cap_vectored_calls,
            }
        )
    )
    print(json.dumps({"metric": "stale_parent_fanout_at_cap_completed", "value": 1}))
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_over_cap_elapsed_ns",
                "value": over_cap_elapsed_ns,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_over_cap_timer_starts",
                "value": over_cap_timer_starts,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_over_cap_vectored_calls",
                "value": over_cap_vectored_calls,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_over_cap_invalid_request",
                "value": int(over_cap["outcome"] == "invalid_request"),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_accepted_to_rejected_elapsed_ratio",
                "value": at_cap_elapsed_ns / over_cap_elapsed_ns,
            }
        )
    )
    print(json.dumps({"metric": "stale_parent_fanout_image_byte_mismatches", "value": 0}))
    print(
        "verified_grpc_get_pages_stale_parent_fanout_cost "
        f"frame_cap={FRAME_CAP} children={CHILD_SHARD_COUNT} "
        f"over_cap={over_cap['outcome']}"
    )
    log.info(
        "verified stale-parent fanout cost at_cap_elapsed_ns=%s at_cap_timers=%s "
        "at_cap_vectored=%s over_cap_elapsed_ns=%s over_cap=%s over_cap_timers=%s "
        "over_cap_vectored=%s",
        at_cap_elapsed_ns,
        at_cap_timer_starts,
        at_cap_vectored_calls,
        over_cap_elapsed_ns,
        over_cap["outcome"],
        over_cap_timer_starts,
        over_cap_vectored_calls,
    )
