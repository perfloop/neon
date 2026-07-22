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


BENCH_BIN_ENV = "PERFLOOP_FRAME_BOUNDARIES_BENCH_BIN"
SMGR_STARTED_METRIC = "pageserver_smgr_query_started_count_total"
GET_VECTORED_COUNT_METRIC = "pageserver_get_vectored_seconds_count"
MAX_GET_VECTORED_KEYS = 32
MAX_GET_PAGE_FRAME_CHUNKS = 4
FRAME_CAP = MAX_GET_VECTORED_KEYS * MAX_GET_PAGE_FRAME_CHUNKS
CHILD_SHARD_COUNT = 8
DIRECT_FRAME_SIZES = (33, 100)


def metric_value(metric_name: str, filters: dict[str, str], env: NeonEnv) -> float:
    value = env.pageserver.http_client().get_metric_value(
        metric_name,
        filters,
        aggregate="sum",
    )
    return 0.0 if value is None else value


def run_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
    mode: str,
    frame_size: int | None = None,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    frame_bench = Path(
        os.environ.get(BENCH_BIN_ENV, str(neon_binpath / "perfloop_get_pages_frame_boundaries"))
    )
    assert frame_bench.is_file(), f"GetPages boundary helper not found at {frame_bench}"
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
    if frame_size is not None:
        command.extend(("--frame-size", str(frame_size)))
    basepath = pg_bin.run_capture(command, with_command_header=False)
    result = json.loads(Path(basepath + ".stdout").read_text())
    assert isinstance(result, dict), f"expected an object from {' '.join(command)}, got {result!r}"
    return result


def test_grpc_get_pages_frame_boundaries(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    # One-page stripes spread consecutive relation blocks over all eight children.
    # With 129 blocks, every child request is locally at or below the 32-key cap,
    # so only an aggregate check before GetPageSplitter can reject the wire frame.
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
        "CREATE TABLE perfloop_grpc_get_pages_boundaries "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_boundaries "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_boundaries'::regclass"
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
    relation = (int(dbnode), int(spcnode), int(relnode))
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )

    # Outside the measured region, independently fetch every page at the same
    # LSN and compare every byte in the direct 33- and 100-page frames. The
    # baseline may report its legacy InternalError; a successful candidate frame
    # can only pass by preserving block order and bytes from all internal chunks.
    direct_completed = 0
    for frame_size in DIRECT_FRAME_SIZES:
        direct = run_frame(
            pg_bin,
            neon_binpath,
            env,
            read_lsn,
            relation,
            "direct",
            frame_size,
        )
        assert set(direct) == {"mode", "outcome", "image_byte_mismatches"}
        assert direct["mode"] == "direct"
        assert direct["outcome"] in ("legacy_internal", "completed")
        assert direct["image_byte_mismatches"] == 0
        direct_completed += int(direct["outcome"] == "completed")

    # Remove the parent and retain a one-pageserver placement so the old parent
    # request can reroute to every child locally. A successful at-cap response
    # proves the child fanout remains legal and retains byte/order equivalence.
    env.storage_controller.tenant_shard_split(
        env.initial_tenant,
        shard_count=CHILD_SHARD_COUNT,
    )
    parent_shard = TenantShardId(env.initial_tenant, 0, 1)
    assert not env.pageserver.tenant_dir(parent_shard).exists()
    child_locations = env.storage_controller.locate(env.initial_tenant)
    assert len(child_locations) == CHILD_SHARD_COUNT
    assert {int(location["node_id"]) for location in child_locations} == {env.pageserver.id}

    at_cap = run_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "stale-at-cap",
    )
    assert at_cap == {
        "mode": "stale_at_cap",
        "outcome": "completed",
        "image_byte_mismatches": 0,
    }

    smgr_filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    get_vectored_filters = {"task_kind": "PageRequestHandler"}
    smgr_before = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_before = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
    over_cap_started = perf_counter_ns()
    over_cap = run_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "stale-over-cap",
    )
    over_cap_elapsed_ns = perf_counter_ns() - over_cap_started
    smgr_after = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_after = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)

    timer_starts = int(smgr_after - smgr_before)
    vectored_calls = int(vectored_after - vectored_before)
    assert over_cap["response_pages"] in (0, FRAME_CAP + 1)
    assert over_cap["outcome"] in ("legacy_accepted", "invalid_request")
    if over_cap["outcome"] == "invalid_request":
        assert over_cap["response_pages"] == 0
        assert timer_starts == 0
        assert vectored_calls == 0
    else:
        assert over_cap["response_pages"] == FRAME_CAP + 1
        assert timer_starts > 0
        assert vectored_calls > 0

    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_over_cap_elapsed_ns",
                "value": over_cap_elapsed_ns,
            }
        )
    )
    print(json.dumps({"metric": "stale_parent_fanout_timer_starts", "value": timer_starts}))
    print(json.dumps({"metric": "stale_parent_fanout_vectored_calls", "value": vectored_calls}))
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_invalid_request",
                "value": int(over_cap["outcome"] == "invalid_request"),
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "chunked_direct_completed_frames",
                "value": direct_completed,
            }
        )
    )
    print(json.dumps({"metric": "chunked_direct_image_byte_mismatches", "value": 0}))
    print(
        "verified_grpc_get_pages_frame_boundaries "
        f"frame_cap={FRAME_CAP} children={CHILD_SHARD_COUNT} direct_images=verified"
    )
    log.info(
        "verified GetPages boundaries direct_completed=%s stale_over_cap=%s "
        "timer_starts=%s vectored_calls=%s",
        direct_completed,
        over_cap["outcome"],
        timer_starts,
        vectored_calls,
    )
