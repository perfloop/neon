from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

from fixtures.common_types import TenantShardId
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.common_types import Lsn
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


CAP = 32
FRAME_CAP = CAP * 4
SMGR = "pageserver_smgr_query_started_count_total"
VECTORED = "pageserver_get_vectored_seconds_count"


def emit(metric: str, value: int | float) -> None:
    print(json.dumps({"metric": metric, "value": value}))


def metric(env: NeonEnv, name: str, filters: dict[str, str]) -> float:
    value = env.pageserver.http_client().get_metric_value(name, filters, aggregate="sum")
    return 0.0 if value is None else value


def setup(builder: NeonEnvBuilder, table: str, min_blocks: int, *, striped: bool = False):
    kwargs = {"initial_tenant_shard_count": 1, "initial_tenant_shard_stripe_size": 1} if striped else {}
    env = builder.init_start(**kwargs)
    endpoint = env.endpoints.create_start("main")
    endpoint.safe_psql(
        f"CREATE TABLE {table} (id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        f"INSERT INTO {table} SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, blocks = endpoint.safe_psql(
        "SELECT (SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        f"FROM pg_class c WHERE c.oid = '{table}'::regclass"
    )[0]
    assert blocks >= min_blocks, f"{table} has {blocks} blocks, need {min_blocks}"
    lsn = wait_for_last_flush_lsn(env, endpoint, env.initial_tenant, env.initial_timeline)
    return env, endpoint, (int(dbnode), int(spcnode), int(relnode)), lsn


def run(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    lsn: Lsn,
    relation: tuple[int, int, int],
    size: int,
    mode: str,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    binary = Path(os.environ.get("PERFLOOP_GET_PAGES_BIN", str(neon_binpath / "perfloop_get_pages")))
    assert binary.is_file(), f"GetPages helper not found at {binary}"
    command = [
        str(binary),
        "--endpoint",
        f"http://localhost:{env.pageserver.service_port.grpc}",
        "--tenant-id",
        str(env.initial_tenant),
        "--timeline-id",
        str(env.initial_timeline),
        "--read-lsn",
        str(lsn),
        "--spcnode",
        str(spcnode),
        "--dbnode",
        str(dbnode),
        "--relnode",
        str(relnode),
        "--frame-size",
        str(size),
        "--fallback-chunk-size",
        str(CAP),
        "--mode",
        mode,
    ]
    basepath = pg_bin.run_capture(command, with_command_header=False)
    result = json.loads(Path(basepath + ".stdout").read_text())
    assert isinstance(result, dict), result
    return result


def test_grpc_get_pages_large_frames(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    sizes = [int(size) for size in os.environ.get("PERFLOOP_GRPC_GET_PAGE_FRAME_SIZES", "33,100").split(",") if size]
    repetitions = int(os.environ.get("PERFLOOP_GRPC_GET_PAGE_FRAME_REPETITIONS", "32"))
    assert sizes and all(size > 0 for size in sizes) and repetitions > 0
    env, _, relation, lsn = setup(neon_env_builder, "perfloop_grpc_get_pages_frame", max(sizes))
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal: Read error.*")
    smgr_filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    for size in sizes:
        cold = run(pg_bin, neon_binpath, env, lsn, relation, size, "frame")
        assert cold["returned_pages"] == size
        before_timers = metric(env, SMGR, smgr_filters)
        before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
        results = [run(pg_bin, neon_binpath, env, lsn, relation, size, "frame") for _ in range(repetitions)]
        timers = int(metric(env, SMGR, smgr_filters) - before_timers)
        vectored = int(metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored)
        assert all(result["requested_pages"] == size and result["returned_pages"] == size for result in results)
        assert timers >= repetitions * size and vectored >= repetitions
        assert timers % repetitions == 0, (timers, repetitions)
        assert vectored % repetitions == 0, (vectored, repetitions)
        requests = sum(result["grpc_request_messages"] for result in results)
        fallbacks = sum(result["oversized_fallbacks"] for result in results)
        pages = sum(result["returned_pages"] for result in results)
        elapsed = sum(result["elapsed_ns"] for result in results)
        assert requests % repetitions == fallbacks % repetitions == pages % repetitions == 0
        if timers // repetitions < 2 * size:
            assert all(result["oversized_fallbacks"] == 0 and result["grpc_request_messages"] == 1 for result in results)
        emit("server_get_page_timer_starts_per_completed_frame", timers // repetitions)
        emit("server_get_vectored_calls_per_completed_frame", vectored // repetitions)
        emit("grpc_request_messages_per_completed_frame", requests // repetitions)
        emit("grpc_oversized_fallbacks_per_completed_frame", fallbacks // repetitions)
        emit("grpc_pages_returned_per_completed_frame", pages // repetitions)
        emit("grpc_frame_elapsed_ns", elapsed / repetitions)
        emit("grpc_frame_cold_elapsed_ns", cold["elapsed_ns"])
        print(f"verified_grpc_get_pages_frame size={size} pages={pages // repetitions} repetitions={repetitions}")


def test_grpc_get_pages_frame_boundaries(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, _, relation, lsn = setup(neon_env_builder, "perfloop_grpc_get_pages_boundaries", FRAME_CAP + 1, striped=True)
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*")
    direct = [run(pg_bin, neon_binpath, env, lsn, relation, size, "verify") for size in (33, 100)]
    for size, result in zip((33, 100), direct):
        assert result["request_id_matches"] and result["image_byte_mismatches"] == 0
        if result["status"] == "ok":
            assert result["reason"] is None and result["response_pages"] == size
        else:
            assert (result["status"], result["reason"], result["response_pages"]) == (
                "internal_error",
                "Read error",
                0,
            )
    completed = sum(result["status"] == "ok" for result in direct)
    assert completed in (0, len(direct))
    env.storage_controller.tenant_shard_split(env.initial_tenant, shard_count=8)
    assert not env.pageserver.tenant_dir(TenantShardId(env.initial_tenant, 0, 1)).exists()
    assert len(env.storage_controller.locate(env.initial_tenant)) == 8
    at_cap = run(pg_bin, neon_binpath, env, lsn, relation, FRAME_CAP, "verify")
    assert at_cap["status"] == "ok" and at_cap["response_pages"] == FRAME_CAP
    assert at_cap["image_byte_mismatches"] == 0 and at_cap["request_id_matches"]
    filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    started = perf_counter_ns()
    over_cap = run(pg_bin, neon_binpath, env, lsn, relation, FRAME_CAP + 1, "raw")
    elapsed = perf_counter_ns() - started
    timers = int(metric(env, SMGR, filters) - before_timers)
    vectored = int(metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored)
    if completed:
        assert over_cap == {
            "status": "invalid_request",
            "reason": f"GetPages request has {FRAME_CAP + 1} blocks, limit is {FRAME_CAP}",
            "response_pages": 0,
            "request_id_matches": True,
            "image_byte_mismatches": 0,
        }
        assert timers == 0 and vectored == 0
    else:
        assert over_cap["status"] == "ok" and over_cap["response_pages"] == FRAME_CAP + 1
        assert over_cap["image_byte_mismatches"] == 0 and timers > 0 and vectored > 0
    emit("stale_parent_fanout_over_cap_elapsed_ns", elapsed)
    emit("stale_parent_fanout_timer_starts", timers)
    emit("stale_parent_fanout_vectored_calls", vectored)
    emit("stale_parent_fanout_invalid_request", int(over_cap["status"] == "invalid_request"))
    print("verified_grpc_get_pages_frame_boundaries")


def test_grpc_get_pages_late_chunk_preflight_and_error_boundary(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, _, relation, lsn = setup(neon_env_builder, "perfloop_grpc_get_pages_late", FRAME_CAP)
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal:.*")
    warmup = run(pg_bin, neon_binpath, env, lsn, relation, FRAME_CAP, "frame")
    preflight = run(pg_bin, neon_binpath, env, lsn, relation, FRAME_CAP, "frame")
    assert warmup["returned_pages"] == preflight["returned_pages"] == FRAME_CAP
    direct = preflight["oversized_fallbacks"] == 0 and preflight["grpc_request_messages"] == 1
    assert direct == (warmup["oversized_fallbacks"] == 0 and warmup["grpc_request_messages"] == 1)
    assert direct or (preflight["oversized_fallbacks"] == 1 and preflight["grpc_request_messages"] == 5)
    env.pageserver.http_client().configure_failpoints(("ps::grpc-get-pages-final-chunk", "return"))
    failure = run(pg_bin, neon_binpath, env, lsn, relation, FRAME_CAP, "raw")
    assert failure["request_id_matches"] and failure["response_pages"] == 0
    assert failure["status"] == "internal_error"
    if direct:
        assert failure["reason"] == "injected final internal GetPages batch error"
    else:
        assert failure["reason"] == "Read error"
    emit("late_chunk_error_reached_final_chunk", int(failure["reason"] == "injected final internal GetPages batch error"))
    emit("late_chunk_error_empty_pages", int(failure["response_pages"] == 0))
    print("verified_grpc_get_pages_late_chunk_preflight")
