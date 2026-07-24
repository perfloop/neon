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
LATE_FRAME = CAP * 4
SMGR = "pageserver_smgr_query_started_count_total"
VECTORED = "pageserver_get_vectored_seconds_count"


def emit(metric: str, value: int | float) -> None:
    print(json.dumps({"metric": metric, "value": value}))


def metric(env: NeonEnv, name: str, filters: dict[str, str]) -> float:
    value = env.pageserver.http_client().get_metric_value(name, filters, aggregate="sum")
    return 0.0 if value is None else value


def make_relation(builder: NeonEnvBuilder, table: str, min_blocks: int, *, striped: bool = False):
    shards = {"initial_tenant_shard_count": 1, "initial_tenant_shard_stripe_size": 1} if striped else {}
    env = builder.init_start(**shards)
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
    return env, (int(dbnode), int(spcnode), int(relnode)), lsn


def run(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    lsn: Lsn,
    relation: tuple[int, int, int],
    size: int,
    mode: str,
    repetitions: int = 1,
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
        "--repetitions",
        str(repetitions),
        "--mode",
        mode,
    ]
    basepath = pg_bin.run_capture(command, with_command_header=False)
    result = json.loads(Path(basepath + ".stdout").read_text())
    assert isinstance(result, dict), result
    return result


def assert_ok(result: dict[str, Any], size: int) -> None:
    assert result["status"] == "ok" and result["reason"] is None
    assert result["response_pages"] == size
    assert result["request_id_matches"] and result["image_byte_mismatches"] == 0


def assert_late_oversized(result: dict[str, Any]) -> None:
    assert (result["status"], result["reason"], result["response_pages"]) == (
        "internal_error",
        "Read error",
        0,
    )
    assert result["request_id_matches"] and result["image_byte_mismatches"] == 0


def test_grpc_get_pages_at_cap_warm_mean(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    repetitions = int(os.environ.get("PERFLOOP_GRPC_GET_PAGE_AT_CAP_REPETITIONS", "1"))
    assert repetitions > 0
    env, relation, lsn = make_relation(neon_env_builder, "perfloop_grpc_get_pages_at_cap", CAP)
    assert_ok(run(pg_bin, neon_binpath, env, lsn, relation, CAP, "raw"), CAP)
    elapsed = 0
    results = []
    for _ in range(repetitions):
        started = perf_counter_ns()
        results.append(run(pg_bin, neon_binpath, env, lsn, relation, CAP, "raw"))
        elapsed += perf_counter_ns() - started
    for result in results:
        assert_ok(result, CAP)
    emit("grpc_get_pages_at_cap_elapsed_ns", elapsed / repetitions)
    emit(
        "grpc_get_pages_at_cap_returned_pages",
        sum(result["response_pages"] for result in results) / repetitions,
    )
    print("verified_grpc_get_pages_at_cap")


def test_grpc_get_pages_frame_boundaries(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(
        neon_env_builder, "perfloop_grpc_get_pages_boundaries", 100, striped=True
    )
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal: Read error.*")
    assert_ok(run(pg_bin, neon_binpath, env, lsn, relation, CAP, "verify"), CAP)
    direct = [run(pg_bin, neon_binpath, env, lsn, relation, size, "verify") for size in (33, 100)]
    direct_completed = all(result["status"] == "ok" for result in direct)
    if direct_completed:
        for size, result in zip((33, 100), direct):
            assert_ok(result, size)
    else:
        for result in direct:
            assert_late_oversized(result)
    env.storage_controller.tenant_shard_split(env.initial_tenant, shard_count=8)
    assert not env.pageserver.tenant_dir(TenantShardId(env.initial_tenant, 0, 1)).exists()
    assert len(env.storage_controller.locate(env.initial_tenant)) == 8
    assert_ok(run(pg_bin, neon_binpath, env, lsn, relation, 100, "verify"), 100)
    print("verified_grpc_get_pages_frame_boundaries")


def test_grpc_get_pages_direct_mixed_goodput(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    repetitions = int(os.environ.get("PERFLOOP_GRPC_GET_PAGE_DIRECT_MIXED_REPETITIONS", "1"))
    assert repetitions > 0
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(neon_env_builder, "perfloop_grpc_get_pages_direct_mixed", 100)
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal: Read error.*")
    verified = run(pg_bin, neon_binpath, env, lsn, relation, 100, "verify")
    direct_completed = verified["status"] == "ok"
    if direct_completed:
        assert_ok(verified, 100)
    else:
        assert_late_oversized(verified)
    filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    started = perf_counter_ns()
    mixed = run(pg_bin, neon_binpath, env, lsn, relation, 100, "mixed", repetitions)
    elapsed = perf_counter_ns() - started
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    assert mixed["normal_pages"] == repetitions * CAP
    if direct_completed:
        assert mixed["wide_pages"] == repetitions * 100
        assert mixed["wide_read_errors"] == 0
        assert vectored_calls >= repetitions * 5
        discarded_timer_starts = 0.0
    else:
        assert mixed["wide_pages"] == 0
        assert mixed["wide_read_errors"] == repetitions
        assert vectored_calls == repetitions
        discarded_timer_starts = timer_starts / repetitions - CAP
    assert timer_starts >= repetitions * (CAP + 100)
    returned_pages = mixed["normal_pages"] + mixed["wide_pages"]
    emit("grpc_get_pages_direct_mixed_goodput_pages_per_second", returned_pages * 1_000_000_000 / elapsed)
    emit("grpc_get_pages_direct_mixed_returned_pages_per_pair", returned_pages / repetitions)
    emit(
        "server_discarded_get_page_timer_starts_per_direct_over_cap_frame",
        discarded_timer_starts,
    )
    emit("server_get_page_timer_starts_per_direct_mixed_pair", timer_starts / repetitions)
    emit("server_get_vectored_calls_per_direct_mixed_pair", vectored_calls / repetitions)
    emit("grpc_get_pages_direct_over_cap_completed", int(direct_completed))
    print("verified_grpc_get_pages_direct_mixed_goodput")


def test_grpc_get_pages_late_chunk_preflight_and_error_boundary(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(neon_env_builder, "perfloop_grpc_get_pages_late", LATE_FRAME)
    env.pageserver.allowed_errors.append(r".*grpc:pageservice.*request failed with Internal:.*")
    warmup = run(pg_bin, neon_binpath, env, lsn, relation, LATE_FRAME, "frame")
    preflight = run(pg_bin, neon_binpath, env, lsn, relation, LATE_FRAME, "frame")
    assert warmup["returned_pages"] == preflight["returned_pages"] == LATE_FRAME
    direct = preflight["oversized_fallbacks"] == 0 and preflight["grpc_request_messages"] == 1
    assert direct == (warmup["oversized_fallbacks"] == 0 and warmup["grpc_request_messages"] == 1)
    assert direct or (preflight["oversized_fallbacks"] == 1 and preflight["grpc_request_messages"] == 5)
    env.pageserver.http_client().configure_failpoints(("ps::grpc-get-pages-final-chunk", "return"))
    failure = run(pg_bin, neon_binpath, env, lsn, relation, LATE_FRAME, "raw")
    assert failure["request_id_matches"] and failure["response_pages"] == 0
    assert failure["status"] == "internal_error"
    if direct:
        assert failure["reason"] == "injected final internal GetPages batch error"
    else:
        assert failure["reason"] == "Read error"
    emit("late_chunk_error_reached_final_chunk", int(failure["reason"] == "injected final internal GetPages batch error"))
    emit("late_chunk_error_empty_pages", int(failure["response_pages"] == 0))
    print("verified_grpc_get_pages_late_chunk_preflight")
