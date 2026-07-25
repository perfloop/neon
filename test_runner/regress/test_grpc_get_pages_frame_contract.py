from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fixtures.common_types import TenantShardId
from fixtures.neon_fixtures import wait_for_last_flush_lsn
from fixtures.utils import wait_until

if TYPE_CHECKING:
    from fixtures.common_types import Lsn
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin

CAP = 32
MAX_RESPONSE_PAGES = 511
SMGR = "pageserver_smgr_query_started_count_total"
VECTORED = "pageserver_get_vectored_seconds_count"


def metric(env: NeonEnv, name: str, filters: dict[str, str]) -> float:
    value = env.pageserver.http_client().get_metric_value(name, filters, aggregate="sum")
    return 0.0 if value is None else value


def make_relation(builder: NeonEnvBuilder, table: str, min_blocks: int, *, striped: bool = False):
    shards = (
        {"initial_tenant_shard_count": 1, "initial_tenant_shard_stripe_size": 1} if striped else {}
    )
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


def frame_contract_binary(neon_binpath: Path) -> Path:
    configured_binary = os.environ.get("GET_PAGES_FRAME_CONTRACT_BIN")
    binary = (
        Path(configured_binary) if configured_binary else neon_binpath / "get_pages_frame_contract"
    )
    if not binary.is_file() and configured_binary is None:
        subprocess.run(
            ["cargo", "build", "--locked", "-p", "pagebench", "--bin", "get_pages_frame_contract"],
            check=True,
            cwd=Path(__file__).parents[2],
        )
    assert binary.is_file(), f"GetPages frame-contract helper not found at {binary}"
    return binary


def run(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    lsn: Lsn,
    relation: tuple[int, int, int],
    size: int,
    mode: str,
    *,
    repeat_block: int | None = None,
    suffix_block: int | None = None,
    start_block: int = 0,
    shard_number: int = 0,
    shard_count: int = 0,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    binary = frame_contract_binary(neon_binpath)
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
        "--start-block",
        str(start_block),
        "--frame-size",
        str(size),
        "--fallback-chunk-size",
        str(CAP),
        "--mode",
        mode,
        "--shard-number",
        str(shard_number),
        "--shard-count",
        str(shard_count),
    ]
    if repeat_block is not None:
        command.extend(("--repeat-block", str(repeat_block)))
    if suffix_block is not None:
        command.extend(("--suffix-block", str(suffix_block)))
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


def frame_filters(env: NeonEnv) -> dict[str, str]:
    return {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }


def sample_direct_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    lsn: Lsn,
    relation: tuple[int, int, int],
    size: int,
    *,
    repeat_block: int | None = None,
) -> tuple[dict[str, Any], float, float, int]:
    filters = frame_filters(env)
    frame_contract_binary(neon_binpath)
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    started = time.perf_counter_ns()
    result = run(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        size,
        "raw",
        repeat_block=repeat_block,
    )
    elapsed_ns = time.perf_counter_ns() - started
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    return result, timer_starts, vectored_calls, elapsed_ns


def emit_control_metric(name: str, value: float | int) -> None:
    print(json.dumps({"metric": name, "value": value}))


def assert_bounded_rejection(result: dict[str, Any]) -> None:
    assert result == {
        "oversized_status": "invalid_request",
        "oversized_reason": f"GetPages request has {MAX_RESPONSE_PAGES + 1} blocks, limit is {MAX_RESPONSE_PAGES}",
        "oversized_pages": 0,
        "following_pages": CAP,
    }


def assert_frame_holds_gc_cutoff(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    lsn: Lsn,
    relation: tuple[int, int, int],
) -> None:
    failpoint = "ps::grpc-get-pages-between-chunks"
    client = env.pageserver.http_client()
    client.configure_failpoints((failpoint, "pause"))
    _, configured_at = wait_until(
        lambda: env.pageserver.assert_log_contains(f"cfg failpoint: {failpoint} pause"), timeout=20
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        frame = executor.submit(run, pg_bin, neon_binpath, env, lsn, relation, CAP + 1, "raw")
        try:
            wait_until(
                lambda: env.pageserver.assert_log_contains(
                    f"at failpoint {failpoint}", configured_at
                ),
                timeout=20,
            )
            # This test-only API advances the same RCU cutoff that real GC updates. The request
            # began below the new cutoff, so a second chunk must retain the old guard to succeed.
            client.timeline_patch_index_part(
                env.initial_tenant,
                env.initial_timeline,
                {"applied_gc_cutoff_lsn": str(lsn + 1)},
            )
        finally:
            client.configure_failpoints((failpoint, "off"))
        assert_ok(frame.result(timeout=20), CAP + 1)


def test_grpc_get_pages_frame_contract(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(neon_env_builder, "grpc_get_pages_frame_contract", 100)
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )

    direct_33, _timers_33, vectored_33, elapsed_33 = sample_direct_frame(
        pg_bin, neon_binpath, env, lsn, relation, CAP + 1
    )
    direct_100 = run(pg_bin, neon_binpath, env, lsn, relation, 100, "raw")
    direct_at_limit, _timers_511, vectored_511, elapsed_511 = sample_direct_frame(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        MAX_RESPONSE_PAGES,
        repeat_block=0,
    )

    direct_completed = direct_33["status"] == "ok"
    if direct_completed:
        assert_ok(direct_33, CAP + 1)
        assert_ok(direct_100, 100)
        assert_ok(direct_at_limit, MAX_RESPONSE_PAGES)
        assert vectored_33 == 2
        assert vectored_511 == 16
    else:
        assert_late_oversized(direct_33)
        assert_late_oversized(direct_100)
        assert_late_oversized(direct_at_limit)

    filters = frame_filters(env)
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    bounded = run(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        MAX_RESPONSE_PAGES + 1,
        "bounded",
        repeat_block=0,
    )
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    if direct_completed:
        assert_bounded_rejection(bounded)
        assert timer_starts == CAP and vectored_calls == 1
        assert_frame_holds_gc_cutoff(pg_bin, neon_binpath, env, lsn, relation)
    elif bounded["oversized_status"] == "ok":
        assert bounded["oversized_pages"] == MAX_RESPONSE_PAGES + 1
        assert bounded["following_pages"] == CAP
    else:
        assert bounded == {
            "oversized_status": "internal_error",
            "oversized_reason": "Read error",
            "oversized_pages": 0,
            "following_pages": CAP,
        }

    emit_control_metric("grpc_get_pages_direct_33_returned_pages", direct_33["response_pages"])
    emit_control_metric("server_get_vectored_calls_per_direct_33_frame", vectored_33)
    emit_control_metric("grpc_get_pages_direct_33_request_ns", elapsed_33)
    emit_control_metric(
        "grpc_get_pages_direct_511_returned_pages", direct_at_limit["response_pages"]
    )
    emit_control_metric("server_get_vectored_calls_per_direct_511_frame", vectored_511)
    emit_control_metric("grpc_get_pages_direct_511_request_ns", elapsed_511)
    print("verified_grpc_get_pages_frame_contract")


def test_grpc_get_pages_stale_parent_frame_contract(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(
        neon_env_builder, "grpc_get_pages_stale_parent_frame_contract", 100, striped=True
    )
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )
    direct = run(pg_bin, neon_binpath, env, lsn, relation, 100, "raw")
    direct_completed = direct["status"] == "ok"
    if direct_completed:
        assert_ok(direct, 100)
    else:
        assert_late_oversized(direct)

    env.storage_controller.tenant_shard_split(env.initial_tenant, shard_count=8)
    assert not env.pageserver.tenant_dir(TenantShardId(env.initial_tenant, 0, 1)).exists()
    assert len(env.storage_controller.locate(env.initial_tenant)) == 8

    filters = frame_filters(env)
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    bounded = run(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        MAX_RESPONSE_PAGES + 1,
        "bounded",
        repeat_block=0,
    )
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    if direct_completed:
        assert_bounded_rejection(bounded)
        assert timer_starts == CAP and vectored_calls == 1
    elif bounded["oversized_status"] == "ok":
        assert bounded["oversized_pages"] == MAX_RESPONSE_PAGES + 1
        assert bounded["following_pages"] == CAP
    else:
        assert bounded == {
            "oversized_status": "internal_error",
            "oversized_reason": "Read error",
            "oversized_pages": 0,
            "following_pages": CAP,
        }
    print("verified_grpc_get_pages_stale_parent_frame_contract")


def test_grpc_get_pages_child_suffix_frame_contract(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(
        neon_env_builder, "grpc_get_pages_child_suffix_frame_contract", 100, striped=True
    )
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )
    direct = run(pg_bin, neon_binpath, env, lsn, relation, 100, "raw")
    direct_completed = direct["status"] == "ok"
    if direct_completed:
        assert_ok(direct, 100)
    else:
        assert_late_oversized(direct)

    env.storage_controller.tenant_shard_split(env.initial_tenant, shard_count=8)
    local_block = None
    remote_block = None
    for block in range(100):
        result = run(
            pg_bin,
            neon_binpath,
            env,
            lsn,
            relation,
            1,
            "raw",
            start_block=block,
            shard_count=8,
        )
        if result["status"] == "ok":
            assert_ok(result, 1)
            local_block = block
        else:
            assert result["status"] == "invalid_request"
            assert result["response_pages"] == 0
            assert result["request_id_matches"] and result["image_byte_mismatches"] == 0
            assert "wrong shard" in result["reason"]
            remote_block = block
        if local_block is not None and remote_block is not None:
            break
    assert local_block is not None and remote_block is not None

    filters = frame_filters(env)
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    prefixed = run(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        CAP,
        "suffix",
        repeat_block=local_block,
        suffix_block=remote_block,
        shard_count=8,
    )
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    if direct_completed:
        assert prefixed["prefixed_status"] == "invalid_request"
        assert prefixed["prefixed_pages"] == 0
        assert f"block {remote_block}" in prefixed["prefixed_reason"]
        assert "wrong shard" in prefixed["prefixed_reason"]
        assert prefixed["following_pages"] == CAP
        assert timer_starts == CAP and vectored_calls == 1
    else:
        assert prefixed["prefixed_status"] in {"internal_error", "invalid_request"}
        assert prefixed["prefixed_pages"] == 0
    print("verified_grpc_get_pages_child_suffix_frame_contract")
