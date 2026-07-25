from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fixtures.common_types import TenantShardId
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.common_types import Lsn
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin

CAP = 32
MAX_RESPONSE_PAGES = 255
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
    cycle_blocks: list[int] | None = None,
    suffix_block: int | None = None,
    start_block: int = 0,
    shard_number: int = 0,
    shard_count: int = 0,
) -> dict[str, Any]:
    dbnode, spcnode, relnode = relation
    binary = Path(
        os.environ.get(
            "PERFLOOP_GET_PAGES_PREFLIGHT_BIN", str(neon_binpath / "perfloop_get_pages_preflight")
        )
    )
    assert binary.is_file(), f"GetPages preflight helper not found at {binary}"
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
    if cycle_blocks is not None:
        command.extend(("--cycle-blocks", ",".join(map(str, cycle_blocks))))
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


def test_grpc_get_pages_aggregate_frame_bound(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(neon_env_builder, "perfloop_grpc_get_pages_bounded", CAP)
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )
    direct = run(pg_bin, neon_binpath, env, lsn, relation, 100, "raw")
    direct_completed = direct["status"] == "ok"
    if direct_completed:
        assert_ok(direct, 100)
    else:
        assert_late_oversized(direct)
    at_limit = run(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        MAX_RESPONSE_PAGES,
        "raw",
        repeat_block=0,
    )
    if direct_completed:
        assert_ok(at_limit, MAX_RESPONSE_PAGES)
    else:
        assert_late_oversized(at_limit)
    filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
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
    assert bounded["following_pages"] == CAP
    if direct_completed:
        assert bounded == {
            "oversized_status": "invalid_request",
            "oversized_reason": f"GetPages request has {MAX_RESPONSE_PAGES + 1} blocks, limit is {MAX_RESPONSE_PAGES}",
            "oversized_pages": 0,
            "following_pages": CAP,
        }
        assert timer_starts == CAP and vectored_calls == 1
    elif bounded["oversized_status"] == "ok":
        assert bounded["oversized_pages"] == MAX_RESPONSE_PAGES + 1
    else:
        assert bounded == {
            "oversized_status": "internal_error",
            "oversized_reason": "Read error",
            "oversized_pages": 0,
            "following_pages": CAP,
        }
    print("verified_grpc_get_pages_aggregate_frame_bound")


def test_grpc_get_pages_stale_parent_frame_bound(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(
        neon_env_builder, "perfloop_grpc_get_pages_stale_bounded", 100, striped=True
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
    child_blocks = []
    for shard_number in range(8):
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
                shard_number=shard_number,
                shard_count=8,
            )
            if result["status"] == "ok":
                assert_ok(result, 1)
                child_blocks.append(block)
                break
            assert result["status"] == "invalid_request"
            assert result["response_pages"] == 0
            assert "wrong shard" in result["reason"]
        else:
            raise AssertionError(f"no local relation block found for child shard {shard_number}")
    filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    normal = run(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        CAP,
        "raw",
        cycle_blocks=child_blocks,
    )
    assert_ok(normal, CAP)
    normal_timer_starts = metric(env, SMGR, filters) - before_timers
    normal_vectored_calls = (
        metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    )
    assert normal_timer_starts == CAP and normal_vectored_calls > 0
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
        cycle_blocks=child_blocks,
    )
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    assert bounded["following_pages"] == CAP
    if direct_completed:
        assert bounded == {
            "oversized_status": "invalid_request",
            "oversized_reason": f"GetPages request has {MAX_RESPONSE_PAGES + 1} blocks, limit is {MAX_RESPONSE_PAGES}",
            "oversized_pages": 0,
            "following_pages": CAP,
        }
        assert timer_starts == normal_timer_starts
        assert vectored_calls == normal_vectored_calls
    elif bounded["oversized_status"] == "ok":
        assert bounded["oversized_pages"] == MAX_RESPONSE_PAGES + 1
    else:
        assert bounded == {
            "oversized_status": "internal_error",
            "oversized_reason": "Read error",
            "oversized_pages": 0,
            "following_pages": CAP,
        }
    print("verified_grpc_get_pages_stale_parent_frame_bound")


def test_grpc_get_pages_child_suffix_preflight(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, relation, lsn = make_relation(
        neon_env_builder, "perfloop_grpc_get_pages_child_suffix", 100, striped=True
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
    filters = {
        "smgr_query_type": "get_page_at_lsn",
        "tenant_id": str(env.initial_tenant),
        "timeline_id": str(env.initial_timeline),
    }
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
    assert prefixed["following_pages"] == CAP
    if direct_completed:
        assert prefixed["prefixed_status"] == "invalid_request"
        assert prefixed["prefixed_pages"] == 0
        assert f"block {remote_block}" in prefixed["prefixed_reason"]
        assert "wrong shard" in prefixed["prefixed_reason"]
        assert timer_starts == CAP and vectored_calls == 1
    else:
        assert prefixed["prefixed_status"] in {"internal_error", "invalid_request"}
        assert prefixed["prefixed_pages"] == 0
    print("verified_grpc_get_pages_child_suffix_preflight")
