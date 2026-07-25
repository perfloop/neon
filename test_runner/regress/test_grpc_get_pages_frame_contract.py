from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fixtures.common_types import Lsn, TenantShardId
from fixtures.neon_fixtures import wait_for_last_flush_lsn
from fixtures.utils import wait_until

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin

CAP = 32
MAX_RESPONSE_PAGES = 511
MAX_GET_PAGES_DECODING_MESSAGE_SIZE = MAX_RESPONSE_PAGES * 6 + 80
SMGR = "pageserver_smgr_query_started_count_total"
VECTORED = "pageserver_get_vectored_seconds_count"


def metric(env: NeonEnv, name: str, filters: dict[str, str]) -> float:
    value = env.pageserver.http_client().get_metric_value(name, filters, aggregate="sum")
    return 0.0 if value is None else value


def make_relation(
    builder: NeonEnvBuilder,
    table: str,
    min_blocks: int,
    *,
    striped: bool = False,
    tenant_conf: dict[str, str] | None = None,
):
    init_kwargs: dict[str, Any] = {}
    if striped:
        init_kwargs.update(
            {
                "initial_tenant_shard_count": 1,
                "initial_tenant_shard_stripe_size": 1,
            }
        )
    if tenant_conf is not None:
        init_kwargs["initial_tenant_conf"] = tenant_conf
    env = builder.init_start(**init_kwargs)
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
    ingress_compression: str = "identity",
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
        "--ingress-compression",
        ingress_compression,
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
    timer_starts = metric(env, SMGR, filters) - before_timers
    vectored_calls = metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) - before_vectored
    reference_pages = result["reference_pages"]
    assert reference_pages == (1 if repeat_block is not None else size)
    # The helper obtains independent single-page references before timing the public frame.
    # Account for that setup so the returned counters describe only the requested frame.
    assert (timer_starts, vectored_calls) == (
        size + reference_pages,
        (size + CAP - 1) // CAP + reference_pages,
    )
    return (
        result,
        timer_starts - reference_pages,
        vectored_calls - reference_pages,
        result["request_elapsed_ns"],
    )


def emit_control_metric(name: str, value: float | int) -> None:
    print(json.dumps({"metric": name, "value": value}))


def assert_bounded_rejection(result: dict[str, Any]) -> None:
    assert {
        key: result[key]
        for key in ("oversized_status", "oversized_reason", "oversized_pages", "following_pages")
    } == {
        "oversized_status": "invalid_request",
        "oversized_reason": f"GetPages request has {MAX_RESPONSE_PAGES + 1} blocks, limit is {MAX_RESPONSE_PAGES}",
        "oversized_pages": 0,
        "following_pages": CAP,
    }
    assert result["following_image_byte_mismatches"] == 0
    assert result["reference_pages"] == 1


def assert_inbound_frame_bound(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    lsn: Lsn,
    relation: tuple[int, int, int],
) -> None:
    """Exercise the identity decoder cap and rejected compressed expansion before page work."""

    client = env.pageserver.http_client()
    failpoint = "ps::grpc-get-pages-after-decode"
    filters = frame_filters(env)
    before_timers = metric(env, SMGR, filters)
    before_vectored = metric(env, VECTORED, {"task_kind": "PageRequestHandler"})
    client.configure_failpoints((failpoint, "return"))
    try:
        # Find adjacent packed repeated-field frame sizes from their actual protobuf encodings.
        # Requests below the cap reach the post-decode hook, which returns the server Vec capacity;
        # requests above it must be rejected by Tonic before that hook can run.
        below_count = 1
        above_count = MAX_GET_PAGES_DECODING_MESSAGE_SIZE + 1
        while below_count + 1 < above_count:
            count = (below_count + above_count) // 2
            result = run(
                pg_bin,
                neon_binpath,
                env,
                lsn,
                relation,
                count,
                "ingress",
                repeat_block=0,
            )
            if result["uncompressed_proto_bytes"] <= MAX_GET_PAGES_DECODING_MESSAGE_SIZE:
                assert result["response_status"] == "invalid_request"
                assert result["response_pages"] == 0
                assert result["transport_status"] is None
                assert result["post_decode_reached"]
                assert result["post_decode_blocks"] == count
                assert result["decoded_block_numbers_capacity_bytes"] >= count * 4
                below_count = count
            else:
                assert result["response_status"] is None
                assert result["transport_status"] == "out_of_range"
                assert not result["post_decode_reached"]
                above_count = count

        below = run(
            pg_bin,
            neon_binpath,
            env,
            lsn,
            relation,
            below_count,
            "ingress",
            repeat_block=0,
        )
        above = run(
            pg_bin,
            neon_binpath,
            env,
            lsn,
            relation,
            above_count,
            "ingress",
            repeat_block=0,
        )
        assert above_count == below_count + 1
        assert below["uncompressed_proto_bytes"] <= MAX_GET_PAGES_DECODING_MESSAGE_SIZE
        assert above["uncompressed_proto_bytes"] > MAX_GET_PAGES_DECODING_MESSAGE_SIZE
        assert below["response_status"] == "invalid_request"
        assert below["response_pages"] == 0
        assert below["transport_status"] is None
        assert below["post_decode_reached"]
        assert below["post_decode_blocks"] == below_count
        assert below["decoded_block_numbers_capacity_bytes"] >= below_count * 4
        assert below["request_elapsed_ns"] > 0
        assert above["response_status"] is None
        assert above["transport_status"] == "out_of_range"
        assert not above["post_decode_reached"]
        assert above["request_elapsed_ns"] > 0

        # These generated-client requests carry real gzip/zstd gRPC frames. A highly
        # compressible body hundreds of times larger than the identity cap remains below that cap
        # on the wire. PageService rejects the unsupported request encoding before Tonic
        # decompression and before the post-decode hook can observe a repeated-field Vec.
        compressed_count = above_count * 256
        for compression in ("gzip", "zstd"):
            compressed = run(
                pg_bin,
                neon_binpath,
                env,
                lsn,
                relation,
                compressed_count,
                "ingress",
                repeat_block=0,
                ingress_compression=compression,
            )
            assert compressed["compression"] == compression
            assert compressed["frame_size"] == compressed_count
            assert compressed["uncompressed_proto_bytes"] > (
                MAX_GET_PAGES_DECODING_MESSAGE_SIZE * 128
            )
            assert compressed["compressed_payload_bytes"] < MAX_GET_PAGES_DECODING_MESSAGE_SIZE
            assert compressed["response_status"] is None
            assert compressed["transport_status"] == "unimplemented"
            assert not compressed["post_decode_reached"]
            assert compressed["request_elapsed_ns"] > 0
            emit_control_metric(
                f"grpc_get_pages_{compression}_compressed_expansion_proto_bytes",
                compressed["uncompressed_proto_bytes"],
            )
            emit_control_metric(
                f"grpc_get_pages_{compression}_compressed_expansion_payload_bytes",
                compressed["compressed_payload_bytes"],
            )
            emit_control_metric(
                f"grpc_get_pages_{compression}_compressed_expansion_request_ns",
                compressed["request_elapsed_ns"],
            )

        emit_control_metric(
            "grpc_get_pages_identity_decoder_near_limit_proto_bytes",
            below["uncompressed_proto_bytes"],
        )
        emit_control_metric(
            "grpc_get_pages_identity_decoder_near_limit_vec_capacity_bytes",
            below["decoded_block_numbers_capacity_bytes"],
        )
        emit_control_metric(
            "grpc_get_pages_identity_decoder_near_limit_request_ns",
            below["request_elapsed_ns"],
        )
        emit_control_metric(
            "grpc_get_pages_identity_decoder_over_limit_request_ns",
            above["request_elapsed_ns"],
        )
    finally:
        client.configure_failpoints((failpoint, "off"))

    assert metric(env, SMGR, filters) == before_timers
    assert metric(env, VECTORED, {"task_kind": "PageRequestHandler"}) == before_vectored
    print("verified_grpc_get_pages_inbound_frame_bound")


def assert_frame_holds_gc_cutoff(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    endpoint: Any,
    request_lsn: Lsn,
    relation: tuple[int, int, int],
    table: str,
) -> None:
    client = env.pageserver.http_client()
    # Materialize a version after request_lsn so immediate GC can advance the cutoff past the
    # paused frame's read point. Both checkpoints make the GC operate on actual timeline history.
    client.timeline_checkpoint(env.initial_tenant, env.initial_timeline)
    endpoint.safe_psql(f"UPDATE {table} SET payload = payload || 'g' WHERE id = 1")
    wait_for_last_flush_lsn(env, endpoint, env.initial_tenant, env.initial_timeline)
    client.timeline_checkpoint(env.initial_tenant, env.initial_timeline)
    before_cutoff = Lsn(
        client.timeline_detail(env.initial_tenant, env.initial_timeline)["applied_gc_cutoff_lsn"]
    )
    assert before_cutoff <= request_lsn

    frame_failpoint = "ps::grpc-get-pages-between-chunks"
    gc_phase_failpoint = "timeline-gc-after-cutoff-store"
    client.configure_failpoints((frame_failpoint, "pause"))
    _, configured_at = wait_until(
        lambda: env.pageserver.assert_log_contains(f"cfg failpoint: {frame_failpoint} pause"),
        timeout=20,
    )
    client.configure_failpoints((gc_phase_failpoint, "return"))
    _, gc_phase_configured_at = wait_until(
        lambda: env.pageserver.assert_log_contains(f"cfg failpoint: {gc_phase_failpoint} return"),
        timeout=20,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        frame = executor.submit(
            run,
            pg_bin,
            neon_binpath,
            env,
            request_lsn,
            relation,
            CAP + 1,
            "raw",
        )
        gc_completion = None
        try:
            wait_until(
                lambda: env.pageserver.assert_log_contains(
                    f"at failpoint {frame_failpoint}", configured_at
                ),
                timeout=20,
            )
            gc_completion = executor.submit(
                client.timeline_gc, env.initial_tenant, env.initial_timeline, 0
            )
            wait_until(
                lambda: Lsn(
                    client.timeline_detail(env.initial_tenant, env.initial_timeline)[
                        "applied_gc_cutoff_lsn"
                    ]
                )
                > request_lsn,
                timeout=20,
            )
            # This failpoint is evaluated after GC stores the new cutoff and before it awaits the
            # RCU waitlist. The nonzero generation count proves the paused public frame still owns
            # the old cutoff view at that precise lifecycle boundary; RcuWaitList::wait cannot
            # complete until the frame is released below.
            wait_until(
                lambda: env.pageserver.assert_log_contains(
                    r"GC cutoff stored before waiting for old readers; [1-9][0-9]* active reader generations",
                    gc_phase_configured_at,
                ),
                timeout=20,
            )
        finally:
            client.configure_failpoints((frame_failpoint, "off"))
            client.configure_failpoints((gc_phase_failpoint, "off"))

        assert_ok(frame.result(timeout=30), CAP + 1)
        assert gc_completion is not None
        gc_completion.result(timeout=30)


def test_grpc_get_pages_frame_contract(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    table = "grpc_get_pages_frame_contract"
    env, endpoint, relation, lsn = make_relation(
        neon_env_builder,
        table,
        100,
        tenant_conf={"pitr_interval": "0 sec", "lsn_lease_length": "0s"},
    )
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument|OutOfRange|Unimplemented):.*"
    )

    direct_33, timer_starts_33, vectored_33, elapsed_33 = sample_direct_frame(
        pg_bin, neon_binpath, env, lsn, relation, CAP + 1
    )
    direct_100, timer_starts_100, vectored_100, elapsed_100 = sample_direct_frame(
        pg_bin, neon_binpath, env, lsn, relation, 100
    )
    direct_255, timer_starts_255, vectored_255, elapsed_255 = sample_direct_frame(
        pg_bin, neon_binpath, env, lsn, relation, 255, repeat_block=0
    )
    direct_at_limit, timer_starts_511, vectored_511, elapsed_511 = sample_direct_frame(
        pg_bin,
        neon_binpath,
        env,
        lsn,
        relation,
        MAX_RESPONSE_PAGES,
        repeat_block=0,
    )

    for result, size in (
        (direct_33, CAP + 1),
        (direct_100, 100),
        (direct_255, 255),
        (direct_at_limit, MAX_RESPONSE_PAGES),
    ):
        assert_ok(result, size)
    assert (timer_starts_33, vectored_33) == (CAP + 1, 2)
    assert (timer_starts_100, vectored_100) == (100, 4)
    assert (timer_starts_255, vectored_255) == (255, 8)
    assert (timer_starts_511, vectored_511) == (MAX_RESPONSE_PAGES, 16)

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
    assert_bounded_rejection(bounded)
    # Apart from one independent reference, only the following valid 32-page request does work.
    assert (
        timer_starts - bounded["reference_pages"],
        vectored_calls - bounded["reference_pages"],
    ) == (
        CAP,
        1,
    )
    assert_inbound_frame_bound(pg_bin, neon_binpath, env, lsn, relation)
    assert_frame_holds_gc_cutoff(pg_bin, neon_binpath, env, endpoint, lsn, relation, table)

    for size, result, vectored_calls, elapsed_ns in (
        (CAP + 1, direct_33, vectored_33, elapsed_33),
        (100, direct_100, vectored_100, elapsed_100),
        (255, direct_255, vectored_255, elapsed_255),
        (MAX_RESPONSE_PAGES, direct_at_limit, vectored_511, elapsed_511),
    ):
        emit_control_metric(
            f"grpc_get_pages_direct_{size}_returned_pages", result["response_pages"]
        )
        emit_control_metric(f"server_get_vectored_calls_per_direct_{size}_frame", vectored_calls)
        emit_control_metric(f"grpc_get_pages_direct_{size}_request_ns", elapsed_ns)
    print("verified_grpc_get_pages_frame_contract")


def test_grpc_get_pages_stale_parent_frame_contract(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, _endpoint, relation, lsn = make_relation(
        neon_env_builder, "grpc_get_pages_stale_parent_frame_contract", 100, striped=True
    )
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )
    direct = run(pg_bin, neon_binpath, env, lsn, relation, 100, "raw")
    assert_ok(direct, 100)

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
    assert_bounded_rejection(bounded)
    # Bound validation is on the original public frame, before stale-parent routing.
    assert (
        timer_starts - bounded["reference_pages"],
        vectored_calls - bounded["reference_pages"],
    ) == (
        CAP,
        1,
    )
    print("verified_grpc_get_pages_stale_parent_frame_contract")


def test_grpc_get_pages_child_suffix_frame_contract(
    neon_env_builder: NeonEnvBuilder, neon_binpath: Path, pg_bin: PgBin
):
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={CAP}"
    env, _endpoint, relation, lsn = make_relation(
        neon_env_builder, "grpc_get_pages_child_suffix_frame_contract", 100, striped=True
    )
    env.pageserver.allowed_errors.append(
        r".*grpc:pageservice.*request failed with (Internal|InvalidArgument):.*"
    )
    direct = run(pg_bin, neon_binpath, env, lsn, relation, 100, "raw")
    assert_ok(direct, 100)

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
            "probe",
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
    assert prefixed["prefixed_status"] == "invalid_request"
    assert prefixed["prefixed_pages"] == 0
    assert f"block {remote_block}" in prefixed["prefixed_reason"]
    assert "wrong shard" in prefixed["prefixed_reason"]
    assert prefixed["following_pages"] == CAP
    assert prefixed["following_image_byte_mismatches"] == 0
    assert prefixed["reference_pages"] == 1
    # Apart from one independent reference, the rejected request must not start work.
    assert (
        timer_starts - prefixed["reference_pages"],
        vectored_calls - prefixed["reference_pages"],
    ) == (
        CAP,
        1,
    )
    print("verified_grpc_get_pages_child_suffix_frame_contract")
