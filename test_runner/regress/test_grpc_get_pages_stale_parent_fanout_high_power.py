from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from fixtures.common_types import TenantShardId
from fixtures.log_helper import log
from fixtures.neon_fixtures import wait_for_last_flush_lsn

if TYPE_CHECKING:
    from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, PgBin


BENCH_BIN_ENV = "PERFLOOP_STALE_PARENT_FANOUT_HIGH_POWER_BENCH_BIN"
REPETITIONS_ENV = "PERFLOOP_STALE_PARENT_FANOUT_HIGH_POWER_REPETITIONS"
SMGR_STARTED_METRIC = "pageserver_smgr_query_started_count_total"
GET_VECTORED_COUNT_METRIC = "pageserver_get_vectored_seconds_count"
MAX_GET_VECTORED_KEYS = 32
MAX_GET_PAGE_FRAME_CHUNKS = 4
FRAME_CAP = MAX_GET_VECTORED_KEYS * MAX_GET_PAGE_FRAME_CHUNKS
CHILD_SHARD_COUNT = 8


def repetitions() -> int:
    value = int(os.environ.get(REPETITIONS_ENV, "32"))
    assert value >= 32, f"{REPETITIONS_ENV} must be at least 32"
    return value


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


def sample_one_frame(
    pg_bin: PgBin,
    neon_binpath: Path,
    env: NeonEnv,
    read_lsn: str,
    relation: tuple[int, int, int],
    smgr_filters: dict[str, str],
    get_vectored_filters: dict[str, str],
) -> tuple[dict[str, Any], int, int]:
    # Each counter bracket encloses exactly one public 128-block stale-parent
    # frame. The setup reference reads and all other repetitions stay outside it.
    smgr_before = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_before = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
    result = run_fanout_frame(
        pg_bin,
        neon_binpath,
        env,
        read_lsn,
        relation,
        "at-cap",
    )
    smgr_after = metric_value(SMGR_STARTED_METRIC, smgr_filters, env)
    vectored_after = metric_value(GET_VECTORED_COUNT_METRIC, get_vectored_filters, env)
    return result, int(smgr_after - smgr_before), int(vectored_after - vectored_before)


def test_grpc_get_pages_stale_parent_fanout_high_power(
    neon_env_builder: NeonEnvBuilder,
    neon_binpath: Path,
    pg_bin: PgBin,
):
    sample_count = repetitions()

    # One-page stripes send consecutive block numbers to each of eight children.
    # Requests use the removed unsharded parent, forcing maybe_split_get_page.
    neon_env_builder.num_pageservers = 1
    neon_env_builder.pageserver_config_override = f"max_get_vectored_keys={MAX_GET_VECTORED_KEYS}"
    env = neon_env_builder.init_start(
        initial_tenant_shard_count=1,
        initial_tenant_shard_stripe_size=1,
    )
    endpoint = env.endpoints.create_start("main")
    endpoint.safe_psql(
        "CREATE TABLE perfloop_grpc_get_pages_fanout_high_power "
        "(id bigint NOT NULL, payload text NOT NULL) WITH (fillfactor = 100)"
    )
    endpoint.safe_psql(
        "INSERT INTO perfloop_grpc_get_pages_fanout_high_power "
        "SELECT n, lpad(n::text, 200, 'x') FROM generate_series(1, 8000) AS n"
    )
    dbnode, spcnode, relnode, relation_blocks = endpoint.safe_psql(
        "SELECT "
        "(SELECT oid FROM pg_database WHERE datname = current_database()), "
        "COALESCE(NULLIF(c.reltablespace, 0), 1663), "
        "c.relfilenode, "
        "pg_relation_size(c.oid) / current_setting('block_size')::integer "
        "FROM pg_class c "
        "WHERE c.oid = 'perfloop_grpc_get_pages_fanout_high_power'::regclass"
    )[0]
    assert relation_blocks >= FRAME_CAP, (
        f"test relation has only {relation_blocks} blocks, need at least {FRAME_CAP}"
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

    relation = (int(dbnode), int(spcnode), int(relnode))
    # Establish image/order equivalence and warm the exact frame outside every
    # timed/counter bracket. The helper obtains public one-page references at the
    # fixed LSN before checking the 128-page stale-parent response.
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
    elapsed_samples: list[int] = []
    timer_starts: list[int] = []
    vectored_calls: list[int] = []

    for _ in range(sample_count):
        result, timers, vectored = sample_one_frame(
            pg_bin,
            neon_binpath,
            env,
            read_lsn,
            relation,
            smgr_filters,
            get_vectored_filters,
        )
        assert result["mode"] == "at_cap"
        assert result["outcome"] == "completed"
        assert result["response_pages"] == FRAME_CAP
        assert timers == FRAME_CAP
        assert vectored == CHILD_SHARD_COUNT
        elapsed_samples.append(int(result["elapsed_ns"]))
        timer_starts.append(timers)
        vectored_calls.append(vectored)

    median_elapsed_ns = float(median(elapsed_samples))
    min_elapsed_ns = min(elapsed_samples)
    max_elapsed_ns = max(elapsed_samples)
    assert min_elapsed_ns > 0
    assert max_elapsed_ns > 0
    assert set(timer_starts) == {FRAME_CAP}
    assert set(vectored_calls) == {CHILD_SHARD_COUNT}

    # One JSONL value per metric is the controller-facing sample. The raw values
    # remain in the captured test output so the review can inspect each frame,
    # while the controller compares independent warm median-of-32 samples.
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_median_elapsed_ns",
                "value": median_elapsed_ns,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_min_elapsed_ns",
                "value": min_elapsed_ns,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_max_elapsed_ns",
                "value": max_elapsed_ns,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_timer_starts_per_frame",
                "value": FRAME_CAP,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_vectored_calls_per_frame",
                "value": CHILD_SHARD_COUNT,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_frames_per_sample",
                "value": sample_count,
            }
        )
    )
    print(
        json.dumps(
            {
                "metric": "stale_parent_fanout_at_cap_high_power_image_byte_mismatches",
                "value": 0,
            }
        )
    )
    print(
        "raw_stale_parent_fanout_at_cap_elapsed_ns="
        + ",".join(str(value) for value in elapsed_samples)
    )
    print(
        "verified_grpc_get_pages_stale_parent_fanout_high_power "
        f"frame_cap={FRAME_CAP} children={CHILD_SHARD_COUNT} repetitions={sample_count}"
    )
    log.info(
        "verified high-power stale-parent fanout median_ns=%s min_ns=%s max_ns=%s "
        "timers_per_frame=%s vectored_per_frame=%s repetitions=%s",
        median_elapsed_ns,
        min_elapsed_ns,
        max_elapsed_ns,
        FRAME_CAP,
        CHILD_SHARD_COUNT,
        sample_count,
    )
