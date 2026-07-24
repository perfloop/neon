from __future__ import annotations

import json
from contextlib import closing
from typing import TYPE_CHECKING

import pytest
from fixtures.neon_fixtures import NeonEnvBuilder, wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build

if TYPE_CHECKING:
    from fixtures.benchmark_fixture import NeonBenchmarker


# The measured tenant has a large, stable catalog and a single hot relation.  The
# setup compaction establishes the stable layer set; only the second compaction is
# timed, after WAL changes one relation.  Keep these constants fixed so samples
# compare the same catalog cardinality and churn volume. The two catalog sizes
# exercise the stable-relation dimension while keeping the hot WAL range fixed.
STABLE_TABLE_COUNTS = (1792, 2048)
HOT_ROWS = 2048
PAYLOAD_BYTES = 512


@skip_in_debug_build("the benchmark compares optimized release compaction")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("stable_tables", STABLE_TABLE_COUNTS)
def test_tiered_compaction_narrow_churn(
    neon_env_builder: NeonEnvBuilder, zenbenchmark: NeonBenchmarker, stable_tables: int
):
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            # The test drives both checkpointing and compaction explicitly.
            "gc_period": "0s",
            "compaction_period": "0s",
            # This makes the post-setup update exceed the repartition interval
            # without turning the measured operation into a bulk-load benchmark.
            "checkpoint_distance": 1024 * 1024,
            "compaction_target_size": 1024 * 1024,
            "compaction_threshold": 1,
            # Benchmark the tiered adapter itself in both arms; it does not
            # depend on a global-default migration.
            "compaction_algorithm": json.dumps({"kind": "tiered"}),
            "image_creation_threshold": 1,
            "image_layer_creation_check_threshold": 0,
        }
    )
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start(
        "main",
        tenant_id=tenant_id,
        config_lines=["max_locks_per_transaction=4096"],
    )

    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE hot (id integer PRIMARY KEY, payload text NOT NULL)")
            cur.execute(
                f"INSERT INTO hot SELECT i, repeat('a', {PAYLOAD_BYTES}) FROM generate_series(1, {HOT_ROWS}) i"
            )
            cur.execute(
                f"""
                DO $$
                DECLARE
                    i integer;
                BEGIN
                    FOR i IN 1..{stable_tables} LOOP
                        EXECUTE format('CREATE TABLE stable_%s (id integer PRIMARY KEY, payload text NOT NULL)', i);
                        EXECUTE format('INSERT INTO stable_%s VALUES (1, %L)', i, 'stable');
                    END LOOP;
                END $$;
                """
            )

    # Establish a stable layer set before the narrow churn. The setup is outside
    # the timed operation; both arms use the same tiered configuration.
    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    pageserver_http.timeline_compact(
        tenant_id,
        timeline_id,
        force_l0_compaction=True,
        wait_until_uploaded=True,
    )

    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE hot SET payload = repeat('b', {PAYLOAD_BYTES})")

    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    l0_before = pageserver_http.layer_map_info(tenant_id, timeline_id).delta_l0_layers()
    assert l0_before, "the narrow update must produce an L0 layer for compaction"

    with zenbenchmark.record_duration(f"narrow_churn_compaction_seconds_stable_{stable_tables}"):
        pageserver_http.timeline_compact(tenant_id, timeline_id, force_l0_compaction=True)

    l0_after = pageserver_http.layer_map_info(tenant_id, timeline_id).delta_l0_layers()
    assert len(l0_after) < len(l0_before), "the timed request must compact the narrow-churn L0"
    assert endpoint.safe_psql("SELECT count(*) FROM hot WHERE payload = repeat('b', 512)") == [
        (HOT_ROWS,)
    ]
    assert endpoint.safe_psql("SELECT count(*) FROM stable_1 WHERE payload = 'stable'") == [(1,)]
