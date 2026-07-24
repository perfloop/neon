from __future__ import annotations

import json
from contextlib import closing
from typing import TYPE_CHECKING

import pytest
from fixtures.benchmark_fixture import MetricReport
from fixtures.neon_fixtures import NeonEnvBuilder, wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build

if TYPE_CHECKING:
    from fixtures.benchmark_fixture import NeonBenchmarker


# A wide post-setup update has more WAL than the stable keyspace. On the
# pre-change planner this selects image materialization; the output metrics make
# that baseline decision and the resulting output range count observable.
STABLE_TABLES = 1792
HOT_ROWS = 2048
UPDATE_ROUNDS = 12
PAYLOAD_BYTES = 512


@skip_in_debug_build("the benchmark compares optimized release compaction")
@pytest.mark.timeout(900)
def test_tiered_compaction_wide_churn(
    neon_env_builder: NeonEnvBuilder, zenbenchmark: NeonBenchmarker
):
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            "gc_period": "0s",
            "compaction_period": "0s",
            "checkpoint_distance": 1024 * 1024,
            "compaction_target_size": 1024 * 1024,
            "compaction_threshold": 1,
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
                    FOR i IN 1..{STABLE_TABLES} LOOP
                        EXECUTE format('CREATE TABLE stable_%s (id integer PRIMARY KEY, payload text NOT NULL)', i);
                        EXECUTE format('INSERT INTO stable_%s VALUES (1, %L)', i, 'stable');
                    END LOOP;
                END $$;
                """
            )

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
            for update in range(UPDATE_ROUNDS):
                marker = chr(ord("b") + update)
                cur.execute(f"UPDATE hot SET payload = repeat('{marker}', {PAYLOAD_BYTES})")

    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    before = pageserver_http.layer_map_info(tenant_id, timeline_id)
    input_l0_layers = before.delta_l0_layers()
    assert input_l0_layers, "the wide update must produce L0 input for compaction"
    before_names = before.historic_by_name()

    with zenbenchmark.record_duration("wide_churn_compaction_seconds"):
        pageserver_http.timeline_compact(tenant_id, timeline_id, force_l0_compaction=True)

    after = pageserver_http.layer_map_info(tenant_id, timeline_id)
    new_layers = [
        layer for layer in after.historic_layers if layer.layer_file_name not in before_names
    ]
    assert new_layers, "the timed compaction must publish output layers"
    assert len(after.delta_l0_layers()) < len(input_l0_layers)

    zenbenchmark.record(
        "wide_churn_input_l0_layers",
        len(input_l0_layers),
        "layers",
        MetricReport.TEST_PARAM,
    )
    zenbenchmark.record(
        "wide_churn_output_ranges",
        len(new_layers),
        "layers",
        MetricReport.TEST_PARAM,
    )
    zenbenchmark.record(
        "wide_churn_new_image_layers",
        sum(layer.kind == "Image" for layer in new_layers),
        "layers",
        MetricReport.TEST_PARAM,
    )
    zenbenchmark.record(
        "wide_churn_new_delta_layers",
        sum(layer.kind == "Delta" for layer in new_layers),
        "layers",
        MetricReport.TEST_PARAM,
    )

    final_marker = chr(ord("b") + UPDATE_ROUNDS - 1)
    assert endpoint.safe_psql(
        f"SELECT count(*) FROM hot WHERE payload = repeat('{final_marker}', {PAYLOAD_BYTES})"
    ) == [(HOT_ROWS,)]
    assert endpoint.safe_psql("SELECT count(*) FROM stable_1 WHERE payload = 'stable'") == [(1,)]
