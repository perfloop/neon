from __future__ import annotations

import json
from contextlib import closing

import pytest
from fixtures.benchmark_fixture import MetricReport
from fixtures.neon_fixtures import NeonEnvBuilder, wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build

CHECKPOINT_DISTANCE = 1024 * 1024
MULTI_OUTPUT_TARGET = 1024 * 1024
ONE_OUTPUT_TARGET = 128 * 1024 * 1024
ROWS = 256
CHUNKS = 256
ROUNDS = 14


@skip_in_debug_build("compare release tiered compaction destinations")
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("checkpoint_distance", "target", "output_shape"),
    [
        (ONE_OUTPUT_TARGET, ONE_OUTPUT_TARGET, "one"),
        (4 * CHECKPOINT_DISTANCE, MULTI_OUTPUT_TARGET, "multi_sparse_l0"),
        (CHECKPOINT_DISTANCE, MULTI_OUTPUT_TARGET, "multi_dense_l0"),
    ],
    ids=["one_output", "multi_sparse_l0", "multi_dense_l0"],
)
def test_tiered_compaction_image_destination(
    neon_env_builder: NeonEnvBuilder,
    zenbenchmark,
    checkpoint_distance: int,
    target: int,
    output_shape: str,
):
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            "gc_period": "0s",
            "compaction_period": "0s",
            "checkpoint_distance": checkpoint_distance,
            "compaction_target_size": target,
            "compaction_threshold": 2,
            "compaction_algorithm": json.dumps({"kind": "tiered"}),
            "image_creation_threshold": 1,
            "image_layer_creation_check_threshold": 0,
        }
    )
    tenant, timeline = env.initial_tenant, env.initial_timeline
    http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start("main", tenant_id=tenant)
    payload = "string_agg(md5((id * 100000 + chunk)::text), '' ORDER BY chunk)"
    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE hot (id integer PRIMARY KEY, payload text NOT NULL)")
            cur.execute(
                f"INSERT INTO hot SELECT id, {payload} FROM generate_series(1, {ROWS}) id "
                f"CROSS JOIN generate_series(1, {CHUNKS}) chunk GROUP BY id"
            )

    for round_ in range(ROUNDS):
        with closing(endpoint.connect()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE hot SET payload = (SELECT {payload.replace('100000', str(100000 + (round_ + 1) * 1000))} "
                    f"FROM generate_series(1, {CHUNKS}) chunk)"
                )
        wait_for_last_flush_lsn(env, endpoint, tenant, timeline)
        http.timeline_checkpoint(tenant, timeline, compact=False)

    before = http.layer_map_info(tenant, timeline)
    inputs = before.delta_l0_layers()
    assert len(inputs) >= ROUNDS
    wal_bytes = sum(layer.layer_file_size for layer in inputs)
    response = http.get(
        f"http://localhost:{http.port}/v1/tenant/{tenant}/timeline/{timeline}/keyspace"
    )
    http.verbose_error(response)
    keyspace_bytes = (
        sum(int(end, 16) - int(start, 16) for start, end in response.json()["keys"]) * 8192
    )
    assert wal_bytes > keyspace_bytes
    names = before.historic_by_name()
    offset = env.pageserver.logfile.stat().st_size
    with zenbenchmark.record_duration("image_destination_compaction_seconds"):
        http.timeline_compact(tenant, timeline, force_l0_compaction=True)

    new = [
        layer
        for layer in http.layer_map_info(tenant, timeline).historic_layers
        if layer.layer_file_name not in names
    ]
    deltas = [layer for layer in new if layer.kind == "Delta"]
    images = [layer for layer in new if layer.kind == "Image"]
    events = env.pageserver.logfile.read_text()[offset:].count(
        "covering with images, because keyspace_size"
    )
    for metric, value in {
        "image_destination_target_bytes": target,
        "image_destination_checkpoint_distance_bytes": checkpoint_distance,
        "image_destination_keyspace_bytes": keyspace_bytes,
        "image_destination_wal_bytes": wal_bytes,
        "image_destination_input_l0_layers": len(inputs),
        "image_destination_output_ranges": len(new),
        "image_destination_new_image_layers": len(images),
        "image_destination_new_delta_layers": len(deltas),
        "image_destination_old_image_decision_events": events,
        # The layer-map endpoint does not expose key ranges, so retain a
        # conservative upper bound for the candidate's per-output input-index
        # loads. The workload makes every update broad enough to overlap the
        # output key ranges.
        "image_destination_delta_job_input_layer_load_upper_bound": len(deltas) * len(inputs),
    }.items():
        zenbenchmark.record(metric, value, "count", MetricReport.TEST_PARAM)
    if output_shape == "one":
        assert len(new) == 1, "the near-target case must retain one output job"
    else:
        assert len(new) > 1, "the selected WAL must produce multiple target-sized outputs"
    assert deltas
    assert len(deltas) == len(new)
    assert not images
    assert endpoint.safe_psql("SELECT count(*) FROM hot") == [(ROWS,)]
