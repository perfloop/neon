import json

import pytest
from fixtures.benchmark_fixture import MetricReport
from fixtures.neon_fixtures import wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build

MIB = 1024 * 1024
CASES = ((128 * MIB, 128 * MIB, "one"), (MIB, MIB, "multi_dense_l0"))
PAYLOAD = "string_agg(md5((id * 100000 + chunk)::text), '' ORDER BY chunk)"


@skip_in_debug_build("compare release tiered compaction destinations")
@pytest.mark.timeout(900)
@pytest.mark.parametrize(("checkpoint_distance", "target", "shape"), CASES)
def test_tiered_compaction_image_destination(
    neon_env_builder, zenbenchmark, checkpoint_distance, target, shape
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
    endpoint.safe_psql("CREATE TABLE hot (id integer PRIMARY KEY, payload text NOT NULL)")
    endpoint.safe_psql(
        f"INSERT INTO hot SELECT id, {PAYLOAD} FROM generate_series(1,256) id "
        f"CROSS JOIN generate_series(1,256) chunk GROUP BY id"
    )
    for round_ in range(14):
        endpoint.safe_psql(
            f"UPDATE hot SET payload=(SELECT {PAYLOAD.replace('100000', str(101000 + round_ * 1000))} "
            "FROM generate_series(1,256) chunk)"
        )
        wait_for_last_flush_lsn(env, endpoint, tenant, timeline)
        http.timeline_checkpoint(tenant, timeline, compact=False)

    before = http.layer_map_info(tenant, timeline)
    inputs = before.delta_l0_layers()
    wal = sum(layer.layer_file_size for layer in inputs)
    response = http.get(
        f"http://localhost:{http.port}/v1/tenant/{tenant}/timeline/{timeline}/keyspace"
    )
    http.verbose_error(response)
    keyspace = sum(int(end, 16) - int(start, 16) for start, end in response.json()["keys"]) * 8192
    assert len(inputs) >= 14 and all(layer.l0 for layer in inputs) and wal > keyspace
    names, offset = before.historic_by_name(), env.pageserver.logfile.stat().st_size
    prefix = f"image_destination_{shape}"
    with zenbenchmark.record_duration(f"{prefix}_compaction_seconds"):
        http.timeline_compact(tenant, timeline, force_l0_compaction=True)

    new = [
        layer
        for layer in http.layer_map_info(tenant, timeline).historic_layers
        if layer.layer_file_name not in names
    ]
    images = sum(layer.kind == "Image" for layer in new)
    deltas = sum(layer.kind == "Delta" for layer in new)
    events = env.pageserver.logfile.read_text()[offset:].count(
        "covering with images, because keyspace_size"
    )
    for name, value in (
        ("target_bytes", target),
        ("keyspace_bytes", keyspace),
        ("wal_bytes", wal),
        ("input_l0_layers", len(inputs)),
        ("output_ranges", len(new)),
        ("per_output_input_layer_overlaps", len(inputs)),
        ("new_image_layers", images),
        ("new_delta_layers", deltas),
        ("old_image_decision_events", events),
    ):
        zenbenchmark.record(f"{prefix}_{name}", value, "count", MetricReport.TEST_PARAM)
    assert len(new) == 1 if shape == "one" else len(new) > 1
    assert images + deltas == len(new) and bool(images) == bool(events)
    assert endpoint.safe_psql("SELECT count(*) FROM hot") == [(256,)]
