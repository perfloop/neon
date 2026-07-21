from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from itertools import pairwise
from typing import TYPE_CHECKING

import pytest
from fixtures.benchmark_fixture import MetricReport, NeonBenchmarker
from fixtures.neon_fixtures import NeonEnv, NeonEnvBuilder, wait_for_last_flush_lsn
from fixtures.pageserver.common_types import DeltaLayerName, parse_layer_file_name
from fixtures.pageserver.makelayers.l0stack import L0StackShape, make_l0_stack

if TYPE_CHECKING:
    from fixtures.common_types import TenantId, TimelineId
    from fixtures.neon_fixtures import Endpoint
    from fixtures.pageserver.http import LayerMapInfo


MIB = 1024 * 1024

# This fixture deliberately names the public workload shape rather than describing
# its Values as a total on-disk DeltaEntry count. make_l0_stack verifies the page
# count and every update count through PostgreSQL before compaction runs.
RSS_SHAPE = L0StackShape(logical_table_size_mib=24, delta_stack_height=48)
RSS_EXPECTED_PAGE_UPDATES = RSS_SHAPE.logical_table_size_mib * 128 * RSS_SHAPE.delta_stack_height


# Keep the compaction fixture on the Legacy public runtime path. Background work
# is disabled so the manual timeline_compact call is the only operation measured.
def _layout_tenant_conf(
    checkpoint_distance: int,
    compaction_target_size: int,
) -> dict[str, str | int]:
    return {
        "gc_period": "0s",
        "compaction_period": "0s",
        "checkpoint_distance": checkpoint_distance,
        "compaction_target_size": compaction_target_size,
        "compaction_threshold": 10_000,
        "compaction_upper_limit": 512,
        "image_creation_threshold": 100_000,
        "image_layer_creation_check_threshold": 100_000,
    }


def _flush_l0(
    env: NeonEnv,
    endpoint: Endpoint,
    tenant_id: TenantId,
    timeline_id: TimelineId,
) -> None:
    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    env.pageserver.http_client().timeline_checkpoint(
        tenant_id,
        timeline_id,
        compact=False,
    )


def _canonical_delta_descriptors(layer_map: LayerMapInfo) -> tuple[tuple[int, int, int, int], ...]:
    descriptors: list[tuple[int, int, int, int]] = []
    for layer in layer_map.delta_layers():
        if layer.l0:
            continue
        descriptor = parse_layer_file_name(layer.layer_file_name)
        assert isinstance(descriptor, DeltaLayerName)
        descriptors.append(
            (
                descriptor.key_start.as_int(),
                descriptor.key_end.as_int(),
                descriptor.lsn_start.as_int(),
                descriptor.lsn_end.as_int(),
            )
        )
    return tuple(sorted(descriptors))


def _new_delta_descriptors(
    layer_map: LayerMapInfo,
    existing_layer_names: set[str],
) -> tuple[tuple[int, int, int, int], ...]:
    descriptors: list[tuple[int, int, int, int]] = []
    for layer in layer_map.delta_layers():
        if layer.l0 or layer.layer_file_name in existing_layer_names:
            continue
        descriptor = parse_layer_file_name(layer.layer_file_name)
        assert isinstance(descriptor, DeltaLayerName)
        descriptors.append(
            (
                descriptor.key_start.as_int(),
                descriptor.key_end.as_int(),
                descriptor.lsn_start.as_int(),
                descriptor.lsn_end.as_int(),
            )
        )
    return tuple(sorted(descriptors))


def _restart_and_assert_descriptors(
    env: NeonEnv,
    tenant_id: TenantId,
    timeline_id: TimelineId,
    endpoint: Endpoint,
    expected_descriptors: tuple[tuple[int, int, int, int], ...],
) -> None:
    # These are ordinary fixture lifecycle operations. Do not retain daemons or
    # suppress teardown failures: a clean restart is part of the check.
    endpoint.stop()
    env.pageserver.restart()
    assert _canonical_delta_descriptors(
        env.pageserver.http_client().layer_map_info(tenant_id, timeline_id)
    ) == expected_descriptors


@pytest.mark.timeout(900)
def test_l0_compaction_streaming_hole_layout(neon_env_builder: NeonEnvBuilder):
    """Keep canonical output gaps selected by Legacy L0 compaction after restart."""
    tenant_conf = _layout_tenant_conf(
        checkpoint_distance=16 * MIB,
        compaction_target_size=32 * 1024,
    )
    tenant_conf["image_creation_threshold"] = 1
    tenant_conf["image_layer_creation_check_threshold"] = 0
    env = neon_env_builder.init_start(initial_tenant_conf=tenant_conf)
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start("main", tenant_id=tenant_id)

    with closing(endpoint.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in ("hole_left", "hole_middle_a", "hole_middle_b", "hole_right"):
                cur.execute(f"CREATE TABLE {table} (id integer PRIMARY KEY, payload text)")
                cur.execute(f"INSERT INTO {table} VALUES (1, '{table}-initial')")

            # First materialize small public image ranges across the eventual
            # left-to-right gap. The second L0 batch contains only its endpoints.
            _flush_l0(env, endpoint, tenant_id, timeline_id)
            pageserver_http.timeline_compact(
                tenant_id,
                timeline_id,
                force_l0_compaction=True,
                force_image_layer_creation=True,
            )
            initial_map = pageserver_http.layer_map_info(tenant_id, timeline_id)
            assert len(initial_map.image_layers()) >= 3
            existing_layer_names = initial_map.historic_by_name()

            for marker in range(3):
                cur.execute("UPDATE hole_left SET payload = %s WHERE id = 1", (f"left-{marker}",))
                cur.execute("UPDATE hole_right SET payload = %s WHERE id = 1", (f"right-{marker}",))
                _flush_l0(env, endpoint, tenant_id, timeline_id)

    before_compaction = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert len(before_compaction.delta_l0_layers()) >= 3
    pageserver_http.timeline_compact(tenant_id, timeline_id, force_l0_compaction=True)
    after_compaction = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not after_compaction.delta_l0_layers()

    new_descriptors = _new_delta_descriptors(after_compaction, existing_layer_names)
    assert len(new_descriptors) >= 2
    # With a 16 MiB phase-1 target, this fixture is smaller than one ordinary
    # output layer. A strict key-range gap is therefore the selected-hole split,
    # not a target-size split.
    assert any(left[1] < right[0] for left, right in pairwise(new_descriptors))

    canonical_descriptors = _canonical_delta_descriptors(after_compaction)
    _restart_and_assert_descriptors(
        env,
        tenant_id,
        timeline_id,
        endpoint,
        canonical_descriptors,
    )
    reader = env.endpoints.create_start("reader", tenant_id=tenant_id)
    assert reader.safe_psql("SELECT payload FROM hole_left") == [("left-2",)]
    assert reader.safe_psql("SELECT payload FROM hole_right") == [("right-2",)]
    reader.stop()


@pytest.mark.timeout(900)
def test_l0_compaction_streaming_duplicate_lsn_layout(neon_env_builder: NeonEnvBuilder):
    """Keep canonical same-key LSN slices and values after a normal restart."""
    env = neon_env_builder.init_start(
        initial_tenant_conf=_layout_tenant_conf(
            checkpoint_distance=16 * 1024,
            compaction_target_size=16 * 1024,
        )
    )
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start("main", tenant_id=tenant_id)
    endpoint.config(["full_page_writes=off"])
    endpoint.reconfigure()

    updates = 128
    with closing(endpoint.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE duplicate_lsn "
                "(id integer PRIMARY KEY, payload text) WITH (fillfactor = 10)"
            )
            cur.execute("INSERT INTO duplicate_lsn VALUES (1, %s)", ("initial",))
            _flush_l0(env, endpoint, tenant_id, timeline_id)
            for marker in range(updates):
                payload = f"value-{marker}-" + ("x" * 192)
                cur.execute("UPDATE duplicate_lsn SET payload = %s WHERE id = 1", (payload,))
                assert cur.rowcount == 1
                _flush_l0(env, endpoint, tenant_id, timeline_id)

    before_compaction = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert len(before_compaction.delta_l0_layers()) >= updates
    pageserver_http.timeline_compact(tenant_id, timeline_id, force_l0_compaction=True)
    after_compaction = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not after_compaction.delta_l0_layers()

    canonical_descriptors = _canonical_delta_descriptors(after_compaction)
    by_key_range: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for key_start, key_end, lsn_start, lsn_end in canonical_descriptors:
        by_key_range[(key_start, key_end)].append((lsn_start, lsn_end))

    duplicate_slices = [
        sorted(lsn_ranges)
        for lsn_ranges in by_key_range.values()
        if len(lsn_ranges) > 1
    ]
    assert duplicate_slices
    assert any(
        all(left[1] == right[0] for left, right in pairwise(lsn_ranges))
        for lsn_ranges in duplicate_slices
    )

    _restart_and_assert_descriptors(
        env,
        tenant_id,
        timeline_id,
        endpoint,
        canonical_descriptors,
    )
    reader = env.endpoints.create_start("reader", tenant_id=tenant_id)
    expected_payload = f"value-{updates - 1}-" + ("x" * 192)
    assert reader.safe_psql("SELECT payload FROM duplicate_lsn") == [(expected_payload,)]
    reader.stop()


@pytest.mark.timeout(3600)
def test_l0_compaction_streaming_rss(
    neon_env_builder: NeonEnvBuilder,
    zenbenchmark: NeonBenchmarker,
):
    """Measure the public Legacy L0 compaction boundary at a fixed fan-in."""
    env = neon_env_builder.init_start()
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start("main", tenant_id=tenant_id)

    make_l0_stack(endpoint, RSS_SHAPE)
    before_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    selected_l0_layers = len(before_layers.delta_l0_layers())
    assert selected_l0_layers == RSS_SHAPE.delta_stack_height + 2

    # Stop normal client writes before the measured manual compaction so no new
    # L0 layer can enter the selected batch.
    endpoint.stop()
    before_rss_kib = zenbenchmark.get_peak_mem(env.pageserver)
    with zenbenchmark.record_duration("l0_compaction_elapsed_s"):
        pageserver_http.timeline_compact(tenant_id, timeline_id, force_l0_compaction=True)
    after_rss_kib = zenbenchmark.get_peak_mem(env.pageserver)

    after_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not after_layers.delta_l0_layers()
    assert after_rss_kib >= before_rss_kib
    zenbenchmark.record(
        "l0_compaction_peak_rss_delta_bytes",
        (after_rss_kib - before_rss_kib) * 1024,
        "bytes",
        MetricReport.LOWER_IS_BETTER,
    )
    zenbenchmark.record(
        "l0_compaction_selected_l0_layers",
        selected_l0_layers,
        "layers",
        MetricReport.TEST_PARAM,
    )
    zenbenchmark.record(
        "l0_compaction_expected_page_updates",
        RSS_EXPECTED_PAGE_UPDATES,
        "page_updates",
        MetricReport.TEST_PARAM,
    )
