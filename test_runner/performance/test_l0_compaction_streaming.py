from __future__ import annotations

import os
from contextlib import closing
from typing import TYPE_CHECKING

import pytest
from fixtures.benchmark_fixture import MetricReport
from fixtures.neon_fixtures import wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build

if TYPE_CHECKING:
    from fixtures.neon_fixtures import Endpoint, NeonEnv, NeonEnvBuilder


PAGE_SIZE = 8192
ROWS_PER_PAGE = 6
L0_PHASE1_METRIC = "pageserver_l0_compaction_phase1_last"


def _flush_to_l0(env: NeonEnv, endpoint: Endpoint, tenant_id, timeline_id) -> None:
    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    env.pageserver.http_client().timeline_checkpoint(tenant_id, timeline_id, compact=False)


def _set_full_page_writes(endpoint: Endpoint, enabled: bool) -> None:
    endpoint.config([f"full_page_writes={'on' if enabled else 'off'}"])
    endpoint.reconfigure()

    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW full_page_writes")
            assert cur.fetchone() == ("on" if enabled else "off",)


def _create_paged_table(endpoint: Endpoint, name: str, page_count: int) -> None:
    assert name.isidentifier()
    rows = page_count * ROWS_PER_PAGE

    with closing(endpoint.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE TABLE {name} (id bigint, value char(92)) WITH (fillfactor=10)")
            cur.execute(
                f"INSERT INTO {name} SELECT i, 'row' || i FROM generate_series(1, {rows}) AS i"
            )
            cur.execute(f"ALTER TABLE {name} SET (fillfactor=100)")
            cur.execute(
                f"""
                WITH tuples_per_page AS (
                    SELECT (ctid::text::point)[0]::bigint AS page_no, count(*) AS tuple_count
                    FROM {name}
                    GROUP BY page_no
                )
                SELECT tuple_count, count(*) FROM tuples_per_page GROUP BY tuple_count
                """
            )
            assert cur.fetchall() == [(ROWS_PER_PAGE, page_count)]


def _update_one_tuple_per_page(
    env: NeonEnv,
    endpoint: Endpoint,
    tenant_id,
    timeline_id,
    name: str,
    update_round: int,
    page_count: int,
) -> None:
    assert name.isidentifier()
    with closing(endpoint.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {name} SET value = value || %s WHERE id %% {ROWS_PER_PAGE} = %s",
                (f",round-{update_round}", update_round % ROWS_PER_PAGE),
            )
            assert cur.rowcount == page_count
    _flush_to_l0(env, endpoint, tenant_id, timeline_id)


def _table_rows(endpoint: Endpoint, names: list[str]) -> dict[str, list[tuple[int, str]]]:
    rows: dict[str, list[tuple[int, str]]] = {}
    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            for name in names:
                assert name.isidentifier()
                cur.execute(f"SELECT id, value FROM {name} ORDER BY id")
                rows[name] = cur.fetchall()
    return rows


def _l0_stat(pageserver_http, tenant_id, timeline_id, stat: str) -> int:
    value = pageserver_http.get_metric_value(
        L0_PHASE1_METRIC,
        {"tenant_id": str(tenant_id), "timeline_id": str(timeline_id), "stat": stat},
    )
    assert value is not None, f"missing structured phase-1 stat {stat}"
    return int(value)


def _compact_l0s(pageserver_http, tenant_id, timeline_id) -> None:
    pageserver_http.timeline_compact(tenant_id, timeline_id, force_l0_compaction=True)


def _assert_new_delta_descriptors(before_layers, after_layers, minimum: int = 1) -> None:
    before_names = before_layers.historic_by_name()
    new_deltas = [
        layer
        for layer in after_layers.delta_layers()
        if not layer.l0 and layer.layer_file_name not in before_names
    ]
    assert len(new_deltas) >= minimum
    assert all(
        layer.lsn_end is not None and layer.lsn_start != layer.lsn_end for layer in new_deltas
    )


def _benchmark_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    assert value > 0
    return value


@skip_in_debug_build("only run with release build")
@pytest.mark.timeout(1800)
def test_l0_compaction_streaming(neon_env_builder: NeonEnvBuilder, zenbenchmark):
    """
    Exercise the legacy L0 metadata path at a real million-entry scale.

    The pageserver reports the exact selected-entry cardinality after compaction;
    the construction estimate is only a lower bound because a WAL layer can also
    contain non-table keys.  The test intentionally varies its SQL update value at
    runtime and consumes the compacted rows after restarting the endpoint.
    """

    update_rounds = _benchmark_int_env("L0_STREAMING_UPDATE_ROUNDS", 128)
    table_mib = _benchmark_int_env("L0_STREAMING_TABLE_MIB", 64)
    page_count = table_mib * 1024 * 1024 // PAGE_SIZE
    expected_table_entries = page_count * update_rounds
    assert expected_table_entries >= 1_000_000

    tenant_conf = {
        "gc_period": "0s",
        "compaction_period": "0s",
        "checkpoint_distance": 64 * 1024 * 1024,
        "compaction_threshold": update_rounds,
        "compaction_upper_limit": update_rounds,
        "image_creation_threshold": 100000,
    }
    env = neon_env_builder.init_start(initial_tenant_conf=tenant_conf)
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start(
        "main", tenant_id=tenant_id, config_lines=["shared_buffers=512MB"]
    )

    _set_full_page_writes(endpoint, enabled=False)
    _create_paged_table(endpoint, "data", page_count)
    _flush_to_l0(env, endpoint, tenant_id, timeline_id)

    # Move setup data out of L0 before building the measured batch.  The normal
    # pageserver restart resets its HWM so the RSS delta is scoped to that batch.
    _compact_l0s(pageserver_http, tenant_id, timeline_id)
    assert not pageserver_http.layer_map_info(tenant_id, timeline_id).delta_l0_layers()
    endpoint.stop()
    env.pageserver.stop()
    env.pageserver.start()
    endpoint.start()

    for update_round in range(update_rounds):
        _update_one_tuple_per_page(
            env,
            endpoint,
            tenant_id,
            timeline_id,
            "data",
            update_round,
            page_count,
        )

    expected_rows = _table_rows(endpoint, ["data"])
    before_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert len(before_layers.delta_l0_layers()) == update_rounds

    endpoint.stop()
    rss_before_kib = pageserver_http.get_metric_value("libmetrics_maxrss_kb")
    assert rss_before_kib is not None
    with zenbenchmark.record_duration("l0_compaction_elapsed_seconds"):
        _compact_l0s(pageserver_http, tenant_id, timeline_id)
    rss_after_kib = pageserver_http.get_metric_value("libmetrics_maxrss_kb")
    assert rss_after_kib is not None
    assert rss_after_kib >= rss_before_kib

    selected_entries = _l0_stat(pageserver_http, tenant_id, timeline_id, "selected_entries")
    selected_layers = _l0_stat(pageserver_http, tenant_id, timeline_id, "selected_l0_layers")
    index_metadata_micros = _l0_stat(
        pageserver_http, tenant_id, timeline_id, "index_metadata_micros"
    )
    assert selected_entries >= expected_table_entries
    assert selected_layers == update_rounds
    assert index_metadata_micros > 0

    after_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not after_layers.delta_l0_layers()
    _assert_new_delta_descriptors(before_layers, after_layers)

    endpoint.start()
    assert _table_rows(endpoint, ["data"]) == expected_rows

    zenbenchmark.record(
        "l0_index_metadata_micros",
        index_metadata_micros,
        "us",
        MetricReport.LOWER_IS_BETTER,
    )
    zenbenchmark.record("l0_selected_entries", selected_entries, "entries", MetricReport.TEST_PARAM)
    zenbenchmark.record("l0_selected_layers", selected_layers, "layers", MetricReport.TEST_PARAM)
    zenbenchmark.record(
        "l0_compaction_rss_growth_bytes",
        (rss_after_kib - rss_before_kib) * 1024,
        "bytes",
        MetricReport.LOWER_IS_BETTER,
    )


@skip_in_debug_build("only run with release build")
@pytest.mark.timeout(900)
def test_l0_compaction_hole_selection(neon_env_builder: NeonEnvBuilder):
    """Use image-covered middle-table gaps and assert that legacy L0 compaction selects them."""

    tenant_conf = {
        "gc_period": "0s",
        "compaction_period": "0s",
        "checkpoint_distance": 64 * 1024,
        "compaction_target_size": 64 * 1024,
        "compaction_threshold": 4,
        "compaction_upper_limit": 4,
        "image_creation_threshold": 1,
        "image_layer_creation_check_threshold": 0,
    }
    env = neon_env_builder.init_start(initial_tenant_conf=tenant_conf)
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start("main", tenant_id=tenant_id)

    _set_full_page_writes(endpoint, enabled=False)
    for name in ["hole_left", "hole_middle", "hole_right"]:
        _create_paged_table(endpoint, name, page_count=128)
    _flush_to_l0(env, endpoint, tenant_id, timeline_id)
    pageserver_http.timeline_compact(
        tenant_id,
        timeline_id,
        force_l0_compaction=True,
        force_repartition=True,
        force_image_layer_creation=True,
    )
    base_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not base_layers.delta_l0_layers()
    # Layer partitioning is implementation-dependent, but the gap must be covered
    # by at least one image before we can verify hole selection below.
    assert base_layers.image_layers()

    for update_round in range(4):
        with closing(endpoint.connect()) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for name in ["hole_left", "hole_right"]:
                    cur.execute(
                        f"UPDATE {name} SET value = value || %s WHERE id %% {ROWS_PER_PAGE} = %s",
                        (f",round-{update_round}", update_round % ROWS_PER_PAGE),
                    )
                    assert cur.rowcount == 128
        # Both tables belong to the same L0, so their key-space gap is present
        # in each of the four selected input layers.
        _flush_to_l0(env, endpoint, tenant_id, timeline_id)

    expected_rows = _table_rows(endpoint, ["hole_left", "hole_middle", "hole_right"])
    before_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert len(before_layers.delta_l0_layers()) == 4

    endpoint.stop()
    _compact_l0s(pageserver_http, tenant_id, timeline_id)

    assert _l0_stat(pageserver_http, tenant_id, timeline_id, "selected_l0_layers") == 4
    assert _l0_stat(pageserver_http, tenant_id, timeline_id, "selected_holes") > 0
    after_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not after_layers.delta_l0_layers()
    _assert_new_delta_descriptors(before_layers, after_layers)

    endpoint.start()
    assert _table_rows(endpoint, ["hole_left", "hole_middle", "hole_right"]) == expected_rows


@skip_in_debug_build("only run with release build")
@pytest.mark.timeout(900)
def test_l0_compaction_duplicate_lsn_splits(neon_env_builder: NeonEnvBuilder):
    """Force a single data-page key through several output LSN segments."""

    update_rounds = 12
    tenant_conf = {
        "gc_period": "0s",
        "compaction_period": "0s",
        "checkpoint_distance": 32 * 1024,
        "compaction_threshold": update_rounds,
        "compaction_upper_limit": update_rounds,
        "image_creation_threshold": 100000,
    }
    env = neon_env_builder.init_start(initial_tenant_conf=tenant_conf)
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    endpoint = env.endpoints.create_start("main", tenant_id=tenant_id)

    _set_full_page_writes(endpoint, enabled=True)
    with closing(endpoint.connect()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE duplicate_key (id bigint PRIMARY KEY, value char(1000))")
            cur.execute("INSERT INTO duplicate_key VALUES (1, %s)", ("initial" + "x" * 993,))
    _flush_to_l0(env, endpoint, tenant_id, timeline_id)
    _compact_l0s(pageserver_http, tenant_id, timeline_id)
    assert not pageserver_http.layer_map_info(tenant_id, timeline_id).delta_l0_layers()

    for update_round in range(update_rounds):
        with closing(endpoint.connect()) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE duplicate_key SET value = %s WHERE id = 1",
                    (f"{update_round:07d}" + "x" * 993,),
                )
                assert cur.rowcount == 1
        _flush_to_l0(env, endpoint, tenant_id, timeline_id)

    expected_rows = _table_rows(endpoint, ["duplicate_key"])
    before_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert len(before_layers.delta_l0_layers()) == update_rounds

    endpoint.stop()
    _compact_l0s(pageserver_http, tenant_id, timeline_id)

    assert _l0_stat(pageserver_http, tenant_id, timeline_id, "selected_l0_layers") == update_rounds
    assert _l0_stat(pageserver_http, tenant_id, timeline_id, "duplicate_lsn_splits") > 0
    after_layers = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert not after_layers.delta_l0_layers()
    _assert_new_delta_descriptors(before_layers, after_layers, minimum=2)

    endpoint.start()
    assert _table_rows(endpoint, ["duplicate_key"]) == expected_rows
