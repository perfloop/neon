from __future__ import annotations

import json
from contextlib import closing

import pytest
from fixtures.neon_fixtures import NeonEnvBuilder, wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build

TARGET_BYTES = 128 * 1024
HISTORY_BYTES = 512 * 1024
HISTORY_CHUNKS = HISTORY_BYTES // 32


def _compact(env, endpoint, tenant_id, timeline_id, pageserver_http) -> None:
    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    assert pageserver_http.layer_map_info(tenant_id, timeline_id).delta_l0_layers()
    pageserver_http.timeline_compact(
        tenant_id,
        timeline_id,
        force_l0_compaction=True,
        wait_until_uploaded=True,
    )


def _write_version(endpoint, version: int) -> tuple[int, int, str]:
    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE history
                SET version = {version},
                    payload = (
                        SELECT string_agg(md5((i + {version} * {HISTORY_CHUNKS})::text), '')
                        FROM generate_series(1, {HISTORY_CHUNKS}) AS i
                    )
                WHERE id = 1
                RETURNING version, length(payload), md5(payload)
                """
            )
            return cur.fetchone()


def _create_anchor_table(endpoint, table: str) -> None:
    with closing(endpoint.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE TABLE "{table}" (id integer PRIMARY KEY, payload text NOT NULL)')
            cur.execute(
                f"""
                INSERT INTO "{table}"
                SELECT id, string_agg(md5((id * 1000 + chunk)::text), '' ORDER BY chunk)
                FROM generate_series(1, 64) AS id
                CROSS JOIN generate_series(1, 32) AS chunk
                GROUP BY id
                """
            )


@skip_in_debug_build("exercise the release tiered compaction path")
@pytest.mark.timeout(900)
def test_tiered_compaction_retains_split_history_after_restart(
    neon_env_builder: NeonEnvBuilder,
):
    """Retain public versions across deliberately sub-value compaction targets."""
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            "gc_period": "0s",
            "compaction_period": "0s",
            "checkpoint_distance": TARGET_BYTES,
            "compaction_target_size": TARGET_BYTES,
            "compaction_threshold": 1,
            "compaction_algorithm": json.dumps({"kind": "tiered"}),
        }
    )
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    main = env.endpoints.create_start("main", tenant_id=tenant_id)

    with closing(main.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE history (id integer PRIMARY KEY, version integer NOT NULL, payload text NOT NULL)"
            )
            cur.execute(
                f"""
                INSERT INTO history
                SELECT 1, 0, string_agg(md5(i::text), '')
                FROM generate_series(1, {HISTORY_CHUNKS}) AS i
                RETURNING version, length(payload), md5(payload)
                """
            )
            expected = {"history_v0": cur.fetchone()}

    _compact(env, main, tenant_id, timeline_id, pageserver_http)
    env.create_branch("history_v0", tenant_id=tenant_id)
    for version in range(1, 4):
        current_value = _write_version(main, version)
        _compact(env, main, tenant_id, timeline_id, pageserver_http)
        if version < 3:
            branch = f"history_v{version}"
            expected[branch] = current_value
            env.create_branch(branch, tenant_id=tenant_id)

    final_layers = pageserver_http.layer_map_info(tenant_id, timeline_id).delta_layers()
    assert len(final_layers) > 1, "the oversized retained row must publish split delta rectangles"
    expected["main"] = current_value
    main.stop()

    env.pageserver.restart(immediate=True)
    for branch, expected_value in expected.items():
        endpoint = env.endpoints.create_start(branch, tenant_id=tenant_id)
        assert endpoint.safe_psql(
            "SELECT version, length(payload), md5(payload) FROM history WHERE id = 1"
        ) == [expected_value]


@skip_in_debug_build("exercise real fanout-one tiered publication")
@pytest.mark.timeout(900)
def test_tiered_compaction_stops_at_singleton_upper_tier(
    neon_env_builder: NeonEnvBuilder,
):
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            "gc_period": "0s",
            "compaction_period": "0s",
            "checkpoint_distance": 1024 * 1024,
            "compaction_target_size": 1024 * 1024,
            "compaction_threshold": 1,
            "compaction_algorithm": json.dumps({"kind": "tiered"}),
        }
    )
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    main = env.endpoints.create_start("main", tenant_id=tenant_id)

    _create_anchor_table(main, "anchor")
    wait_for_last_flush_lsn(env, main, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    initial_map = pageserver_http.layer_map_info(tenant_id, timeline_id)
    assert initial_map.delta_l0_layers()

    offset = env.pageserver.logfile.stat().st_size
    pageserver_http.timeline_compact(
        tenant_id,
        timeline_id,
        force_l0_compaction=True,
        wait_until_uploaded=True,
    )
    # The first L0 pass publishes one upper tier and then reaches L1 in the
    # same planner invocation. This guards the fanout-one early-stop branch,
    # rather than issuing a later request that returns before the planner.
    assert "Level 1 identified" in env.pageserver.logfile.read_text()[offset:]
    final_map = pageserver_http.layer_map_info(tenant_id, timeline_id)
    final_deltas = final_map.delta_layers()
    names = [layer.layer_file_name for layer in final_deltas]
    assert len(names) == len(set(names))
    assert not final_map.delta_l0_layers()
    assert len([layer for layer in final_deltas if not layer.l0]) == 1

    expected = main.safe_psql(
        "SELECT count(*), md5(string_agg(payload, '' ORDER BY id)) FROM anchor"
    )
    main.stop()
    env.pageserver.restart(immediate=True)
    restarted = env.endpoints.create_start("main", tenant_id=tenant_id)
    assert (
        restarted.safe_psql("SELECT count(*), md5(string_agg(payload, '' ORDER BY id)) FROM anchor")
        == expected
    )
