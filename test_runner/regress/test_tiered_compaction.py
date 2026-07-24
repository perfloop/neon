from __future__ import annotations

import json
from contextlib import closing

import pytest
from fixtures.neon_fixtures import NeonEnvBuilder, wait_for_last_flush_lsn
from fixtures.utils import skip_in_debug_build


@skip_in_debug_build("exercise the release compaction path")
@pytest.mark.timeout(900)
def test_tiered_compaction_preserves_branch_history_after_restart(
    neon_env_builder: NeonEnvBuilder,
):
    """Exercise publication, relation-directory metadata, and branch reads after compaction."""
    # Match the measured large-catalog shape.  Smaller sparse inputs exercise a
    # separate known limitation of the unfinished tiered implementation.
    stable_tables = 2048
    hot_rows = 2048
    payload_bytes = 512
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            "gc_period": "0s",
            "compaction_period": "0s",
            "checkpoint_distance": 1024 * 1024,
            "compaction_target_size": 1024 * 1024,
            "compaction_threshold": 1,
            "image_creation_threshold": 1,
            "image_layer_creation_check_threshold": 0,
            "compaction_algorithm": json.dumps({"kind": "tiered"}),
        }
    )
    tenant_id = env.initial_tenant
    timeline_id = env.initial_timeline
    pageserver_http = env.pageserver.http_client()
    main = env.endpoints.create_start(
        "main",
        tenant_id=tenant_id,
        config_lines=["max_locks_per_transaction=4096"],
    )

    with closing(main.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE hot (id integer PRIMARY KEY, payload text NOT NULL)")
            cur.execute(
                f"INSERT INTO hot SELECT i, repeat('a', {payload_bytes}) FROM generate_series(1, {hot_rows}) i"
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

    # Persist an initial layer set before retaining it through a branch.
    wait_for_last_flush_lsn(env, main, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    pageserver_http.timeline_compact(
        tenant_id,
        timeline_id,
        force_l0_compaction=True,
        wait_until_uploaded=True,
    )
    env.create_branch("historic", tenant_id=tenant_id)

    with closing(main.connect()) as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE hot SET payload = repeat('b', {payload_bytes})")

    wait_for_last_flush_lsn(env, main, tenant_id, timeline_id)
    pageserver_http.timeline_checkpoint(tenant_id, timeline_id, compact=False)
    assert pageserver_http.layer_map_info(tenant_id, timeline_id).delta_l0_layers()
    pageserver_http.timeline_compact(
        tenant_id,
        timeline_id,
        force_l0_compaction=True,
        wait_until_uploaded=True,
    )

    assert main.safe_psql(
        f"SELECT count(*) FROM hot WHERE payload = repeat('b', {payload_bytes})"
    ) == [(hot_rows,)]
    main.stop()

    # An immediate restart drops in-memory publication state.  New endpoints then
    # read both the compacted main timeline and the retained branch from layers.
    env.pageserver.restart(immediate=True)
    main_after_restart = env.endpoints.create_start("main", tenant_id=tenant_id)
    historic = env.endpoints.create_start("historic", tenant_id=tenant_id)

    assert main_after_restart.safe_psql(
        f"SELECT count(*) FROM hot WHERE payload = repeat('b', {payload_bytes})"
    ) == [(hot_rows,)]
    assert historic.safe_psql(
        f"SELECT count(*) FROM hot WHERE payload = repeat('a', {payload_bytes})"
    ) == [(hot_rows,)]
    assert main_after_restart.safe_psql(
        f"SELECT count(*) FROM stable_{stable_tables} WHERE payload = 'stable'"
    ) == [(1,)]
    assert historic.safe_psql(
        f"SELECT count(*) FROM stable_{stable_tables} WHERE payload = 'stable'"
    ) == [(1,)]


@skip_in_debug_build("exercise the release collect-keyspace path")
@pytest.mark.timeout(900)
def test_collect_keyspace_reflects_stable_relation_cardinality(
    neon_env_builder: NeonEnvBuilder,
):
    """Confirm that a large stable catalog is visible to the keyspace reconstruction API."""
    stable_tables = 512
    env = neon_env_builder.init_start(
        initial_tenant_conf={
            "gc_period": "0s",
            "compaction_period": "0s",
            "compaction_algorithm": json.dumps({"kind": "tiered"}),
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

    wait_for_last_flush_lsn(env, endpoint, tenant_id, timeline_id)
    response = pageserver_http.get(
        f"{pageserver_http.base_url}/v1/tenant/{tenant_id}/timeline/{timeline_id}/keyspace"
    )
    pageserver_http.verbose_error(response)
    keyspace = response.json()
    assert len(keyspace["keys"]) >= stable_tables
