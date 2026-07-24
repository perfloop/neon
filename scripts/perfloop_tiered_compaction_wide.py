#!/usr/bin/env python3
"""Emit one native wide-churn tiered-compaction proof sample."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import perfloop_tiered_compaction as narrow

PRIMARY_METRIC = "wide_churn_compaction_seconds"
REQUIRED_METRICS = (
    "wide_churn_input_l0_layers",
    "wide_churn_output_ranges",
    "wide_churn_new_image_layers",
    "wide_churn_new_delta_layers",
)
BENCHMARK_NODE = (
    "test_runner/performance/test_tiered_compaction_wide.py::test_tiered_compaction_wide_churn"
)


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def sample() -> None:
    env = narrow.command_environment()
    with narrow.temporary_output_dir(env, "wide-sample") as run_dir:
        out_dir = Path(run_dir) / "native-metrics"
        try:
            narrow.run_pytest(BENCHMARK_NODE, out_dir=out_dir, capture=True)
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr, end="")
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, end="")
            raise

        metric_files = list(out_dir.glob("*.json"))
        if len(metric_files) != 1:
            die(f"expected one native benchmark report, found {len(metric_files)}")
        try:
            report = json.loads(metric_files[0].read_text())
            expected_metrics = (PRIMARY_METRIC, *REQUIRED_METRICS)
            records: dict[str, int | float] = {}
            for result in report["result"]:
                for record in result["data"]:
                    metric = record["name"]
                    if metric not in expected_metrics:
                        continue
                    if metric in records:
                        die(f"native benchmark emitted {metric} more than once")
                    records[metric] = record["value"]
            if set(records) != set(expected_metrics):
                die(f"expected metrics {expected_metrics}, found {tuple(records)}")
            for metric, value in records.items():
                if not isinstance(value, int | float) or value < 0:
                    die(f"invalid {metric} value: {value!r}")
            if records[PRIMARY_METRIC] <= 0:
                die(f"invalid {PRIMARY_METRIC} value: {records[PRIMARY_METRIC]!r}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            die(f"could not parse native benchmark report: {exc}")

        for metric in (PRIMARY_METRIC, *REQUIRED_METRICS):
            print(json.dumps({"metric": metric, "value": records[metric]}, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if args.build:
        # Controller staging worktrees are intentionally remote-free. Resolve the
        # repository-owned relative submodule URLs explicitly before delegating
        # to the established native build.
        env = narrow.command_environment()
        for version in ("v14", "v15", "v16", "v17"):
            narrow.run(
                [
                    "git",
                    "config",
                    f"submodule.vendor/postgres-{version}.url",
                    "https://github.com/perfloop/postgres.git",
                ],
                env=env,
            )
        narrow.build()
    else:
        sample()


if __name__ == "__main__":
    main()
