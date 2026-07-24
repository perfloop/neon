#!/usr/bin/env python3
"""Run the native tiered-compaction benchmark as one Perfloop proof sample."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

PRIMARY_METRIC = "narrow_churn_compaction_seconds_stable_2048"
REQUIRED_METRICS = ("narrow_churn_compaction_seconds_stable_1792",)
BENCHMARK_NODE = (
    "test_runner/performance/test_tiered_compaction.py::test_tiered_compaction_narrow_churn"
)
CHECK_NODES = {
    "branch-history": (
        "test_runner/regress/test_tiered_compaction.py::"
        "test_tiered_compaction_preserves_branch_history_after_restart[release-pg17]"
    ),
    "keyspace-shape": (
        "test_runner/regress/test_tiered_compaction.py::"
        "test_collect_keyspace_reflects_stable_relation_cardinality[release-pg17]"
    ),
}


def repository_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "Cargo.toml").is_file() or not (root / "pyproject.toml").is_file():
        raise RuntimeError("run this script from the Neon repository root")
    return root


def command_environment() -> dict[str, str]:
    env = os.environ.copy()
    # Dependency downloads may be shared. The sandbox Cargo launcher keys
    # compiled project artifacts to the current measured worktree.
    env["CARGO_HOME"] = "/workspace/deps/cargo"
    env["POETRY_CACHE_DIR"] = "/workspace/deps/poetry-cache"
    env["POETRY_VIRTUALENVS_IN_PROJECT"] = "true"
    env.pop("CARGO_TARGET_DIR", None)
    return env


def run(
    command: list[str], *, env: dict[str, str], capture: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=repository_root(),
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def cargo_target_dir(env: dict[str, str]) -> Path:
    metadata = run(["cargo", "metadata", "--format-version=1", "--no-deps"], env=env, capture=True)
    return Path(json.loads(metadata.stdout)["target_directory"])


def poetry_command(env: dict[str, str]) -> Path:
    poetry_home = Path("/workspace/deps/perfloop-poetry")
    poetry = poetry_home / "bin" / "poetry"
    if poetry.exists():
        return poetry

    poetry_home.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "venv", str(poetry_home)], check=True)
    subprocess.run(
        [str(poetry_home / "bin" / "pip"), "install", "poetry==2.4.1"],
        check=True,
        env=env,
    )
    return poetry


def install_benchmark_artifact() -> None:
    artifact = os.environ.get("PERFLOOP_BENCH_BIN")
    if artifact is None:
        return
    destination = Path(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repository_root() / "scripts" / "perfloop_tiered_compaction.py", destination)
    destination.chmod(0o755)


def build() -> None:
    env = command_environment()
    run(
        ["git", "submodule", "update", "--init", "--recursive", "--depth", "1", "--jobs", "8"],
        env=env,
    )
    poetry = poetry_command(env)
    run([str(poetry), "install", "--no-interaction", "--no-ansi"], env=env)

    target_dir = cargo_target_dir(env)
    run(
        ["cargo", "build", "--release", "--package", "communicator", "--features", "testing"],
        env=env,
    )
    make_env = env | {
        "BUILD_TYPE": "release",
        "CARGO_BUILD_FLAGS": "--features=testing",
    }
    # The repository Makefile assigns this variable itself, so pass the Cargo
    # launcher's worktree-keyed artifact directory as a command-line override.
    run(
        [
            "make",
            "-s",
            "-j8",
            f"NEON_CARGO_ARTIFACT_TARGET_DIR={target_dir / 'release'}",
            "neon",
        ],
        env=make_env,
    )
    install_benchmark_artifact()


def require_runtime(env: dict[str, str]) -> tuple[Path, Path]:
    root = repository_root()
    python = root / ".venv" / "bin" / "python"
    release_dir = cargo_target_dir(env) / "release"
    if not python.is_file() or not (release_dir / "pageserver").is_file():
        raise RuntimeError("missing benchmark runtime; run --build first")
    return python, release_dir


def revision(env: dict[str, str]) -> str:
    return run(["git", "rev-parse", "HEAD"], env=env, capture=True).stdout.strip()


def test_environment(env: dict[str, str], output_dir: Path, release_dir: Path) -> dict[str, str]:
    test_env = env | {
        "BUILD_TYPE": "release",
        "DEFAULT_PG_VERSION": "17",
        "NEON_BIN": str(release_dir),
        "TEST_OUTPUT": str(output_dir),
        "GITHUB_SHA": revision(env),
    }
    # The benchmark supplies its own per-tenant setting. Do not let a caller
    # silently alter the selected algorithm through the test-fixture override.
    test_env.pop("PAGESERVER_DEFAULT_TENANT_CONFIG_COMPACTION_ALGORITHM", None)
    return test_env


def temporary_output_dir(env: dict[str, str], purpose: str) -> tempfile.TemporaryDirectory[str]:
    base = Path("/dev/shm") / "perfloop-tiered-compaction" / revision(env) / purpose
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="run-", dir=base)


def run_pytest(
    node: str, *, out_dir: Path | None, capture: bool, build_if_missing: bool = False
) -> subprocess.CompletedProcess[str]:
    env = command_environment()
    try:
        python, release_dir = require_runtime(env)
    except RuntimeError:
        if not build_if_missing:
            raise
        build()
        env = command_environment()
        python, release_dir = require_runtime(env)
    with temporary_output_dir(env, "pytest") as run_dir:
        test_output = Path(run_dir) / "test-output"
        test_output.mkdir()
        test_env = test_environment(env, test_output, release_dir)
        command = [str(python), "-m", "pytest", "-q", "--timeout=900"]
        if out_dir is not None:
            out_dir.mkdir()
            command.extend(["--out-dir", str(out_dir)])
        command.append(node)
        return run(command, env=test_env, capture=capture)


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def sample() -> None:
    env = command_environment()
    with temporary_output_dir(env, "sample") as run_dir:
        out_dir = Path(run_dir) / "native-metrics"
        try:
            run_pytest(BENCHMARK_NODE, out_dir=out_dir, capture=True)
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
                if not isinstance(value, int | float) or value <= 0:
                    die(f"invalid {metric} value: {value!r}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            die(f"could not parse native benchmark report: {exc}")

        # Keep the native pytest transcript out of stdout: the controller needs
        # exactly one JSONL object per emitted metric from this invocation.
        for metric in (PRIMARY_METRIC, *REQUIRED_METRICS):
            print(json.dumps({"metric": metric, "value": records[metric]}, separators=(",", ":")))


def planner_unit_check() -> None:
    env = command_environment()
    run(["cargo", "test", "--package", "pageserver_compaction", "--test", "tests"], env=env)


def check(name: str) -> None:
    if name == "planner-unit":
        planner_unit_check()
        return
    if name == "correctness":
        for node in CHECK_NODES.values():
            run_pytest(node, out_dir=None, capture=False, build_if_missing=True)
        planner_unit_check()
        return
    try:
        node = CHECK_NODES[name]
    except KeyError:
        die(f"unknown check: {name}")
    run_pytest(node, out_dir=None, capture=False, build_if_missing=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--sample", action="store_true")
    mode.add_argument("--check", choices=[*CHECK_NODES, "planner-unit", "correctness"])
    args = parser.parse_args()

    if args.build:
        build()
    elif args.sample:
        sample()
    else:
        check(args.check)


if __name__ == "__main__":
    main()
