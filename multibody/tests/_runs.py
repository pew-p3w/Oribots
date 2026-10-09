"""Helpers for the check-suite tests that start real training runs.

Named with a leading underscore so the runner's ``test_*.py`` glob skips it.
"""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import contest_paths

ROOT = contest_paths.ROOT
MULTIBODY = contest_paths.MULTIBODY
RUNNER = MULTIBODY / "run_multibody.py"
TINY = MULTIBODY / "configs" / "_tiny.py"
FIXTURES = MULTIBODY / "tests" / "fixtures"


def tiny_config(workspace: Path, name: str = "tiny_run", **overrides: Any) -> Path:
    """
    Write a copy of ``configs/_tiny.py`` (absolute fixture paths) with settings replaced.

    :returns: The config path.
    """
    lines = []
    for line in TINY.read_text(encoding="utf-8").splitlines():
        setting = line.split("=")[0].strip() if "=" in line and not line.startswith((" ", "#")) else None
        if setting in overrides:
            continue
        lines.append(line)
    text = "\n".join(lines).replace('"../tests/fixtures/', f'"{FIXTURES}/')
    for setting, value in overrides.items():
        text += f"\n{setting} = {value!r}"
    path = workspace / f"{name}.py"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def start(args: list[str], log: Path) -> subprocess.Popen:
    """Start ``run_multibody.py`` with ``args`` from the Oribots root; output goes to ``log``."""
    handle = open(log, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(RUNNER), *args], cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT
    )


def run(args: list[str], log: Path, timeout: float = 1200) -> int:
    """Run ``run_multibody.py`` to completion; returns its exit status."""
    process = start(args, log)
    return process.wait(timeout=timeout)


def wait_for(path: Path, process: subprocess.Popen, timeout: float = 600) -> None:
    """Wait until ``path`` exists (or fail if the process ends first)."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise AssertionError(f"the run ended (status {process.returncode}) before {path.name} appeared")
        if time.monotonic() > deadline:
            raise AssertionError(f"{path.name} did not appear within {timeout} s")
        time.sleep(0.2)


def descendants(pid: int) -> list[int]:
    """All descendant process ids of ``pid``."""
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2:
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, stack = [], list(children.get(pid, []))
    while stack:
        current = stack.pop()
        found.append(current)
        stack.extend(children.get(current, []))
    return found


def alive(pids: list[int]) -> list[int]:
    """The pids that still exist (zombies count as gone)."""
    result = []
    for pid in pids:
        try:
            state = subprocess.check_output(["ps", "-o", "stat=", "-p", str(pid)], text=True).strip()
        except subprocess.CalledProcessError:
            continue
        if state and not state.startswith("Z"):
            result.append(pid)
    return result


def kill_training_tree(runner: subprocess.Popen, signum: int) -> None:
    """Send ``signum`` to the training child's process group (child and workers) and the runner."""
    for pid in descendants(runner.pid):
        try:
            os.killpg(pid, signum)
        except (ProcessLookupError, PermissionError):
            continue
    try:
        runner.send_signal(signum)
    except ProcessLookupError:
        pass


def record_rows(folder: Path) -> dict[str, list[str]]:
    """The two record files' lines (header included), keyed by kind."""
    rows = {}
    for kind in ("candidates_", "per_robot_"):
        files = sorted(folder.glob(f"{kind}*_run_*.csv"))
        assert len(files) == 1, f"{folder}: {len(files)} {kind} record files"
        rows[kind] = files[0].read_text(encoding="utf-8").splitlines()
    return rows


def load(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def next_candidates(snapshot: dict[str, Any]) -> np.ndarray:
    """The candidates the stored CMA-ES state would draw next (consumes a copy)."""
    import layer_cmaes

    state = pickle.loads(pickle.dumps(snapshot["cma_state"]))
    rng = np.random.get_state()
    np.random.set_state(snapshot["numpy_rng_state"])
    try:
        return np.array(layer_cmaes.restore_optimizer(state).ask())
    finally:
        np.random.set_state(rng)


def snapshot_differences(a: dict[str, Any], b: dict[str, Any], ignore: tuple[str, ...] = ("record_paths",)) -> list[str]:
    """Items that differ between two snapshots (CMA state compared by what it draws next)."""
    differ = []
    if set(a) != set(b):
        differ.append(f"keys {sorted(set(a) ^ set(b))}")
    for key in sorted(set(a) & set(b)):
        if key in ignore:
            continue
        x, y = a[key], b[key]
        if key == "cma_state":
            same = np.array_equal(next_candidates(a), next_candidates(b)) and np.array_equal(x["mean"], y["mean"]) and x["sigma"] == y["sigma"]
        elif key == "numpy_rng_state":
            same = all(np.array_equal(p, q) if isinstance(p, np.ndarray) else p == q for p, q in zip(x, y))
        elif key == "settings_values":
            same = {k: v for k, v in x.items() if k != "trained_weights_source"} == {k: v for k, v in y.items() if k != "trained_weights_source"}
        elif isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            same = isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
        else:
            same = x == y
        if not same:
            differ.append(key)
    return differ
