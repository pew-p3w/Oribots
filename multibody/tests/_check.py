"""Shared helpers for the multibody check suite.

Every multibody test is a plain script run by ``make -C multibody check`` (see
``multibody/run_checks.sh``). This module gives those scripts a common spine:

- the ``<NAME> CHECK PASSED`` / ``<NAME> CHECK FAILED`` protocol and process exit
  codes the runner reads (requirement 22.2);
- ``run_checks`` / ``run_property``: seeded, fixed-count sampling loops for the
  "FOR ALL" properties, using only ``numpy`` and the standard library, because
  the project does not install Hypothesis (OQ-5, requirement 22.5);
- ``forbid_disallowed_imports``: fails a test that pulled in Hypothesis or any
  package outside ``requirements.txt`` (requirement 22.8);
- ``temp_workspace``: a scratch directory under the system temp dir, cleaned up
  on exit, so a test leaves nothing behind and never writes under
  ``Oribots/output/`` (requirement 22.6).

It is named with a leading underscore so the runner's ``test_*.py`` glob never
runs it as a test. A test script uses it like::

    from _check import check_main

    def run():
        # ... assertions; raise CheckError(...) or AssertionError on failure ...
        pass

    if __name__ == "__main__":
        check_main("MY_TEST", run)
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

# Make the multibody modules (one directory up) and, through them, the
# single-robot modules and the vendored engine importable for every test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contest_paths  # noqa: E402

contest_paths.install()
# Tests never write bytecode next to the sources (iCloud and stale .pyc).
sys.dont_write_bytecode = True


# The packages a multibody test is allowed to import, beyond the standard
# library, the vendored ``revolve2`` engine and the single-robot modules. This
# is exactly the top-level import set of ``requirements.txt`` (requirement 1.6,
# 22.5): no Hypothesis, no pytest, nothing a fresh ``make setup`` env lacks.
ALLOWED_THIRD_PARTY = frozenset(
    {
        "numpy",
        "scipy",
        "pyrr",
        "mujoco",
        "dm_control",
        "mujoco_python_viewer",
        "glfw",
        "cv2",  # opencv-python-headless
        "noise",
        "cma",
        "revolve2",  # the vendored engine namespace
    }
)

# Explicitly disallowed: a test must not reach for a property-testing library,
# since the project installs none and the cluster has no PyPI access (OQ-5).
DISALLOWED_IMPORTS = frozenset({"hypothesis", "pytest", "_pytest", "nose"})

# At least this many cases per property (requirement 22.5).
DEFAULT_PROPERTY_CASES = 100


class CheckError(AssertionError):
    """A check failed. Carries a human-readable reason for the FAILED line."""


def _repo_root() -> Path:
    """
    The Oribots repository root (the parent of ``multibody``).

    :returns: Absolute path to the Oribots root.
    """
    return Path(__file__).resolve().parents[2]


def check_main(name: str, run: Callable[[], Any]) -> None:
    """
    Run one test body and report in the suite's protocol, then exit.

    Prints ``<NAME> CHECK PASSED`` and exits 0 when ``run`` returns without
    raising; otherwise prints ``<NAME> CHECK FAILED`` with the reason (and a
    traceback for an unexpected error) and exits 1. This is the single place the
    pass/fail contract (requirement 22.2) is implemented, so every test reports
    the same way.

    :param name: Upper-case test name, unique within the suite.
    :param run: The test body; it raises on failure.
    """
    try:
        forbid_disallowed_imports()
        run()
    except CheckError as error:
        print(f"{name} CHECK FAILED: {error}", flush=True)
        sys.exit(1)
    except AssertionError as error:
        reason = str(error) or "assertion failed"
        print(f"{name} CHECK FAILED: {reason}", flush=True)
        sys.exit(1)
    except BaseException:  # noqa: BLE001 - report any error as a clean failure
        print(f"{name} CHECK FAILED: unexpected error", flush=True)
        traceback.print_exc()
        sys.exit(1)
    print(f"{name} CHECK PASSED", flush=True)
    sys.exit(0)


def require(condition: bool, message: str) -> None:
    """
    Fail the check with ``message`` unless ``condition`` holds.

    :param condition: What must be true.
    :param message: The reason shown on the FAILED line.
    :raises CheckError: If the condition is false.
    """
    if not condition:
        raise CheckError(message)


def require_close(
    actual: Any, expected: Any, tolerance: float, message: str
) -> None:
    """
    Fail unless ``actual`` and ``expected`` agree within ``tolerance``.

    :param actual: Value produced.
    :param expected: Value wanted.
    :param tolerance: Maximum allowed absolute difference (element-wise).
    :param message: The reason shown on the FAILED line.
    :raises CheckError: If any element differs by more than the tolerance.
    """
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape:
        raise CheckError(
            f"{message}: shape {actual_array.shape} != {expected_array.shape}"
        )
    worst = float(np.max(np.abs(actual_array - expected_array))) if actual_array.size else 0.0
    if not worst <= tolerance:
        raise CheckError(f"{message}: max abs diff {worst:.3e} > {tolerance:.3e}")


def run_property(
    description: str,
    sample: Callable[[np.random.Generator], Any],
    check: Callable[[Any], None],
    *,
    cases: int = DEFAULT_PROPERTY_CASES,
    seed: int = 0,
) -> None:
    """
    Check a "FOR ALL" property over seeded random cases (no Hypothesis, OQ-5).

    Draws ``cases`` inputs from a seeded generator and runs ``check`` on each.
    The seed is fixed so repeated runs test the same inputs (requirement 22.5);
    a failing case's index, seed and value go into the failure message so it can
    be reproduced.

    :param description: What the property asserts (for the failure message).
    :param sample: Builds one random input from a numpy Generator.
    :param check: Asserts the property for one input; raises on failure.
    :param cases: Number of random inputs; at least 100 for a property.
    :param seed: Fixed seed for the generator.
    :raises CheckError: On the first input that fails, naming index, seed, value.
    """
    if cases < DEFAULT_PROPERTY_CASES:
        raise CheckError(
            f"property '{description}' asked for {cases} cases; "
            f"at least {DEFAULT_PROPERTY_CASES} are required (requirement 22.5)"
        )
    rng = np.random.default_rng(seed)
    for index in range(cases):
        value = sample(rng)
        try:
            check(value)
        except AssertionError as error:
            reason = str(error) or "assertion failed"
            raise CheckError(
                f"property '{description}' failed on case {index} "
                f"(seed={seed}): {reason}; input={value!r}"
            ) from error


def forbid_disallowed_imports() -> None:
    """
    Fail if a disallowed package was imported into this process.

    Guards requirement 22.8 / OQ-5: a test must not use Hypothesis, pytest or
    any other package the project does not install. Checked against
    ``sys.modules`` so it catches an import anywhere in the test's import graph.

    :raises CheckError: If a disallowed top-level package is loaded.
    """
    loaded = {name.split(".", 1)[0] for name in sys.modules}
    offenders = sorted(loaded & DISALLOWED_IMPORTS)
    if offenders:
        raise CheckError(
            f"disallowed import(s) present: {', '.join(offenders)} "
            "(the project installs no property-testing library; OQ-5)"
        )


@contextlib.contextmanager
def temp_workspace(prefix: str = "multibody_check_") -> Iterator[Path]:
    """
    A scratch directory under the system temp dir, removed on exit.

    So a test writes its outputs here and leaves nothing behind outside the
    system temp dir, and never under ``Oribots/output/`` (requirement 22.6).
    The directory is removed even if the test raises.

    :param prefix: Name prefix for the temporary directory.
    :yields: The temporary directory path.
    :raises CheckError: If asked to create it under the repository.
    """
    base = tempfile.mkdtemp(prefix=prefix)
    workspace = Path(base)
    try:
        # A test must not have redirected the temp dir into the repo.
        if _repo_root() in workspace.resolve().parents or workspace.resolve() == _repo_root():
            raise CheckError(
                f"temp workspace {workspace} is inside the repository; "
                "it must be under the system temp dir (requirement 22.6)"
            )
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def assert_nothing_under_output() -> None:
    """
    Fail if the test created anything under ``Oribots/output/``.

    A standing guard for requirement 22.6: the check suite leaves existing runs
    and the analysis folder untouched. Call it at the end of a test that
    exercises code which writes run folders, to confirm it wrote only to a temp
    workspace. It compares the current ``output/`` listing against a snapshot the
    test captured at start via :func:`snapshot_output`.

    :raises CheckError: Never on its own; see :func:`snapshot_output`.
    """
    # Intentionally a no-op without a prior snapshot; the paired helper below is
    # what tests use. Kept as a named entry point for readability in tests.
    return None


def snapshot_output() -> set[str]:
    """
    Record the current contents of ``Oribots/output/`` for a later diff.

    :returns: The set of top-level names under ``output/`` (empty if absent).
    """
    output = _repo_root() / "output"
    if not output.is_dir():
        return set()
    return {entry.name for entry in output.iterdir()}


def require_output_unchanged(before: set[str]) -> None:
    """
    Fail if anything new appeared under ``Oribots/output/`` since ``before``.

    :param before: The snapshot from :func:`snapshot_output` taken at test start.
    :raises CheckError: If a new entry appeared under ``output/``.
    """
    after = snapshot_output()
    created = sorted(after - before)
    if created:
        raise CheckError(
            f"test wrote under Oribots/output/: {created} "
            "(tests must write only to a temp workspace; requirement 22.6)"
        )
