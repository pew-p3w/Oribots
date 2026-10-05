"""Check that this environment runs Oribots from its own files only.

Run with `make verify` (or `python tests/test_environment.py`). It fails if any
`revolve2.*` module resolves outside `Oribots/engine/`, which is what happens
when a virtual environment still carries editable installs of some other copy of
Revolve2.
"""

import importlib
import sys
import warnings
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"

EXPECTED_NUMPY = "1.26.4"

REVOLVE2_MODULES = [
    "revolve2.simulation.scene",
    "revolve2.modular_robot",
    "revolve2.modular_robot.body.v2",
    "revolve2.modular_robot_simulation",
    "revolve2.experimentation.rng",
    "revolve2.experimentation.logging",
    "revolve2.ci_group.terrains",
    "revolve2.ci_group.simulation_parameters",
    "revolve2.ci_group.interactive_objects",
    "revolve2.simulators.mujoco_simulator",
]

THIRD_PARTY_MODULES = ["numpy", "scipy", "pyrr", "mujoco", "dm_control", "glfw", "mujoco_viewer", "cv2", "noise", "cma"]


def engine_roots() -> list[Path]:
    """
    The six directories under engine/ that each contain a `revolve2` namespace slice.

    Oribots puts these on sys.path itself (run.py does the same) instead of
    relying on the .pth files that `pip install -e` writes. Python silently
    skips a .pth file that carries the macOS "hidden" flag, and iCloud Drive
    sets that flag on files under ~/Documents after installation.
    """
    roots = sorted(
        {p.parent for pattern in ("*/revolve2", "simulators/*/revolve2") for p in ENGINE.glob(pattern) if p.is_dir()}
    )
    if len(roots) != 6:
        raise SystemExit(f"expected 6 Revolve2 packages under engine/, found {len(roots)}: {[str(r) for r in roots]}")
    return roots


def hidden_editable_pth_files() -> list[Path]:
    """Editable-install .pth files that Python will skip because they are flagged hidden."""
    import site

    hidden_flag = getattr(__import__("stat"), "UF_HIDDEN", 0)
    return [
        pth
        for site_dir in site.getsitepackages()
        for pth in sorted(Path(site_dir).glob("revolve2_*.pth"))
        if getattr(pth.lstat(), "st_flags", 0) & hidden_flag
    ]


def duplicated_library_files() -> list[Path]:
    """
    Find copies like `libsensor 2.dylib` sitting next to installed libraries.

    macOS creates these when a synced folder resolves a conflict, which happens
    to virtual environments kept in iCloud Drive. They are not harmless: MuJoCo
    loads every shared library in its plugin directory, so a duplicate
    registers the same plugin twice and MuJoCo aborts the process with
    `plugin ... is already registered` before any test can report anything.

    :returns: The duplicate files found, worst offenders first.
    """
    import site

    duplicates: list[Path] = []
    for site_dir in site.getsitepackages():
        for pattern in ("**/* [0-9].dylib", "**/* [0-9].so", "**/* [0-9].dll"):
            duplicates.extend(Path(site_dir).glob(pattern))
    return sorted(duplicates)


def mjpython_start_error() -> str | None:
    """
    Check that `mjpython` can actually start, when it is present.

    `mjpython` is how the interactive MuJoCo viewer gets the macOS main thread.
    A venv built from a uv/standalone Python cannot start it — mjpython fails to
    `dlopen @rpath/libpython3.11.dylib`, because a standalone build does not
    expose libpython next to the venv the way a framework build (Homebrew's
    `python@3.11`) does. When that happens the interactive viewer is unusable
    even though headless training works fine.

    Checked on macOS only, and only when the venv actually ships an `mjpython`.
    On Linux (the Docker gate, the cluster) the interactive viewer is never used
    and the mujoco wheel still ships an `mjpython`, so running it there would be
    both pointless and a false failure; the check is skipped off macOS.

    :returns: An error string if mjpython is present but cannot start, else None.
    """
    import subprocess

    # macOS-only: the main-thread window problem this guards is a macOS concern,
    # and `mjpython` is shipped by the mujoco wheel on Linux too (so an .exists()
    # check alone would wrongly run it in the Docker gate and on the cluster,
    # where the interactive viewer is never used).
    if sys.platform != "darwin":
        return None
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        return None  # headless environment (e.g. HPC conda) - nothing to check
    try:
        result = subprocess.run(
            [str(mjpython), "-c", "pass"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"mjpython is present but could not be launched: {error}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit code {result.returncode}"
        return (
            f"mjpython is present but fails to start ({tail}). The interactive "
            "viewer will not work. This happens when the venv is built from a "
            "uv/standalone Python; rebuild it from a framework Python "
            "(Homebrew's python@3.11): `make setup PYTHON=/opt/homebrew/opt/python@3.11/bin/python3.11`."
        )
    return None


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def report(problems: list[str]) -> None:
    """
    Print the problems found and stop.

    :param problems: The problems to report.
    :raises SystemExit: Always, with a failing status.
    """
    print("ENVIRONMENT CHECK FAILED")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)


def main() -> None:
    problems: list[str] = []

    for root in reversed(engine_roots()):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python 3.11 required, running {sys.version.split()[0]}")

    # Check this before importing anything heavy: a duplicated plugin library
    # makes MuJoCo abort on import, which would kill this script mid-run and
    # report nothing useful.
    duplicates = duplicated_library_files()
    if duplicates:
        problems.append(
            f"{len(duplicates)} duplicated library files in site-packages, such as "
            f"{duplicates[0].name!r}. MuJoCo aborts when a plugin is registered twice. "
            "Rebuild the environment with `make setup`, and keep it out of iCloud Drive."
        )
        for duplicate in duplicates[:5]:
            problems.append(f"    {duplicate}")
        # Stop here rather than importing: MuJoCo would abort the process and
        # this report would never be printed.
        report(problems)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        modules = {name: importlib.import_module(name) for name in THIRD_PARTY_MODULES + REVOLVE2_MODULES}

    numpy = modules["numpy"]
    if numpy.__version__ != EXPECTED_NUMPY:
        problems.append(f"numpy {numpy.__version__} installed, expected {EXPECTED_NUMPY}")

    # Every revolve2 module, and every entry of the shared `revolve2` namespace,
    # must live under Oribots/engine/.
    for name in REVOLVE2_MODULES:
        module = modules[name]
        origin = getattr(module, "__file__", None)
        locations = [Path(origin)] if origin else [Path(p) for p in module.__path__]
        for location in locations:
            if not _is_inside(location, ENGINE):
                problems.append(f"{name} resolves outside engine/: {location}")
    revolve2 = importlib.import_module("revolve2")
    for entry in revolve2.__path__:
        if not _is_inside(Path(entry), ENGINE):
            problems.append(f"revolve2 namespace has an entry outside engine/: {entry}")

    # Warnings that point at a numpy/OpenCV binary-compatibility problem.
    for warning in caught:
        text = f"{warning.category.__name__}: {warning.message}".lower()
        if any(key in text for key in ("numpy", "opencv", "cv2", "abi")):
            problems.append(f"import warning: {warning.category.__name__}: {warning.message}")

    # If mjpython is shipped in this venv, it must actually start, or the
    # interactive viewer is silently unusable. Absent mjpython (headless conda
    # on the HPC) is fine and skipped.
    mjpython_error = mjpython_start_error()
    if mjpython_error is not None:
        problems.append(mjpython_error)

    if problems:
        report(problems)

    print("ENVIRONMENT CHECK PASSED")
    print(f"  python   {sys.version.split()[0]}  ({sys.executable})")
    for distribution in ("numpy", "scipy", "mujoco", "dm-control", "opencv-python-headless", "cma"):
        print(f"  {distribution:<24} {metadata.version(distribution)}")
    print(f"  revolve2 namespace: {len(revolve2.__path__)} entries, all under engine/")
    skipped = hidden_editable_pth_files()
    if skipped:
        print(
            f"  WARNING: Python is skipping {len(skipped)} editable-install .pth files (macOS 'hidden' flag,\n"
            "           typically set by iCloud Drive on a checkout under ~/Documents). This is harmless for\n"
            "           run.py and this check, which add engine/ to sys.path themselves, but a bare\n"
            "           `python -c 'import revolve2'` in this venv will fail. Keeping the checkout and .venv\n"
            "           outside iCloud Drive avoids it."
        )
    ignored = [w for w in caught if not any(k in f"{w.message}".lower() for k in ("numpy", "opencv", "cv2", "abi"))]
    for warning in ignored:
        print(f"  ignored unrelated warning: {warning.category.__name__}: {str(warning.message).splitlines()[0]}")


if __name__ == "__main__":
    main()
