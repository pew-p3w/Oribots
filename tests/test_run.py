"""Check run.py's guards on the five workflow modes.

run.py is the only thing a user drives directly, so a mistake here is a mistake
in every experiment. These checks cover the rules that protect existing work,
without running a simulation: a full end-to-end run belongs in the Linux gate.

Run with `make check-run`, or `python tests/test_run.py`.
"""

import importlib.util
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402


def load_run_module():
    """
    Import run.py without executing a workflow.

    :returns: The loaded run module.
    """
    spec = importlib.util.spec_from_file_location("oribots_run", paths.ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["oribots_run"] = module
    spec.loader.exec_module(module)
    return module


def check_overwrite_guard(run) -> list[str]:
    """
    Check that a new run refuses to write into a folder that already holds one.

    Generation snapshots cannot be recovered once overwritten, and the run CSV
    survives under a fresh name, so the loss would be easy to miss.

    :param run: The loaded run module.
    :returns: A list of problems; empty when the guard behaves.
    """
    problems = []
    work = Path(tempfile.mkdtemp(prefix="oribots_guard_"))
    try:
        # An empty folder, or one that does not exist yet, is fine to run into.
        empty = work / "fresh"
        empty.mkdir()
        for folder, description in ((empty, "an empty folder"), (work / "missing", "a new folder")):
            try:
                run._require_fresh_output_dir(folder)
            except SystemExit as error:
                problems.append(f"refused {description}: {error}")

        # A folder holding a crashed run's CSV but no snapshots is still fine:
        # nothing irreplaceable is there.
        csv_only = work / "csv_only"
        csv_only.mkdir()
        (csv_only / "parameters_x_run_20260101_000000.csv").write_text("num_of_generation\n")
        try:
            run._require_fresh_output_dir(csv_only)
        except SystemExit as error:
            problems.append(f"refused a folder with only a CSV: {error}")

        # A run killed before its first snapshot has already recorded individuals.
        # A new run would truncate weights.csv, so such a folder is refused too.
        for name in ("weights.csv", "individual_performance_x_run_20260101_000000.csv"):
            recorded = work / f"recorded_{name.split('_')[0]}"
            recorded.mkdir()
            (recorded / name).write_text("individual_id\n")
            try:
                run._require_fresh_output_dir(recorded)
            except SystemExit as error:
                if name not in str(error):
                    problems.append(f"refusal for a folder holding {name} does not name it: {error}")
            else:
                problems.append(f"accepted a folder that already holds {name}")

        # A folder holding snapshots must be refused, and the message has to say
        # what is there and what to do instead.
        used = work / "spider_run"
        used.mkdir()
        for generation in (1, 2, 10):
            (used / f"gen{generation}.pkl").write_bytes(b"snapshot")
        try:
            run._require_fresh_output_dir(used)
        except SystemExit as error:
            message = str(error)
            for expected in ("3 generation snapshots", "gen1.pkl", "gen10.pkl", "-c"):
                if expected not in message:
                    problems.append(f"refusal message does not mention {expected!r}: {message}")
        else:
            problems.append("a folder holding 3 snapshots was NOT refused")

        # Legacy folders can hold names like gen100_extend200.pkl; the guard must
        # still refuse rather than crash while sorting them.
        legacy = work / "legacy"
        legacy.mkdir()
        (legacy / "gen100.pkl").write_bytes(b"snapshot")
        (legacy / "gen100_extend200.pkl").write_bytes(b"snapshot")
        try:
            run._require_fresh_output_dir(legacy)
        except SystemExit:
            pass
        except Exception as error:  # noqa: BLE001 - reported, not raised
            problems.append(f"guard crashed on a legacy snapshot name: {type(error).__name__}: {error}")
        else:
            problems.append("a legacy folder holding snapshots was NOT refused")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return problems


def check_resume_is_exempt(run) -> list[str]:
    """
    Check that resuming is not blocked by the new-run guard.

    Resuming deliberately writes more snapshots into the folder it came from,
    so it must not be routed through the guard.

    :param run: The loaded run module.
    :returns: A list of problems; empty when resume stays exempt.
    """
    problems = []
    source = (paths.ROOT / "run.py").read_text()
    guard_calls = source.count("_require_fresh_output_dir(")
    if guard_calls != 2:  # one definition, one call site
        problems.append(
            f"_require_fresh_output_dir appears {guard_calls} times; expected one definition "
            "and one call, on the -r path only"
        )
    continue_body = source.split("def _continue_training(")[1].split("\ndef ")[0]
    if "_require_fresh_output_dir" in continue_body:
        problems.append("_continue_training calls the new-run guard; resuming would be blocked")
    return problems


def check_resume_rejects_unknown_algorithm(run) -> list[str]:
    """
    Check that resuming a snapshot with no algorithm field fails loudly.

    :param run: The loaded run module.
    :returns: A list of problems; empty when it refuses clearly.
    """
    problems = []
    work = Path(tempfile.mkdtemp(prefix="oribots_resume_"))
    try:
        snapshot_path = work / "gen1.pkl"
        with open(snapshot_path, "wb") as handle:
            pickle.dump({"generation": 1, "completed_generations": 1}, handle)
        try:
            run._continue_training(snapshot_path)
        except SystemExit as error:
            if "algorithm" not in str(error):
                problems.append(f"refusal does not mention the algorithm field: {error}")
        except Exception as error:  # noqa: BLE001 - reported, not raised
            problems.append(f"resume raised {type(error).__name__} instead of exiting clearly: {error}")
        else:
            problems.append("resuming a snapshot with no algorithm field was allowed")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return problems


def check_viewer_resolution(run) -> list[str]:
    """
    Check the viewer is paired to the launcher and bad combinations are refused.

    Verified behaviour on macOS: only mjpython + native displays; plain python +
    custom is invisible; mjpython + custom crashes. `_resolve_viewer` must pick
    the working viewer for `auto` and refuse the combinations that cannot work,
    rather than silently producing a dead window. mjpython detection is faked
    here so both launchers can be checked without actually being under mjpython.

    :param run: The loaded run module.
    :returns: A list of problems; empty when resolution behaves.
    """
    problems = []
    original = run._running_under_mjpython
    is_macos = sys.platform == "darwin"
    try:
        # Under mjpython: auto and native -> native; custom is the crash combo.
        run._running_under_mjpython = lambda: True
        if run._resolve_viewer("auto") != "native":
            problems.append("under mjpython, auto should select native")
        if run._resolve_viewer("native") != "native":
            problems.append("under mjpython, native should stay native")
        try:
            run._resolve_viewer("custom")
        except SystemExit:
            pass
        else:
            problems.append("under mjpython, custom should be refused (it crashes)")

        # Under plain python: auto -> custom; native is invisible on macOS.
        run._running_under_mjpython = lambda: False
        if run._resolve_viewer("auto") != "custom":
            problems.append("under plain python, auto should select custom")
        if run._resolve_viewer("custom") != "custom":
            problems.append("under plain python, custom should stay custom")
        if is_macos:
            try:
                run._resolve_viewer("native")
            except SystemExit:
                pass
            else:
                problems.append("under plain python on macOS, native should be refused")
    finally:
        run._running_under_mjpython = original
    return problems


def main() -> None:
    """Run the checks and report."""
    paths.install()
    run = load_run_module()

    problems = check_overwrite_guard(run)
    problems.extend(check_resume_is_exempt(run))
    problems.extend(check_resume_rejects_unknown_algorithm(run))
    problems.extend(check_viewer_resolution(run))

    if problems:
        print("RUN CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("RUN CHECK PASSED")
    print("  a new run refuses to overwrite a folder that already holds snapshots")
    print("  empty folders, new folders and CSV-only folders are still accepted")
    print("  a folder that already holds per-individual recording files is refused")
    print("  legacy snapshot names (gen100_extend200.pkl) are handled, not crashed on")
    print("  resuming stays exempt from the guard, and refuses snapshots with no algorithm field")
    print("  the viewer is paired to the launcher; invisible/crashing combinations are refused")


if __name__ == "__main__":
    main()
