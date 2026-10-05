"""Run the real workflow end to end, for both algorithms, and check what it leaves behind.

The other checks prove the parts; this one proves they work together, through
the same `run.py` used from the command line. It trains for real (MuJoCo, four simulator
processes, so the multiprocessing path is exercised), exports, resumes, and
checks that a mistake cannot destroy earlier work.

`-t` opens a window and renders for tens of seconds, so it cannot run unattended.
Its path is driven in-process with only the simulator call stubbed: loading the
snapshot, installing the config it was trained with, the legacy body shim, and the
parameter-count guard all run for real.

Everything is written to a temporary directory that is removed afterwards, so
this never touches `output/`.

Run with `make check-workflow`, or `python tests/test_workflow.py`. It takes about
a minute.
"""

import csv
import hashlib
import importlib.util
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

# Snapshots hold revolve2 objects, so unpickling them here needs the engine on
# the path. Install it explicitly, as every entry point does, rather than
# relying on editable-install .pth files (the Lengau image has none).
paths.install()

RUN = paths.ROOT / "run.py"
EA_CONFIG = paths.CONFIG / "_tiny.py"
CMAES_CONFIG = paths.CONFIG / "_tiny_cmaes.py"
EA_MAIN = paths.ROOT / "src" / "ea" / "main.py"
CMAES_MAIN = paths.ROOT / "src" / "cmaes" / "main.py"

CSV_COLUMNS = [
    "num_of_generation", "best_fitness", "worst_fitness", "best_parent_fitness",
    "best_offspring_fitness", "best_ever_fitness", "best_robot_weights", "worst_robot_weights",
]
# Four simulator processes, so the multiprocessing start method is exercised
# (`spawn` on macOS, `fork` on Linux) rather than a single in-process simulator.
NUM_SIMULATORS = "4"
# An optional snapshot written by an earlier version of the code (its config uses
# the former body module name). It is checked only if present at this path,
# next to the repository.
LEGACY_SNAPSHOT = paths.ROOT.parent / "output" / "simple" / "gen37.pkl"


def run_cli(*args, env_extra=None):
    """
    Run `run.py` as a user would.

    :param args: Command-line arguments.
    :param env_extra: Extra environment variables.
    :returns: The completed process, with stdout and stderr combined.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["SLURM_NTASKS"] = NUM_SIMULATORS
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(RUN), *map(str, args)],
        capture_output=True, text=True, env=env, cwd=paths.ROOT,
    )


def load_pickle(path: Path) -> dict:
    """
    Load a snapshot.

    :param path: The pickle file.
    :returns: Its contents.
    """
    with open(path, "rb") as handle:
        return pickle.load(handle)


def checksum(path: Path) -> str:
    """
    Fingerprint a file's contents.

    :param path: The file.
    :returns: Its SHA-256.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Read a CSV.

    :param path: The CSV file.
    :returns: Header and rows.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def tail(result: subprocess.CompletedProcess, lines: int = 6) -> str:
    """
    Summarize a failed command for a report.

    :param result: The completed process.
    :param lines: How many trailing lines to keep.
    :returns: The last lines of its output.
    """
    text = (result.stdout + result.stderr).strip().splitlines()
    return " | ".join(line.strip() for line in text[-lines:])


def check_ea(work: Path) -> list[str]:
    """
    Train, protect, export and resume with the EA.

    :param work: A scratch directory.
    :returns: A list of problems; empty when the EA workflow holds.
    """
    problems = []
    out = work / "ea"

    result = run_cli("-r", EA_CONFIG, EA_MAIN, out)
    if result.returncode != 0:
        return [f"EA: -r failed (exit {result.returncode}): {tail(result)}"]

    for name in ("gen1.pkl", "gen2.pkl"):
        if not (out / name).exists():
            problems.append(f"EA: {name} was not written")
    if problems:
        return problems
    csvs = sorted(out.glob("parameters_*_run_*.csv"))
    if len(csvs) != 1:
        problems.append(f"EA: expected one run CSV, found {len(csvs)}")
    elif read_csv(csvs[0])[0] != CSV_COLUMNS:
        problems.append(f"EA: run CSV columns are {read_csv(csvs[0])[0]}")
    elif len(read_csv(csvs[0])[1]) != 3:
        problems.append(f"EA: run CSV has {len(read_csv(csvs[0])[1])} rows, expected 3 (generations 0-2)")
    if not list(out.glob("parameters_*.npy")):
        problems.append("EA: best weights .npy was not written")

    snapshot = load_pickle(out / "gen1.pkl")
    if snapshot.get("algorithm") != "ea":
        problems.append(f"EA: snapshot algorithm is {snapshot.get('algorithm')!r}, expected 'ea'")

    # A second run into the same folder must be refused and leave the first intact.
    before = {name: checksum(out / name) for name in ("gen1.pkl", "gen2.pkl")}
    result = run_cli("-r", EA_CONFIG, EA_MAIN, out)
    if result.returncode == 0:
        problems.append("EA: a second -r into an existing run folder was allowed")
    for name, digest in before.items():
        if checksum(out / name) != digest:
            problems.append(f"EA: {name} was overwritten by a refused run")
    if "already holds" not in result.stdout + result.stderr:
        problems.append(f"EA: refusal did not explain itself: {tail(result)}")

    # Export gives exactly the canonical columns, one row per snapshot.
    result = run_cli("-o", out)
    exported = out / "generations.csv"
    if result.returncode != 0 or not exported.exists():
        problems.append(f"EA: -o failed: {tail(result)}")
    else:
        header, rows = read_csv(exported)
        if header != CSV_COLUMNS:
            problems.append(f"EA: exported columns are {header}")
        if len(rows) != 2:
            problems.append(f"EA: export has {len(rows)} rows, expected 2")
        # Every value must match what training itself recorded for that
        # generation, not merely have the right shape. The run CSV also holds the
        # initial population as generation 0, which has no snapshot.
        recorded = {row[0]: row for row in read_csv(csvs[0])[1]} if len(csvs) == 1 else {}
        for row in rows:
            if recorded.get(row[0]) != row:
                differing = [c for c, a, b in zip(CSV_COLUMNS, row, recorded.get(row[0], [])) if a != b]
                problems.append(
                    f"EA: exported generation {row[0]} differs from the run CSV in {differing or 'all columns'}"
                )

    # Resume from generation 1 must pick up and finish generation 2.
    result = run_cli("-c", out / "gen1.pkl")
    output = result.stdout + result.stderr
    if result.returncode != 0:
        problems.append(f"EA: -c failed: {tail(result)}")
    elif "Resuming from checkpoint after generation 1" not in output:
        problems.append("EA: -c did not report resuming after generation 1")
    elif not (out / "gen2.pkl").exists():
        problems.append("EA: -c did not write generation 2")
    if len(list(out.glob("parameters_*_run_*.csv"))) != 1:
        problems.append("EA: resuming started a new CSV instead of continuing the run's")
    return problems


def check_cmaes(work: Path) -> list[str]:
    """
    Train, export and resume with CMA-ES, including that resume continues the covariance.

    :param work: A scratch directory.
    :returns: A list of problems; empty when the CMA-ES workflow holds.
    """
    problems = []
    out = work / "cmaes"

    result = run_cli("-r", CMAES_CONFIG, CMAES_MAIN, out)
    if result.returncode != 0:
        return [f"CMA-ES: -r failed (exit {result.returncode}): {tail(result)}"]
    if not (out / "gen1.pkl").exists() or not (out / "gen2.pkl").exists():
        return ["CMA-ES: snapshots were not written"]

    snapshot = load_pickle(out / "gen1.pkl")
    if snapshot.get("algorithm") != "cmaes":
        problems.append(f"CMA-ES: snapshot algorithm is {snapshot.get('algorithm')!r}, expected 'cmaes'")
    if "cma_state" not in snapshot:
        problems.append("CMA-ES: snapshot carries no optimizer state, so it cannot be resumed")

    csvs = sorted(out.glob("parameters_*_run_*.csv"))
    if len(csvs) != 1 or read_csv(csvs[0])[0] != CSV_COLUMNS:
        problems.append("CMA-ES: run CSV missing or has the wrong columns")
    else:
        for row in read_csv(csvs[0])[1]:
            if row[3] or row[4]:
                problems.append("CMA-ES: parent/offspring columns should be blank")
                break

    result = run_cli("-o", out)
    if result.returncode != 0:
        problems.append(f"CMA-ES: -o failed: {tail(result)}")
    elif read_csv(out / "generations.csv")[0] != CSV_COLUMNS:
        problems.append("CMA-ES: exported columns are wrong")

    # Resuming from generation 1 must reproduce generation 2 exactly. That can
    # only happen if the covariance was restored: a restart would draw different
    # candidates and reach a different fitness.
    original_best = float(load_pickle(out / "gen2.pkl")["best_fitness"])
    result = run_cli("-c", out / "gen1.pkl")
    if result.returncode != 0:
        problems.append(f"CMA-ES: -c failed: {tail(result)}")
    else:
        resumed_best = float(load_pickle(out / "gen2.pkl")["best_fitness"])
        if abs(resumed_best - original_best) > 1e-9:
            problems.append(
                f"CMA-ES: resuming from generation 1 gave {resumed_best}, not the original "
                f"{original_best}; the covariance was not restored"
            )
    return problems


def check_bad_invocations(work: Path) -> list[str]:
    """
    Check mistakes are refused clearly, not with a traceback.

    :param work: A scratch directory.
    :returns: A list of problems; empty when they are.
    """
    problems = []

    # An EA config with the CMA-ES main, and the other way round.
    for config, main, label in (
        (EA_CONFIG, CMAES_MAIN, "an EA config with the CMA-ES main"),
        (CMAES_CONFIG, EA_MAIN, "a CMA-ES config with the EA main"),
    ):
        out = work / f"mismatch_{main.parent.name}"
        result = run_cli("-r", config, main, out)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            problems.append(f"{label} was accepted")
        elif "Traceback" in output:
            problems.append(f"{label} failed with a traceback instead of a clear message")
        if list(out.glob("gen*.pkl")):
            problems.append(f"{label} still wrote snapshots")

    # Resuming a snapshot without an 'algorithm' field is refused.
    stray = work / "stray.pkl"
    with open(stray, "wb") as handle:
        pickle.dump({"generation": 1, "completed_generations": 1}, handle)
    result = run_cli("-c", stray)
    if result.returncode == 0 or "algorithm" not in result.stdout + result.stderr:
        problems.append("-c on a snapshot with no algorithm field was not refused with an explanation")
    return problems


def check_test_mode(work: Path) -> list[str]:
    """
    Drive `-t` through everything except the simulator itself.

    :param work: A scratch directory.
    :returns: A list of problems; empty when the -t path holds.
    """
    problems = []
    spec = importlib.util.spec_from_file_location("oribots_run_for_t", RUN)
    run = importlib.util.module_from_spec(spec)
    sys.modules["oribots_run_for_t"] = run
    spec.loader.exec_module(run)

    seen = {}

    def record(**kwargs):
        seen.update(kwargs)

    run._simulate_and_report = record

    def attempt(snapshot_path: Path):
        seen.clear()
        try:
            run._run_test(snapshot_path, "custom")
        except SystemExit as error:
            return str(error)
        except Exception as error:  # noqa: BLE001 - reported as a failure, not a crash
            return f"{type(error).__name__}: {error}"
        return None

    # A snapshot written by this code: loads, installs its config, hands the right
    # number of weights to the simulator.
    fresh = work / "ea" / "gen2.pkl"
    if fresh.exists():
        refusal = attempt(fresh)
        expected = len(load_pickle(fresh)["parameters"])
        if refusal:
            problems.append(f"-t refused a fresh snapshot: {refusal}")
        elif len(seen.get("weights", [])) != expected:
            problems.append(f"-t passed {len(seen.get('weights', []))} weights, expected {expected}")

    # A snapshot whose weights do not fit its body is refused, not simulated.
    if fresh.exists():
        broken = load_pickle(fresh)
        broken["parameters"] = broken["parameters"][:-3]
        broken["best_parameters"] = broken["parameters"]
        broken_path = work / "broken.pkl"
        with open(broken_path, "wb") as handle:
            pickle.dump(broken, handle)
        refusal = attempt(broken_path)
        if refusal is None:
            problems.append("-t simulated a snapshot with the wrong number of parameters")
        elif "parameters" not in refusal:
            problems.append(f"-t refusal does not mention parameters: {refusal}")
        if seen:
            problems.append("-t reached the simulator despite a parameter mismatch")

    # A snapshot in the legacy style: its config imports a body from the former
    # engine-namespace module, which holds only upstream's four bodies, and its
    # stored paths point at a cluster directory that does not exist here. simple_v2
    # exists only in this project's own catalogue, so the import succeeds only if
    # the former module name is redirected to it.
    if fresh.exists():
        legacy_style = load_pickle(fresh)
        source = legacy_style["config_source"]
        if "from bodies import simple_v2" not in source:
            problems.append("test setup: fresh config no longer imports simple_v2 from bodies")
        else:
            legacy_style["config_source"] = source.replace(
                "from bodies import simple_v2",
                "from revolve2.ci_group.modular_robots_v2 import simple_v2",
            )
            legacy_style["code_paths"] = dict(
                legacy_style["code_paths"],
                config_path="/scratch/nobody/earlier-checkout/config/1_examples/simple.py",
                main_path="/scratch/nobody/earlier-checkout/examples/1a/main.py",
                module_dir="/scratch/nobody/earlier-checkout/examples/1a",
            )
            legacy_path = work / "legacy_style.pkl"
            with open(legacy_path, "wb") as handle:
                pickle.dump(legacy_style, handle)
            for cached in ("revolve2.ci_group.modular_robots_v2", "bodies"):
                sys.modules.pop(cached, None)
            refusal = attempt(legacy_path)
            if refusal:
                problems.append(f"-t refused a legacy-style snapshot: {refusal}")
            elif len(seen.get("weights", [])) != 22:
                problems.append(f"-t passed {len(seen.get('weights', []))} weights for a legacy-style snapshot")
            else:
                redirected = getattr(sys.modules.get("revolve2.ci_group.modular_robots_v2"), "__file__", None)
                if redirected is None or Path(redirected).resolve() != (paths.CONFIG / "bodies.py").resolve():
                    problems.append(
                        f"the former body module name resolves to {redirected}, not config/bodies.py"
                    )

    # The optional earlier-version snapshot: its config imports the former body
    # module, and its stored paths point at directories that do not exist here.
    if LEGACY_SNAPSHOT.exists():
        refusal = attempt(LEGACY_SNAPSHOT)
        if refusal:
            problems.append(f"-t refused the legacy snapshot {LEGACY_SNAPSHOT.name}: {refusal}")
        elif len(seen.get("weights", [])) != 22:
            problems.append(f"-t passed {len(seen.get('weights', []))} weights for the legacy snapshot, expected 22")
    return problems


def main() -> None:
    """Run the workflow checks and report."""
    work = Path(tempfile.mkdtemp(prefix="oribots_workflow_"))
    try:
        problems = check_ea(work)
        problems.extend(check_cmaes(work))
        problems.extend(check_bad_invocations(work))
        problems.extend(check_test_mode(work))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if problems:
        print("WORKFLOW CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("WORKFLOW CHECK PASSED")
    print(f"  EA and CMA-ES each train for real with {NUM_SIMULATORS} simulator processes")
    print("  snapshots are tagged with their algorithm; CSVs have the eight canonical columns")
    print("  a second run into an existing folder is refused and leaves it byte-identical")
    print("  -o exports one row per snapshot, matching what training recorded; -c resumes the EA,")
    print("    and CMA-ES resumes with its covariance")
    print("  mismatched config/main pairings and unresumable snapshots are refused clearly")
    legacy = "and the legacy snapshot loads" if LEGACY_SNAPSHOT.exists() else "(legacy snapshot not present, skipped)"
    print(f"  -t loads snapshots, refuses a parameter mismatch, redirects the former body module {legacy}")


if __name__ == "__main__":
    main()
