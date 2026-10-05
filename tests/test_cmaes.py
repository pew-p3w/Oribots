"""Check that the CMA-ES loop makes the same decisions as its recorded reference.

CMA-ES chooses each generation's candidates from the covariance it has adapted
so far, so a change to the loop is only safe if the optimizer makes the same
sequence of choices. With a stubbed evaluator and a fixed seed, the whole `ask`/`tell`
sequence and the resulting mean/sigma trajectory are deterministic and need no
simulator. That trace is the equivalence contract, compared exactly against
`cmaes_reference.json`.

Two snapshot fields are newer than the reference and are checked separately:
every snapshot carries `algorithm="cmaes"` and the full optimizer state
(`cma_state`), so `run.py -c` can resume a CMA-ES run and continue the
covariance rather than restart it.

Run with `make check-cmaes`, or `python tests/test_cmaes.py`.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import cmaes_fingerprint as cf  # noqa: E402
from compare import differences  # noqa: E402

MAIN = paths.ROOT / "src" / "cmaes" / "main.py"
CONFIG = paths.CONFIG / "_tiny_cmaes.py"

EXPECTED_HEADER = [
    "num_of_generation", "best_fitness", "worst_fitness", "best_parent_fitness",
    "best_offspring_fitness", "best_ever_fitness", "best_robot_weights", "worst_robot_weights",
]


def check_snapshot_additions(current: dict) -> list[str]:
    """
    Check the snapshot behaviour that the reference does not cover.

    :param current: The fingerprint of the CMA-ES run.
    :returns: A list of problems; empty when all hold.
    """
    problems = []

    for name, snapshot in current["snapshots"].items():
        if snapshot["algorithm"] != "cmaes":
            problems.append(f"{name}: algorithm is {snapshot['algorithm']!r}, expected 'cmaes'")
        # The resume fix: the optimizer state must travel with the snapshot.
        if not snapshot["has_cma_state"]:
            problems.append(f"{name}: has no cma_state, so run.py -c could not resume it")
        row = snapshot["csv_row"] or {}
        if row.get("best_parent_fitness") != "" or row.get("best_offspring_fitness") != "":
            problems.append(f"{name}: parent/offspring columns are not blank for CMA-ES")

    if current["csv"]["header"] != EXPECTED_HEADER:
        problems.append(f"CSV header is {current['csv']['header']}")
    if current["num_csv_files"] != 1:
        problems.append(f"run wrote {current['num_csv_files']} run CSV files, expected 1")
    problems.extend(check_recording(current))

    return problems


def check_recording(current: dict) -> list[str]:
    """
    Check every individual of every generation was recorded, and correctly.

    :param current: The fingerprint of the CMA-ES run.
    :returns: A list of problems; empty when all hold.
    """
    import json

    problems = []
    performance = current["recording"]["performance"]
    weights = current["recording"]["weights"]
    settings = current["settings"]
    snapshots = current["snapshots"]
    expected_individuals = settings["num_generations"] * settings["cma_population_size"]
    if len(weights) != expected_individuals:
        problems.append(f"weights.csv has {len(weights)} rows, expected {expected_individuals}")
    if len(performance) != expected_individuals * settings["num_training_ball_poses"]:
        problems.append(f"the performance CSV has {len(performance)} rows")
    if len({row["individual_id"] for row in weights}) != len(weights):
        problems.append("weights.csv repeats an individual_id")
    recorded_weights = {row["individual_id"]: json.loads(row["robot_weights"]) for row in weights}
    for name, snapshot in snapshots.items():
        generation = snapshot["generation"]
        if not all(snapshot["recording_paths"]):
            problems.append(f"{name}: does not carry the recording file paths")
        for index, (parameters, fitness) in enumerate(
            zip(snapshot["population_parameters"], snapshot["population_fitnesses"])
        ):
            individual = f"g{generation}_i{index}"
            if individual not in recorded_weights:
                problems.append(f"{individual} is not in weights.csv")
                continue
            if cf._round(recorded_weights[individual]) != parameters:
                problems.append(f"{individual}: recorded weights differ from the snapshot's population")
            rows = [row for row in performance if row["individual_id"] == individual]
            if not rows or any(cf._round(float(row["overall_fitness"])) != fitness for row in rows):
                problems.append(f"{individual}: recorded overall_fitness is not the raw fitness CMA-ES used")
    return problems


def main() -> None:
    """Run the CMA-ES loop and compare it with the reference."""
    work = Path(tempfile.mkdtemp(prefix="oribots_cmaes_check_"))
    try:
        current = cf.cma_fingerprint(main_path=MAIN, config_path=CONFIG, output_dir=work / "run")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    reference = cf.load_reference()

    # The trace is the equivalence contract: identical optimizer decisions.
    trace_problems = differences(
        {"settings": reference["settings"], "trace": reference["trace"], "best_weights": reference["best_weights"]},
        {"settings": current["settings"], "trace": current["trace"], "best_weights": current["best_weights"]},
    )
    problems = [f"optimizer decisions differ: {problem}" for problem in trace_problems]
    problems.extend(check_snapshot_additions(current))

    if problems:
        print("CMA-ES CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        raise SystemExit(1)

    trace = current["trace"]
    settings = current["settings"]
    print("CMA-ES CHECK PASSED")
    print(
        f"  a {settings['cma_population_size']}-candidate, {settings['num_generations']}-generation run "
        "reproduces the reference optimizer trace exactly"
    )
    print(f"  {len(trace['ask'])} ask/tell rounds; mean and sigma trajectories identical")
    print(f"  final step-size sigma = {trace['sigma'][-1]}")
    print("  fitness is raw and signed; parent/offspring CSV columns are blank")
    print("  every snapshot carries algorithm='cmaes' and cma_state, so run.py -c can resume it")
    print(
        f"  every individual recorded: {len(current['recording']['weights'])} weights rows, "
        f"{len(current['recording']['performance'])} trial rows, matching the snapshots"
    )


if __name__ == "__main__":
    main()
