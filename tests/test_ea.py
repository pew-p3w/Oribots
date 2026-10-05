"""Check that the EA makes the same choices as its recorded reference.

The EA decides which controllers survive, and the `[0, 1]` fitness clip lives in
the EA rather than in the evaluator. If the clip were applied anywhere but in
the selection path, different robots would win and nothing else would notice.

So the comparison is deliberately asymmetric. The reference was recorded from
an earlier implementation running against an evaluator that clipped for it.
This runs the current EA, with clamped selection, against an evaluator that
returns raw signed fitness. Matching results show that the clip is applied in
the right place. The default unclamped selection rule is checked separately.

Run with `make check-ea`, or `python tests/test_ea.py`.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import ea_fingerprint  # noqa: E402
from compare import differences  # noqa: E402

MAIN = paths.ROOT / "src" / "ea" / "main.py"
CONFIG = paths.CONFIG / "_tiny.py"


def check_expectations(current: dict) -> list[str]:
    """
    Check the things the recorded comparison cannot state on its own.

    :param current: The fingerprint of the EA's run.
    :returns: A list of problems; empty when all hold.
    """
    problems = []

    # Snapshots must say which algorithm wrote them, so resuming dispatches on
    # a stated fact rather than guessing from which keys are present.
    if current["algorithm_field"] != ["ea"]:
        problems.append(
            f"snapshots report algorithm {current['algorithm_field']}, expected ['ea']"
        )

    # If no score had fallen outside [0, 1] the run would agree wherever the
    # clip lived, and this whole check would prove nothing.
    clip = current["clip_exercised"]
    if clip["below_zero"] + clip["above_one"] == 0:
        problems.append(
            "no fitness fell outside [0, 1], so this run does not actually test the clip"
        )
    if clip["within_range"] == 0:
        problems.append(
            "every fitness was clipped, so selection had nothing to distinguish"
        )

    # The CSV schema is fixed and shared with CMA-ES.
    expected_header = [
        "num_of_generation", "best_fitness", "worst_fitness", "best_parent_fitness",
        "best_offspring_fitness", "best_ever_fitness", "best_robot_weights",
        "worst_robot_weights",
    ]
    if current["csv"]["header"] != expected_header:
        problems.append(f"CSV header is {current['csv']['header']}")
    if current["num_csv_files"] != 1:
        problems.append(f"run wrote {current['num_csv_files']} run CSV files, expected 1")

    # A resumable snapshot needs the state to continue from.
    required = {
        "population_parameters", "population_fitnesses", "rng_state",
        "reproducer_rng_state", "best_ever_parameters", "best_ever_fitness",
        "num_params", "training_ball_poses",
    }
    for name, snapshot in current["snapshots"].items():
        missing = sorted(required - set(snapshot["keys"]))
        if missing:
            problems.append(f"{name} cannot be resumed from: missing {', '.join(missing)}")

    # Fitness reported to the CSV must be inside the clipped range.
    for row in current["csv"]["rows"]:
        for column, value in zip(expected_header, row):
            if column.endswith("fitness") and value:
                if not -1e-12 <= float(value) <= 1.0 + 1e-12:
                    problems.append(f"{column} is {value}, outside the clipped range")

    return problems


def without_additions(fingerprint: dict) -> dict:
    """
    Drop what the reference cannot contain.

    :param fingerprint: A run's fingerprint.
    :returns: A copy without the algorithm field and the snapshots' additions.
    """
    trimmed = {key: value for key, value in fingerprint.items() if key != "algorithm_field"}
    trimmed["snapshots"] = {
        name: {key: value for key, value in snapshot.items() if key != "_additions"}
        for name, snapshot in fingerprint["snapshots"].items()
    }
    return trimmed


def check_selection_rule(clamped: dict, raw: dict) -> list[str]:
    """
    Check the two selection rules do what they say.

    :param clamped: Fingerprint of a run with SELECT_ON_RAW_FITNESS = False.
    :param raw: Fingerprint of a run with the default (unclamped) rule.
    :returns: A list of problems; empty when all hold.
    """
    problems = []
    outside = 0
    for name, snapshot in clamped["snapshots"].items():
        added = snapshot["_additions"]
        if added.get("selection_on_raw_fitness") is not False:
            problems.append(f"clamped run {name}: selection_on_raw_fitness is not False")
        if added.get("population_selection_fitnesses") != snapshot["population_fitnesses"]:
            problems.append(f"clamped run {name}: selection values differ from the recorded ones")
    for name, snapshot in raw["snapshots"].items():
        added = snapshot["_additions"]
        if added.get("selection_on_raw_fitness") is not True:
            problems.append(f"raw run {name}: selection_on_raw_fitness is not True")
        selection = added.get("population_selection_fitnesses", [])
        recorded = snapshot["population_fitnesses"]
        # Recorded fitness is the clamp of what selection compared, nothing else.
        if [ea_fingerprint._round(min(max(v, 0.0), 1.0)) for v in selection] != recorded:
            problems.append(f"raw run {name}: recorded fitness is not the clamp of the selection value")
        outside += sum(1 for v in selection if v < 0.0 or v > 1.0)
    if outside == 0:
        problems.append("no selection value fell outside [0, 1], so the raw rule was not exercised")
    if raw["snapshots"] == clamped["snapshots"]:
        problems.append("the unclamped rule made exactly the same choices as the clamped one")
    return problems


def check_tournament_compares_selection_fitness(work: Path) -> list[str]:
    """
    Check the tournament is handed the selection values, not the recorded ones.

    A whole run cannot pin this down on its own (other choices differ between the
    two rules anyway), so the reproducer is driven directly with a population
    whose recorded fitness is tied at 1.0 while the selection values differ, and
    the values the tournament receives are captured.

    :param work: A scratch folder.
    :returns: A list of problems; empty when all hold.
    """
    import numpy as np

    main_module = ea_fingerprint.load_main_module(MAIN, CONFIG, False, work / "tournament")
    genotype_module = sys.modules["genotype"]
    selection_values = [1.30, 0.40, 1.05, -0.20, 1.10, 0.95]
    population = [
        main_module.Individual(
            genotype_module.Genotype(np.full(4, float(index))),
            min(max(value, 0.0), 1.0),
            value,
        )
        for index, value in enumerate(selection_values)
    ]
    seen: list[list[float]] = []
    original = main_module._tournament

    def spy(rng, fitnesses, k):
        seen.append(list(fitnesses))
        return original(rng, fitnesses, k)

    main_module._tournament = spy
    try:
        main_module.TournamentCloneReproducer().reproduce(population)
    finally:
        main_module._tournament = original
    if not seen:
        return ["the reproducer never ran a tournament"]
    if any(values != selection_values for values in seen):
        return [f"the tournament compared {seen[0]}, not the selection values {selection_values}"]
    return []


def check_recording(current: dict) -> list[str]:
    """
    Check every individual of every generation (including 0) was recorded.

    :param current: A run's fingerprint.
    :returns: A list of problems; empty when all hold.
    """
    import json

    problems = []
    settings = current["settings"]
    weights = current["_recording"]["weights"]
    performance = current["_recording"]["performance"]
    population = settings["population_size"]
    generations = settings["num_generations"] + 1  # generation 0 is recorded too
    if len(weights) != generations * population:
        problems.append(f"weights.csv has {len(weights)} rows, expected {generations * population}")
    if len(performance) != generations * population * settings["num_training_ball_poses"]:
        problems.append(f"the performance CSV has {len(performance)} rows")
    if len({row["individual_id"] for row in weights}) != len(weights):
        problems.append("weights.csv repeats an individual_id")
    recorded_weights = {row["individual_id"]: json.loads(row["robot_weights"]) for row in weights}
    for name, snapshot in current["snapshots"].items():
        generation = snapshot["generation"]
        if not (snapshot["_additions"].get("detailed_csv_path") and snapshot["_additions"].get("weights_csv_path")):
            problems.append(f"{name}: does not carry the recording file paths")
        for index, (parameters, fitness) in enumerate(
            zip(snapshot["population_parameters"], snapshot["population_fitnesses"])
        ):
            individual = f"g{generation}_i{index}"
            if ea_fingerprint._round(recorded_weights.get(individual, [])) != parameters:
                problems.append(f"{individual}: recorded weights differ from the snapshot's population")
            rows = [row for row in performance if row["individual_id"] == individual]
            if not rows or any(ea_fingerprint._round(float(row["overall_fitness"])) != fitness for row in rows):
                problems.append(f"{individual}: recorded overall_fitness is not the recorded (clamped) fitness")
    return problems


def main() -> None:
    """Run the EA under both selection rules and check them."""
    work = Path(tempfile.mkdtemp(prefix="oribots_ea_check_"))
    try:
        # The reference EA selected on the clamped fitness: with that rule this
        # EA must reproduce it exactly.
        clamped_config = work / "_tiny_clamped.py"
        clamped_config.write_text(CONFIG.read_text() + "\nSELECT_ON_RAW_FITNESS = False\n")
        current = ea_fingerprint.fingerprint(
            main_path=MAIN,
            config_path=clamped_config,
            evaluator_clips=False,
            output_dir=work / "run",
        )
        raw = ea_fingerprint.fingerprint(
            main_path=MAIN,
            config_path=CONFIG,
            evaluator_clips=False,
            output_dir=work / "run_raw",
        )
        tournament_problems = check_tournament_compares_selection_fitness(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    reference = ea_fingerprint.load_reference()
    problems = differences(without_additions(reference), without_additions(current))
    problems.extend(check_expectations(current))
    problems.extend(check_expectations(raw))
    problems.extend(check_selection_rule(current, raw))
    problems.extend(tournament_problems)
    problems.extend(check_recording(current))
    problems.extend(check_recording(raw))

    if problems:
        print("EA CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        raise SystemExit(1)

    clip = current["clip_exercised"]
    settings = current["settings"]
    print("EA CHECK PASSED")
    print(
        f"  a {settings['population_size']}-controller, {settings['num_generations']}-generation run "
        "reproduces the reference EA exactly"
    )
    print("  with SELECT_ON_RAW_FITNESS = False it clips raw fitness itself and selects on the clipped value")
    print("  by default it selects on the unclamped value, records the clamped one, and chooses differently")
    print(
        f"  every individual recorded, generation 0 included: {len(current['_recording']['weights'])} "
        "weights rows per run, matching the snapshots"
    )
    print(
        f"  the clip was exercised: {clip['below_zero']} of {clip['scores']} scores fell below 0, "
        f"{clip['within_range']} were inside the range"
    )
    print(f"  {len(current['snapshots'])} snapshots, each resumable and tagged algorithm='ea'")
    print("  tournament selection and clone crossover make identical choices")


if __name__ == "__main__":
    main()
