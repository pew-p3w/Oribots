"""Reference-value fingerprints of the evolutionary algorithm itself.

The EA decides which controllers survive, so a change to it is only safe if the
changed code makes the same choices. This module runs a complete short EA and
records everything it produced: every CSV row, every generation's fitnesses and
weights, and the contents of each snapshot.

No simulation is involved. The evaluator is replaced with a stub that derives a
fitness from the parameter vector arithmetically, and the time-seeded random
generators are replaced with seeded ones, so a run is reproducible to the last
digit.

The point of comparison needs care. The reference was recorded from an earlier
implementation in which the evaluator clipped fitness to `[0, 1]` before the EA
saw it. In the current code the shared evaluator returns raw signed fitness and
the EA applies the clip itself. With clamped selection
(`SELECT_ON_RAW_FITNESS = False`) the two are equivalent only if the EA clips in
the right place: the clipped value is what tournament selection, `best_ever` and
the CSV all use, so clipping a moment too late would change which robots win.
The reference is therefore recorded as:

    reference EA  +  an evaluator that clips        (the earlier arrangement)

and the current code is checked as:

    current EA    +  an evaluator that returns raw  (the current arrangement)

Identical results show that moving the clip did not change the search.
"""

import csv
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

REFERENCE_PATH = Path(__file__).resolve().parent / "ea_reference.json"

SEED = 20260921
DECIMALS = 12
# Snapshot keys that are storage detail rather than a decision the EA made.
IGNORED_SNAPSHOT_KEYS = {"csv_path", "save_path", "config_source", "code_paths", "artifacts"}
# Current snapshots carry this; the reference snapshots have no such field.
ALGORITHM_KEY = "algorithm"
# Per-individual recording paths and the unclamped selection values. Recorded
# separately (see `additions`), because the reference snapshots do not contain
# them.
ADDED_SNAPSHOT_KEYS = {
    "detailed_csv_path",
    "weights_csv_path",
    "selection_on_raw_fitness",
    "raw_selection_switched_after_generation",
    "population_selection_fitnesses",
    "best_ever_selection_fitness",
}


def _round(value: Any) -> Any:
    """Round a number, or a nested sequence of numbers."""
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [_round(item) for item in value]
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def raw_fitness(parameters: Any) -> float:
    """
    Score a parameter vector without simulating anything.

    Scaled so most scores land inside `[0, 1]` while some fall outside in both
    directions. Both halves matter: if nothing were ever clipped the comparison
    would prove nothing about where the clip lives, and if everything were
    clipped every score would collapse onto 0 or 1 and selection would be
    meaningless. `fingerprint()` records how often the clip actually bit.

    :param parameters: A controller parameter vector.
    :returns: A reproducible signed fitness, roughly within [-0.1, 1.1].
    """
    import numpy as np

    values = np.asarray(getattr(parameters, "parameters", parameters), dtype=float)
    return float(np.sin(values.sum() * 3.7) * 0.6 + 0.5)


def make_stub_evaluator_module(clips: bool, num_parameters: int) -> ModuleType:
    """
    Build a stand-in `evaluator` module that scores without simulating.

    :param clips: Whether the evaluator clips fitness to `[0, 1]` before
        returning it, as in the reference implementation.
    :param num_parameters: The controller size to report.
    :returns: A module exposing an `Evaluator` class.
    """
    module = ModuleType("evaluator")

    class StubEvaluator:
        """Scores parameter vectors arithmetically, in place of MuJoCo."""

        def __init__(self) -> None:
            self.calls = 0

        @property
        def num_parameters(self) -> int:
            """
            The controller size.

            :returns: Number of parameters.
            """
            return num_parameters

        def sample_ball_pose(self) -> Any:
            """
            A stand-in training ball pose.

            :returns: A pose-like object with the attributes the EA logs.
            """
            return _StubPose()

        def sample_ball_poses(self, num_poses: int) -> list[Any]:
            """
            Stand-in training ball poses.

            :param num_poses: How many to sample.
            :returns: The poses.
            """
            return [_StubPose() for _ in range(num_poses)]

        def evaluate_on_ball_poses(self, population: list[Any], ball_poses: list[Any]) -> list[float]:
            """
            Score a population.

            :param population: Genotypes or parameter vectors.
            :param ball_poses: Ignored; kept for signature compatibility.
            :returns: One fitness per member.
            :raises ValueError: If no ball poses are given, as the real one does.
            """
            if len(ball_poses) == 0:
                raise ValueError("At least one ball pose is required.")
            self.calls += 1
            scores = [raw_fitness(member) for member in population]
            module.raw_scores.extend(scores)
            if clips:
                return [min(max(score, 0.0), 1.0) for score in scores]
            return scores

        def evaluate_on_ball_poses_full(
            self, population: list[Any], ball_poses: list[Any]
        ) -> tuple[list[float], list[list[float]], list[list[dict[str, float]]]]:
            """
            Score a population, with the per-trial detail recording needs.

            Every trial scores the same as the mean, so the mean is exactly what
            `evaluate_on_ball_poses` returns.

            :param population: Genotypes or parameter vectors.
            :param ball_poses: One trial per pose.
            :returns: (mean per member, trials [pose][member], movements [pose][member]).
            """
            means = self.evaluate_on_ball_poses(population, ball_poses)
            movement = {
                "forward_m": 0.0,
                "backward_m": 0.0,
                "sideways_m": 0.0,
                "net_along_heading_m": 0.0,
                "start_heading_deg": 0.0,
                "end_heading_deg": 0.0,
                "end_angle_to_ball_deg": 0.0,
            }
            return (
                means,
                [list(means) for _ in ball_poses],
                [[dict(movement) for _ in population] for _ in ball_poses],
            )

    module.Evaluator = StubEvaluator
    module.raw_scores = []
    return module


class _StubPosition:
    """A stand-in ball position."""

    x = 4.0
    y = 3.0
    z = 0.3


class _StubPose:
    """A stand-in ball pose, carrying only what the EA logs."""

    def __init__(self) -> None:
        self.position = _StubPosition()


def load_main_module(main_path: Path, config_path: Path, evaluator_clips: bool, output_dir: Path) -> ModuleType:
    """
    Load an EA `main.py` with stubs in place of the simulator and the clock.

    :param main_path: The `main.py` to load.
    :param config_path: The config it should run.
    :param evaluator_clips: Whether the stub evaluator clips before returning.
    :param output_dir: Where the run should write its artifacts.
    :returns: The loaded module, ready for `main()`.
    """
    import numpy as np

    os.environ["REVOLVE2_RUN_CONFIG_PATH"] = str(config_path)
    os.environ["REVOLVE2_RUN_MAIN_PATH"] = str(main_path)
    os.environ["REVOLVE2_RUN_OUTPUT_DIR"] = str(output_dir)

    for cached in ("config", "evaluator", "bodies", "genotype", "ball_aware_brain", "paths"):
        sys.modules.pop(cached, None)
    paths.install(config_path.parent)

    # The config decides the controller size; ask it via the real body.
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)

    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )

    hinges = config_module.BODY.find_modules_of_type(ActiveHinge)
    structure, mapping = active_hinges_to_cpg_network_structure_neighbor(hinges)
    num_parameters = structure.num_connections + len(mapping) * config_module.FEEDBACK_NUM_INPUTS

    sys.modules["evaluator"] = make_stub_evaluator_module(evaluator_clips, num_parameters)

    spec = importlib.util.spec_from_file_location(f"ea_main_{evaluator_clips}", main_path)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    # Both the EA and its reproducer seed themselves from the clock; replace
    # that so a run is reproducible.
    main_module.make_rng_time_seed = lambda: np.random.default_rng(SEED)
    return main_module


def _reproducer_trace(main_module: ModuleType) -> dict[str, Any]:
    """
    Record the reproducer's choices directly, apart from a full run.

    :param main_module: The loaded EA module.
    :returns: Tournament picks and the offspring produced from a fixed population.
    """
    import numpy as np

    genotype_module = sys.modules["genotype"]

    fitnesses = [0.10, 0.90, 0.35, 0.35, 0.00, 0.72, 0.51, 0.99]
    rng = np.random.default_rng(SEED)
    tournaments = [
        main_module._tournament(rng, fitnesses, k)
        for k in (2, 2, 3, 4, 2, 5)
    ]

    rng = np.random.default_rng(SEED)
    population = [
        main_module.Individual(genotype_module.Genotype(rng.uniform(-1.0, 1.0, 12)), fitness)
        for fitness in fitnesses
    ]
    reproducer = main_module.TournamentCloneReproducer()
    reproducer._rng = np.random.default_rng(SEED)
    children = reproducer.reproduce(population)

    return {
        "crossover_probability": _round(main_module.CROSSOVER_PROBABILITY),
        "tournament_winners": tournaments,
        "parent_fitnesses": _round(fitnesses),
        "num_children": len(children),
        "children": [_round(child.parameters) for child in children],
    }


def _read_csv(path: Path) -> dict[str, Any]:
    """
    Read a run's CSV.

    :param path: The CSV file.
    :returns: Its header and rows.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    return {"header": rows[0], "rows": rows[1:]}


def _read_snapshots(output_dir: Path) -> dict[str, Any]:
    """
    Read every generation snapshot and record the decisions it captured.

    :param output_dir: The run's output directory.
    :returns: Per-snapshot recorded values.
    """
    snapshots = {}
    for path in sorted(output_dir.glob("gen*.pkl"), key=lambda p: int(p.stem.removeprefix("gen"))):
        with open(path, "rb") as handle:
            snapshot = pickle.load(handle)
        recorded = {
            "keys": sorted(set(snapshot) - IGNORED_SNAPSHOT_KEYS - {ALGORITHM_KEY} - ADDED_SNAPSHOT_KEYS),
            "generation": snapshot["generation"],
            "num_params": snapshot["num_params"],
            "fitness": _round(snapshot["fitness"]),
            "best_fitness": _round(snapshot["best_fitness"]),
            "worst_fitness": _round(snapshot["worst_fitness"]),
            "best_parent_fitness": _round(snapshot["best_parent_fitness"]),
            "best_offspring_fitness": _round(snapshot["best_offspring_fitness"]),
            "best_ever_fitness": _round(snapshot["best_ever_fitness"]),
            "population_fitnesses": _round(snapshot["population_fitnesses"]),
            "best_parameters": _round(snapshot["best_parameters"]),
            "worst_parameters": _round(snapshot["worst_parameters"]),
            "best_ever_parameters": _round(snapshot["best_ever_parameters"]),
            "population_parameters": _round(snapshot["population_parameters"]),
            "csv_row": {key: str(value) for key, value in snapshot["csv_row"].items()},
            "_additions": {
                key: (
                    _round(snapshot[key])
                    if key in ("population_selection_fitnesses", "best_ever_selection_fitness")
                    else snapshot[key]
                )
                for key in ADDED_SNAPSHOT_KEYS
                if key in snapshot
            },
        }
        snapshots[path.stem] = recorded
    return snapshots


def fingerprint(main_path: Path, config_path: Path, evaluator_clips: bool, output_dir: Path) -> dict[str, Any]:
    """
    Run a complete short EA and record everything it decided.

    :param main_path: The EA `main.py` to run.
    :param config_path: The config to run it with.
    :param evaluator_clips: Whether the stub evaluator clips before returning,
        as in the reference implementation.
    :param output_dir: Where the run writes its artifacts.
    :returns: A JSON-serializable fingerprint of the run.
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    main_module = load_main_module(main_path, config_path, evaluator_clips, output_dir)

    reproducer = _reproducer_trace(main_module)

    argv = sys.argv
    sys.argv = [str(main_path)]
    try:
        main_module.main()
    finally:
        sys.argv = argv

    config_module = sys.modules["config"]
    # The run CSV is compared with the reference; the per-individual recording
    # files (which the reference implementation did not write) are read separately.
    csv_files = sorted(
        path
        for path in output_dir.glob("*.csv")
        if path.name != "weights.csv" and not path.name.startswith("individual_performance_")
    )

    def read_rows(pattern: str) -> list[dict[str, str]]:
        found = sorted(output_dir.glob(pattern))
        if len(found) != 1:
            return []
        with open(found[0], newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    weights_path = output_dir / config_module.parameter_filename()

    snapshot_algorithms = set()
    for path in output_dir.glob("gen*.pkl"):
        with open(path, "rb") as handle:
            snapshot_algorithms.add(pickle.load(handle).get(ALGORITHM_KEY))

    # Proof that this run actually exercised the clip. If nothing fell outside
    # [0, 1], the run would agree no matter where the clip was applied, and the
    # comparison would be worthless.
    scores = sys.modules["evaluator"].raw_scores
    below = sum(1 for score in scores if score < 0.0)
    above = sum(1 for score in scores if score > 1.0)

    return {
        "settings": {
            "seed": SEED,
            "decimals": DECIMALS,
            "population_size": config_module.POPULATION_SIZE,
            "num_generations": config_module.NUM_GENERATIONS,
            "num_training_ball_poses": config_module.NUM_TRAINING_BALL_POSES,
        },
        "reproducer": reproducer,
        "csv": _read_csv(csv_files[0]),
        "num_csv_files": len(csv_files),
        "snapshots": _read_snapshots(output_dir),
        "best_weights": _round(np.load(weights_path)),
        "algorithm_field": sorted(value for value in snapshot_algorithms if value is not None),
        "clip_exercised": {
            "scores": len(scores),
            "below_zero": below,
            "above_one": above,
            "within_range": len(scores) - below - above,
        },
        "_recording": {
            "performance": read_rows("individual_performance_*.csv"),
            "weights": read_rows("weights.csv"),
        },
    }


def load_reference() -> dict[str, Any]:
    """
    Load the stored reference fingerprint.

    :returns: The reference values of the EA.
    """
    return json.loads(REFERENCE_PATH.read_text())
