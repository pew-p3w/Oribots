"""Reference-value fingerprints of the CMA-ES optimization loop.

CMA-ES decides which controllers to try next from the fitness of the ones it
just tried, so a change to the loop is only safe if it makes the same sequence
of decisions. This module runs a complete short CMA-ES and records what the
optimizer produced: the solutions it asked for each generation, the fitnesses it
was told, and the mean and step-size (sigma) trajectory that its covariance
adaptation drives.

No simulation is involved. The evaluator is replaced with a stub that scores a
parameter vector arithmetically, and the CMA-ES seed is fixed, so a run is
reproducible to the last digit. `cma_fingerprint` takes the main module as an
argument, so the same scenario can be run against any version of the loop and
the results compared.

`tests/cmaes_reference.json` holds the reference values, recorded from an
earlier implementation of the loop; `tests/test_cmaes.py` checks the current
loop against them.
"""

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

REFERENCE_PATH = Path(__file__).resolve().parent / "cmaes_reference.json"

SEED = 20260922
DECIMALS = 10
IGNORED_SNAPSHOT_KEYS = {"csv_path", "save_path", "config_source", "code_paths", "artifacts", "cma_state"}


def _round(value: Any) -> Any:
    """Round a number, or a nested sequence of numbers."""
    if hasattr(value, "__len__") and not isinstance(value, str):
        return [_round(item) for item in value]
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def raw_fitness(parameters: Any) -> float:
    """
    Score a parameter vector without simulating anything.

    Signed, so CMA-ES sees candidates that ended up worse than they started, and
    smooth in the parameters so the covariance adaptation has a gradient to
    follow across generations.

    :param parameters: A controller parameter vector.
    :returns: A reproducible signed fitness.
    """
    import numpy as np

    values = np.asarray(parameters, dtype=float)
    return float(np.cos(values.sum() * 2.3) * 0.5 + values.mean() * 0.1)


def make_stub_evaluator_module(num_parameters: int) -> ModuleType:
    """
    Build a stand-in `evaluator` module that scores without simulating.

    :param num_parameters: The controller size to report.
    :returns: A module exposing an `Evaluator` class.
    """
    module = ModuleType("evaluator")

    class StubEvaluator:
        """Scores parameter vectors arithmetically, in place of MuJoCo."""

        @property
        def num_parameters(self) -> int:
            """
            The controller size.

            :returns: Number of parameters.
            """
            return num_parameters

        def sample_ball_poses(self, num_poses: int) -> list[Any]:
            """
            Stand-in training ball poses.

            :param num_poses: How many to sample.
            :returns: The poses.
            """
            return [_StubPose() for _ in range(num_poses)]

        def evaluate_on_ball_poses(self, population: list[Any], ball_poses: list[Any]) -> list[float]:
            """
            Score a population, raw and signed.

            :param population: Parameter vectors.
            :param ball_poses: Ignored; kept for signature compatibility.
            :returns: One fitness per member.
            :raises ValueError: If no ball poses are given.
            """
            if len(ball_poses) == 0:
                raise ValueError("At least one ball pose is required.")
            return [raw_fitness(member) for member in population]

        def evaluate_on_ball_poses_full(
            self, population: list[Any], ball_poses: list[Any]
        ) -> tuple[list[float], list[list[float]], list[list[dict[str, float]]]]:
            """
            Score a population, with the per-trial detail recording needs.

            Every trial scores the same as the mean, so the mean handed to the
            optimizer is exactly `raw_fitness`, as in `evaluate_on_ball_poses`.

            :param population: Parameter vectors.
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
    return module


class _StubPosition:
    """A stand-in ball position."""

    x = 4.0
    y = 3.0
    z = 0.3


class _StubPose:
    """A stand-in ball pose, carrying only what the loop logs."""

    def __init__(self) -> None:
        self.position = _StubPosition()


def _load_main_module(main_path: Path, config_path: Path, output_dir: Path) -> ModuleType:
    """
    Load a CMA-ES `main.py` with a stub evaluator and a fixed seed.

    :param main_path: The `main.py` to load.
    :param config_path: The config it should run.
    :param output_dir: Where the run writes its artifacts.
    :returns: The loaded module, ready for `main()`.
    """
    os.environ["REVOLVE2_RUN_CONFIG_PATH"] = str(config_path)
    os.environ["REVOLVE2_RUN_MAIN_PATH"] = str(main_path)
    os.environ["REVOLVE2_RUN_OUTPUT_DIR"] = str(output_dir)

    for cached in ("config", "evaluator", "bodies", "genotype", "ball_aware_brain", "paths"):
        sys.modules.pop(cached, None)
    paths.install(config_path.parent)

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

    sys.modules["evaluator"] = make_stub_evaluator_module(num_parameters)

    spec = importlib.util.spec_from_file_location(f"cmaes_main_{main_path.parent.name}", main_path)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)

    # CMA-ES seeds from the clock; fix it so the ask/tell sequence is reproducible.
    main_module.seed_from_time = lambda: SEED
    return main_module


def _read_snapshots(output_dir: Path) -> dict[str, Any]:
    """
    Read every generation snapshot and record the decisions it captured.

    The optimizer state (`cma_state`) is excluded from the recorded values - it
    is large, contains derived arrays, and its effect is captured indirectly and
    more legibly by the mean/sigma trajectory below.

    :param output_dir: The run's output directory.
    :returns: Per-snapshot recorded values.
    """
    snapshots = {}
    for path in sorted(output_dir.glob("gen*.pkl"), key=lambda p: int(p.stem.removeprefix("gen"))):
        with open(path, "rb") as handle:
            snapshot = pickle.load(handle)
        # The reference snapshots predate the algorithm, cma_state and csv_row
        # fields, so read defensively: the fitness/parameter decisions are the
        # shared contract, and the newer fields are recorded separately for the
        # test to check.
        csv_row = snapshot.get("csv_row")
        snapshots[path.stem] = {
            "has_cma_state": "cma_state" in snapshot,
            "algorithm": snapshot.get("algorithm"),
            "generation": snapshot["generation"],
            "best_fitness": _round(snapshot["best_fitness"]),
            "worst_fitness": _round(snapshot["worst_fitness"]),
            "best_ever_fitness": _round(snapshot["best_ever_fitness"]),
            "population_fitnesses": _round(snapshot["population_fitnesses"]),
            "best_parameters": _round(snapshot["best_parameters"]),
            "recording_paths": [snapshot.get("detailed_csv_path"), snapshot.get("weights_csv_path")],
            "population_parameters": _round([list(p) for p in snapshot["population_parameters"]]),
            "csv_row": {key: str(value) for key, value in csv_row.items()} if csv_row else None,
        }
    return snapshots


def cma_fingerprint(main_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """
    Run a complete short CMA-ES and record everything the optimizer decided.

    Wraps the optimizer's `ask`/`tell` to record the solutions requested, the
    fitnesses supplied, and the mean/sigma after each `tell` - the trajectory the
    covariance adaptation produces.

    :param main_path: The CMA-ES `main.py` to run.
    :param config_path: The config to run it with.
    :param output_dir: Where the run writes its artifacts.
    :returns: A JSON-serializable fingerprint of the run.
    """
    import cma
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    main_module = _load_main_module(main_path, config_path, output_dir)

    trace: dict[str, list] = {"ask": [], "tell_fitnesses": [], "mean": [], "sigma": []}
    original_init = cma.CMAEvolutionStrategy.__init__
    original_ask = cma.CMAEvolutionStrategy.ask
    original_tell = cma.CMAEvolutionStrategy.tell

    def traced_ask(self, *args, **kwargs):
        solutions = original_ask(self, *args, **kwargs)
        trace["ask"].append(_round([list(s) for s in solutions]))
        return solutions

    def traced_tell(self, solutions, values, *args, **kwargs):
        trace["tell_fitnesses"].append(_round(list(values)))
        result = original_tell(self, solutions, values, *args, **kwargs)
        trace["mean"].append(_round(list(self.mean)))
        trace["sigma"].append(_round(self.sigma))
        return result

    cma.CMAEvolutionStrategy.ask = traced_ask
    cma.CMAEvolutionStrategy.tell = traced_tell
    argv = sys.argv
    sys.argv = [str(main_path)]
    try:
        main_module.main()
    finally:
        sys.argv = argv
        cma.CMAEvolutionStrategy.ask = original_ask
        cma.CMAEvolutionStrategy.tell = original_tell

    import csv as csv_module

    config_module = sys.modules["config"]
    # The run CSV is compared with the reference; the per-individual recording
    # files (which the reference implementation did not write) are read separately.
    recording_files = {"weights.csv"}
    csv_files = sorted(
        path
        for path in output_dir.glob("*.csv")
        if path.name not in recording_files and not path.name.startswith("individual_performance_")
    )
    weights_path = output_dir / config_module.parameter_filename()

    with open(csv_files[0], newline="", encoding="utf-8") as handle:
        rows = list(csv_module.reader(handle))

    def read_rows(pattern: str) -> list[dict[str, str]]:
        found = sorted(output_dir.glob(pattern))
        if len(found) != 1:
            return []
        with open(found[0], newline="", encoding="utf-8") as handle:
            return list(csv_module.DictReader(handle))

    return {
        "settings": {
            "seed": SEED,
            "decimals": DECIMALS,
            "num_generations": config_module.NUM_GENERATIONS,
            "cma_population_size": config_module.CMA_POPULATION_SIZE,
            "cma_initial_std": config_module.CMA_INITIAL_STD,
            "cma_initial_mean": config_module.CMA_INITIAL_MEAN,
            "num_training_ball_poses": config_module.NUM_TRAINING_BALL_POSES,
        },
        "trace": trace,
        "csv": {"header": rows[0], "rows": rows[1:]},
        "num_csv_files": len(csv_files),
        "snapshots": _read_snapshots(output_dir),
        "best_weights": _round(list(np.load(weights_path))),
        "recording": {
            "performance": read_rows("individual_performance_*.csv"),
            "weights": read_rows("weights.csv"),
        },
    }


def load_reference() -> dict[str, Any]:
    """
    Load the stored reference fingerprint.

    :returns: The reference values of the CMA-ES loop.
    """
    return json.loads(REFERENCE_PATH.read_text())
