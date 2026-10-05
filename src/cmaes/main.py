"""CMA-ES optimization of the CPG brain weights, for the robot-to-ball task.

The scene, brain and fitness are identical to the EA; only the optimizer differs.
CMA-ES (via the `cma` package) works on raw signed fitness: the shared evaluator
returns it unclipped, and this file minimizes negative fitness with
`opt.tell(solutions, (-raw).tolist())`. Unlike the EA, this neither clips nor
records a clamped value: the CMA-ES search needs to rank candidates that a clip
to `[0, 1]` would tie, both at 0 (no progress, or moving away from the ball) and
at 1.

CSV columns match the EA exactly, but `best_parent_fitness` and
`best_offspring_fitness` are blank: CMA-ES has no parent/offspring split.

Snapshots carry the full CMA optimizer state (`cma_state`) as well as
`algorithm="cmaes"`. Storing the optimizer state in every `gen#.pkl` is what
lets `run.py -c` resume a CMA-ES run and continue the covariance adaptation,
rather than restart it.

Every individual of every generation is recorded (`src/recording.py`): each
trial's raw score and movement in `individual_performance_<name>_run_<ts>.csv`,
and each controller's weights in `weights.csv`. The recording files' paths travel
in every snapshot, so a resumed run keeps writing into them, after dropping any
rows the interrupted run had already written past the resume point (the run CSV
is trimmed the same way).

`cma` is imported only in this file, so EA runs and visual tests never require it.

Run through `run.py -r config/<body>_cmaes.py src/cmaes/main.py output/<run>`.
"""

import os
import sys
from pathlib import Path

# src/ holds the path helper (src/paths.py) but is not importable until it is on
# sys.path, so add it first and then import paths.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

_config_path = os.environ.get("REVOLVE2_RUN_CONFIG_PATH")
paths.install(Path(_config_path).parent if _config_path else None)

# Simulator workers re-import this module (macOS spawn), and a directly-launched
# child inherits neither the parent's sys.path nor its bytecode policy. Disable
# bytecode writing here so a same-second config edit cannot be masked by a stale
# .pyc. See src/paths.py.
sys.dont_write_bytecode = True

import csv
import importlib.util
import json
import logging
import math
import pickle
from datetime import datetime
from typing import Any

import cma
import numpy as np
import recording

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import seed_from_time


SNAPSHOT_VERSION = 2
# Recorded in every snapshot so `run.py -c` dispatches on a stated fact.
ALGORITHM = "cmaes"
CONFIG_PATH_ENV = "REVOLVE2_RUN_CONFIG_PATH"
MAIN_PATH_ENV = "REVOLVE2_RUN_MAIN_PATH"
OUTPUT_DIR_ENV = "REVOLVE2_RUN_OUTPUT_DIR"


def _load_config_module():
    """
    Load the selected run config as the module named config.

    :returns: Loaded config module.
    """
    config_path = os.environ.get(CONFIG_PATH_ENV)
    if config_path is None:
        import config as default_config

        return default_config
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config file: {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    return config_module


config = _load_config_module()


def main() -> None:
    """Run the CMA-ES optimization loop."""
    setup_logging()
    logging.info("Starting CMA-ES main.")
    logging.info("Importing evaluator and MuJoCo simulator dependencies.")
    from evaluator import Evaluator

    checkpoint_path = _checkpoint_path_from_args(sys.argv)

    generation_output_dir = os.environ.get(OUTPUT_DIR_ENV)
    output_dir = (
        os.path.abspath(generation_output_dir)
        if generation_output_dir is not None
        else os.path.dirname(__file__)
    )
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, config.parameter_filename())
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        output_dir,
        f"{config.parameter_filename().removesuffix('.npy')}_run_{run_timestamp}.csv",
    )
    detailed_csv_path, weights_csv_path = recording.default_paths(
        output_dir, config.parameter_filename().removesuffix(".npy"), run_timestamp
    )

    evaluator = Evaluator()
    num_params = evaluator.num_parameters
    logging.info(f"Controller has {num_params} parameters to optimize.")

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "rb") as handle:
            checkpoint = pickle.load(handle)
        _validate_resume(checkpoint, num_params)
        opt = _restore_optimizer(checkpoint["cma_state"])
        completed_generations = int(checkpoint["completed_generations"])
        training_ball_poses = checkpoint["training_ball_poses"]
        best_ever_params = np.asarray(checkpoint["best_ever_parameters"])
        best_ever_fitness = float(checkpoint["best_ever_fitness"])
        csv_path = checkpoint.get("csv_path", csv_path)
        save_path = checkpoint.get("save_path", save_path)
        # Keep appending to the recording files the run started with.
        saved_detailed_csv_path = checkpoint.get("detailed_csv_path")
        saved_weights_csv_path = checkpoint.get("weights_csv_path")
        if saved_detailed_csv_path and saved_weights_csv_path:
            detailed_csv_path = saved_detailed_csv_path
            weights_csv_path = saved_weights_csv_path
        else:
            logging.warning(
                "This snapshot was written by a run that did not record individuals. "
                f"Recording starts at generation {completed_generations + 1}, into new "
                f"files: {detailed_csv_path} and {weights_csv_path}"
            )
        # Drop rows the interrupted run already wrote past the resume point, so
        # nothing is written (or recorded) twice.
        recording.trim_after_generation(
            csv_path, completed_generations, lambda row: int(row["num_of_generation"])
        )
        recording.trim_recording_after_generation(
            detailed_csv_path, weights_csv_path, completed_generations
        )
        logging.info(f"Resumed after generation {completed_generations}.")
        _log_training_ball_poses(training_ball_poses)
    else:
        completed_generations = 0
        training_ball_poses = evaluator.sample_ball_poses(config.NUM_TRAINING_BALL_POSES)
        logging.info(
            f"Training on {len(training_ball_poses)} fixed random ball positions."
        )
        _log_training_ball_poses(training_ball_poses)
        logging.info(f"Run CSV will be saved to: {csv_path}")
        logging.info(f"Recording every individual to: {detailed_csv_path}")
        logging.info(f"Recording every individual's weights to: {weights_csv_path}")

        initial_mean = [config.CMA_INITIAL_MEAN] * num_params
        options = cma.CMAOptions()
        options.set("bounds", list(config.CMA_BOUNDS))
        options.set("seed", seed_from_time() % 2**32)
        options.set("maxiter", config.NUM_GENERATIONS)
        options.set("verbose", -9)
        if getattr(config, "CMA_POPULATION_SIZE", None) is not None:
            options.set("popsize", config.CMA_POPULATION_SIZE)
        opt = cma.CMAEvolutionStrategy(initial_mean, config.CMA_INITIAL_STD, options)
        logging.info(
            f"CMA-ES initialized: initial_mean={config.CMA_INITIAL_MEAN}, "
            f"initial_std={config.CMA_INITIAL_STD}, bounds={config.CMA_BOUNDS}, "
            f"population_size={opt.popsize}."
        )
        best_ever_params = np.array(initial_mean)
        best_ever_fitness = -float("inf")

    logging.info("Starting CMA-ES optimization.")
    for generation in range(completed_generations, config.NUM_GENERATIONS):
        logging.info(f"Generation {generation + 1} / {config.NUM_GENERATIONS}")

        solutions = opt.ask()
        # The per-trial scores and movements for recording come from the same
        # simulations, at no extra cost.
        mean_fitnesses, trial_fitnesses, trial_movements = evaluator.evaluate_on_ball_poses_full(
            solutions, training_ball_poses
        )
        raw_fitnesses = np.array(mean_fitnesses)
        logging.info(
            f"Best raw fitness this generation: {raw_fitnesses.max():.4f}  "
            f"Mean: {raw_fitnesses.mean():.4f}  Worst: {raw_fitnesses.min():.4f}"
        )

        best_idx = int(raw_fitnesses.argmax())
        if raw_fitnesses[best_idx] > best_ever_fitness:
            best_ever_fitness = float(raw_fitnesses[best_idx])
            best_ever_params = np.array(solutions[best_idx])
            np.save(save_path, best_ever_params)
            logging.info(f"New best-ever raw fitness: {best_ever_fitness:.4f}")

        # CMA-ES minimizes, and fitness is maximized, so it is told the negatives.
        opt.tell(solutions, (-raw_fitnesses).tolist())

        _write_csv_row(
            csv_path=csv_path,
            generation_index=generation + 1,
            solutions=solutions,
            raw_fitnesses=raw_fitnesses,
            best_ever_fitness=best_ever_fitness,
            mode="w" if generation == 0 else "a",
        )
        recording.write_generation(
            performance_csv_path=detailed_csv_path,
            weights_csv_path=weights_csv_path,
            generation_index=generation + 1,
            parameters=solutions,
            overall_fitnesses=mean_fitnesses,
            trial_fitnesses=trial_fitnesses,
            training_ball_poses=training_ball_poses,
            mode="w" if generation == 0 else "a",
            trial_movements=trial_movements,
        )
        if generation_output_dir is not None:
            _save_snapshot(
                output_dir=output_dir,
                generation_index=generation + 1,
                opt=opt,
                solutions=solutions,
                raw_fitnesses=raw_fitnesses,
                best_ever_params=best_ever_params,
                best_ever_fitness=best_ever_fitness,
                training_ball_poses=training_ball_poses,
                save_path=save_path,
                csv_path=csv_path,
                num_params=num_params,
                detailed_csv_path=detailed_csv_path,
                weights_csv_path=weights_csv_path,
            )
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                completed_generations=generation + 1,
                opt=opt,
                best_ever_params=best_ever_params,
                best_ever_fitness=best_ever_fitness,
                training_ball_poses=training_ball_poses,
                csv_path=csv_path,
                save_path=save_path,
                num_params=num_params,
                detailed_csv_path=detailed_csv_path,
                weights_csv_path=weights_csv_path,
            )
        if opt.stop():
            logging.info(f"CMA-ES stopped early: {opt.stop()}")
            break

    logging.info("CMA-ES optimization complete.")
    logging.info(f"Best raw fitness: {best_ever_fitness:.4f}")
    np.save(save_path, best_ever_params)
    logging.info(f"Best weights saved to: {save_path}")
    logging.info(f"Run CSV saved to: {csv_path}")


def _restore_optimizer(cma_state: dict[str, Any]) -> "cma.CMAEvolutionStrategy":
    """
    Rebuild a CMA-ES optimizer from a stored state dictionary.

    :param cma_state: The optimizer's `__dict__` as stored in a snapshot.
    :returns: A restored optimizer that continues the covariance adaptation.
    """
    opt = cma.CMAEvolutionStrategy.__new__(cma.CMAEvolutionStrategy)
    opt.__dict__.update(cma_state)
    return opt


def _validate_resume(checkpoint: dict[str, Any], num_params: int) -> None:
    """
    Check a resume checkpoint matches the current controller and carries CMA state.

    :param checkpoint: The loaded resume snapshot/checkpoint.
    :param num_params: Current expected number of parameters.
    :raises SystemExit: If the checkpoint cannot be resumed here.
    """
    if "cma_state" not in checkpoint:
        raise SystemExit(
            "This CMA-ES snapshot has no optimizer state (cma_state) and cannot be "
            "resumed. Only snapshots that store the optimizer state are resumable."
        )
    if int(checkpoint.get("num_params", num_params)) != num_params:
        raise SystemExit(
            f"Snapshot has {checkpoint['num_params']} parameters, but this setup "
            f"expects {num_params}."
        )


def _checkpoint_path_from_args(argv: list[str]) -> str | None:
    """
    Get optional checkpoint path from command-line arguments.

    :param argv: Command-line arguments.
    :returns: Absolute checkpoint path or None.
    :raises SystemExit: If too many arguments are passed.
    """
    if len(argv) > 2:
        raise SystemExit("Usage: main.py [checkpoint.pkl]")
    if len(argv) == 1:
        return None
    return os.path.abspath(argv[1])


def _log_training_ball_poses(training_ball_poses: list[Any]) -> None:
    """
    Log fixed training ball poses and their starting distances.

    :param training_ball_poses: Fixed training ball poses.
    """
    for index, ball_pose in enumerate(training_ball_poses):
        distance = math.sqrt(ball_pose.position.x**2 + ball_pose.position.y**2)
        logging.info(
            f"Training ball {index}: x={ball_pose.position.x:.4f}, "
            f"y={ball_pose.position.y:.4f}, initial robot-to-ball distance={distance:.4f} m"
        )


CSV_FIELDNAMES = [
    "num_of_generation",
    "best_fitness",
    "worst_fitness",
    "best_parent_fitness",
    "best_offspring_fitness",
    "best_ever_fitness",
    "best_robot_weights",
    "worst_robot_weights",
]


def _write_csv_row(
    csv_path: str,
    generation_index: int,
    solutions: list[Any],
    raw_fitnesses: np.ndarray,
    best_ever_fitness: float,
    mode: str,
) -> None:
    """
    Write one generation row to the run CSV.

    :param csv_path: Path to the run CSV.
    :param generation_index: Current generation (1-based).
    :param solutions: Solutions evaluated this generation.
    :param raw_fitnesses: Raw signed fitness values used by CMA-ES.
    :param best_ever_fitness: Best raw fitness over the whole run.
    :param mode: File open mode ('w' for the first row, 'a' after).
    """
    best_idx = int(raw_fitnesses.argmax())
    worst_idx = int(raw_fitnesses.argmin())
    with open(csv_path, mode, encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if mode == "w":
            writer.writeheader()
        writer.writerow(_csv_row(generation_index, solutions, raw_fitnesses, best_ever_fitness))


def _csv_row(
    generation_index: int,
    solutions: list[Any],
    raw_fitnesses: np.ndarray,
    best_ever_fitness: float,
) -> dict[str, Any]:
    """
    Build one CSV row. Parent/offspring columns are blank for CMA-ES.

    :param generation_index: Current generation (1-based).
    :param solutions: Solutions evaluated this generation.
    :param raw_fitnesses: Raw signed fitness values.
    :param best_ever_fitness: Best raw fitness over the whole run.
    :returns: A row keyed by the canonical CSV fields.
    """
    best_idx = int(raw_fitnesses.argmax())
    worst_idx = int(raw_fitnesses.argmin())
    return {
        "num_of_generation": generation_index,
        "best_fitness": f"{raw_fitnesses[best_idx]:.17g}",
        "worst_fitness": f"{raw_fitnesses[worst_idx]:.17g}",
        "best_parent_fitness": "",
        "best_offspring_fitness": "",
        "best_ever_fitness": f"{best_ever_fitness:.17g}",
        "best_robot_weights": _serialize(solutions[best_idx]),
        "worst_robot_weights": _serialize(solutions[worst_idx]),
    }


def _serialize(parameters: Any) -> str:
    """
    Serialize a parameter vector for a CSV cell.

    :param parameters: Controller parameter vector.
    :returns: JSON list string.
    """
    return json.dumps([float(value) for value in np.asarray(parameters)])


def _save_snapshot(
    output_dir: str,
    generation_index: int,
    opt: "cma.CMAEvolutionStrategy",
    solutions: list[Any],
    raw_fitnesses: np.ndarray,
    best_ever_params: np.ndarray,
    best_ever_fitness: float,
    training_ball_poses: list[Any],
    save_path: str,
    csv_path: str,
    num_params: int,
    detailed_csv_path: str,
    weights_csv_path: str,
) -> None:
    """
    Save a generation snapshot that is both testable and resumable.

    The snapshot stores the full optimizer state (`cma_state`), so `run.py -c`
    can continue the covariance adaptation from a `gen#.pkl` rather than
    restart it.

    :param output_dir: Folder where snapshots are written.
    :param generation_index: One-based generation number.
    :param opt: The CMA-ES optimizer (its state is stored).
    :param solutions: Solutions evaluated this generation.
    :param raw_fitnesses: Raw signed fitness values.
    :param best_ever_params: Best parameter vector so far.
    :param best_ever_fitness: Best raw fitness so far.
    :param training_ball_poses: Fixed training ball poses.
    :param save_path: Path to the best-ever npy file.
    :param csv_path: Path to the run CSV.
    :param num_params: Number of controller parameters.
    :param detailed_csv_path: Per-trial recording CSV path.
    :param weights_csv_path: Recorded weights CSV path.
    """
    best_idx = int(raw_fitnesses.argmax())
    worst_idx = int(raw_fitnesses.argmin())
    snapshot_path = os.path.join(output_dir, f"gen{generation_index}.pkl")
    module_dir = os.path.dirname(_main_path())
    snapshot = {
        "version": SNAPSHOT_VERSION,
        "algorithm": ALGORITHM,
        "generation": generation_index,
        "completed_generations": generation_index,
        "parameters": np.array(solutions[best_idx]),
        "fitness": float(raw_fitnesses[best_idx]),
        "best_parameters": np.array(solutions[best_idx]),
        "best_fitness": float(raw_fitnesses[best_idx]),
        "worst_parameters": np.array(solutions[worst_idx]),
        "worst_fitness": float(raw_fitnesses[worst_idx]),
        "best_ever_parameters": best_ever_params.copy(),
        "best_ever_fitness": best_ever_fitness,
        "best_parent_fitness": None,
        "best_offspring_fitness": None,
        "population_parameters": [np.array(solution) for solution in solutions],
        "population_fitnesses": raw_fitnesses.tolist(),
        "training_ball_poses": training_ball_poses,
        # The optimizer state travels with the snapshot, so a resume continues it.
        "cma_state": opt.__dict__.copy(),
        "num_params": num_params,
        "csv_path": csv_path,
        "save_path": save_path,
        "detailed_csv_path": detailed_csv_path,
        "weights_csv_path": weights_csv_path,
        "config": _config_values(),
        "config_source": _config_source(),
        "code_paths": {
            "main_path": _main_path(),
            "module_dir": module_dir,
            "config_path": _config_path_abs(),
            "ball_aware_brain_path": os.path.join(module_dir, "ball_aware_brain.py"),
            "evaluator_path": os.path.join(module_dir, "evaluator.py"),
        },
        "artifacts": {
            "best_weights_path": save_path,
            "training_csv_path": csv_path,
            "output_dir": output_dir,
        },
        "csv_row": _csv_row(generation_index, solutions, raw_fitnesses, best_ever_fitness),
    }
    temp_path = f"{snapshot_path}.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(snapshot, handle)
    os.replace(temp_path, snapshot_path)
    logging.info(f"Generation snapshot saved to: {snapshot_path}")


def _save_checkpoint(
    checkpoint_path: str,
    completed_generations: int,
    opt: "cma.CMAEvolutionStrategy",
    best_ever_params: np.ndarray,
    best_ever_fitness: float,
    training_ball_poses: list[Any],
    csv_path: str,
    save_path: str,
    num_params: int,
    detailed_csv_path: str,
    weights_csv_path: str,
) -> None:
    """
    Save all state required to resume CMA-ES later.

    :param checkpoint_path: Path to the checkpoint file.
    :param completed_generations: Number of completed generations.
    :param opt: The CMA-ES optimizer.
    :param best_ever_params: Best parameter vector so far.
    :param best_ever_fitness: Best fitness so far.
    :param training_ball_poses: Fixed training ball poses.
    :param csv_path: Path to the run CSV.
    :param save_path: Path to the best-ever npy file.
    :param num_params: Number of controller parameters.
    :param detailed_csv_path: Per-trial recording CSV path.
    :param weights_csv_path: Recorded weights CSV path.
    """
    checkpoint = {
        "algorithm": ALGORITHM,
        "completed_generations": completed_generations,
        "cma_state": opt.__dict__.copy(),
        "best_ever_parameters": best_ever_params.copy(),
        "best_ever_fitness": best_ever_fitness,
        "training_ball_poses": training_ball_poses,
        "csv_path": csv_path,
        "save_path": save_path,
        "num_params": num_params,
        "detailed_csv_path": detailed_csv_path,
        "weights_csv_path": weights_csv_path,
        "config_source": _config_source(),
        "code_paths": {"main_path": _main_path(), "config_path": _config_path_abs()},
    }
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(checkpoint, handle)
    os.replace(temp_path, checkpoint_path)
    logging.info(
        f"Checkpoint saved after generation {completed_generations}: {checkpoint_path}"
    )


def _config_values() -> dict[str, Any]:
    """
    Collect config values for snapshot storage.

    :returns: Serializable config summary.
    """
    return {
        "test_file": config.TEST_FILE,
        "parameter_filename": config.parameter_filename(),
        "num_generations": config.NUM_GENERATIONS,
        "num_training_ball_poses": config.NUM_TRAINING_BALL_POSES,
        "simulation_time": config.SIMULATION_TIME,
        "cma_initial_std": config.CMA_INITIAL_STD,
        "cma_initial_mean": config.CMA_INITIAL_MEAN,
        "cma_bounds": list(config.CMA_BOUNDS),
        "cma_population_size": getattr(config, "CMA_POPULATION_SIZE", None),
        "num_simulators": config.NUM_SIMULATORS,
        "headless": config.HEADLESS,
        "ball_radius": config.BALL_RADIUS,
        "ball_mass": config.BALL_MASS,
        "ball_reached_distance": config.BALL_REACHED_DISTANCE,
    }


def _config_source() -> str:
    """
    Read the active config source text.

    :returns: Config Python source text.
    """
    with open(_config_path_abs(), "r", encoding="utf-8") as handle:
        return handle.read()


def _config_path_abs() -> str:
    """
    Get the absolute path to the active config file.

    :returns: Absolute config path.
    """
    return os.path.abspath(os.environ.get(CONFIG_PATH_ENV, config.__file__))


def _main_path() -> str:
    """
    Get the path to this main file as seen by the runner.

    :returns: Absolute main path.
    """
    return os.path.abspath(os.environ.get(MAIN_PATH_ENV, __file__))


if __name__ == "__main__":
    main()
