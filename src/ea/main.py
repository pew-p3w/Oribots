"""The evolutionary algorithm: breed controllers that walk to the ball.

Each generation the whole population is replaced: a tournament picks a winner,
the winner is cloned twice, the clones may be crossed over, and one child is
kept and mutated. Because both parents are clones of the same winner, crossover
produces no recombination, so nearly all variation comes from mutation. This is
the intended design of the algorithm, not an oversight.

There is no elitism: `best_ever` is tracked only so the best controller found
can be saved and logged, never to protect it from being replaced.

Two fitness values exist for every individual. `Individual.fitness` is the
RECORDED value: the mean over the training balls, clamped to `[0, 1]`, so graphs
and analysis share one clear scale; every CSV and the snapshot fitness fields
show it. `Individual.selection_fitness` is what selection compares: the same
mean WITHOUT the clamp (`SELECT_ON_RAW_FITNESS`, default on), so individuals
above 1 (which reach the ball and earn the reach and speed bonuses) and below 0
are still told apart. With the clamp, most of a strong population ties at 1.0
and tournament selection can no longer tell them apart. A run keeps the rule it started with; a clamped run can be
switched to the unclamped rule when resumed, with
`REVOLVE2_SWITCH_TO_RAW_SELECTION=1`.

Every individual of every generation is recorded (`src/recording.py`), and a
resumed run keeps writing into the same files after trimming what the
interrupted run wrote past the resume point.

The shared evaluator returns raw signed fitness; this file clamps it for the
record (`_make_individuals`). `tests/test_ea.py` checks that with
`SELECT_ON_RAW_FITNESS = False` this EA reproduces its recorded reference run
exactly, and that with it on, selection compares the unclamped values.

Run through `run.py -r`, which sets the three `REVOLVE2_RUN_*` variables this
file reads.
"""

import os
import sys
from pathlib import Path

# Nothing below can be imported until the repository's own directories are on
# the path, and src/ holds the helper that puts them there, hence these two
# lines before `import paths`. See src/paths.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

_config_path = os.environ.get("REVOLVE2_RUN_CONFIG_PATH")
paths.install(Path(_config_path).parent if _config_path else None)

# This process may have been started directly rather than by run.py, and
# simulator workers re-import this module. Cached bytecode is treated as current
# whenever a source file's size and whole-second mtime are unchanged, so a
# config edited and rerun within a second could otherwise use its old values.
sys.dont_write_bytecode = True

import csv
import importlib.util
import json
import logging
import math
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import recording
from genotype import Genotype

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed


CHECKPOINT_VERSION = 3
GENERATION_SNAPSHOT_VERSION = 1
CONFIG_PATH_ENV = "REVOLVE2_RUN_CONFIG_PATH"
MAIN_PATH_ENV = "REVOLVE2_RUN_MAIN_PATH"
OUTPUT_DIR_ENV = "REVOLVE2_RUN_OUTPUT_DIR"
# Recorded in every snapshot and checkpoint so `run.py -c` knows which algorithm
# to resume with, instead of inferring it from which keys happen to be present.
ALGORITHM = "ea"

# Probability that a tournament winner's two clone genotypes are crossed over.
CROSSOVER_PROBABILITY = 0.80

# Whether selection compares the unclamped mean fitness (True) or the recorded,
# clamped one (False). A config may set
# SELECT_ON_RAW_FITNESS; a run continued from a checkpoint keeps the rule it started
# with (see main()). Set once config is loaded, below.
SELECT_ON_RAW_FITNESS = True

# Set to "1" when continuing a run that selects on the clamped fitness to switch it
# to the unclamped rule from the next generation on (one way only: a run already on
# the unclamped rule stays on it). The switch is stored, so later resumes keep the
# unclamped rule without the variable.
SWITCH_TO_RAW_SELECTION_ENV = "REVOLVE2_SWITCH_TO_RAW_SELECTION"
# The last generation produced under the clamped rule, for a run switched with the
# variable above; None for a run that never switched.
RAW_SELECTION_SWITCHED_AFTER_GENERATION: int | None = None


def _load_config_module():
    """
    Load the selected run config as the module named config.

    run.py can point to config files such as simple.py or spider.py. Loading the
    file explicitly avoids accidentally importing config.py from the same folder.

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
SELECT_ON_RAW_FITNESS = bool(getattr(config, "SELECT_ON_RAW_FITNESS", True))


class Individual:
    """A single member of the population."""

    genotype: Genotype
    fitness: float
    selection_fitness: float

    def __init__(
        self,
        genotype: Genotype,
        fitness: float,
        selection_fitness: float | None = None,
    ) -> None:
        """
        Initialize an individual.

        :param genotype: The evolved controller parameter vector.
        :param fitness: The recorded fitness, clamped to [0, 1]. Every CSV and
            snapshot field shows this value.
        :param selection_fitness: The value selection compares. Defaults to
            ``fitness`` (clamped selection).
        """
        self.genotype = genotype
        self.fitness = fitness
        self.selection_fitness = (
            fitness if selection_fitness is None else selection_fitness
        )


def _make_individuals(
    genotypes: list[Genotype], mean_fitnesses: list[float]
) -> list[Individual]:
    """
    Turn evaluated genotypes into individuals.

    The evaluator hands back the unclamped mean over the training balls. The
    recorded fitness is that mean clamped to [0, 1]; selection uses the unclamped mean unless
    ``SELECT_ON_RAW_FITNESS`` is off.

    :param genotypes: The evaluated genotypes.
    :param mean_fitnesses: Unclamped mean fitness of each genotype.
    :returns: One individual per genotype.
    """
    individuals = []
    for genotype, mean in zip(genotypes, mean_fitnesses):
        recorded = float(np.clip(mean, 0.0, 1.0))
        individuals.append(
            Individual(
                genotype,
                recorded,
                float(mean) if SELECT_ON_RAW_FITNESS else recorded,
            )
        )
    return individuals


class TournamentCloneReproducer:
    """Produces offspring from tournament-winning clone crossover and mutation."""

    _rng: np.random.Generator

    def __init__(self) -> None:
        """Initialize the reproducer."""
        self._rng = make_rng_time_seed()

    def reproduce(self, population: list[Individual]) -> list[Genotype]:
        """
        Create a full replacement population using tournament selection.

        :param population: Evaluated current population.
        :returns: New child genotypes of size config.POPULATION_SIZE.
        :raises ValueError: If the population is empty.
        """
        if len(population) == 0:
            raise ValueError("Population cannot be empty.")

        children = []
        while len(children) < config.POPULATION_SIZE:
            winner = population[
                _tournament(
                    self._rng,
                    [individual.selection_fitness for individual in population],
                    k=config.TOURNAMENT_SIZE,
                )
            ]
            clone1 = winner.genotype.copy()
            clone2 = winner.genotype.copy()
            if self._rng.random() < CROSSOVER_PROBABILITY:
                child1, child2 = Genotype.one_point_crossover(
                    clone1,
                    clone2,
                    self._rng,
                )
            else:
                child1, child2 = clone1, clone2
            child = child1 if self._rng.random() < 0.5 else child2
            children.append(
                child.mutate(
                    self._rng,
                    config.MUTATE_STD,
                    config.MUTATION_PROBABILITY,
                )
            )
        return children


def _tournament(rng: np.random.Generator, fitnesses: list[float], k: int) -> int:
    """
    Select the best individual from a random tournament.

    :param rng: Random number generator.
    :param fitnesses: Fitness values.
    :param k: Tournament size.
    :returns: Winning index.
    """
    participants = rng.choice(range(len(fitnesses)), size=k)
    return int(max(participants, key=lambda index: fitnesses[index]))


def main() -> None:
    """Run the EA optimization loop."""
    global SELECT_ON_RAW_FITNESS, RAW_SELECTION_SWITCHED_AFTER_GENERATION
    setup_logging()
    logging.info("Starting main.")
    logging.info("Importing evaluator and MuJoCo simulator dependencies.")
    from evaluator import Evaluator

    checkpoint_path = _checkpoint_path_from_args(sys.argv)
    if checkpoint_path is not None:
        logging.info(f"Checkpoint path: {checkpoint_path}")

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
        f"{config.parameter_filename().removesuffix('.npy')}_run_"
        f"{run_timestamp}.csv",
    )
    detailed_csv_path, weights_csv_path = recording.default_paths(
        output_dir,
        config.parameter_filename().removesuffix(".npy"),
        run_timestamp,
    )

    # The evaluator returns the unclamped mean; _make_individuals clamps it to [0, 1]
    # for recording and decides what selection compares.
    evaluator = Evaluator()
    num_params = evaluator.num_parameters
    logging.info(f"Controller has {num_params} parameters to optimize.")

    reproducer = TournamentCloneReproducer()
    logging.info(f"Crossover probability: {CROSSOVER_PROBABILITY:.2f}")
    logging.info(
        "Config selection rule: "
        + (
            "unclamped mean fitness."
            if SELECT_ON_RAW_FITNESS
            else "clamped [0, 1] fitness."
        )
        + " A resumed run may override this from its checkpoint (see below)."
    )

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = _load_checkpoint(checkpoint_path)
        _validate_checkpoint(checkpoint, num_params)
        rng = _rng_from_state(checkpoint["rng_state"])
        reproducer._rng.bit_generator.state = checkpoint["reproducer_rng_state"]
        csv_path = checkpoint["csv_path"]
        save_path = checkpoint["save_path"]
        # Keep appending to the files the run started with, never mint new ones.
        saved_detailed_csv_path = checkpoint.get("detailed_csv_path")
        saved_weights_csv_path = checkpoint.get("weights_csv_path")
        if saved_detailed_csv_path and saved_weights_csv_path:
            detailed_csv_path = saved_detailed_csv_path
            weights_csv_path = saved_weights_csv_path
        training_ball_poses = checkpoint["training_ball_poses"]
        # A run keeps the selection rule it started with. A checkpoint without
        # this field comes from a run that selected on the clamped fitness, so
        # it continues that way (changing rule mid-run would confound
        # the analysis of the run).
        SELECT_ON_RAW_FITNESS = bool(
            checkpoint.get("selection_on_raw_fitness", False)
        )
        RAW_SELECTION_SWITCHED_AFTER_GENERATION = checkpoint.get(
            "raw_selection_switched_after_generation"
        )
        # The one allowed change: clamped -> unclamped, on request. Selection first
        # reads the parents' fitness after they are scored again at the top of the
        # next generation, so no clamped value from the checkpoint reaches it.
        if os.environ.get(SWITCH_TO_RAW_SELECTION_ENV) == "1":
            if SELECT_ON_RAW_FITNESS:
                logging.info(
                    f"{SWITCH_TO_RAW_SELECTION_ENV}=1 ignored: this run already "
                    "selects on the unclamped fitness."
                )
            else:
                SELECT_ON_RAW_FITNESS = True
                RAW_SELECTION_SWITCHED_AFTER_GENERATION = int(
                    checkpoint["completed_generations"]
                )
                logging.info(
                    f"{SWITCH_TO_RAW_SELECTION_ENV}=1: switching this run from the "
                    "clamped to the unclamped selection rule. Generations up to "
                    f"{RAW_SELECTION_SWITCHED_AFTER_GENERATION} were selected on "
                    "the clamped fitness, later ones on the unclamped fitness."
                )
        logging.info(
            "Selection compares "
            + (
                "the unclamped mean fitness."
                if SELECT_ON_RAW_FITNESS
                else "the clamped [0, 1] fitness (rule of the run being continued)."
            )
        )
        population = _population_from_checkpoint(checkpoint)
        best_ever = Individual(
            Genotype(np.asarray(checkpoint["best_ever_parameters"])),
            float(checkpoint["best_ever_fitness"]),
            _stored_selection_fitness(
                checkpoint.get("best_ever_selection_fitness"),
                float(checkpoint["best_ever_fitness"]),
            ),
        )
        completed_generations = int(checkpoint["completed_generations"])
        # Drop rows the interrupted run already wrote past the resume point, so
        # nothing is written (and recorded) twice.
        recording.trim_after_generation(
            csv_path,
            completed_generations,
            lambda row: int(row["num_of_generation"]),
        )
        recording.trim_recording_after_generation(
            detailed_csv_path, weights_csv_path, completed_generations
        )
        if not (saved_detailed_csv_path and saved_weights_csv_path):
            logging.warning(
                "This snapshot was written by a run that did not record "
                "individuals. Recording starts at generation "
                f"{completed_generations + 1}, into NEW files (generations up to "
                f"{completed_generations} cannot be recovered): "
                f"{detailed_csv_path} and {weights_csv_path}"
            )
        logging.info(
            f"Resuming from checkpoint after generation {completed_generations}."
        )
        logging.info(f"Continuing CSV: {csv_path}")
        logging.info(f"Continuing best weights file: {save_path}")
        _log_training_ball_poses(training_ball_poses)
    else:
        rng = make_rng_time_seed()
        completed_generations = 0
        training_ball_poses = evaluator.sample_ball_poses(
            config.NUM_TRAINING_BALL_POSES
        )
        logging.info(
            f"Training on {len(training_ball_poses)} fixed random ball positions."
        )
        _log_training_ball_poses(training_ball_poses)
        logging.info(f"Run CSV will be saved to: {csv_path}")

        logging.info("Generating initial population.")
        initial_genotypes = [
            Genotype.random(num_params, rng)
            for _ in range(config.POPULATION_SIZE)
        ]

        logging.info("Evaluating initial population.")
        (
            initial_fitnesses,
            initial_trial_fitnesses,
            initial_trial_movements,
        ) = evaluator.evaluate_on_ball_poses_full(
            [genotype.parameters for genotype in initial_genotypes],
            training_ball_poses,
        )

        population = _make_individuals(initial_genotypes, initial_fitnesses)
        best_ever = max(population, key=lambda ind: ind.selection_fitness)
        np.save(save_path, best_ever.genotype.parameters)
        _write_generation_csv_row(
            csv_path=csv_path,
            generation_index=0,
            population=population,
            best_parent_fitness=None,
            best_offspring_fitness=None,
            best_ever_fitness=best_ever.fitness,
            mode="w",
        )
        _record_generation(
            detailed_csv_path=detailed_csv_path,
            weights_csv_path=weights_csv_path,
            generation_index=0,
            population=population,
            trial_fitnesses=initial_trial_fitnesses,
            trial_movements=initial_trial_movements,
            training_ball_poses=training_ball_poses,
            mode="w",
        )
        logging.info(f"Initial best normalized fitness: {best_ever.fitness:.4f}")
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                completed_generations=completed_generations,
                population=population,
                best_ever=best_ever,
                training_ball_poses=training_ball_poses,
                rng=rng,
                reproducer=reproducer,
                csv_path=csv_path,
                save_path=save_path,
                num_params=num_params,
                detailed_csv_path=detailed_csv_path,
                weights_csv_path=weights_csv_path,
            )

    logging.info("Starting optimization.")
    for generation in range(completed_generations, config.NUM_GENERATIONS):
        logging.info(f"Generation {generation + 1} / {config.NUM_GENERATIONS}")

        parent_genotypes = [individual.genotype for individual in population]
        population = _make_individuals(
            parent_genotypes,
            evaluator.evaluate_on_ball_poses(
                [genotype.parameters for genotype in parent_genotypes],
                training_ball_poses,
            ),
        )
        # The recorded (clamped) fitness of each parent.
        parent_fitnesses = [individual.fitness for individual in population]

        offspring_genotypes = reproducer.reproduce(population)
        (
            offspring_means,
            offspring_trial_fitnesses,
            offspring_trial_movements,
        ) = evaluator.evaluate_on_ball_poses_full(
            [genotype.parameters for genotype in offspring_genotypes],
            training_ball_poses,
        )
        offspring_population = _make_individuals(offspring_genotypes, offspring_means)
        # The recorded (clamped) fitness of each child.
        offspring_fitnesses = [ind.fitness for ind in offspring_population]

        logging.info(f"Best parent normalized fitness: {max(parent_fitnesses):.4f}")
        logging.info(
            f"Best offspring normalized fitness: {max(offspring_fitnesses):.4f}"
        )
        if SELECT_ON_RAW_FITNESS:
            logging.info(
                "Best offspring unclamped fitness (used for selection): "
                f"{max(ind.selection_fitness for ind in offspring_population):.4f}"
            )

        population = offspring_population
        generation_best = max(population, key=lambda ind: ind.selection_fitness)
        if generation_best.selection_fitness > best_ever.selection_fitness:
            best_ever = generation_best
            np.save(save_path, best_ever.genotype.parameters)
            logging.info(
                f"New best-ever normalized fitness: {best_ever.fitness:.4f}"
                f" (unclamped {best_ever.selection_fitness:.4f})"
            )
        _record_generation(
            detailed_csv_path=detailed_csv_path,
            weights_csv_path=weights_csv_path,
            generation_index=generation + 1,
            population=population,
            trial_fitnesses=offspring_trial_fitnesses,
            trial_movements=offspring_trial_movements,
            training_ball_poses=training_ball_poses,
            mode="a",
        )
        _write_generation_csv_row(
            csv_path=csv_path,
            generation_index=generation + 1,
            population=population,
            best_parent_fitness=max(parent_fitnesses),
            best_offspring_fitness=max(offspring_fitnesses),
            best_ever_fitness=best_ever.fitness,
            mode="a",
        )
        if generation_output_dir is not None:
            _save_generation_snapshot(
                output_dir=output_dir,
                generation_index=generation + 1,
                population=population,
                training_ball_poses=training_ball_poses,
                rng=rng,
                reproducer=reproducer,
                best_parent_fitness=max(parent_fitnesses),
                best_offspring_fitness=max(offspring_fitnesses),
                best_ever=best_ever,
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
                population=population,
                best_ever=best_ever,
                training_ball_poses=training_ball_poses,
                rng=rng,
                reproducer=reproducer,
                csv_path=csv_path,
                save_path=save_path,
                num_params=num_params,
                detailed_csv_path=detailed_csv_path,
                weights_csv_path=weights_csv_path,
            )

    logging.info("Optimization complete.")
    logging.info(f"Best normalized fitness: {best_ever.fitness:.4f}")
    logging.info(f"Best controller parameters: {best_ever.genotype.parameters}")
    np.save(save_path, best_ever.genotype.parameters)
    logging.info(f"Best weights saved to: {save_path}")
    logging.info(f"Run CSV saved to: {csv_path}")


def _checkpoint_path_from_args(argv: list[str]) -> str | None:
    """
    Get optional checkpoint path from command-line arguments.

    :param argv: Command-line arguments.
    :returns: Absolute checkpoint path or None.
    :raises SystemExit: If too many arguments are passed.
    """
    if len(argv) > 2:
        raise SystemExit(
            "Usage: main.py [checkpoint.pkl]"
        )
    if len(argv) == 1:
        return None
    return os.path.abspath(argv[1])


def _save_checkpoint(
    checkpoint_path: str,
    completed_generations: int,
    population: list[Individual],
    best_ever: Individual,
    training_ball_poses: list[Any],
    rng: np.random.Generator,
    reproducer: TournamentCloneReproducer,
    csv_path: str,
    save_path: str,
    num_params: int,
    detailed_csv_path: str,
    weights_csv_path: str,
) -> None:
    """
    Save all state required to resume optimization later.

    :param checkpoint_path: Path to checkpoint file.
    :param completed_generations: Number of completed EA generations.
    :param population: Current generation population.
    :param best_ever: Best individual found so far.
    :param training_ball_poses: Fixed training ball poses.
    :param rng: Main random number generator.
    :param reproducer: Reproducer with RNG state.
    :param csv_path: Path to run CSV.
    :param save_path: Path to best-ever npy file.
    :param num_params: Expected number of controller parameters.
    """
    checkpoint = {
        "algorithm": ALGORITHM,
        "version": CHECKPOINT_VERSION,
        "completed_generations": completed_generations,
        "population_parameters": [
            individual.genotype.parameters.copy()
            for individual in population
        ],
        "population_fitnesses": [
            float(individual.fitness)
            for individual in population
        ],
        "selection_on_raw_fitness": SELECT_ON_RAW_FITNESS,
        "raw_selection_switched_after_generation": (
            RAW_SELECTION_SWITCHED_AFTER_GENERATION
        ),
        "population_selection_fitnesses": [
            float(individual.selection_fitness)
            for individual in population
        ],
        "best_ever_selection_fitness": float(best_ever.selection_fitness),
        "best_ever_parameters": best_ever.genotype.parameters.copy(),
        "best_ever_fitness": float(best_ever.fitness),
        "training_ball_poses": training_ball_poses,
        "rng_state": rng.bit_generator.state,
        "reproducer_rng_state": reproducer._rng.bit_generator.state,
        "csv_path": csv_path,
        "save_path": save_path,
        "num_params": num_params,
        "detailed_csv_path": detailed_csv_path,
        "weights_csv_path": weights_csv_path,
        "config": {
            "test_file": config.TEST_FILE,
            "population_size": config.POPULATION_SIZE,
            "tournament_size": config.TOURNAMENT_SIZE,
            "num_generations": config.NUM_GENERATIONS,
            "num_training_ball_poses": config.NUM_TRAINING_BALL_POSES,
            "min_training_ball_distance_fraction": getattr(
                config,
                "MIN_TRAINING_BALL_DISTANCE_FRACTION",
                0.0,
            ),
            "simulation_time": config.SIMULATION_TIME,
            "mutate_std": config.MUTATE_STD,
            "mutation_probability": config.MUTATION_PROBABILITY,
        },
    }
    temp_path = f"{checkpoint_path}.tmp"
    with open(temp_path, "wb") as checkpoint_file:
        pickle.dump(checkpoint, checkpoint_file)
    os.replace(temp_path, checkpoint_path)
    logging.info(
        f"Checkpoint saved after generation {completed_generations}: "
        f"{checkpoint_path}"
    )


def _save_generation_snapshot(
    output_dir: str,
    generation_index: int,
    population: list[Individual],
    training_ball_poses: list[Any],
    rng: np.random.Generator,
    reproducer: TournamentCloneReproducer,
    best_parent_fitness: float,
    best_offspring_fitness: float,
    best_ever: Individual,
    save_path: str,
    csv_path: str,
    num_params: int,
    detailed_csv_path: str,
    weights_csv_path: str,
) -> None:
    """
    Save the current generation's best robot as a testable snapshot.

    The snapshot also carries the recording settings. ``run.py -c`` resumes
    from a snapshot, so without them a resumed run would silently stop
    recording (or start new, orphaned files).

    :param output_dir: Folder where generation snapshots are written.
    :param generation_index: One-based generation number.
    :param population: Current generation population.
    :param training_ball_poses: Fixed training ball poses.
    :param rng: Main random number generator.
    :param reproducer: Reproducer with RNG state.
    :param best_parent_fitness: Best fitness before replacement.
    :param best_offspring_fitness: Best fitness among children.
    :param best_ever: Best individual found over the whole run so far.
    :param save_path: Path to the best-ever npy file.
    :param csv_path: Path to the live training CSV.
    :param num_params: Number of controller parameters.
    :param detailed_csv_path: Per-trial performance CSV path.
    :param weights_csv_path: Weights CSV path.
    """
    best = max(population, key=lambda ind: ind.selection_fitness)
    worst = min(population, key=lambda ind: ind.selection_fitness)
    snapshot_path = os.path.join(output_dir, f"gen{generation_index}.pkl")
    module_dir = os.path.dirname(_main_path())
    snapshot = {
        "version": CHECKPOINT_VERSION,
        "snapshot_version": GENERATION_SNAPSHOT_VERSION,
        "algorithm": ALGORITHM,
        "detailed_csv_path": detailed_csv_path,
        "weights_csv_path": weights_csv_path,
        "completed_generations": generation_index,
        "generation": generation_index,
        "parameters": best.genotype.parameters.copy(),
        "fitness": float(best.fitness),
        "population_parameters": [
            individual.genotype.parameters.copy()
            for individual in population
        ],
        "population_fitnesses": [
            float(individual.fitness)
            for individual in population
        ],
        "selection_on_raw_fitness": SELECT_ON_RAW_FITNESS,
        "raw_selection_switched_after_generation": (
            RAW_SELECTION_SWITCHED_AFTER_GENERATION
        ),
        "population_selection_fitnesses": [
            float(individual.selection_fitness)
            for individual in population
        ],
        "best_ever_selection_fitness": float(best_ever.selection_fitness),
        "training_ball_poses": training_ball_poses,
        "rng_state": rng.bit_generator.state,
        "reproducer_rng_state": reproducer._rng.bit_generator.state,
        "best_parameters": best.genotype.parameters.copy(),
        "best_fitness": float(best.fitness),
        "worst_parameters": worst.genotype.parameters.copy(),
        "worst_fitness": float(worst.fitness),
        "best_parent_fitness": float(best_parent_fitness),
        "best_offspring_fitness": float(best_offspring_fitness),
        "best_ever_parameters": best_ever.genotype.parameters.copy(),
        "best_ever_fitness": float(best_ever.fitness),
        "num_params": num_params,
        "csv_path": csv_path,
        "save_path": save_path,
        "config": _config_values(),
        "config_source": _config_source(),
        "code_paths": {
            "main_path": _main_path(),
            "module_dir": module_dir,
            "config_path": _config_path(),
            "ball_aware_brain_path": os.path.join(
                module_dir, "ball_aware_brain.py"
            ),
            "evaluator_path": os.path.join(module_dir, "evaluator.py"),
            "genotype_path": os.path.join(module_dir, "genotype.py"),
        },
        "artifacts": {
            "best_weights_path": save_path,
            "training_csv_path": csv_path,
            "output_dir": output_dir,
        },
        "csv_row": _generation_csv_row(
            generation_index=generation_index,
            population=population,
            best_parent_fitness=best_parent_fitness,
            best_offspring_fitness=best_offspring_fitness,
            best_ever_fitness=best_ever.fitness,
        ),
    }
    temp_path = f"{snapshot_path}.tmp"
    with open(temp_path, "wb") as snapshot_file:
        pickle.dump(snapshot, snapshot_file)
    os.replace(temp_path, snapshot_path)
    logging.info(f"Generation snapshot saved to: {snapshot_path}")


def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    :param checkpoint_path: Checkpoint path.
    :returns: Loaded checkpoint dictionary.
    """
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, "rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    return checkpoint


def _validate_checkpoint(checkpoint: dict[str, Any], num_params: int) -> None:
    """
    Validate that a checkpoint matches the current controller setup.

    :param checkpoint: Loaded checkpoint dictionary.
    :param num_params: Current expected number of parameters.
    :raises ValueError: If the checkpoint is incompatible.
    """
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version: {checkpoint.get('version')}"
        )
    if int(checkpoint["num_params"]) != num_params:
        raise ValueError(
            f"Checkpoint has {checkpoint['num_params']} parameters, "
            f"but current setup expects {num_params}."
        )
    if len(checkpoint["population_parameters"]) != config.POPULATION_SIZE:
        raise ValueError(
            f"Checkpoint population size is "
            f"{len(checkpoint['population_parameters'])}, but config expects "
            f"{config.POPULATION_SIZE}."
        )


def _population_from_checkpoint(checkpoint: dict[str, Any]) -> list[Individual]:
    """
    Recreate population objects from a checkpoint.

    :param checkpoint: Loaded checkpoint dictionary.
    :returns: Population as Individual objects.
    """
    stored_selection = checkpoint.get("population_selection_fitnesses")
    if stored_selection is None:
        stored_selection = [None] * len(checkpoint["population_parameters"])
    return [
        Individual(
            Genotype(np.asarray(parameters)),
            float(fitness),
            _stored_selection_fitness(selection, float(fitness)),
        )
        for parameters, fitness, selection in zip(
            checkpoint["population_parameters"],
            checkpoint["population_fitnesses"],
            stored_selection,
        )
    ]


def _stored_selection_fitness(stored: float | None, recorded: float) -> float:
    """
    Pick the selection fitness to restore from a checkpoint.

    A checkpoint without this field stores none, and a run that selects on the
    clamped fitness ignores what is stored: both use the recorded value.

    :param stored: The stored unclamped fitness, if any.
    :param recorded: The stored recorded (clamped) fitness.
    :returns: The selection fitness to use.
    """
    if stored is None or not SELECT_ON_RAW_FITNESS:
        return recorded
    return float(stored)


def _rng_from_state(state: dict[str, Any]) -> np.random.Generator:
    """
    Recreate a numpy random generator from a bit-generator state.

    :param state: Bit-generator state.
    :returns: Restored random generator.
    """
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng


def _log_training_ball_poses(training_ball_poses: list[Any]) -> None:
    """
    Log fixed training ball poses and their starting distances.

    :param training_ball_poses: Fixed training ball poses.
    """
    for index, ball_pose in enumerate(training_ball_poses):
        distance = _initial_robot_to_ball_distance(ball_pose)
        logging.info(
            f"Training ball {index}: "
            f"x={ball_pose.position.x:.4f}, y={ball_pose.position.y:.4f}, "
            f"initial robot-to-ball distance={distance:.4f} m"
        )


def _write_generation_csv_row(
    csv_path: str,
    generation_index: int,
    population: list[Individual],
    best_parent_fitness: float | None,
    best_offspring_fitness: float | None,
    best_ever_fitness: float,
    mode: str,
) -> None:
    """
    Write one generation's best and worst controller weights to the run CSV.

    :param csv_path: Path to the run CSV.
    :param generation_index: Generation index. Initial population is 0.
    :param population: Current population for this generation.
    :param best_parent_fitness: Best normalized fitness in the old generation.
    :param best_offspring_fitness: Best normalized fitness among children.
    :param best_ever_fitness: Best normalized fitness seen in the whole run so far.
    :param mode: File mode to use.
    """
    fieldnames = [
        "num_of_generation",
        "best_fitness",
        "worst_fitness",
        "best_parent_fitness",
        "best_offspring_fitness",
        "best_ever_fitness",
        "best_robot_weights",
        "worst_robot_weights",
    ]
    with open(csv_path, mode, encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if recording.needs_header(mode, csv_file):
            writer.writeheader()
        writer.writerow(
            _generation_csv_row(
                generation_index=generation_index,
                population=population,
                best_parent_fitness=best_parent_fitness,
                best_offspring_fitness=best_offspring_fitness,
                best_ever_fitness=best_ever_fitness,
            )
        )


def _generation_csv_row(
    generation_index: int,
    population: list[Individual],
    best_parent_fitness: float | None,
    best_offspring_fitness: float | None,
    best_ever_fitness: float,
) -> dict[str, str | int]:
    """
    Create a CSV-compatible row for one generation.

    :param generation_index: Generation index. Initial population is 0.
    :param population: Current generation population.
    :param best_parent_fitness: Best normalized fitness in the old generation.
    :param best_offspring_fitness: Best normalized fitness among children.
    :param best_ever_fitness: Best normalized fitness seen in the whole run so far.
    :returns: CSV row dictionary.
    """
    best = max(population, key=lambda ind: ind.selection_fitness)
    worst = min(population, key=lambda ind: ind.selection_fitness)
    return {
        "num_of_generation": generation_index,
        "best_fitness": _format_optional_float(best.fitness),
        "worst_fitness": _format_optional_float(worst.fitness),
        "best_parent_fitness": _format_optional_float(best_parent_fitness),
        "best_offspring_fitness": _format_optional_float(best_offspring_fitness),
        "best_ever_fitness": _format_optional_float(best_ever_fitness),
        "best_robot_weights": _serialize_parameters(best.genotype.parameters),
        "worst_robot_weights": _serialize_parameters(worst.genotype.parameters),
    }


def _record_generation(
    detailed_csv_path: str,
    weights_csv_path: str,
    generation_index: int,
    population: list[Individual],
    trial_fitnesses: list[list[float]],
    trial_movements: list[list[dict[str, float]]],
    training_ball_poses: list[Any],
    mode: str,
) -> None:
    """
    Record every individual of one generation (see recording.py).

    The overall fitness written is each individual's recorded fitness (the
    mean clamped to [0, 1] in ``_make_individuals``), so nothing is recomputed
    here.

    :param detailed_csv_path: Per-trial performance CSV path.
    :param weights_csv_path: Weights CSV path.
    :param generation_index: Generation the population was evaluated in.
    :param population: Evaluated population.
    :param trial_fitnesses: Raw score, indexed ``[trial][individual]``.
    :param trial_movements: How each trial moved relative to the robot's heading,
        indexed ``[trial][individual]``.
    :param training_ball_poses: Fixed training ball poses.
    :param mode: File mode: "w" for a new run, "a" to append.
    """
    recording.write_generation(
        performance_csv_path=detailed_csv_path,
        weights_csv_path=weights_csv_path,
        generation_index=generation_index,
        parameters=[individual.genotype.parameters for individual in population],
        overall_fitnesses=[individual.fitness for individual in population],
        trial_fitnesses=trial_fitnesses,
        training_ball_poses=training_ball_poses,
        mode=mode,
        trial_movements=trial_movements,
    )


_serialize_parameters = recording.serialize_parameters
_format_optional_float = recording.format_float


def _initial_robot_to_ball_distance(ball_pose) -> float:
    """
    Calculate initial xy-plane distance from the default robot start to the ball.

    :param ball_pose: Ball pose.
    :returns: Distance from robot start at xy=(0, 0) to the ball.
    """
    return math.sqrt(ball_pose.position.x**2 + ball_pose.position.y**2)


def _config_values() -> dict[str, Any]:
    """
    Store the configuration values needed to describe this run.

    :returns: Serializable configuration summary.
    """
    return {
        "test_file": config.TEST_FILE,
        "parameter_filename": config.parameter_filename(),
        "population_size": config.POPULATION_SIZE,
        "tournament_size": config.TOURNAMENT_SIZE,
        "num_generations": config.NUM_GENERATIONS,
        "num_training_ball_poses": config.NUM_TRAINING_BALL_POSES,
        "min_training_ball_distance_fraction": getattr(
            config,
            "MIN_TRAINING_BALL_DISTANCE_FRACTION",
            0.0,
        ),
        "no_progress_penalty": getattr(config, "NO_PROGRESS_PENALTY", 0.01),
        "no_progress_epsilon": getattr(config, "NO_PROGRESS_EPSILON", 1e-6),
        "fitness_progress_weight": getattr(config, "FITNESS_PROGRESS_WEIGHT", 1.0),
        "fitness_reached_bonus_weight": getattr(
            config,
            "FITNESS_REACHED_BONUS_WEIGHT",
            0.20,
        ),
        "fitness_time_to_reach_weight": getattr(
            config,
            "FITNESS_TIME_TO_REACH_WEIGHT",
            0.10,
        ),
        "fitness_alignment_weight": getattr(config, "FITNESS_ALIGNMENT_WEIGHT", 0.05),
        "simulation_time": config.SIMULATION_TIME,
        "mutate_std": config.MUTATE_STD,
        "mutation_probability": config.MUTATION_PROBABILITY,
        "num_simulators": config.NUM_SIMULATORS,
        "headless": config.HEADLESS,
        "ball_radius": config.BALL_RADIUS,
        "ball_mass": config.BALL_MASS,
        "ball_reached_distance": config.BALL_REACHED_DISTANCE,
    }


def _config_source() -> str:
    """
    Read the config source used for this run.

    :returns: Config Python source text.
    """
    config_path = _config_path()
    with open(config_path, "r", encoding="utf-8") as config_file:
        return config_file.read()


def _config_path() -> str:
    """
    Get the path to the active config file.

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
