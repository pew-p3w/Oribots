"""The training process: CMA-ES over the residual layer only (the Layer_Optimizer, R16-R18).

Launched as a child process by ``run_multibody.py``:

- a new run reads ``MULTIBODY_CONFIG_PATH`` and ``MULTIBODY_OUTPUT_DIR``;
- a resume gets the snapshot path as its one argument and works in the
  snapshot's folder (R17.8).

Per generation: ask -> score every candidate on every layout -> write the
generation's record rows -> tell CMA-ES (negated raw scores, no clipping) ->
write ``gen<N>.pkl`` atomically. A generation that does not complete leaves the
previous snapshot as the resume point.

Search vector: the Layer_Parameters only (W1 then W2), never the trained
weights. Initial mean: W2 = 0 and a seeded W1 (R16.2), so generation 0's mean is
exactly the trained controller.

Reproducibility: CMA-ES draws its samples from its own seeded
``numpy.random.RandomState`` (pickled inside the optimizer state), and numpy's
global random state is saved in every snapshot too, so a resumed run continues
bit for bit (R18.2). The CMA-ES seed is ``W1_INIT_SEED + 1``.

A snapshot is a plain dict of builtins, numpy arrays and the CMA-ES optimizer
state: it loads with only the pinned packages installed (R17.7).
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()

import cma  # noqa: E402

import arena  # noqa: E402
import config_loader  # noqa: E402
import match  # noqa: E402
from arena_evaluator import MatchEvaluator, MatchWorkerCrashed  # noqa: E402
from layered_brain import pack_layer_params  # noqa: E402
from match_recording import Recorder, new_record_names  # noqa: E402

SNAPSHOT_VERSION = 1
CONFIG_PATH_ENV = "MULTIBODY_CONFIG_PATH"
OUTPUT_DIR_ENV = "MULTIBODY_OUTPUT_DIR"

# Every item a multibody snapshot holds (R17.2); a file missing any is refused (R17.9).
SNAPSHOT_KEYS = (
    "algorithm",
    "snapshot_version",
    "generation",
    "completed_generations",
    "cma_state",
    "numpy_rng_state",
    "best_layer_params",
    "best_score",
    "best_ever_layer_params",
    "best_ever_score",
    "trained_weights",
    "base_config_source",
    "multibody_config_source",
    "layouts",
    "layout_seed",
    "w1_init",
    "w1_init_seed",
    "reference_layer",
    "opponents",
    "k",
    "hidden_size",
    "num_active_hinges",
    "record_paths",
    "settings_values",
)

EXIT_WORKER_CRASH = 3
EXIT_TOO_MANY_FAILED_MATCHES = 4
EXIT_NON_FINITE_SCORE = 5


class SnapshotRefused(Exception):
    """A file given as a multibody snapshot cannot be used (R17.9, R18.5)."""


def load_snapshot(path: Path) -> dict[str, Any]:
    """
    Load and check a multibody snapshot.

    :param path: The ``gen<N>.pkl``.
    :returns: The snapshot dict.
    :raises SnapshotRefused: Naming the file and the algorithm found or the missing items.
    """
    path = Path(path)
    if not path.is_file():
        raise SnapshotRefused(f"{path}: no such file")
    try:
        with open(path, "rb") as handle:
            snapshot = pickle.load(handle)
    except Exception as error:  # noqa: BLE001
        raise SnapshotRefused(f"{path}: cannot be loaded as a snapshot ({type(error).__name__}: {error})") from error
    if not isinstance(snapshot, dict):
        raise SnapshotRefused(f"{path}: not a snapshot (a {type(snapshot).__name__})")
    algorithm = snapshot.get("algorithm")
    if algorithm != config_loader.MULTIBODY_ALGORITHM:
        raise SnapshotRefused(
            f"{path}: not a multibody snapshot (algorithm {algorithm!r}, expected "
            f"{config_loader.MULTIBODY_ALGORITHM!r})"
        )
    missing = [key for key in SNAPSHOT_KEYS if key not in snapshot]
    if missing:
        raise SnapshotRefused(f"{path}: incomplete multibody snapshot, missing {', '.join(missing)}")
    return snapshot


def make_optimizer(settings: config_loader.ContestSettings, mean: np.ndarray) -> cma.CMAEvolutionStrategy:
    """
    A fresh CMA-ES over the Layer_Parameters (R16.1, R16.5, R16.6).

    :param settings: The contest settings.
    :param mean: The initial mean (W1 seeded, W2 = 0).
    :returns: The optimizer.
    """
    seed = settings.w1_init_seed + 1
    options = cma.CMAOptions()
    options.set("bounds", list(settings.cma_bounds))
    options.set("popsize", settings.cma_population_size)
    options.set("seed", seed)
    options.set("randn", np.random.RandomState(seed).randn)
    options.set("maxiter", float("inf"))
    options.set("verbose", -9)
    options.set("verb_disp", 0)
    options.set("verb_log", 0)  # never write outcmaes/ files into the working directory
    optimizer = cma.CMAEvolutionStrategy(list(mean), settings.cma_initial_std, options)
    # CMA-ES's constructor can reseed numpy's global generator from the clock. Nothing
    # here draws from it (samples come from the RandomState above), but it is saved in
    # every snapshot, so seed it deterministically: two fresh runs then match exactly.
    np.random.seed(seed)
    return optimizer


def restore_optimizer(cma_state: dict[str, Any]) -> cma.CMAEvolutionStrategy:
    """
    Rebuild CMA-ES from a stored state (the single-robot ``_restore_optimizer`` approach).

    :param cma_state: The optimizer's ``__dict__`` as stored.
    :returns: The optimizer.
    """
    optimizer = cma.CMAEvolutionStrategy.__new__(cma.CMAEvolutionStrategy)
    optimizer.__dict__.update(cma_state)
    return optimizer


def initial_mean(settings: config_loader.ContestSettings) -> tuple[np.ndarray, np.ndarray]:
    """
    The initial W1 values and the packed initial mean (W2 = 0) (R16.2).

    :returns: (w1 flat, mean).
    """
    w1 = config_loader.initial_w1(settings.hidden_size, settings.w1_init_std, settings.w1_init_seed)
    w2 = np.zeros((settings.num_active_hinges, settings.hidden_size))
    return w1, pack_layer_params(w1.reshape(settings.hidden_size, 4), w2)


def write_snapshot(output_dir: Path, generation: int, snapshot: dict[str, Any]) -> Path:
    """
    Write ``gen<N>.pkl`` atomically: a kill at any moment leaves either the old set of
    snapshots or the new one complete, never a partial ``gen<N>.pkl`` (R17.5).

    :returns: The snapshot path.
    """
    path = Path(output_dir) / f"gen{generation}.pkl"
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
        stream=sys.stderr,
        force=True,
    )


def main() -> None:
    """Run (or continue) a contest training run."""
    _setup_logging()
    if len(sys.argv) > 2:
        raise SystemExit("usage: layer_cmaes.py [gen<N>.pkl]")

    if len(sys.argv) == 2:
        snapshot_path = Path(sys.argv[1]).resolve()
        snapshot = load_snapshot(snapshot_path)
        settings = config_loader.settings_from_snapshot(snapshot)
        output_dir = snapshot_path.parent
        layouts = arena.layouts_from_plain(snapshot["layouts"])
        optimizer = restore_optimizer(snapshot["cma_state"])
        np.random.set_state(snapshot["numpy_rng_state"])
        completed = int(snapshot["completed_generations"])
        w1 = np.asarray(snapshot["w1_init"], dtype=np.float64)
        record_names = dict(snapshot["record_paths"])
        best_ever = (np.asarray(snapshot["best_ever_layer_params"]), float(snapshot["best_ever_score"]))
        recorder = Recorder(output_dir, record_names)
        recorder.trim_after(completed)
        logging.info(f"Resuming {output_dir} after generation {completed} of {settings.num_generations}.")
    else:
        config_path = Path(os.environ[CONFIG_PATH_ENV]).resolve()
        output_dir = Path(os.environ[OUTPUT_DIR_ENV]).resolve()
        settings = config_loader.load_contest(config_path)
        layouts = arena.sample_layouts(
            k=settings.k,
            num_layouts=settings.num_layouts,
            seed=settings.layout_seed,
            start_radius=settings.start_radius,
            min_arc_gap=settings.min_arc_gap,
            footprint=settings.footprint_radius,
            head_offset=settings.head_offset,
            possession_distance=settings.possession_distance,
        )
        w1, mean = initial_mean(settings)
        optimizer = make_optimizer(settings, mean)
        completed = 0
        record_names = new_record_names(config_path.stem)
        best_ever = None
        recorder = Recorder(output_dir, record_names)
        logging.info(f"New contest run {output_dir}: {config_path.name}, K={settings.k}, L={settings.hidden_size}, "
                     f"{settings.cma_population_size} candidates x {settings.num_layouts} layouts, "
                     f"{settings.simulation_time:g} s matches, {settings.worker_count} worker(s).")

    config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    structure, mapping = config_loader.body_wiring(settings.body)
    evaluator = MatchEvaluator(settings, structure, mapping, layouts)
    reference = settings.reference_layer if settings.reference_layer is not None else match.zero_output_layer(settings)

    if completed >= settings.num_generations:
        logging.info("The run is already complete.")
        return

    for generation in range(completed + 1, settings.num_generations + 1):
        logging.info(f"Generation {generation} / {settings.num_generations}")
        candidates = [np.asarray(x, dtype=np.float64) for x in optimizer.ask()]
        try:
            scores, records = evaluator.score_candidates(candidates, reference)
        except MatchWorkerCrashed as crash:
            logging.error(f"Generation {generation}: {crash}. Stopping; gen{generation - 1}.pkl stays the resume point.")
            sys.exit(EXIT_WORKER_CRASH)
        if records.failed_fraction > settings.max_failed_match_fraction:
            logging.error(
                f"Generation {generation}: {records.failed_count} of {len(records.matches)} matches "
                f"({records.failed_fraction:.3f}) failed the deterministic checks, more than "
                f"MAX_FAILED_MATCH_FRACTION = {settings.max_failed_match_fraction}. Stopping before "
                f"writing gen{generation}.pkl."
            )
            sys.exit(EXIT_TOO_MANY_FAILED_MATCHES)
        bad = [index for index, score in enumerate(scores) if not math.isfinite(score)]
        if bad:
            logging.error(f"Generation {generation}: candidate {bad[0]} has a non-finite score. Stopping.")
            sys.exit(EXIT_NON_FINITE_SCORE)

        recorder.write_generation(generation, candidates, scores, records, settings.failed_match_score)
        optimizer.tell(candidates, [-score for score in scores])

        best_index = int(np.argmax(scores))
        best = (candidates[best_index].copy(), float(scores[best_index]))
        if best_ever is None or best[1] > best_ever[1]:
            best_ever = best
        logging.info(
            f"Generation {generation}: best {best[1]:.6f}, mean {float(np.mean(scores)):.6f}, "
            f"best ever {best_ever[1]:.6f}, failed matches {records.failed_count}"
        )
        snapshot = {
            "algorithm": config_loader.MULTIBODY_ALGORITHM,
            "snapshot_version": SNAPSHOT_VERSION,
            "generation": generation,
            "completed_generations": generation,
            "cma_state": optimizer.__dict__.copy(),
            "numpy_rng_state": np.random.get_state(),
            "best_layer_params": best[0],
            "best_score": best[1],
            "best_ever_layer_params": best_ever[0],
            "best_ever_score": best_ever[1],
            "trained_weights": np.asarray(settings.trained_weights, dtype=np.float64).copy(),
            "base_config_source": settings.base_config_source,
            "multibody_config_source": settings.multibody_config_source,
            "layouts": arena.layouts_to_plain(layouts),
            "layout_seed": settings.layout_seed,
            "w1_init": np.asarray(w1, dtype=np.float64).copy(),
            "w1_init_seed": settings.w1_init_seed,
            "reference_layer": None if settings.reference_layer is None else settings.reference_layer.copy(),
            "opponents": settings.opponents,
            "k": settings.k,
            "hidden_size": settings.hidden_size,
            "num_active_hinges": settings.num_active_hinges,
            "record_paths": dict(record_names),
            "settings_values": config_loader.settings_values(settings),
        }
        path = write_snapshot(output_dir, generation, snapshot)
        logging.info(f"Saved {path.name}.")
    logging.info("Contest training complete.")


if __name__ == "__main__":
    main()
