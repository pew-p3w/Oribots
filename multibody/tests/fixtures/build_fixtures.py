"""Rebuild the test fixtures in this folder (run by hand; the outputs are committed).

- ``trained_weights_seed.npy``: a seeded random 44-value trained-weights vector for
  the pentagon spider (12 CPG + 32 steering), so tests never need the real
  trained weights (those come from a UCT run and are needed only from the
  timing pilot on).
- ``stored_spider_8ball_config.txt``: the config source text a real single-robot
  ``spider_8ball`` snapshot stores (copied verbatim from the old repository's
  run). It imports ``pentagon_body``, a module that exists only in that
  repository, so it exercises the expression-comparison path of the
  trained-weights cross-check (R4.9).
- ``fake_single_robot_gen1.pkl``: a minimal single-robot snapshot dict holding
  that config source, ``algorithm = "ea"`` and two 44-value parameter fields. It
  pickles only builtins and numpy arrays, so it loads anywhere.

Usage, from ``Oribots/``::

    .venv.nosync/bin/python multibody/tests/fixtures/build_fixtures.py
"""

import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NUM_SPIDER_PARAMETERS = 44
SEED = 20261003


def main() -> None:
    """Write the fixtures."""
    rng = np.random.default_rng(SEED)
    weights = rng.uniform(-1.0, 1.0, size=NUM_SPIDER_PARAMETERS)
    other = rng.uniform(-1.0, 1.0, size=NUM_SPIDER_PARAMETERS)
    np.save(HERE / "trained_weights_seed.npy", weights)
    config_source = (HERE / "stored_spider_8ball_config.txt").read_text(encoding="utf-8")
    snapshot = {
        "algorithm": "ea",
        "version": 3,
        "snapshot_version": 1,
        "generation": 1,
        "completed_generations": 1,
        "best_parameters": weights.copy(),
        "best_ever_parameters": other,
        "config_source": config_source,
    }
    with open(HERE / "fake_single_robot_gen1.pkl", "wb") as handle:
        pickle.dump(snapshot, handle, protocol=4)


if __name__ == "__main__":
    main()
