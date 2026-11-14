"""Compare benchmark results with a baseline: speed and whether results changed.

For each result it prints the speedup over the baseline and how far the
fitnesses moved:

- ``max |diff|``: the largest change in any single trial's fitness. Zero (or
  about 1e-12) means the change kept the physics exactly.
- ``rank corr``: Spearman correlation of the controllers' mean fitness. This is
  what matters to CMA-ES and the EA, which only use the ranking. 1.0 means the
  same order; above about 0.9 means a physics change still ranks controllers
  nearly the same.
- ``top-k kept``: how many of the baseline's best k controllers are still in the
  best k.

Usage:

    .venv/Scripts/python bench/compare.py bench/results/baseline.json bench/results/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation, with average ranks for ties.

    :param a: First values.
    :param b: Second values.
    :returns: Correlation in [-1, 1] (nan if either side is constant).
    """
    from scipy.stats import spearmanr

    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def compare(baseline: dict, other: dict, top_k: int = 5) -> dict:
    """
    Compare one result with the baseline.

    :param baseline: The baseline result.
    :param other: The result to compare.
    :param top_k: How many of the best controllers to track.
    :returns: Speedup and agreement figures.
    :raises ValueError: If the two did not simulate the same trials.
    """
    for key in ("config", "candidates", "balls", "seed", "sim_time"):
        if baseline[key] != other[key]:
            raise ValueError(
                f"{other['label']} differs from {baseline['label']} in {key} "
                f"({other[key]!r} vs {baseline[key]!r}); they did not simulate the same trials"
            )
    base_trials = np.asarray(baseline["trial_fitness"], dtype=float)
    other_trials = np.asarray(other["trial_fitness"], dtype=float)
    base_mean = np.asarray(baseline["mean_fitness"], dtype=float)
    other_mean = np.asarray(other["mean_fitness"], dtype=float)
    k = min(top_k, len(base_mean))
    base_top = set(np.argsort(-base_mean)[:k].tolist())
    other_top = set(np.argsort(-other_mean)[:k].tolist())
    return {
        "label": other["label"],
        "speedup": baseline["wall_seconds"] / other["wall_seconds"],
        "seconds_per_trial": other["seconds_per_trial"],
        "max_abs_diff": float(np.max(np.abs(base_trials - other_trials))),
        "rank_corr": spearman(base_mean, other_mean),
        "top_k_kept": len(base_top & other_top),
        "top_k": k,
    }


def main(argv: list[str] | None = None) -> list[dict]:
    """
    Print the comparison table.

    :param argv: The baseline path, then the results to compare.
    :returns: The comparison rows.
    :raises SystemExit: Without at least two results.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit("usage: compare.py BASELINE.json RESULT.json [RESULT.json ...]")
    baseline = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    rows = []
    for path in argv[1:]:
        other = json.loads(Path(path).read_text(encoding="utf-8"))
        if other["label"] == baseline["label"]:
            continue
        rows.append(compare(baseline, other))

    print(
        f"baseline {baseline['label']}: {baseline['trials']} trials, "
        f"{baseline['seconds_per_trial']:.3f} s/trial, {baseline['workers']} worker(s)"
    )
    print(f"{'result':<28}{'speedup':>9}{'s/trial':>10}{'max |diff|':>13}{'rank corr':>11}{'top-k kept':>12}")
    for row in rows:
        print(
            f"{row['label']:<28}{row['speedup']:>8.2f}x{row['seconds_per_trial']:>10.3f}"
            f"{row['max_abs_diff']:>13.2e}{row['rank_corr']:>11.3f}"
            f"{row['top_k_kept']:>8}/{row['top_k']}"
        )
    return rows


if __name__ == "__main__":
    main()
