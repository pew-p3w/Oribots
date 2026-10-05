"""Load and plot fitness curves from Oribots training runs.

A training run writes a `generations.csv` (via `run.py -o`, or the live CSV a
run writes as it goes) with one row per generation and these columns:

    num_of_generation, best_fitness, worst_fitness, best_parent_fitness,
    best_offspring_fitness, best_ever_fitness, best_robot_weights,
    worst_robot_weights

This module finds those CSVs under `output/`, loads them, and plots the fitness
curves. It is imported by `fitness_curves.ipynb`; keeping the logic here (rather
than in notebook cells) means it can be reused and tested.

Two points affect how a plot should be read:

- Paths are resolved from the repository root, found by walking up from this
  file, never from the current working directory. So the notebook plots the
  same runs no matter where Jupyter was started.

- `best_fitness` does not mean the same thing across algorithms. For the EA it
  is the recorded value, the mean clamped to `[0, 1]`. For CMA-ES it is the raw
  signed optimizer fitness (it can be negative). Do not compare the two axes
  directly without accounting for that; `best_ever_fitness` is the most
  comparable single curve within one algorithm.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

# The columns every run CSV carries, in order. Kept in sync with run.py's
# CSV_FIELDNAMES; the numeric ones are the plottable series.
NUMERIC_COLUMNS = [
    "best_fitness",
    "worst_fitness",
    "best_parent_fitness",
    "best_offspring_fitness",
    "best_ever_fitness",
]
GENERATION_COLUMN = "num_of_generation"


def repo_root() -> Path:
    """
    Find the Oribots repository root by walking up from this file.

    Robust to the notebook's working directory: it looks for the repo's marker
    files (`run.py` next to an `engine/` directory) rather than assuming a CWD.

    :returns: The repository root path.
    :raises RuntimeError: If the root cannot be located.
    """
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "run.py").is_file() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("Could not find the Oribots repo root above this file.")


def output_dir() -> Path:
    """
    The directory holding training runs.

    :returns: The `output/` directory path.
    """
    return repo_root() / "output"


def find_run_csvs(base: Path | None = None) -> dict[str, Path]:
    """
    Find one representative CSV per run folder under `output/`.

    A run folder may hold a `generations.csv` (from `run.py -o`) and/or a live
    `parameters_*_run_*.csv` written during training. `generations.csv` is
    preferred when present; otherwise the most recent live CSV is used.

    :param base: Directory to search. Defaults to `output/`.
    :returns: Mapping of run name (the folder name) to its CSV path.
    """
    base = base if base is not None else output_dir()
    runs: dict[str, Path] = {}
    if not base.is_dir():
        return runs
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        if folder.name == "analysis":
            continue
        generations = folder / "generations.csv"
        if generations.is_file():
            runs[folder.name] = generations
            continue
        live = sorted(folder.glob("*_run_*.csv"))
        if live:
            runs[folder.name] = live[-1]
    return runs


def load_run(csv_path: Path) -> dict[str, list[float]]:
    """
    Load a run CSV into columns of floats, keyed by column name.

    Blank cells (for example the parent/offspring columns of a CMA-ES run)
    become `float('nan')`, so a plot simply skips them. The weight columns are
    ignored here; this module plots fitness, not parameters.

    :param csv_path: Path to a run CSV.
    :returns: Mapping of column name to a list of values (NaN where blank).
    """
    columns: dict[str, list[float]] = {GENERATION_COLUMN: []}
    for name in NUMERIC_COLUMNS:
        columns[name] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            columns[GENERATION_COLUMN].append(float(row[GENERATION_COLUMN]))
            for name in NUMERIC_COLUMNS:
                value = row.get(name, "")
                columns[name].append(float(value) if value not in ("", None) else float("nan"))
    return columns


def _has_any_value(series: list[float]) -> bool:
    """
    Whether a series has at least one non-NaN value worth plotting.

    :param series: A column of values.
    :returns: True if any value is finite.
    """
    return any(value == value for value in series)  # NaN != NaN


def plot_run(csv_path: Path, run_name: str, axis: Any) -> None:
    """
    Plot one run's fitness curves onto a Matplotlib axis.

    :param csv_path: Path to the run CSV.
    :param run_name: Name to title the plot with.
    :param axis: A Matplotlib axis to draw on.
    """
    data = load_run(csv_path)
    generations = data[GENERATION_COLUMN]
    plotted = 0
    for name in NUMERIC_COLUMNS:
        if _has_any_value(data[name]):
            axis.plot(generations, data[name], label=name, marker=".", linewidth=1)
            plotted += 1
    axis.set_title(run_name)
    axis.set_xlabel(GENERATION_COLUMN)
    axis.set_ylabel("fitness")
    if plotted:
        axis.legend(fontsize="small")
    axis.grid(True, alpha=0.3)


def plot_all_runs(base: Path | None = None):
    """
    Plot every run found under `output/`, one subplot each.

    :param base: Directory to search. Defaults to `output/`.
    :returns: The Matplotlib figure, or None if no runs were found.
    """
    import matplotlib.pyplot as plt

    runs = find_run_csvs(base)
    if not runs:
        print(
            f"No run CSVs found under {output_dir()}.\n"
            "Train a run first, e.g.:\n"
            "    python run.py -r config/simple.py src/ea/main.py output/simple\n"
            "then re-run this notebook."
        )
        return None

    count = len(runs)
    figure, axes = plt.subplots(count, 1, figsize=(9, 4 * count), squeeze=False)
    for axis, (run_name, csv_path) in zip(axes[:, 0], runs.items()):
        plot_run(csv_path, run_name, axis)
    figure.tight_layout()
    return figure


def best_ever_comparison(base: Path | None = None):
    """
    Plot every run's best-ever fitness on one axis, for a cross-run comparison.

    Best-ever is the most comparable single curve within an algorithm. Across
    algorithms, mind the scale difference noted in the module docstring.

    :param base: Directory to search. Defaults to `output/`.
    :returns: The Matplotlib figure, or None if no runs were found.
    """
    import matplotlib.pyplot as plt

    runs = find_run_csvs(base)
    if not runs:
        print(f"No run CSVs found under {output_dir()}.")
        return None

    figure, axis = plt.subplots(figsize=(9, 5))
    plotted = 0
    for run_name, csv_path in runs.items():
        data = load_run(csv_path)
        if _has_any_value(data["best_ever_fitness"]):
            axis.plot(
                data[GENERATION_COLUMN],
                data["best_ever_fitness"],
                label=run_name,
                marker=".",
                linewidth=1,
            )
            plotted += 1
    axis.set_title("best-ever fitness across runs")
    axis.set_xlabel(GENERATION_COLUMN)
    axis.set_ylabel("best_ever_fitness")
    if plotted:
        axis.legend(fontsize="small")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    return figure
