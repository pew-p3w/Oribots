"""Record every candidate and every robot of every match (the Recorder, R19).

Two CSV files per run, in the run's output folder:

- ``candidates_<stem>_run_<timestamp>.csv``: one row per candidate per generation:
  ``generation, candidate_index, score, layer_params``;
- ``per_robot_<stem>_run_<timestamp>.csv``: one row per candidate, layout and
  slot: ``generation, candidate_index, layout_index, slot, role, d0,
  final_distance, min_distance, time_bonus, possession_share, opposed_share,
  first_reach_bonus, most_possession_bonus, score, failed``.

``generation`` is N of the generation's snapshot ``gen<N>.pkl``. A generation's
candidate rows are ordered by ascending score, ties by candidate index (as the
single-robot recorder), and its per-robot rows follow the same candidate order,
then layout, then slot. ``time_bonus`` is T_i in [-1, 1]. A failed match's rows
carry ``failed = True``, the FAILED_MATCH_SCORE as ``score`` and ``nan`` for the
terms it never computed. Floats are written with 17 significant digits, so they
read back to the identical 64-bit value (R19.7). Every row of a generation is
written and flushed to disk before that generation's snapshot (R19.6).
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any

import contest_paths

contest_paths.install()

import recording  # noqa: E402  (single-robot helpers: float format, header, trim)

CANDIDATE_COLUMNS = ["generation", "candidate_index", "score", "layer_params"]
PER_ROBOT_COLUMNS = [
    "generation",
    "candidate_index",
    "layout_index",
    "slot",
    "role",
    "d0",
    "final_distance",
    "min_distance",
    "time_bonus",
    "possession_share",
    "opposed_share",
    "first_reach_bonus",
    "most_possession_bonus",
    "score",
    "failed",
]
METRIC_COLUMNS = PER_ROBOT_COLUMNS[5:14]
CANDIDATE_PREFIX = "candidates_"
PER_ROBOT_PREFIX = "per_robot_"

# Names that mark a folder as already holding a run (R19.5).
RUN_FILE_PATTERNS = (
    "gen*.pkl",
    "weights.csv",
    "individual_performance_*.csv",
    f"{CANDIDATE_PREFIX}*_run_*.csv",
    f"{PER_ROBOT_PREFIX}*_run_*.csv",
)


def new_record_names(stem: str) -> dict[str, str]:
    """
    File names for a new run's two record files (relative to the output folder).

    :param stem: The multibody config's file stem.
    :returns: {"candidates": ..., "per_robot": ...}.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return {
        "candidates": f"{CANDIDATE_PREFIX}{stem}_run_{stamp}.csv",
        "per_robot": f"{PER_ROBOT_PREFIX}{stem}_run_{stamp}.csv",
    }


def existing_run_files(output_dir: Path) -> list[str]:
    """
    Files at the top of ``output_dir`` that show it already holds a run (R19.5).

    :returns: Sorted file names; empty for a fresh folder.
    """
    if not output_dir.is_dir():
        return []
    found = set()
    for pattern in RUN_FILE_PATTERNS:
        found.update(path.name for path in output_dir.glob(pattern) if path.is_file())
    return sorted(found)


def _complete_row_generation(columns: list[str]) -> Any:
    """
    A ``generation_of_row`` for the trim that also rejects rows cut short by a kill.

    :returns: The function.
    """

    def generation_of_row(row: dict[str, str]) -> int:
        if any(row.get(column) in (None, "") for column in columns):
            raise ValueError("incomplete row")
        if None in row:  # extra fields: a line merged with a cut-off one
            raise ValueError("malformed row")
        last = columns[-1]
        if last == "failed" and row["failed"] not in ("True", "False"):
            raise ValueError("cut-off failed flag")
        if last == "layer_params" and not row["layer_params"].endswith("]"):
            raise ValueError("cut-off parameter list")
        return int(row["generation"])

    return generation_of_row


class Recorder:
    """Appends a run's rows; trims rows past the resume point once on resume."""

    def __init__(self, output_dir: Path, names: dict[str, str]) -> None:
        """
        :param output_dir: The run's output folder (the snapshot's folder on resume).
        :param names: The record file names (relative to the folder).
        """
        self.output_dir = Path(output_dir)
        self.names = dict(names)
        self.candidates_path = self.output_dir / names["candidates"]
        self.per_robot_path = self.output_dir / names["per_robot"]

    def trim_after(self, generation: int) -> None:
        """
        Remove rows of later generations and incomplete rows, once, before a resumed
        run writes its first row (R18.4).

        :param generation: The resumed snapshot's generation N.
        """
        # A line cut off by a kill has no newline: drop it first, then drop rows
        # of later generations and any row that is not complete.
        for path in (self.candidates_path, self.per_robot_path):
            _drop_partial_last_line(path)
        recording.trim_after_generation(
            str(self.candidates_path), generation, _complete_row_generation(CANDIDATE_COLUMNS)
        )
        recording.trim_after_generation(
            str(self.per_robot_path), generation, _complete_row_generation(PER_ROBOT_COLUMNS)
        )

    def write_generation(
        self,
        generation: int,
        candidate_layers: list[Any],
        scores: list[float],
        records: Any,
        failed_match_score: float,
    ) -> None:
        """
        Append one generation's rows and flush them to disk (R19.2, R19.3, R19.6).

        :param generation: N of the snapshot this generation will write.
        :param candidate_layers: The candidates, by candidate index.
        :param scores: Their scores, by candidate index.
        :param records: The evaluator's ``CandidateRecords``.
        :param failed_match_score: The score a failed match's rows carry.
        """
        order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
        by_candidate: dict[int, list[Any]] = {}
        for record in records.matches:
            by_candidate.setdefault(record.candidate_index, []).append(record)

        candidate_rows = [
            {
                "generation": generation,
                "candidate_index": index,
                "score": recording.format_float(scores[index]),
                "layer_params": recording.serialize_parameters(candidate_layers[index]),
            }
            for index in order
        ]
        robot_rows = []
        for index in order:
            for record in sorted(by_candidate[index], key=lambda r: r.layout_index):
                for slot, role in enumerate(record.roles):
                    row = {
                        "generation": generation,
                        "candidate_index": index,
                        "layout_index": record.layout_index,
                        "slot": slot,
                        "role": role,
                        "failed": "True" if record.failure is not None else "False",
                    }
                    if record.metrics is None:
                        for column in METRIC_COLUMNS:
                            row[column] = "nan"
                        row["score"] = recording.format_float(failed_match_score)
                    else:
                        metrics = record.metrics[slot]
                        for column in METRIC_COLUMNS:
                            row[column] = recording.format_float(getattr(metrics, column))
                    robot_rows.append(row)
        _append_rows(self.candidates_path, CANDIDATE_COLUMNS, candidate_rows)
        _append_rows(self.per_robot_path, PER_ROBOT_COLUMNS, robot_rows)


def _append_rows(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Append rows, writing the header if the file is new or empty, then fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if recording.needs_header("a", handle):
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _drop_partial_last_line(path: Path) -> None:
    """Remove a final line that has no newline (cut off by a kill)."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n")
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data[: cut + 1] if cut >= 0 else b"")
    os.replace(temp, path)


def read_rows(path: Path) -> list[dict[str, str]]:
    """
    Read a record file.

    :returns: Its rows.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
