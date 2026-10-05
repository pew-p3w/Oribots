"""Per-individual recording, shared by both optimizer mains.

Every run records every individual of every generation into two CSV files next
to the run CSV:

``individual_performance_<name>_run_<timestamp>.csv``
    One row per individual per training-ball trial: ``generation``,
    ``individual_id``, ``trial``, ``trial_fitness``, ``overall_fitness``,
    ``ball_pose``, and how that trial moved relative to the robot's own heading:
    ``forward_m``, ``backward_m``, ``sideways_m`` (metres travelled along and
    across the heading), ``end_heading_deg`` (where it ended up pointing, degrees
    from +x) and ``end_angle_to_ball_deg`` (angle from its heading to the ball at
    the end; 0 means pointing straight at it). See ``robot_frame.movement_report``.

``weights.csv``
    One row per individual: ``individual_id`` and ``robot_weights``.

``individual_id`` is ``g<generation>_i<index>``, where ``index`` is the
individual's position in the population handed to the evaluator. Ids are
unique across the whole run.

``trial_fitness`` is always the raw signed score of one simulation.
``overall_fitness`` is the fitness the optimizer recorded for the individual:
the mean of its ``trial_fitness`` rows, clamped to [0, 1] for the EA and raw for
CMA-ES. This module never clips or recomputes anything; it writes what the
optimizer hands it. Rows are written in ascending ``overall_fitness`` order, so
the END of the file holds each generation's best individuals, not typical ones.

A resumed run keeps writing into the files it started with (their paths travel
in every snapshot) and first drops any rows the interrupted run had already
written past the resume point, including a half-written last line.
"""

import csv
import json
import logging
import os
from typing import Any, Callable, Sequence

import numpy as np


MOVEMENT_FIELDNAMES = [
    "forward_m",
    "backward_m",
    "sideways_m",
    "end_heading_deg",
    "end_angle_to_ball_deg",
]
PERFORMANCE_FIELDNAMES = [
    "generation",
    "individual_id",
    "trial",
    "trial_fitness",
    "overall_fitness",
    "ball_pose",
] + MOVEMENT_FIELDNAMES
WEIGHTS_FIELDNAMES = ["individual_id", "robot_weights"]
WEIGHTS_FILENAME = "weights.csv"


def individual_id(generation_index: int, individual_index: int) -> str:
    """
    Build the id of one individual.

    :param generation_index: Generation number the individual was evaluated in.
    :param individual_index: Position in the population given to the evaluator.
    :returns: Id such as ``g56_i398``.
    """
    return f"g{generation_index}_i{individual_index}"


def generation_of_individual_id(value: str) -> int:
    """
    Extract the generation number from an individual id.

    :param value: Id such as ``g56_i398``.
    :returns: The generation number.
    :raises ValueError: If the id is malformed.
    """
    if not value.startswith("g") or "_i" not in value:
        raise ValueError(f"Malformed individual id: {value!r}")
    return int(value[1 : value.index("_i")])


def default_paths(
    output_dir: str, parameter_stem: str, run_timestamp: str
) -> tuple[str, str]:
    """
    Choose the two recording file paths for a new run.

    :param output_dir: Run output folder.
    :param parameter_stem: Parameter filename without the ``.npy`` suffix.
    :param run_timestamp: Timestamp shared with the run CSV name.
    :returns: Tuple of (performance CSV path, weights CSV path).
    """
    return (
        os.path.join(
            output_dir,
            f"individual_performance_{parameter_stem}_run_{run_timestamp}.csv",
        ),
        os.path.join(output_dir, WEIGHTS_FILENAME),
    )


def serialize_parameters(parameters: Any) -> str:
    """
    Serialize a controller parameter vector for one CSV cell.

    :param parameters: Parameter vector.
    :returns: JSON list string.
    """
    return json.dumps([float(parameter) for parameter in np.asarray(parameters)])


def format_float(value: float | None) -> str:
    """
    Format a float for a CSV cell without losing precision.

    :param value: Value to format, or None.
    :returns: Empty string for None, otherwise a round-trippable string.
    """
    return "" if value is None else f"{float(value):.17g}"


def write_generation(
    performance_csv_path: str,
    weights_csv_path: str,
    generation_index: int,
    parameters: Sequence[Any],
    overall_fitnesses: Sequence[float],
    trial_fitnesses: Sequence[Sequence[float]],
    training_ball_poses: Sequence[Any],
    mode: str,
    trial_movements: Sequence[Sequence[dict[str, float]]] | None = None,
) -> None:
    """
    Record every individual of one generation.

    Individuals are written in ascending order of overall fitness (ties by
    index), but each keeps the id of its position in the population.

    :param performance_csv_path: Per-trial performance CSV.
    :param weights_csv_path: Weights CSV.
    :param generation_index: Generation the individuals were evaluated in.
    :param parameters: Controller parameter vector of each individual.
    :param overall_fitnesses: Fitness the optimizer used for each individual.
    :param trial_fitnesses: Raw score, indexed ``[trial][individual]``.
    :param training_ball_poses: The fixed training ball poses, one per trial.
    :param mode: ``"w"`` to start new files, ``"a"`` to append.
    :param trial_movements: Movement record of each trial, indexed
        ``[trial][individual]`` (see ``robot_frame.movement_report``). Left empty
        in the file if not given.
    :raises ValueError: If the inputs have inconsistent shapes.
    """
    population_size = len(parameters)
    if population_size == 0:
        raise ValueError("Cannot record an empty population.")
    if len(overall_fitnesses) != population_size:
        raise ValueError("Need one overall fitness per individual.")
    if len(trial_fitnesses) != len(training_ball_poses):
        raise ValueError("Number of trial-fitness rows does not match ball poses.")
    for trial_values in trial_fitnesses:
        if len(trial_values) != population_size:
            raise ValueError("Each trial row needs one value per individual.")
    if trial_movements is not None:
        if len(trial_movements) != len(training_ball_poses) or any(
            len(row) != population_size for row in trial_movements
        ):
            raise ValueError("Movement records must match [trial][individual].")

    order = sorted(
        range(population_size),
        key=lambda index: (float(overall_fitnesses[index]), index),
    )

    # When continuing a file that was started with a different column set, keep
    # ITS columns, so the rows written now line up with the header already there.
    fieldnames = _existing_header(performance_csv_path) if mode == "a" else None
    fieldnames = fieldnames or PERFORMANCE_FIELDNAMES
    with open(performance_csv_path, mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore", restval=""
        )
        if needs_header(mode, file):
            writer.writeheader()
        for index in order:
            for trial_index, ball_pose in enumerate(training_ball_poses):
                position = ball_pose.position
                writer.writerow(
                    {
                        "generation": generation_index,
                        "individual_id": individual_id(generation_index, index),
                        "trial": trial_index + 1,
                        "trial_fitness": format_float(trial_fitnesses[trial_index][index]),
                        "overall_fitness": format_float(overall_fitnesses[index]),
                        "ball_pose": json.dumps(
                            [float(position.x), float(position.y), float(position.z)]
                        ),
                        **(
                            {
                                name: format_float(
                                    trial_movements[trial_index][index][name]
                                )
                                for name in MOVEMENT_FIELDNAMES
                            }
                            if trial_movements is not None
                            else {}
                        ),
                    }
                )

    with open(weights_csv_path, mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=WEIGHTS_FIELDNAMES)
        if needs_header(mode, file):
            writer.writeheader()
        for index in order:
            writer.writerow(
                {
                    "individual_id": individual_id(generation_index, index),
                    "robot_weights": serialize_parameters(parameters[index]),
                }
            )


def _existing_header(csv_path: str) -> list[str] | None:
    """
    Read the column names of an existing, non-empty CSV file.

    :param csv_path: The file.
    :returns: The column names, or None if the file is missing or empty.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        first_line = file.readline().strip()
    return first_line.split(",") if first_line else None


def needs_header(mode: str, file: Any) -> bool:
    """
    Decide whether a CSV file being opened needs its header row.

    A header is needed for a fresh file, and also when appending to a file that
    is missing or empty (for example after a run is resumed on another machine),
    so a CSV is never left without a header.

    :param mode: The mode the file was opened with.
    :param file: The open file object.
    :returns: True if the header should be written.
    """
    return mode == "w" or file.tell() == 0


def trim_after_generation(
    csv_path: str,
    max_generation: int,
    generation_of_row: Callable[[dict[str, str]], int],
) -> int:
    """
    Drop rows of generations later than ``max_generation`` from a CSV.

    Used when a run is resumed from a snapshot: any rows the interrupted run had
    already written for generations after the snapshot would otherwise be
    written a second time, breaking the uniqueness of ``individual_id``. Rows
    that cannot be parsed (for example a line cut short by a killed process) are
    dropped too. Does nothing if the file does not exist.

    :param csv_path: CSV file to trim in place.
    :param max_generation: Last generation to keep.
    :param generation_of_row: Function returning the generation of a row.
    :returns: Number of rows removed.
    """
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0

    kept: list[dict[str, str]] = []
    removed = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        for row in reader:
            try:
                keep = generation_of_row(row) <= max_generation
            except (ValueError, TypeError, KeyError):
                keep = False
            if keep:
                kept.append(row)
            else:
                removed += 1
    if removed == 0 or fieldnames is None:
        return 0

    temp_path = f"{csv_path}.tmp"
    with open(temp_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(temp_path, csv_path)
    logging.info(
        f"Removed {removed} row(s) after generation {max_generation} from {csv_path}"
    )
    return removed


def trim_recording_after_generation(
    performance_csv_path: str, weights_csv_path: str, max_generation: int
) -> None:
    """
    Trim both recording files so that a resumed run continues cleanly.

    :param performance_csv_path: Per-trial performance CSV.
    :param weights_csv_path: Weights CSV.
    :param max_generation: Last generation to keep (the resume point).
    """
    trim_after_generation(
        performance_csv_path, max_generation, lambda row: int(row["generation"])
    )
    trim_after_generation(
        weights_csv_path,
        max_generation,
        lambda row: generation_of_individual_id(row["individual_id"]),
    )
