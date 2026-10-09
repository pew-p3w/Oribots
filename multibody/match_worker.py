"""Simulate and score one match (this runs inside a worker process, or inline).

The task (built by ``match.make_match_task`` in the main process) carries the
already-converted scene, so the worker only simulates and scores and the result
does not depend on which worker runs it (R15.5). The worker first installs the
base config as the module named ``config`` (R4.5), for any single-robot helper
that reads it.

The engine returns the state at t = 0, then a state each time a sample interval
has passed, then one last state at the end of the match: N + 1 states for
N = floor(simulation_time x sampling_frequency) samples, the last of which is
the end-of-match state. A different count is a deterministic failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()

import config_loader  # noqa: E402
from match_scoring import MatchFailure, RobotMetrics, RobotTrack, score_match  # noqa: E402


@dataclass
class MatchOutcome:
    """A match's result: K metrics, or the reason it failed deterministically."""

    metrics: list[RobotMetrics] | None
    failure: str | None


def simulate_match_states(task: dict[str, Any], headless: bool = True, viewer_type: Any = None) -> list[Any]:
    """
    Run the simulation for a match task.

    :param task: The match task.
    :param headless: False opens the viewer (playback only).
    :param viewer_type: The viewer type when not headless.
    :returns: The scene simulation states (t = 0 first).
    """
    from revolve2.modular_robot_simulation._scene_simulation_state import SceneSimulationState
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

    states = simulate_scene(
        task["scene_id"],
        task["scene"],
        headless,
        None,
        False,
        task["control_step"],
        task["sample_step"],
        task["simulation_time"],
        task["simulation_timestep"],
        task["cast_shadows"],
        task["fast_sim"],
        task["viewer_type"] if viewer_type is None else viewer_type,
    )
    return [SceneSimulationState(state, task["mapping"]) for state in states]


def extract_tracks(
    scene_states: list[Any],
    robots: list[Any],
    ball: Any,
    head_offset: tuple[float, float, float],
    num_samples: int,
    sampling_frequency: float,
) -> tuple[list[RobotTrack], np.ndarray]:
    """
    The compact per-robot series the scorer needs, read from the simulation states.

    Head positions use the robot's true orientation (``robot_frame.reference_position``
    with the head offset passed explicitly), the same point the brain measures from.

    :returns: (one RobotTrack per robot in slot order, ball xy of shape (N+1, 2)).
    :raises MatchFailure: If the engine returned an unexpected number of states.
    """
    import robot_frame

    if len(scene_states) != num_samples + 1:
        raise MatchFailure(
            f"the simulation returned {len(scene_states)} states, expected {num_samples + 1} "
            "(t = 0 plus one per sample)"
        )
    sample_times = np.arange(1, num_samples + 1, dtype=np.float64) / float(sampling_frequency)
    ball_xy = np.empty((num_samples + 1, 2))
    heads = np.empty((len(robots), num_samples + 1, 2))
    cores = np.empty((len(robots), num_samples + 1, 2))
    for index, state in enumerate(scene_states):
        ball_position = state._simulation_state.get_multi_body_system_pose(ball).position
        ball_xy[index] = (ball_position.x, ball_position.y)
        for slot, robot in enumerate(robots):
            pose = state.get_modular_robot_simulation_state(robot).get_pose()
            head = robot_frame.reference_position(pose, head_offset)
            heads[slot, index] = (head.x, head.y)
            cores[slot, index] = (pose.position.x, pose.position.y)
    tracks = []
    for slot in range(len(robots)):
        distances = np.hypot(heads[slot, :, 0] - ball_xy[:, 0], heads[slot, :, 1] - ball_xy[:, 1])
        tracks.append(
            RobotTrack(
                d0=float(distances[0]),
                distances=distances[1:].copy(),
                sample_times=sample_times,
                core_xy=cores[slot].copy(),
            )
        )
    return tracks, ball_xy


def score_states(task: dict[str, Any], scene_states: list[Any]) -> MatchOutcome:
    """
    Score a simulated match; a deterministic failure becomes an outcome, not an error.

    :returns: The outcome.
    """
    values = task["settings_values"]
    settings = SimpleNamespace(**values)
    n = int(math.floor(task["simulation_time"] * task["sampling_frequency"] + 1e-9))
    try:
        tracks, ball_xy = extract_tracks(
            scene_states,
            task["robots"],
            task["ball"],
            tuple(values["head_offset"]),
            n,
            task["sampling_frequency"],
        )
        metrics = score_match(settings, tracks, ball_xy, task["roles"], task["layout_index"])
    except MatchFailure as failure:
        return MatchOutcome(metrics=None, failure=str(failure))
    return MatchOutcome(metrics=metrics, failure=None)


def simulate_and_score_match(task: dict[str, Any]) -> MatchOutcome:
    """
    The worker entry point: install ``config``, simulate, score.

    :param task: A match task from ``match.make_match_task``.
    :returns: The outcome.
    """
    config_loader.install_base_config_as_config(task["base_config_source"], task["base_config_name"])
    return score_states(task, simulate_match_states(task))
