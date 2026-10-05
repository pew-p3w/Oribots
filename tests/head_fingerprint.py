"""Reference-value fingerprints of head mode and the pentagon spider.

Head mode measures everything from a head block on the robot's front instead of
from the core: the brain's steering inputs, the fitness, the "ball reached"
stop condition and the movement report. It also relies on the engine's
``physical_orientation`` (the simulator reports orientations permuted) and on
the core-builder hook that gives the pentagon spider its five-sided core.

This module replays FROZEN REAL simulation states (recorded once by simulating
an evolved spider that reaches the ball and a random one that does not) through
those functions and records what comes out. No simulator runs here, so the
values are the same on macOS and Linux.

`tests/head_reference.json` holds the frozen inputs and the reference values
computed from them; `tests/test_head.py` checks the current code against them.

The modules are passed in, so the same scenario can be run against any version
of the code.
"""

import hashlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

REFERENCE_PATH = Path(__file__).resolve().parent / "head_reference.json"

DECIMALS = 12
# Every Nth sample is traced for the per-sample values (brain inputs, measuring
# point, heading). Fitness, movement and the stop condition use every sample.
TRACE_EVERY = 25
# A training-ball layout is drawn with this seed and checked too.
BALL_LAYOUT_SEED = 20260927
BALL_LAYOUT_COUNT = 8

# The fitness settings of spider_8ball, fixed here so the fingerprint does not
# depend on whichever config is present. HEAD_OFFSET comes from the reference.
CONFIG_VALUES = {
    "BALL_RADIUS": 0.3,
    "BALL_REACHED_DISTANCE": 0.3 * (1.0 + 0.1),
    "NO_PROGRESS_PENALTY": 0.01,
    "NO_PROGRESS_EPSILON": 1e-6,
    "FITNESS_PROGRESS_WEIGHT": 1.0,
    "FITNESS_REACHED_BONUS_WEIGHT": 0.20,
    "FITNESS_TIME_TO_REACH_WEIGHT": 0.10,
    "FITNESS_ALIGNMENT_WEIGHT": 0.05,
    "FEEDBACK_DISTANCE_SCALE": 20.0,
}


def _round(value: Any) -> float:
    """Round a number so float noise cannot change a fingerprint."""
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _round_all(values: Any) -> list[float]:
    """Round every number in a flat sequence."""
    return [_round(value) for value in values]


def make_config(head_offset: list[float], use_head: bool, simulation_time: float) -> ModuleType:
    """
    Build the `config` module the code under test reads.

    :param head_offset: The head's offset in the robot frame.
    :param use_head: Whether to measure from the head.
    :param simulation_time: The trial length the states were simulated with.
    :returns: The module.
    """
    module = ModuleType("config")
    for name, value in CONFIG_VALUES.items():
        setattr(module, name, value)
    module.HEAD_OFFSET = tuple(head_offset)
    module.USE_HEAD_COORDINATES = use_head
    module.SIMULATION_TIME = simulation_time
    return module


class _Robot:
    """Stands in for the robot handle scene states are queried with."""


class _Ball:
    """Stands in for the ball handle scene states are queried with."""


class _RobotState:
    def __init__(self, pose: Any) -> None:
        self._pose = pose

    def get_pose(self) -> Any:
        return self._pose


class _SimulationState:
    """Answers pose queries for the robot's multi-body system and the ball."""

    def __init__(self, robot_system: Any, robot_pose: Any, ball: Any, ball_pose: Any) -> None:
        self._poses = {id(robot_system): robot_pose, id(ball): ball_pose}

    def get_multi_body_system_pose(self, system: Any) -> Any:
        return self._poses[id(system)]


class _SceneState:
    """Stands in for a SceneSimulationState built from one frozen sample."""

    def __init__(self, robot: Any, robot_pose: Any, simulation_state: _SimulationState) -> None:
        self._robot = robot
        self._robot_pose = robot_pose
        self._simulation_state = simulation_state

    def get_modular_robot_simulation_state(self, robot: Any) -> _RobotState:
        assert robot is self._robot
        return _RobotState(self._robot_pose)


class _Mapping:
    def __init__(self, system: Any) -> None:
        self.multi_body_system = system


class _SensorState:
    """What the brain reads its ball inputs from."""

    def __init__(self, simulation_state: _SimulationState, system: Any) -> None:
        self._simulation_state = simulation_state
        self._body_to_multi_body_system_mapping = _Mapping(system)


def _frozen_states(samples: list[dict[str, list[float]]], robot: Any, robot_system: Any, ball: Any) -> list[tuple[Any, Any]]:
    """
    Rebuild real pyrr poses from frozen samples.

    :returns: (scene state, simulation state) per sample.
    """
    from pyrr import Quaternion, Vector3

    from revolve2.simulation.scene import Pose

    states = []
    for sample in samples:
        robot_pose = Pose(
            Vector3(sample["robot_position"]), Quaternion(sample["robot_orientation"])
        )
        ball_pose = Pose(Vector3(sample["ball_position"]), Quaternion())
        simulation = _SimulationState(robot_system, robot_pose, ball, ball_pose)
        states.append((_SceneState(robot, robot_pose, simulation), simulation))
    return states


def _trace_sequence(
    modules: dict[str, ModuleType],
    samples: list[dict[str, list[float]]],
    head_offset: list[float],
    simulation_time: float,
    sampling_frequency: float,
) -> dict[str, Any]:
    """
    Run one frozen state sequence through head mode and legacy mode.

    :param modules: robot_frame, brain, evaluator, scene (the engine's
        _modular_robot_scene) and set_config (a function installing `config`).
    :returns: Everything the sequence produces.
    """
    robot, robot_system, ball = _Robot(), object(), _Ball()
    states = _frozen_states(samples, robot, robot_system, ball)
    scene_states = [scene for scene, _ in states]
    result: dict[str, Any] = {"num_samples": len(samples)}
    for mode, use_head in (("head", True), ("legacy", False)):
        modules["set_config"](make_config(head_offset, use_head, simulation_time))
        robot_frame = modules["robot_frame"]
        offset = robot_frame.reference_offset()
        bound_stop = modules["scene"]._BoundStopOnRobotObjectDistance(
            robot_multi_body_system=robot_system,
            obj=ball,
            distance=CONFIG_VALUES["BALL_REACHED_DISTANCE"],
            reference_offset=offset,
        )
        stops = [bool(bound_stop(simulation)) for _, simulation in states]
        # Distance to the threshold at every sample, so a result that sits on
        # the edge (and could flip with the last bit of a float) is visible.
        margins = []
        for scene, _ in states:
            point = robot_frame.reference_position(
                scene.get_modular_robot_simulation_state(robot).get_pose(), offset
            )
            ball_position = scene._simulation_state.get_multi_body_system_pose(ball).position
            margins.append(
                math.hypot(point.x - ball_position.x, point.y - ball_position.y)
                - CONFIG_VALUES["BALL_REACHED_DISTANCE"]
            )
        traced = []
        for index in range(0, len(states), TRACE_EVERY):
            scene, simulation = states[index]
            pose = scene.get_modular_robot_simulation_state(robot).get_pose()
            inputs = modules["brain"]._ball_relative_inputs(
                _SensorState(simulation, robot_system),
                ball,
                CONFIG_VALUES["FEEDBACK_DISTANCE_SCALE"],
                reference_offset=offset,
            )
            traced.append(
                {
                    "index": index,
                    "brain_inputs": _round_all(inputs),
                    "measuring_point": _round_all(robot_frame.reference_position(pose, offset)),
                    "heading": _round_all(robot_frame.heading_direction(pose)),
                    "physical_orientation": _round_all(modules["scene"].physical_orientation(pose)),
                }
            )
        fitness = modules["evaluator"]._trial_fitness(
            scene_states=scene_states,
            robot=robot,
            ball=ball,
            sampling_frequency=sampling_frequency,
            simulation_time=simulation_time,
        )
        movement = robot_frame.movement_report(scene_states, robot, ball)
        result[mode] = {
            "offset": _round_all(offset),
            "first_stop_index": stops.index(True) if True in stops else None,
            "num_stops": sum(stops),
            "min_threshold_margin": _round(min(abs(m) for m in margins)),
            "fitness": _round(fitness),
            "movement": {key: _round(value) for key, value in sorted(movement.items())},
            "trace": traced,
        }
    return result


def fingerprint(modules: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Fingerprint head mode, the pentagon spider and the training-ball layout.

    :param modules: robot_frame, brain, evaluator, scene, set_config, plus
        ``spider_pentagon`` (a function returning the body and its head module)
        and ``make_training_ball_poses`` (the config function).
    :param inputs: The frozen inputs (see ``tests/head_reference.json``).
    :returns: The fingerprint.
    """
    import numpy as np

    import body_fingerprint
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot_simulation._build_multi_body_systems import (
        BodyToMultiBodySystemConverter,
    )
    from revolve2.simulation.scene import Pose

    body, head = modules["spider_pentagon"]()
    system, _ = BodyToMultiBodySystemConverter().convert_robot_body(body, Pose(), False)
    core = system.root
    rigid_bodies = [core]
    for rigid_body in rigid_bodies:
        for joint in system.get_joints_for_rigid_body(rigid_body):
            for other in (joint.rigid_body1, joint.rigid_body2):
                if all(other is not seen for seen in rigid_bodies):
                    rigid_bodies.append(other)
    aabb_position, aabb = system.calculate_aabb()
    pentagon = {
        "head_offset": _round_all(modules["robot_frame"].head_offset_of(body, head)),
        "num_active_hinges": len(body.find_modules_of_type(ActiveHinge)),
        "core_geometries": [
            {
                "position": _round_all(geometry.pose.position),
                "orientation": _round_all(geometry.pose.orientation),
                "mass": _round(geometry.mass),
                "size": _round_all(geometry.aabb.size),
            }
            for geometry in core.geometries
        ],
        "num_rigid_bodies": len(rigid_bodies),
        "num_geometries": sum(len(rigid_body.geometries) for rigid_body in rigid_bodies),
        "aabb": [_round_all(aabb_position), _round_all(aabb.size)],
        # The same structural fingerprint every catalogue body gets.
        "structure": body_fingerprint.fingerprint_body(body),
    }
    layout = modules["make_training_ball_poses"](
        np.random.default_rng(BALL_LAYOUT_SEED), BALL_LAYOUT_COUNT
    )
    sequences = {
        name: _trace_sequence(
            modules,
            samples,
            inputs["head_offset"],
            inputs["simulation_time"],
            inputs["sampling_frequency"],
        )
        for name, samples in sorted(inputs["sequences"].items())
    }
    return {
        "pentagon": pentagon,
        "ball_layout": [_round_all(pose.position) for pose in layout],
        "sequences": sequences,
    }


def load_reference() -> dict[str, Any]:
    """
    Load the frozen inputs and the recorded fingerprint.

    :returns: ``{"inputs": ..., "fingerprint": ...}``.
    """
    return json.loads(REFERENCE_PATH.read_text())


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts and lists into dotted paths."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def compare(reference: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """
    List every value that differs between two fingerprints.

    :returns: One line per difference, empty when identical.
    """
    ref, cur = _flatten(reference), _flatten(current)
    problems = []
    for key in sorted(set(ref) | set(cur)):
        if key not in cur:
            problems.append(f"{key}: missing (expected {ref[key]!r})")
        elif key not in ref:
            problems.append(f"{key}: unexpected value {cur[key]!r}")
        elif ref[key] != cur[key]:
            problems.append(f"{key}: expected {ref[key]!r}, got {cur[key]!r}")
    return problems


def value_count(data: dict[str, Any]) -> int:
    """Count the leaf values in a fingerprint."""
    return len(_flatten(data))


def digest(data: dict[str, Any]) -> str:
    """A short hash of a fingerprint, for the test's report."""
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]
