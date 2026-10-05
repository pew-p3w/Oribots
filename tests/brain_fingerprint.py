"""Reference-value fingerprints of the ball-aware CPG brain.

The brain decides what every hinge does, so a change to it is only safe if the
changed code produces the same numbers. This module drives a
brain module through a fixed, scripted scenario and records what comes out:
the steering inputs it derives from a ball position, the CPG state it
integrates, and the hinge targets it commands.

It takes the brain module as an argument, so the same scenario can be run
against any version of the brain and the results compared.
`tests/brain_reference.json` holds the reference values; `tests/test_brain.py`
checks the current brain against them.

No simulator is involved: the robot pose, ball pose and control interface are
stubs, which is what makes the run reproducible.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REFERENCE_PATH = Path(__file__).resolve().parent / "brain_reference.json"

# Fixed scenario constants. These are test inputs, not project configuration.
BODIES = ("gecko_v2", "spider_v2", "salamander_v2")
NUM_STEERING_INPUTS = 4
STEERING_OUTPUT_SCALE = 0.5
DISTANCE_SCALE = 20.0
INITIAL_STATE_UNIFORM = math.sqrt(2) * 0.5
DT = 0.05
NUM_CONTROL_STEPS = 12
PARAMETER_SEED = 12345
DECIMALS = 9  # loose enough to absorb cross-platform float noise, tight enough to catch logic changes


def _round(value: Any) -> float:
    """Round a value so last-bit float differences do not change a fingerprint."""
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0.0 else rounded


def _round_all(values: Any) -> list[Any]:
    """Round a nested sequence of numbers."""
    return [_round_all(v) if hasattr(v, "__len__") else _round(v) for v in values]


class _Pose:
    """A stand-in for a simulated pose."""

    def __init__(self, position: Any, orientation: Any) -> None:
        self.position = position
        self.orientation = orientation


class _SimulationState:
    """A stand-in simulation state that returns scripted poses."""

    def __init__(self, poses: dict[int, _Pose]) -> None:
        self._poses = poses

    def get_multi_body_system_pose(self, multi_body_system: Any) -> _Pose:
        """
        Get the scripted pose of a body.

        :param multi_body_system: The robot or ball sentinel.
        :returns: Its scripted pose.
        """
        return self._poses[id(multi_body_system)]


class _Mapping:
    """A stand-in for the body-to-multi-body-system mapping."""

    def __init__(self, multi_body_system: Any) -> None:
        self.multi_body_system = multi_body_system


class _SensorState:
    """A stand-in sensor state exposing what the brain reads from it."""

    def __init__(self, simulation_state: _SimulationState, mapping: _Mapping) -> None:
        self._simulation_state = simulation_state
        self._body_to_multi_body_system_mapping = mapping


class _ControlInterface:
    """A stand-in control interface that records hinge targets instead of applying them."""

    def __init__(self) -> None:
        self.targets: dict[int, float] = {}

    def set_active_hinge_target(self, active_hinge: Any, target: float) -> None:
        """
        Record a commanded hinge target.

        :param active_hinge: The hinge being commanded.
        :param target: The commanded target angle.
        """
        self.targets[id(active_hinge)] = float(target)


def _scripted_poses(step: int, robot: Any, ball: Any) -> _SimulationState:
    """
    Build the scripted robot and ball poses for one control step.

    The robot walks away from the origin while turning, and the ball drifts
    across it, so the scenario sweeps a range of heading errors and distances
    rather than testing a single geometry.

    :param step: Control step index.
    :param robot: Sentinel standing for the robot's multi-body system.
    :param ball: Sentinel standing for the ball.
    :returns: A simulation state holding both poses.
    """
    from pyrr import Quaternion, Vector3

    return _SimulationState(
        {
            id(robot): _Pose(
                Vector3([0.10 * step, 0.05 * step, 0.0]),
                Quaternion.from_z_rotation(0.2 * step),
            ),
            id(ball): _Pose(
                Vector3([5.0 - 0.30 * step, 2.0 + 0.10 * step, 0.3]),
                Quaternion(),
            ),
        }
    )


def _pure_functions(brain_module: Any) -> dict[str, Any]:
    """
    Record the module-level helper functions' outputs over fixed sweeps.

    :param brain_module: The brain module under test.
    :returns: Recorded values keyed by function name.
    """
    import numpy as np

    angles = [-math.pi, -math.pi / 2, -0.1, 0.0, 0.1, math.pi / 2, math.pi, 3 * math.pi, -3 * math.pi]

    rng = np.random.default_rng(PARAMETER_SEED)
    state = rng.uniform(-1.0, 1.0, 5)
    weights = rng.uniform(-1.0, 1.0, (5, 5))
    weights = weights - weights.T  # the CPG weight matrix is antisymmetric
    rk45_steps = []
    current = state.copy()
    for _ in range(5):
        current = brain_module._rk45(current, weights, DT)
        rk45_steps.append(_round_all(current))

    return {
        "wrap_angle": {f"{angle:.6f}": _round(brain_module._wrap_angle(angle)) for angle in angles},
        "rk45": {
            "initial_state": _round_all(state),
            "steps": rk45_steps,
        },
    }


def _body_trace(brain_module: Any, bodies_module: Any, body_name: str) -> dict[str, Any]:
    """
    Drive the brain through the scripted scenario on one body.

    :param brain_module: The brain module under test.
    :param bodies_module: The body catalogue module.
    :param body_name: Name of the body constructor to use.
    :returns: Recorded parameter layout, steering inputs, CPG states and hinge targets.
    """
    import numpy as np
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )

    body = getattr(bodies_module, body_name)()
    active_hinges = body.find_modules_of_type(ActiveHinge)
    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    num_steering = brain_module.steering_parameter_count(
        output_mapping=output_mapping,
        num_steering_inputs=NUM_STEERING_INPUTS,
    )
    num_params = cpg_network_structure.num_connections + num_steering
    params = np.random.default_rng(PARAMETER_SEED).uniform(-1.0, 1.0, num_params)

    robot = type("RobotBody", (), {})()
    ball = type("BallBody", (), {})()

    brain = brain_module.BallAwareCpgBrain.from_params(
        params=params,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=INITIAL_STATE_UNIFORM,
        output_mapping=output_mapping,
        ball=ball,
        num_steering_inputs=NUM_STEERING_INPUTS,
        steering_output_scale=STEERING_OUTPUT_SCALE,
        distance_scale=DISTANCE_SCALE,
    )
    instance = brain.make_instance()
    mapping = _Mapping(robot)

    steering_inputs = []
    cpg_states = []
    hinge_targets = []
    for step in range(NUM_CONTROL_STEPS):
        simulation_state = _scripted_poses(step, robot, ball)
        sensor_state = _SensorState(simulation_state, mapping)

        steering_inputs.append(
            _round_all(
                brain_module._ball_relative_inputs(
                    sensor_state=sensor_state,
                    ball=ball,
                    distance_scale=DISTANCE_SCALE,
                )
            )
        )
        interface = _ControlInterface()
        instance.control(DT, sensor_state, interface)
        cpg_states.append(_round_all(instance._state))
        hinge_targets.append([_round(interface.targets[id(hinge)]) for _, hinge in output_mapping])

    # Wrong-length parameter vectors must be rejected, not silently reshaped.
    rejects = {}
    for delta in (-1, 1):
        try:
            brain_module.BallAwareCpgBrain.from_params(
                params=np.zeros(num_params + delta),
                cpg_network_structure=cpg_network_structure,
                initial_state_uniform=INITIAL_STATE_UNIFORM,
                output_mapping=output_mapping,
                ball=ball,
                num_steering_inputs=NUM_STEERING_INPUTS,
                steering_output_scale=STEERING_OUTPUT_SCALE,
                distance_scale=DISTANCE_SCALE,
            )
        except ValueError:
            rejects[str(delta)] = "ValueError"
        else:
            rejects[str(delta)] = "ACCEPTED"

    return {
        "num_active_hinges": len(active_hinges),
        "num_cpg_connections": cpg_network_structure.num_connections,
        "num_steering_parameters": num_steering,
        "num_parameters": num_params,
        "hinge_ranges": _round_all([hinge.range for _, hinge in output_mapping]),
        "steering_matrix_shape": list(brain._steering_matrix.shape),
        "weight_matrix": _round_all(brain._weight_matrix),
        "initial_state": _round_all(brain._initial_state),
        "steering_inputs_per_step": steering_inputs,
        "cpg_state_per_step": cpg_states,
        "hinge_targets_per_step": hinge_targets,
        "wrong_length_params": rejects,
    }


def fingerprint(brain_module: Any, bodies_module: Any) -> dict[str, Any]:
    """
    Record everything the brain produces in the fixed scenario.

    :param brain_module: The brain module under test.
    :param bodies_module: The body catalogue module.
    :returns: A JSON-serializable fingerprint.
    """
    return {
        "scenario": {
            "bodies": list(BODIES),
            "num_steering_inputs": NUM_STEERING_INPUTS,
            "steering_output_scale": STEERING_OUTPUT_SCALE,
            "distance_scale": DISTANCE_SCALE,
            "initial_state_uniform": _round(INITIAL_STATE_UNIFORM),
            "dt": DT,
            "num_control_steps": NUM_CONTROL_STEPS,
            "parameter_seed": PARAMETER_SEED,
            "decimals": DECIMALS,
        },
        "pure_functions": _pure_functions(brain_module),
        "bodies": {name: _body_trace(brain_module, bodies_module, name) for name in BODIES},
    }


def load_reference() -> dict[str, Any]:
    """
    Load the stored reference fingerprint.

    :returns: The reference values of the brain.
    """
    return json.loads(REFERENCE_PATH.read_text())


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """
    Flatten nested dicts to dotted paths so differences can be located precisely.

    :param value: The structure to flatten.
    :param prefix: Path prefix accumulated so far.
    :returns: Mapping of dotted path to leaf value.
    """
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def compare(reference: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """
    Compare two brain fingerprints.

    :param reference: The expected values.
    :param current: The values produced by the brain under test.
    :returns: Human-readable differences; empty when identical.
    """
    if reference.get("scenario") != current.get("scenario"):
        return ["scenario constants differ: the reference was captured under different settings"]

    # Keys starting with "_" are provenance notes about the reference file, not
    # recorded brain behaviour.
    def recorded(data: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in data.items() if not key.startswith("_")}

    expected, actual = _flatten(recorded(reference)), _flatten(recorded(current))
    problems = []
    for path in sorted(set(expected) | set(actual)):
        if path not in actual:
            problems.append(f"{path}: missing from the current brain")
        elif path not in expected:
            problems.append(f"{path}: not present in the reference")
        elif expected[path] != actual[path]:
            summary = ""
            if not isinstance(expected[path], list):
                summary = f" (expected {expected[path]!r}, got {actual[path]!r})"
            problems.append(f"{path}: differs{summary}")
    return problems


def value_count(fingerprint_data: dict[str, Any]) -> int:
    """
    Count the recorded leaf numbers, for reporting how much was compared.

    :param fingerprint_data: A fingerprint.
    :returns: Number of recorded values.
    """

    def count(value: Any) -> int:
        if isinstance(value, dict):
            return sum(count(v) for v in value.values())
        if isinstance(value, list):
            return sum(count(v) for v in value)
        return 1

    return count(fingerprint_data["bodies"]) + count(fingerprint_data["pure_functions"])


def digest(fingerprint_data: dict[str, Any]) -> str:
    """
    Short content hash of a fingerprint, for logging.

    :param fingerprint_data: A fingerprint.
    :returns: First 12 hex characters of its SHA-256.
    """
    return hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()[:12]
