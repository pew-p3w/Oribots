"""The robot controller: a CPG brain with live ball-position feedback.

This is the only brain in the project. Both the EA and CMA-ES import it, so
both algorithms always train the same controller.

Each hinge is driven by a central pattern generator, which produces the gait.
On top of that sits an evolved steering layer that reads where the ball is
right now and nudges the hinges toward it. An evolved parameter vector holds
both parts: the CPG connection weights first, then the steering weights.

Per control step the steering layer sees four inputs:
`sin(angle_error)`, `cos(angle_error)`, the ball distance normalized by
`distance_scale` and clipped to `[0, 1]`, and a constant bias. Its output is
squashed with `tanh`, scaled by `steering_output_scale`, and added to the CPG
target; the result is clipped to the hinge's range of motion.

`tests/test_brain.py` checks this module against the reference values in
`tests/brain_reference.json`. Changing the control behaviour here fails that
test.

Head mode: with a non-zero `reference_offset` (a head's position in the robot
frame, see `src/robot_frame.py`) the ball's angle and distance are measured from
the head, using the robot's true orientation. With the default `(0, 0, 0)` they
are measured from the core (see `_ball_relative_inputs` for a caveat). `tests/test_head.py` checks head mode.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pyrr import Vector3

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.brain.cpg import CpgNetworkStructure
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.modular_robot_simulation._modular_robot_scene import physical_orientation
from revolve2.simulation.scene import MultiBodySystem


class BallAwareCpgBrain(Brain):
    """A CPG brain with an evolved live ball-position steering layer."""

    _initial_state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _steering_matrix: npt.NDArray[np.float_]
    _ball: MultiBodySystem
    _steering_output_scale: float
    _distance_scale: float
    _reference_offset: tuple[float, float, float]

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        steering_matrix: npt.NDArray[np.float_],
        ball: MultiBodySystem,
        steering_output_scale: float,
        distance_scale: float,
        reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Initialize this brain.

        :param initial_state: Initial CPG state.
        :param weight_matrix: CPG weight matrix.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :param steering_matrix: Evolved steering layer weights.
        :param ball: Ball object to track during simulation.
        :param steering_output_scale: Maximum steering output as a hinge-range fraction.
        :param distance_scale: Distance used to normalize ball distance input.
        :param reference_offset: Point on the robot the ball is measured from, in the
            robot frame. (0, 0, 0) is the core; a head offset measures from the head.
        """
        self._initial_state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping
        self._steering_matrix = steering_matrix
        self._ball = ball
        self._steering_output_scale = steering_output_scale
        self._distance_scale = distance_scale
        self._reference_offset = tuple(reference_offset)

    @classmethod
    def from_params(
        cls,
        params: npt.NDArray[np.float_],
        cpg_network_structure: CpgNetworkStructure,
        initial_state_uniform: float,
        output_mapping: list[tuple[int, ActiveHinge]],
        ball: MultiBodySystem,
        num_steering_inputs: int,
        steering_output_scale: float,
        distance_scale: float,
        reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> BallAwareCpgBrain:
        """
        Create a ball-aware CPG brain from evolved CPG and steering parameters.

        :param params: CPG parameters followed by steering-layer parameters.
        :param cpg_network_structure: CPG network structure.
        :param initial_state_uniform: Initial value for every CPG state.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :param ball: Ball object to track during simulation.
        :param num_steering_inputs: Number of live feedback inputs.
        :param steering_output_scale: Maximum steering output as a hinge-range fraction.
        :param distance_scale: Distance used to normalize ball distance input.
        :param reference_offset: Point on the robot the ball is measured from, in the
            robot frame. (0, 0, 0) is the core; a head offset measures from the head.
        :returns: The created brain.
        :raises ValueError: If the parameter vector length is incorrect.
        """
        num_cpg_params = cpg_network_structure.num_connections
        num_steering_params = steering_parameter_count(
            output_mapping=output_mapping,
            num_steering_inputs=num_steering_inputs,
        )
        expected_num_params = num_cpg_params + num_steering_params
        if len(params) != expected_num_params:
            raise ValueError(
                f"Expected {expected_num_params} parameters "
                f"({num_cpg_params} CPG + {num_steering_params} steering), "
                f"got {len(params)}."
            )

        initial_state = cpg_network_structure.make_uniform_state(
            initial_state_uniform
        )
        weight_matrix = cpg_network_structure.make_connection_weights_matrix_from_params(
            list(params[:num_cpg_params])
        )
        steering_matrix = np.asarray(params[num_cpg_params:]).reshape(
            len(output_mapping),
            num_steering_inputs,
        )
        return cls(
            initial_state=initial_state,
            weight_matrix=weight_matrix,
            output_mapping=output_mapping,
            steering_matrix=steering_matrix,
            ball=ball,
            steering_output_scale=steering_output_scale,
            distance_scale=distance_scale,
            reference_offset=reference_offset,
        )

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return BallAwareCpgBrainInstance(
            initial_state=self._initial_state.copy(),
            weight_matrix=self._weight_matrix.copy(),
            output_mapping=self._output_mapping,
            steering_matrix=self._steering_matrix.copy(),
            ball=self._ball,
            steering_output_scale=self._steering_output_scale,
            distance_scale=self._distance_scale,
            reference_offset=self._reference_offset,
        )


class BallAwareCpgBrainInstance(BrainInstance):
    """Stateful instance of a ball-aware CPG brain."""

    _state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _steering_matrix: npt.NDArray[np.float_]
    _ball: MultiBodySystem
    _steering_output_scale: float
    _distance_scale: float
    _reference_offset: tuple[float, float, float]

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        steering_matrix: npt.NDArray[np.float_],
        ball: MultiBodySystem,
        steering_output_scale: float,
        distance_scale: float,
        reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """
        Initialize this brain instance.

        :param initial_state: Initial CPG state.
        :param weight_matrix: CPG weight matrix.
        :param output_mapping: Mapping from CPG outputs to active hinges.
        :param steering_matrix: Evolved steering layer weights.
        :param ball: Ball object to track during simulation.
        :param steering_output_scale: Maximum steering output as a hinge-range fraction.
        :param distance_scale: Distance used to normalize ball distance input.
        :param reference_offset: Point on the robot the ball is measured from, in the
            robot frame. (0, 0, 0) is the core; a head offset measures from the head.
        """
        self._state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping
        self._steering_matrix = steering_matrix
        self._ball = ball
        self._steering_output_scale = steering_output_scale
        self._distance_scale = distance_scale
        self._reference_offset = reference_offset

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the robot using CPG output plus live ball-position feedback.

        :param dt: Elapsed seconds since the last control step.
        :param sensor_state: Interface for reading current sensor state.
        :param control_interface: Interface for controlling active hinges.
        """
        self._state = _rk45(self._state, self._weight_matrix, dt)
        steering_inputs = _ball_relative_inputs(
            sensor_state=sensor_state,
            ball=self._ball,
            distance_scale=self._distance_scale,
            reference_offset=self._reference_offset,
        )

        for output_index, (state_index, active_hinge) in enumerate(
            self._output_mapping
        ):
            cpg_target = float(self._state[state_index]) * active_hinge.range
            steering_fraction = (
                np.tanh(
                    float(
                        np.dot(
                            self._steering_matrix[output_index],
                            steering_inputs,
                        )
                    )
                )
                * self._steering_output_scale
            )
            steering_offset = steering_fraction * active_hinge.range
            target = np.clip(
                cpg_target + steering_offset,
                -active_hinge.range,
                active_hinge.range,
            )
            control_interface.set_active_hinge_target(active_hinge, float(target))


def steering_parameter_count(
    output_mapping: list[tuple[int, ActiveHinge]], num_steering_inputs: int) -> int:
    """
    Calculate the number of parameters needed by the steering layer.

    :param output_mapping: Mapping from CPG outputs to active hinges.
    :param num_steering_inputs: Number of live feedback inputs.
    :returns: Number of steering parameters.
    """
    return len(output_mapping) * num_steering_inputs


def _ball_relative_inputs(
    sensor_state: ModularRobotSensorState,
    ball: MultiBodySystem,
    distance_scale: float,
    reference_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> npt.NDArray[np.float_]:
    """
    Calculate live feedback inputs describing the ball relative to the robot.

    The angle is between the robot's heading (its +x axis, toward the head when
    it has one) and the direction to the ball, and the distance is measured from
    the reference point (the head, or the core if the offset is zero).

    With a head (non-zero offset) both come from the robot's true orientation.
    With no offset the calculation uses the orientation the simulator reports,
    which is a permutation of the physical one, so the "heading" is effectively
    the fixed world +x axis whichever way the robot faces (see robot_frame.py and
    engine/PATCHES.md). This core-mode behaviour is kept unchanged so that
    configs and controllers trained with it keep behaving the same; head mode is
    the orientation-correct alternative.

    :param sensor_state: Current robot sensor state.
    :param ball: Ball object.
    :param distance_scale: Distance used to normalize ball distance input.
    :param reference_offset: Point on the robot to measure from, in the robot frame.
    :returns: Feedback inputs: sin(angle), cos(angle), distance, bias.
    """
    simulation_state = getattr(sensor_state, "_simulation_state")
    mapping = getattr(sensor_state, "_body_to_multi_body_system_mapping")

    robot_pose = simulation_state.get_multi_body_system_pose(mapping.multi_body_system)
    ball_pose = simulation_state.get_multi_body_system_pose(ball)

    robot_position = robot_pose.position
    orientation = robot_pose.orientation
    if reference_offset != (0.0, 0.0, 0.0):
        orientation = physical_orientation(robot_pose)
        robot_position = robot_position + orientation * Vector3(reference_offset)
    to_ball = ball_pose.position - robot_position
    distance = math.sqrt(to_ball.x**2 + to_ball.y**2)
    target_angle = math.atan2(to_ball.y, to_ball.x)
    heading = orientation * Vector3([1.0, 0.0, 0.0])
    heading_angle = math.atan2(heading.y, heading.x)
    angle_error = _wrap_angle(target_angle - heading_angle)

    return np.array(
        [
            math.sin(angle_error),
            math.cos(angle_error),
            np.clip(distance / distance_scale, 0.0, 1.0),
            1.0,
        ]
    )


def _wrap_angle(angle: float) -> float:
    """
    Wrap an angle to [-pi, pi].

    :param angle: Angle in radians.
    :returns: Wrapped angle.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rk45(
    state: npt.NDArray[np.float_], weight_matrix: npt.NDArray[np.float_], dt: float
) -> npt.NDArray[np.float_]:
    """
    Calculate the next CPG state using the same RK4 step as the stock CPG brain.

    :param state: Current CPG state.
    :param weight_matrix: CPG weight matrix.
    :param dt: Time step.
    :returns: The new state.
    """
    a1: npt.NDArray[np.float_] = np.matmul(weight_matrix, state)
    a2: npt.NDArray[np.float_] = np.matmul(weight_matrix, state + dt / 2.0 * a1)
    a3: npt.NDArray[np.float_] = np.matmul(weight_matrix, state + dt / 2.0 * a2)
    a4: npt.NDArray[np.float_] = np.matmul(weight_matrix, state + dt * a3)
    return np.clip(state + dt / 6.0 * (a1 + 2.0 * (a2 + a3) + a4), -1.0, 1.0)
