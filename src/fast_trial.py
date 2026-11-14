"""Run one training trial without the per-tick Python objects. Results are bit-identical.

A training trial with the original code (``simulate_scene``, then
``evaluator._trial_fitness``) spends most of its time outside the physics. At
every control tick (20 Hz) and sample (5 Hz) it copies every MuJoCo state
array, wraps the copies in state, sensor and pose objects, looks up bodies and
joints through UUID-keyed dictionaries, and rotates vectors through pyrr's
multiple-dispatch objects. At implicitfast with a 5 ms timestep that is about
two thirds of the trial.

This module runs the same trial and returns the same fitness and movement
record, down to the last bit, with that overhead removed:

- body ids, hinge control indices and hinge ranges are looked up once;
- the physics is stepped inside MuJoCo (``mj_step(..., nstep=k)``) between
  control ticks and samples, instead of one Python call per step. The steps at
  which control and sampling happen are found with exactly the original
  timing rules on exactly the times MuJoCo produces (checked after every
  chunk), so they happen at the same steps;
- the brain and the stop condition read MuJoCo's arrays directly;
- a sample keeps only what scoring reads: the robot core's position and
  orientation, and the ball's position.

"Bit-identical" is kept by doing every floating-point operation the original
does, on the same types, in the same order. Where the original calls a
library routine whose rounding is not fixed by IEEE 754 (``np.dot``, which may
use BLAS; ``np.tanh``; ``np.float64 ** 2``; ``np.mean``), this calls the same
routine on the same kind of value. Only exactly-rounded arithmetic (``+ - * /``,
comparisons, ``min``/``max`` instead of ``np.clip`` on finite values) is done
differently, and those give the same result by definition. Pyrr's quaternion
rotation is written out with its exact expressions, operand order and zero
terms included, so even the sign of a zero is the same.

Only the scenes training builds are handled (``Evaluator._make_scene``: one
robot with a ``BallAwareCpgBrain``, a ball, "ball reached" stop conditions, no
cameras, sampling on). ``supports`` says whether a task qualifies; anything else
goes through the original code. ``tests/test_fast_trial.py`` checks that both
paths agree exactly, for every shipped config, integrator and head mode.

Turn it off with ``FAST_TRIAL = False`` in a config, or the environment variable
``REVOLVE2_FAST_TRIAL=0``.
"""

from __future__ import annotations

import math
import os
from typing import Any

import mujoco
import numpy as np

import config
import robot_frame
from ball_aware_brain import BallAwareCpgBrainInstance, _wrap_angle

from revolve2.modular_robot_simulation._modular_robot_scene import (
    _BoundStopOnRobotObjectDistance,
)
from revolve2.modular_robot_simulation._modular_robot_simulation_handler import (
    ModularRobotSimulationHandler,
)
from revolve2.simulation.scene import UUIDKey

ZERO = (0.0, 0.0, 0.0)


def enabled() -> bool:
    """
    Whether the fast path is switched on (it is unless turned off).

    :returns: False if the config or the environment turns it off.
    """
    return os.environ.get("REVOLVE2_FAST_TRIAL", "1") != "0" and bool(
        getattr(config, "FAST_TRIAL", True)
    )


def supports(task: dict[str, Any]) -> bool:
    """
    Whether this trial is one the fast path reproduces exactly.

    :param task: A worker task (see ``Evaluator._evaluate_poses_in_workers``).
    :returns: True for the scenes training builds; False sends the trial
        through the original code.
    """
    scene = task.get("scene")
    handler = getattr(scene, "handler", None)
    if type(handler) is not ModularRobotSimulationHandler:
        return False
    brains = getattr(handler, "_brains", [])
    if len(brains) != 1 or type(brains[0][0]) is not BallAwareCpgBrainInstance:
        return False
    if any(
        type(stop) is not _BoundStopOnRobotObjectDistance
        for stop in getattr(handler, "_stop_conditions", [])
    ):
        return False
    body_mapping = brains[0][1]
    if body_mapping.camera_to_sim_camera or body_mapping.imu_to_sim_imu:
        return False
    if task.get("sample_step") is None:
        return False
    return True


# --------------------------------------------------------------------------
# Exact copies of the floating-point work the original code does.
# --------------------------------------------------------------------------


def _rotate(qx, qy, qz, qw, vx, vy, vz):
    """
    Rotate a vector by a quaternion exactly as pyrr's ``apply_to_vector`` does.

    pyrr computes ``cross(q, cross(v4, conjugate(q)))`` with ``v4 = (v, 0)``.
    The expressions below are pyrr's ``cross``, term for term and in the same
    order, including the terms multiplied by the zero w of ``v4``.

    :returns: The rotated vector (x, y, z).
    """
    # conjugate(q)
    cx, cy, cz, cw = -qx, -qy, -qz, qw
    # cross(v4, conjugate(q)), v4 = (vx, vy, vz, 0.0)
    vw = 0.0
    ix = vx * cw + vy * cz - vz * cy + vw * cx
    iy = -vx * cz + vy * cw + vz * cx + vw * cy
    iz = vx * cy - vy * cx + vz * cw + vw * cz
    iw = -vx * cx - vy * cy - vz * cz + vw * cw
    # cross(q, inner)
    ox = qx * iw + qy * iz - qz * iy + qw * ix
    oy = -qx * iz + qy * iw + qz * ix + qw * iy
    oz = qx * iy - qy * ix + qz * iw + qw * iz
    return ox, oy, oz


def _rk45_exact(state: np.ndarray, weights: np.ndarray, dt: float) -> np.ndarray:
    """
    ``ball_aware_brain._rk45``, with the same operations in the same order.

    The four ``np.matmul`` calls are kept as they are (they may use BLAS, whose
    summation order is the library's). Only ``np.clip`` is replaced by
    ``np.minimum(np.maximum(...))``, which is how NumPy defines clip for
    floats (NaN propagating), without ``np.clip``'s Python-level overhead.

    :returns: The new CPG state.
    """
    matmul = np.matmul
    half = dt / 2.0
    a1 = matmul(weights, state)
    a2 = matmul(weights, state + half * a1)
    a3 = matmul(weights, state + half * a2)
    a4 = matmul(weights, state + dt * a3)
    new_state = state + dt / 6.0 * (a1 + 2.0 * (a2 + a3) + a4)
    return np.minimum(np.maximum(new_state, -1.0), 1.0)


def _reference_point(px, py, pz, quat, offset):
    """
    The measuring point (core, or head) for a robot pose.

    Same as ``robot_frame.reference_position`` (and the brain's and stop
    condition's versions of it): the core position, or the core position plus
    the offset rotated by the robot's physical orientation.

    :param px: Core x (``np.float64``, as a pose's ``position.x`` is).
    :param py: Core y.
    :param pz: Core z.
    :param quat: The core's MuJoCo quaternion ``xquat`` (w, x, y, z).
    :param offset: Robot-frame offset, ``ZERO`` for the core.
    :returns: The point (x, y, z).
    """
    if offset == ZERO:
        return px, py, pz
    # physical_orientation(): pyrr (x, y, z, w) = MuJoCo (x, y, z, w), via float().
    rx, ry, rz = _rotate(
        float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]),
        float(offset[0]), float(offset[1]), float(offset[2]),
    )
    # px is np.float64, so the sum is too, as a Vector3 element is.
    return px + rx, py + ry, pz + rz


class _Trial:
    """Everything one trial needs, looked up once."""

    def __init__(self, task: dict[str, Any]) -> None:
        from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model

        scene = task["scene"]
        handler = scene.handler
        brain, body_mapping = handler._brains[0]

        self.model, mapping = scene_to_model(
            scene,
            task["simulation_timestep"],
            cast_shadows=task["cast_shadows"],
            fast_sim=task["fast_sim"],
            integrator=task["integrator"],
        )
        self.data = mujoco.MjData(self.model)
        self.mapping = mapping

        def body_id(multi_body_system) -> int:
            return mapping.multi_body_system[UUIDKey(multi_body_system)].id

        self.robot_id = body_id(body_mapping.multi_body_system)
        self.ball_id = body_id(brain._ball)

        # Brain.
        self.state = brain._state.copy()
        self.weights = brain._weight_matrix
        self.steering = brain._steering_matrix
        self.steering_scale = brain._steering_output_scale
        self.distance_scale = brain._distance_scale
        self.brain_offset = tuple(brain._reference_offset)
        self.outputs = []
        position_indices, velocity_indices = [], []
        for output_index, (state_index, active_hinge) in enumerate(brain._output_mapping):
            joint = body_mapping.active_hinge_to_joint_hinge[UUIDKey(active_hinge)]
            joint_mujoco = mapping.hinge_joint[UUIDKey(joint)]
            self.outputs.append(
                (state_index, active_hinge.range, self.steering[output_index])
            )
            position_indices.append(joint_mujoco.ctrl_index_position)
            velocity_indices.append(joint_mujoco.ctrl_index_velocity)
        # Written with one indexed assignment each. The original writes each
        # hinge's position target and a zero velocity target one at a time; no
        # two hinges share a control index, so the result is the same.
        self.position_indices = np.array(position_indices, dtype=np.intp)
        self.velocity_indices = np.array(velocity_indices, dtype=np.intp)
        if len(set(position_indices) | set(velocity_indices)) != 2 * len(position_indices):
            raise ValueError("Hinges share MuJoCo control indices; the fast trial assumes they do not.")
        self.targets = np.empty(len(position_indices))

        # Stop conditions: (robot body, object body, distance, offset).
        self.stops = [
            (
                body_id(stop.robot_multi_body_system),
                body_id(stop.obj),
                stop.distance,
                tuple(stop.reference_offset),
            )
            for stop in handler._stop_conditions
        ]

    # -- control -------------------------------------------------------------

    def control(self, dt: float) -> None:
        """One brain step: ``BallAwareCpgBrainInstance.control``, exactly."""
        xpos = self.data.xpos
        robot = xpos[self.robot_id]
        quat = self.data.xquat[self.robot_id]
        ball = xpos[self.ball_id]

        self.state = _rk45_exact(self.state, self.weights, dt)

        # _ball_relative_inputs: positions are np.float64, as Vector3 elements are.
        rx, ry, rz = robot[0], robot[1], robot[2]
        if self.brain_offset != ZERO:
            rx, ry, rz = _reference_point(rx, ry, rz, quat, self.brain_offset)
            # physical_orientation()
            q = (float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0]))
        else:
            # The reported orientation: MuJoCo's (w, x, y, z) read as (x, y, z, w).
            q = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        # to_x and to_y stay np.float64 (as a Vector3 element is), because the
        # original squares them with np.float64's `**`.
        to_x = ball[0] - rx
        to_y = ball[1] - ry
        distance = math.sqrt(to_x**2 + to_y**2)
        target_angle = math.atan2(to_y, to_x)
        # Plain floats: + - * are exactly rounded, so the type does not matter.
        hx, hy, _ = _rotate(q[0], q[1], q[2], q[3], 1.0, 0.0, 0.0)
        heading_angle = math.atan2(hy, hx)
        angle_error = _wrap_angle(target_angle - heading_angle)
        # np.clip(x, lo, hi) on a scalar is min(max(x, lo), hi), NaN included
        # (Python's max/min return the NaN when it is the first argument).
        inputs = np.array(
            [
                math.sin(angle_error),
                math.cos(angle_error),
                min(max(distance / self.distance_scale, 0.0), 1.0),
                1.0,
            ]
        )

        state = self.state
        targets = self.targets
        scale = self.steering_scale
        dot = np.dot
        tanh = np.tanh
        for k, (state_index, hinge_range, steering_row) in enumerate(self.outputs):
            cpg_target = float(state[state_index]) * hinge_range
            steering_fraction = float(tanh(float(dot(steering_row, inputs)))) * scale
            steering_offset = steering_fraction * hinge_range
            # The brain clips to the hinge range; the control interface clips the
            # result again, which leaves an in-range value unchanged.
            targets[k] = min(max(cpg_target + steering_offset, -hinge_range), hinge_range)
        ctrl = self.data.ctrl
        ctrl[self.position_indices] = targets
        ctrl[self.velocity_indices] = 0.0

    def should_stop(self) -> bool:
        """The "ball reached" stop conditions, exactly as they compute it."""
        xpos = self.data.xpos
        for robot_id, object_id, threshold, offset in self.stops:
            robot = xpos[robot_id]
            rx, ry, _ = _reference_point(
                robot[0], robot[1], robot[2], self.data.xquat[robot_id], offset
            )
            obj = xpos[object_id]
            distance = math.sqrt((rx - obj[0]) ** 2 + (ry - obj[1]) ** 2)
            if distance <= threshold:
                return True
        return False


def _run(trial: _Trial, task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the trial and keep the samples scoring needs.

    Follows ``simulate_scene``'s loop exactly: the same initial sample, the
    same control and sampling rules evaluated on the same ``data.time`` values,
    the state appended when the stop condition fires, and the final sample.

    :returns: Robot positions (n, 3), robot quaternions (n, 4) and ball
        positions (n, 3), one row per sampled state.
    """
    model, data = trial.model, trial.data
    control_step = task["control_step"]
    sample_step = task["sample_step"]
    end = float("inf") if task["simulation_time"] is None else task["simulation_time"]
    timestep = model.opt.timestep

    robot_pos, robot_quat, ball_pos = [], [], []

    def sample() -> None:
        robot_pos.append(data.xpos[trial.robot_id].copy())
        robot_quat.append(data.xquat[trial.robot_id].copy())
        ball_pos.append(data.xpos[trial.ball_id].copy())

    mujoco.mj_forward(model, data)
    sample()

    last_control_time = 0.0
    last_sample_time = 0.0
    time = data.time
    while time < end:
        if time >= last_control_time + control_step:
            last_control_time = math.floor(time / control_step) * control_step
            trial.control(control_step)
            if trial.should_stop():
                sample()
                break
        if time >= last_sample_time + sample_step:
            last_sample_time = int(time / sample_step) * sample_step
            sample()

        # Steps until the loop would next do anything: the original checks
        # these conditions before every step, on the time MuJoCo reports.
        next_control = last_control_time + control_step
        next_sample = last_sample_time + sample_step
        predicted = time
        steps = 0
        while True:
            predicted = predicted + timestep
            steps += 1
            if not predicted < end or predicted >= next_control or predicted >= next_sample:
                break
        mujoco.mj_step(model, data, nstep=steps)
        time = data.time
        if time != predicted:
            raise RuntimeError(
                f"MuJoCo's time {time!r} differs from the predicted {predicted!r}; "
                "the fast trial cannot reproduce the original step timing. Set "
                "REVOLVE2_FAST_TRIAL=0 and report this."
            )

    sample()
    return np.array(robot_pos), np.array(robot_quat), np.array(ball_pos)


def _score(
    robot_pos: np.ndarray,
    robot_quat: np.ndarray,
    ball_pos: np.ndarray,
    task: dict[str, Any],
) -> tuple[float, dict[str, float] | None]:
    """
    ``evaluator._trial_fitness`` and ``robot_frame.movement_report``, exactly.

    :returns: The trial fitness and, if the task asks for it, the movement record.
    """
    import evaluator

    offset = robot_frame.reference_offset()
    num = len(robot_pos)
    points = [
        _reference_point(robot_pos[i, 0], robot_pos[i, 1], robot_pos[i, 2], robot_quat[i], offset)
        for i in range(num)
    ]
    distances = [
        math.sqrt((points[i][0] - ball_pos[i, 0]) ** 2 + (points[i][1] - ball_pos[i, 1]) ** 2)
        for i in range(num)
    ]

    progress = evaluator._signed_distance_progress(distances[0], min(distances))

    reached_bonus, time_bonus = 0.0, 0.0
    simulation_time = config.SIMULATION_TIME
    for index, distance in enumerate(distances):
        if distance <= config.BALL_REACHED_DISTANCE:
            reach_time = evaluator._sample_index_to_time(
                index=index,
                num_samples=num,
                sampling_frequency=task["sampling_frequency"],
                simulation_time=simulation_time,
            )
            if simulation_time <= 0.0:
                reached_bonus, time_bonus = 1.0, 1.0
            else:
                reached_bonus = 1.0
                time_bonus = max(0.0, min(1.0, 1.0 - reach_time / simulation_time))
            break

    alignments = []
    epsilon = getattr(config, "NO_PROGRESS_EPSILON", 1e-6)
    for i in range(num - 1):
        previous, current = points[i], points[i + 1]
        move_x = current[0] - previous[0]
        move_y = current[1] - previous[1]
        target_x = ball_pos[i, 0] - previous[0]
        target_y = ball_pos[i, 1] - previous[1]
        move_norm = math.sqrt(move_x**2 + move_y**2)
        target_norm = math.sqrt(target_x**2 + target_y**2)
        if move_norm <= epsilon or target_norm <= epsilon:
            continue
        alignments.append((move_x * target_x + move_y * target_y) / (move_norm * target_norm))
    alignment = 0.0 if len(alignments) == 0 else float(np.mean(alignments))

    fitness = (
        getattr(config, "FITNESS_PROGRESS_WEIGHT", 1.0) * progress
        + getattr(config, "FITNESS_REACHED_BONUS_WEIGHT", 0.20) * reached_bonus
        + getattr(config, "FITNESS_TIME_TO_REACH_WEIGHT", 0.10) * time_bonus
        + getattr(config, "FITNESS_ALIGNMENT_WEIGHT", 0.05) * alignment
    )

    movement = None
    if task["with_movement"]:
        movement = robot_frame.movement_report_from_arrays(
            position=robot_pos,
            reported=robot_quat,
            ball_xy=(ball_pos[-1, 0], ball_pos[-1, 1]),
            offset=offset,
        )
    return fitness, movement


def simulate_and_score(task: dict[str, Any]) -> tuple[float, dict[str, float] | None]:
    """
    Simulate and score one trial; the same answer as the original worker path.

    :param task: A worker task for which ``supports(task)`` is True.
    :returns: The trial fitness and, if requested, its movement record.
    """
    trial = _Trial(task)
    robot_pos, robot_quat, ball_pos = _run(trial, task)
    return _score(robot_pos, robot_quat, ball_pos, task)
