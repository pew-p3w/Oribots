"""Where a robot's front is, and the point on it that the task measures from.

A robot's front is the +x axis of its core. A body such as the spider is
symmetric, so that front is invisible. A *head* is a small passive block bolted
to the core on the +x axis: it makes the front visible, and it is the point the
task measures from (distance to the ball, reaching the ball, movement).

The heading and the head position are computed from the robot's TRUE orientation
(``physical_orientation``, in the engine's ``_modular_robot_scene.py``). The
orientation the simulator reports is a permuted version of it (see
``engine/PATCHES.md``), so it cannot be used to find the front directly.

Everything that needs that point reads it from here, so the brain, the fitness
function, the "ball reached" stop condition and the movement report cannot
disagree.

A config turns it on by defining::

    HEAD_OFFSET = robot_frame.head_offset_of(BODY, HEAD)   # robot-frame offset
    USE_HEAD_COORDINATES = True

Without ``HEAD_OFFSET`` everything measures from the core. With it but
``USE_HEAD_COORDINATES = False`` the head is only a visible marker and nothing is
measured from it.

``tests/test_head.py`` checks this module against reference values recorded
from real simulation states.
"""

import math
from typing import Any

import numpy as np
from pyrr import Vector3

from revolve2.modular_robot_simulation._modular_robot_scene import physical_orientation


ZERO_OFFSET = (0.0, 0.0, 0.0)


def head_offset_of(body: Any, head: Any) -> tuple[float, float, float]:
    """
    Find where a head module sits, in the robot's own frame.

    A passive brick on the core is a box inside the core's rigid body, so its
    position relative to the core is fixed. It is read from the geometry the
    simulator actually builds, not calculated separately.

    :param body: The robot body.
    :param head: The head module (a brick attached directly to the core).
    :returns: The head's centre as (x, y, z) in the core's frame, in metres.
    :raises ValueError: If the head cannot be found, or does not sit on the
        heading axis (+x), where "toward the head" would not be "forward".
    """
    from revolve2.modular_robot_simulation._build_multi_body_systems import (
        BodyToMultiBodySystemConverter,
    )
    from revolve2.simulation.scene import Pose

    multi_body_system, _ = BodyToMultiBodySystemConverter().convert_robot_body(
        body, Pose(), False
    )
    matches = [
        geometry
        for geometry in multi_body_system.root.geometries
        if abs(geometry.mass - head.mass) < 1e-9
        and np.allclose(geometry.aabb.size, head.bounding_box)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one head block in the core's body, found "
            f"{len(matches)}. The head must be a brick attached directly to the core."
        )
    position = matches[0].pose.position
    offset = (float(position.x), float(position.y), float(position.z))
    if offset[0] <= 0.0 or abs(offset[1]) > 1e-3:
        raise ValueError(
            f"The head is at {offset}, not on the +x heading axis. The controller "
            "steers relative to +x, so a head anywhere else would point somewhere "
            "other than where the robot is steering."
        )
    return offset


def reference_offset() -> tuple[float, float, float]:
    """
    Get the offset of the point the task measures from, in the robot's frame.

    :returns: The head offset if the config uses head coordinates, otherwise
        (0, 0, 0), which is the core.
    """
    import config

    if getattr(config, "USE_HEAD_COORDINATES", False):
        return tuple(config.HEAD_OFFSET)
    return ZERO_OFFSET


def reference_position(pose: Any, offset: tuple[float, float, float]) -> Vector3:
    """
    Get the world position of the measuring point for a robot pose.

    :param pose: The robot's pose (its core).
    :param offset: Offset of the measuring point in the robot's own frame.
    :returns: The measuring point's world position.
    """
    if offset == ZERO_OFFSET:
        return pose.position
    return pose.position + physical_orientation(pose) * Vector3(offset)


def heading_direction(pose: Any) -> Vector3:
    """
    Get the direction the robot is pointing (its front) in the world.

    :param pose: The robot's pose.
    :returns: A unit vector along the robot's +x axis (toward the head).
    """
    return physical_orientation(pose) * Vector3([1.0, 0.0, 0.0])


def make_stop_condition(robot: Any, ball: Any) -> Any:
    """
    Make the "ball reached" stop condition, measured from the same point as the
    fitness function so the two always agree on whether the ball was reached.

    :param robot: The robot.
    :param ball: The ball.
    :returns: The stop condition to add to the scene.
    """
    import config
    from revolve2.modular_robot_simulation import StopOnRobotObjectDistance

    return StopOnRobotObjectDistance(
        robot=robot,
        obj=ball,
        distance=config.BALL_REACHED_DISTANCE,
        reference_offset=reference_offset(),
    )


def _rotate(quaternions_xyzw: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    """
    Rotate one fixed vector by many quaternions at once.

    :param quaternions_xyzw: Array (n, 4) of unit quaternions in (x, y, z, w) order.
    :param vector: The vector to rotate.
    :returns: Array (n, 3) of rotated vectors.
    """
    q = quaternions_xyzw[:, :3]
    w = quaternions_xyzw[:, 3:4]
    v = np.asarray(vector, dtype=float)
    t = 2.0 * np.cross(q, v)
    return v + w * t + np.cross(q, t)


def movement_report(scene_states: list[Any], robot: Any, ball: Any) -> dict[str, float]:
    """
    Describe how the robot moved relative to where it was pointing.

    Each step of the measuring point is split into the part along the robot's
    heading (forward, or backward if negative) and the part across it. This reads
    every pose once and does the arithmetic in numpy, because it runs for every
    trial of every individual during training.

    :param scene_states: Sampled scene states of one trial.
    :param robot: The robot.
    :param ball: The ball.
    :returns: Distances in metres and angles in degrees.
    """
    offset = reference_offset()
    poses = [
        state.get_modular_robot_simulation_state(robot).get_pose()
        for state in scene_states
    ]
    position = np.array([[p.position.x, p.position.y, p.position.z] for p in poses])
    # The simulator reports the orientation as (w, x, y, z) read as (x, y, z, w);
    # physical_orientation() undoes that, and so does this reordering.
    reported = np.array([[float(v) for v in p.orientation] for p in poses])
    physical = reported[:, [1, 2, 3, 0]]
    heading = _rotate(physical, (1.0, 0.0, 0.0))
    point = position if offset == ZERO_OFFSET else position + _rotate(physical, offset)

    step = point[1:, :2] - point[:-1, :2]
    heading_xy = heading[:-1, :2]
    norm = np.hypot(heading_xy[:, 0], heading_xy[:, 1])
    usable = norm > 0.0
    unit = heading_xy[usable] / norm[usable, None]
    step = step[usable]
    along = step[:, 0] * unit[:, 0] + step[:, 1] * unit[:, 1]
    across = -step[:, 0] * unit[:, 1] + step[:, 1] * unit[:, 0]
    forward = float(along[along >= 0.0].sum())
    backward = float(-along[along < 0.0].sum())
    sideways = float(np.abs(across).sum())

    def yaw_degrees(index: int) -> float:
        return math.degrees(math.atan2(heading[index, 1], heading[index, 0]))

    ball_position = (
        scene_states[-1]._simulation_state.get_multi_body_system_pose(ball).position
    )
    to_ball = math.degrees(
        math.atan2(ball_position.y - point[-1, 1], ball_position.x - point[-1, 0])
    )
    end_yaw = yaw_degrees(-1)
    angle_to_ball = (to_ball - end_yaw + 180.0) % 360.0 - 180.0
    return {
        "forward_m": forward,
        "backward_m": backward,
        "sideways_m": sideways,
        "net_along_heading_m": forward - backward,
        "start_heading_deg": yaw_degrees(0),
        "end_heading_deg": end_yaw,
        "end_angle_to_ball_deg": angle_to_ball,
    }
