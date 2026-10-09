"""The walled arena and the robots' start layouts.

Arena (R6): a flat floor of ``terrain_size`` centred on (0, 0) with its surface at
height 0, enclosed by four static box walls. Each wall's inner face lies on a
floor edge, so the area inside the walls is exactly ``terrain_size`` (D-5). The
two walls at x = +-size.x/2 run the full y extent plus both wall thicknesses, so
the four corners are closed. ``revolve2.ci_group.terrains.flat`` is not used or
changed.

Layouts (R7): the ball at rest at (0, 0, ball_radius) and K robot cores on a
ring of ``start_radius`` around it, each with its own ring angle and its own
yaw. Ring angles are drawn uniformly from [-pi, pi) and the whole set is redrawn
until every pair is at least ``min_arc_gap`` apart along the ring and at least
two footprint radii apart in a straight line; yaws are drawn uniformly from
[-pi, pi), independently. Every robot's head must start farther than the
possession distance from the ball. All draws come from one
``numpy.random.default_rng(seed)``, so a seed reproduces its layouts bit for bit.

A run stores its layouts in every snapshot as plain lists (``layouts_to_plain``)
so a snapshot loads without importing this module (R17.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()
from pyrr import Quaternion, Vector3

from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import AABB, Color, Pose
from revolve2.simulation.scene.geometry import GeometryBox, GeometryPlane
from revolve2.simulation.scene.geometry.textures import Texture
from revolve2.simulation.scene.vector2 import Vector2

# Wall colour: a light grey that reads clearly against the darker floor.
WALL_COLOR = Color(170, 170, 180, 255)

# How many full redraws a layout may take before the sampler gives up. The
# pre-flight checks (feasibility_problems) rule out impossible settings, so this
# only bounds pathological-but-feasible cases.
MAX_LAYOUT_ATTEMPTS = 100_000


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #


def build_walled_terrain(
    terrain_size: tuple[float, float], wall_height: float, wall_thickness: float
) -> Terrain:
    """
    Build the arena: a flat floor plus four static box walls (R6.1, R6.2).

    :param terrain_size: Floor size (x, y) in metres; the walls' inner faces lie on its edges.
    :param wall_height: Wall height in metres, from the floor up.
    :param wall_thickness: Wall thickness in metres, outward from the inner face.
    :returns: The terrain.
    """
    size_x, size_y = float(terrain_size[0]), float(terrain_size[1])
    height, thickness = float(wall_height), float(wall_thickness)
    texture = Texture(base_color=WALL_COLOR)
    floor = GeometryPlane(pose=Pose(), mass=0.0, size=Vector2([size_x, size_y]))
    walls = []
    for centre, extent in wall_boxes(size_x, size_y, height, thickness):
        walls.append(
            GeometryBox(
                pose=Pose(Vector3(list(centre))),
                mass=0.0,
                texture=texture,
                aabb=AABB(Vector3(list(extent))),
            )
        )
    return Terrain(static_geometry=[floor, *walls])


def wall_boxes(
    size_x: float, size_y: float, height: float, thickness: float
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """
    The four wall boxes as (centre, full extent) pairs, in the order +x, -x, +y, -y.

    :param size_x: Floor size along x.
    :param size_y: Floor size along y.
    :param height: Wall height.
    :param thickness: Wall thickness.
    :returns: Four (centre, extent) pairs.
    """
    half_x, half_y = size_x / 2.0, size_y / 2.0
    z = height / 2.0
    # x-walls span y including both corner squares, so the corners are closed.
    x_extent = (thickness, size_y + 2.0 * thickness, height)
    y_extent = (size_x, thickness, height)
    return [
        ((half_x + thickness / 2.0, 0.0, z), x_extent),
        ((-(half_x + thickness / 2.0), 0.0, z), x_extent),
        ((0.0, half_y + thickness / 2.0, z), y_extent),
        ((0.0, -(half_y + thickness / 2.0), z), y_extent),
    ]


def _t_pose_aabb(body: Any) -> tuple[Vector3, Any]:
    """
    The body's T-pose axis-aligned bounding box in its own frame (core at the origin).

    The same box the engine uses to lift a robot onto the floor.

    :param body: The robot body.
    :returns: The box centre and the AABB.
    """
    from revolve2.modular_robot_simulation._build_multi_body_systems import (
        BodyToMultiBodySystemConverter,
    )

    multi_body_system, _ = BodyToMultiBodySystemConverter().convert_robot_body(
        body, Pose(), False
    )
    return multi_body_system.calculate_aabb()


def footprint_radius(body: Any) -> float:
    """
    r_f: the largest horizontal distance from the core centre to a corner of the
    body's T-pose bounding box, so a disc of this radius covers the body at any yaw (R7.4).

    :param body: The robot body.
    :returns: The footprint radius in metres.
    """
    centre, aabb = _t_pose_aabb(body)
    half_x, half_y = aabb.size.x / 2.0, aabb.size.y / 2.0
    return max(
        math.hypot(centre.x + sx * half_x, centre.y + sy * half_y)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    )


def standing_height(body: Any) -> float:
    """
    The height of the body's highest point when it stands on the floor (R6.5).

    A robot is spawned with its T-pose bounding box resting on the floor, so this
    is the box's height.

    :param body: The robot body.
    :returns: The standing height in metres.
    """
    _, aabb = _t_pose_aabb(body)
    return float(aabb.size.z)


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RobotStart:
    """Where one robot starts: core position on the floor and yaw about the vertical."""

    xy: tuple[float, float]
    yaw: float


@dataclass(frozen=True)
class Layout:
    """One start configuration: the K robots' starts (the ball is always at the centre)."""

    starts: tuple[RobotStart, ...]


def ball_start_position(ball_radius: float) -> tuple[float, float, float]:
    """
    The ball's start position: at rest on the floor at the centre (R7.1).

    :param ball_radius: Ball radius.
    :returns: (0, 0, ball_radius).
    """
    return (0.0, 0.0, float(ball_radius))


def spawn_orientation(yaw: float) -> Quaternion:
    """
    The orientation to hand the engine so a robot spawns at ``yaw`` about the vertical.

    The engine writes a multi-body system's orientation into the MuJoCo model as
    ``[*pose.orientation]`` (``_scene_to_model.py``), i.e. in pyrr's (x, y, z, w)
    order, while MuJoCo reads a quaternion as (w, x, y, z). It is the same
    permutation ``physical_orientation`` undoes on the way out. A plain yaw
    quaternion therefore spawns the robot tilted or upside down. Passing the
    components pre-permuted, (cos(yaw/2), 0, 0, sin(yaw/2)) in pyrr's slots, makes
    MuJoCo receive w = cos(yaw/2), z = sin(yaw/2): an upright robot at ``yaw``.
    Single-robot runs never hit this because they spawn at the identity.
    The engine is vendored and not edited (R1), so the fix lives here.

    :param yaw: Yaw in radians.
    :returns: The quaternion to put in the spawn pose.
    """
    half = float(yaw) / 2.0
    return Quaternion([math.cos(half), 0.0, 0.0, math.sin(half)])


def start_pose(start: RobotStart) -> Pose:
    """
    The pose a robot is added to the scene with (R7.11). The engine then lifts it
    so its bounding box rests on the floor (a yaw does not change that lift).

    :param start: The robot's start.
    :returns: The pose (z = 0, yaw-only orientation in the engine's input order).
    """
    return Pose(
        Vector3([float(start.xy[0]), float(start.xy[1]), 0.0]),
        spawn_orientation(start.yaw),
    )


def head_xy(start: RobotStart, head_offset: tuple[float, float, float]) -> tuple[float, float]:
    """
    The head reference point's floor position for a robot at its start pose.

    :param start: The robot's start.
    :param head_offset: The head offset in the robot frame.
    :returns: The head's (x, y).
    """
    cos_yaw, sin_yaw = math.cos(start.yaw), math.sin(start.yaw)
    ox, oy = float(head_offset[0]), float(head_offset[1])
    return (
        start.xy[0] + cos_yaw * ox - sin_yaw * oy,
        start.xy[1] + sin_yaw * ox + cos_yaw * oy,
    )


def _arc_gap(angle_a: float, angle_b: float, radius: float) -> float:
    """
    Arc length along the shorter arc between two ring angles.

    :param angle_a: First angle.
    :param angle_b: Second angle.
    :param radius: Ring radius.
    :returns: The arc length.
    """
    difference = abs(angle_a - angle_b) % (2.0 * math.pi)
    return radius * min(difference, 2.0 * math.pi - difference)


def feasibility_problems(
    *,
    k: int,
    terrain_size: tuple[float, float],
    start_radius: float,
    min_arc_gap: float,
    footprint: float,
    ball_radius: float,
    head_offset: tuple[float, float, float],
    possession_distance: float,
) -> list[str]:
    """
    The pre-flight layout checks of R7.6 and R7.10, run before any sampling.

    :returns: One message per violated condition, naming the conflicting values;
        empty when the layout settings are feasible.
    """
    problems = []
    half_short_side = min(float(terrain_size[0]), float(terrain_size[1])) / 2.0
    if start_radius + footprint > half_short_side:
        problems.append(
            f"START_RADIUS ({start_radius}) + footprint radius r_f ({footprint:.4f}) = "
            f"{start_radius + footprint:.4f} m exceeds half the shorter TERRAIN_SIZE side "
            f"({half_short_side}): a robot could start through a wall"
        )
    circumference = 2.0 * math.pi * start_radius
    if k * min_arc_gap >= circumference:
        problems.append(
            f"K ({k}) x MIN_ARC_GAP ({min_arc_gap}) = {k * min_arc_gap:.4f} m is not less "
            f"than the ring circumference 2*pi*START_RADIUS = {circumference:.4f} m"
        )
    if start_radius > 0:
        chord = 2.0 * start_radius * math.sin(min(min_arc_gap / (2.0 * start_radius), math.pi / 2.0))
        if chord < 2.0 * footprint:
            problems.append(
                f"two robots MIN_ARC_GAP ({min_arc_gap}) apart on a ring of START_RADIUS "
                f"({start_radius}) are only {chord:.4f} m apart, less than 2 x r_f "
                f"({2.0 * footprint:.4f} m): their footprints could overlap"
            )
    if start_radius - footprint <= ball_radius:
        problems.append(
            f"START_RADIUS ({start_radius}) - r_f ({footprint:.4f}) is not greater than "
            f"BALL_RADIUS ({ball_radius}): a robot could start on the ball"
        )
    head_reach = math.hypot(float(head_offset[0]), float(head_offset[1]))
    if start_radius - head_reach <= possession_distance:
        problems.append(
            f"START_RADIUS ({start_radius}) - head offset length ({head_reach:.4f}) is not "
            f"greater than POSSESSION_DISTANCE ({possession_distance}): a robot could "
            "start holding the ball"
        )
    return problems


def sample_layouts(
    *,
    k: int,
    num_layouts: int,
    seed: int,
    start_radius: float,
    min_arc_gap: float,
    footprint: float,
    head_offset: tuple[float, float, float],
    possession_distance: float,
) -> list[Layout]:
    """
    Sample a run's layouts (R7.2-7.5, R7.7).

    :returns: ``num_layouts`` layouts, bit-identical for the same arguments.
    :raises RuntimeError: If a layout cannot be found within the attempt limit.
    """
    rng = np.random.default_rng(int(seed))
    layouts = []
    for layout_index in range(int(num_layouts)):
        for _ in range(MAX_LAYOUT_ATTEMPTS):
            angles = rng.uniform(-math.pi, math.pi, size=k)
            yaws = rng.uniform(-math.pi, math.pi, size=k)
            starts = tuple(
                RobotStart(
                    xy=(
                        float(start_radius * math.cos(angle)),
                        float(start_radius * math.sin(angle)),
                    ),
                    yaw=float(yaw),
                )
                for angle, yaw in zip(angles, yaws)
            )
            if _layout_ok(starts, angles, start_radius, min_arc_gap, footprint, head_offset, possession_distance):
                layouts.append(Layout(starts=starts))
                break
        else:
            raise RuntimeError(
                f"Could not sample layout {layout_index} in {MAX_LAYOUT_ATTEMPTS} attempts "
                f"(K={k}, START_RADIUS={start_radius}, MIN_ARC_GAP={min_arc_gap}, r_f={footprint:.4f})."
            )
    return layouts


def _layout_ok(
    starts: tuple[RobotStart, ...],
    angles: np.ndarray,
    start_radius: float,
    min_arc_gap: float,
    footprint: float,
    head_offset: tuple[float, float, float],
    possession_distance: float,
) -> bool:
    """
    Whether a candidate layout meets R7.2, R7.4 and R7.5.

    :returns: True if every pair is far enough apart and every head is clear of the ball.
    """
    count = len(starts)
    for a in range(count):
        for b in range(a + 1, count):
            if _arc_gap(float(angles[a]), float(angles[b]), start_radius) < min_arc_gap:
                return False
            ax, ay = starts[a].xy
            bx, by = starts[b].xy
            if math.hypot(ax - bx, ay - by) < 2.0 * footprint:
                return False
    for start in starts:
        hx, hy = head_xy(start, head_offset)
        if math.hypot(hx, hy) <= possession_distance:
            return False
    return True


def layouts_to_plain(layouts: list[Layout]) -> list[list[list[float]]]:
    """
    Layouts as plain nested lists [[x, y, yaw], ...] for a snapshot (R17.7).

    :param layouts: The layouts.
    :returns: One list of [x, y, yaw] per layout.
    """
    return [[[s.xy[0], s.xy[1], s.yaw] for s in layout.starts] for layout in layouts]


def layouts_from_plain(plain: list[list[list[float]]]) -> list[Layout]:
    """
    Rebuild layouts stored by :func:`layouts_to_plain`, bit for bit.

    :param plain: The stored lists.
    :returns: The layouts.
    """
    return [
        Layout(starts=tuple(RobotStart(xy=(float(x), float(y)), yaw=float(yaw)) for x, y, yaw in layout))
        for layout in plain
    ]
