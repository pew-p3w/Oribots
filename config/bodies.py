"""The robot body catalogue.

Each function builds one robot body out of v2 modules. Configs pick a body from
here, for example `from bodies import spider_v2`.

These are the classic Revolve2 robot designs built with v2 modules
(`ActiveHingeV2`, `BrickV2`, `BodyV2`), plus `simple_v2` and the pentagon
spider. They are deliberately kept here rather than inside `engine/`, so the
vendored Revolve2 source stays untouched and can be updated without colliding
with this catalogue.

`tests/test_bodies.py` checks every body in here against the reference
structures in `tests/body_reference.json`. A change to a body's structure fails
that test.

This file is imported as a plain top-level module named `bodies`, not as
`config.bodies`: the runner registers the chosen config file as the module
`config`, which would shadow the package name.
"""

import math

import numpy as np
from pyrr import Quaternion, Vector3

from revolve2.modular_robot.body import AttachmentPoint, Color, RightAngles
from revolve2.modular_robot.body.base import Body, Core
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BodyV2, BrickV2
from revolve2.modular_robot.body.v2._attachment_face_core_v2 import (
    AttachmentFaceCoreV2,
)
from revolve2.simulation.scene import AABB, Pose
from revolve2.simulation.scene.geometry import GeometryBox


def all_bodies() -> list[BodyV2]:
    """
    Get a list of all robot bodies in this catalogue.

    Named `all_bodies` rather than `all`, so it cannot shadow the `all()`
    builtin when a config does `from bodies import ...`.

    :returns: The list of robots.
    """
    return [
        babya_v2(),
        babyb_v2(),
        blokky_v2(),
        garrix_v2(),
        gecko_v2(),
        insect_v2(),
        linkin_v2(),
        longleg_v2(),
        penguin_v2(),
        pentapod_v2(),
        queen_v2(),
        salamander_v2(),
        squarish_v2(),
        snake_v2(),
        spider_v2(),
        stingray_v2(),
        tinlicker_v2(),
        turtle_v2(),
        ww_v2(),
        zappa_v2(),
        ant_v2(),
        park_v2(),
        simple_v2(),
    ]


def simple_v2() -> BodyV2:
    """
    Create the small two-legged body used for quick experiments.

    Each side carries two hinges in series ending in a brick, which is the
    fewest moving parts that can still produce a gait. With four active hinges
    it evolves 22 controller parameters, making it the fastest body to train.

    :returns: The created body.
    """
    body = BodyV2()
    body.core_v2.left_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.left_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(RightAngles.DEG_0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(RightAngles.DEG_0)
    return body


def spider_v2() -> BodyV2:
    """
    Get the spider modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(0.0)

    return body


def gecko_v2() -> BodyV2:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body


def babya_v2() -> BodyV2:
    """
    Get the babya modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body


def ant_v2() -> BodyV2:
    """
    Get the ant modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.right.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.attachment.right.attachment = BrickV2(0.0)

    return body


def salamander_v2() -> BodyV2:
    """
    Get the salamander modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment = BrickV2(-np.pi / 2.0)

    body.core_v2.back_face.bottom.attachment.front.front.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment.left = BrickV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment.front = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.left.attachment.front.attachment = ActiveHingeV2(
        -np.pi / 2.0
    )

    body.core_v2.back_face.bottom.attachment.front.front.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.left = (
        ActiveHingeV2(0.0)
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front.attachment = BrickV2(
        -np.pi / 2.0
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front.attachment.left = BrickV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.front.front.attachment.front.front.front.front.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def blokky_v2() -> BodyV2:
    """
    Get the blokky modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom = BrickV2(0.0)
    body.core_v2.back_face.bottom.right = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.front.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.front.attachment.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front.right.left = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.front.right.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right.front.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.front.attachment.attachment.right.front.front = ActiveHingeV2(0.0)

    return body


def park_v2() -> BodyV2:
    """
    Get the park modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.right = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.right = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.front = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.right = ActiveHingeV2(
        -np.pi / 2.0
    )
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.left = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.front = ActiveHingeV2(
        0.0
    )
    body.core_v2.back_face.bottom.attachment.attachment.front.left.attachment.front.attachment = (
        BrickV2(0.0)
    )

    return body


def babyb_v2() -> BodyV2:
    """
    Get the babyb modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.front_face.bottom.attachment.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.front_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(-np.pi / 2.0)

    return body


def garrix_v2() -> BodyV2:
    """
    Get the garrix modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.front_face.bottom = ActiveHingeV2(np.pi / 2.0)

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.left = ActiveHingeV2(0.0)

    part2 = BrickV2(0.0)
    part2.right = ActiveHingeV2(np.pi / 2.0)
    part2.front = ActiveHingeV2(np.pi / 2.0)
    part2.left = ActiveHingeV2(0.0)
    part2.left.attachment = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.left.attachment.attachment.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom.attachment.attachment.attachment.left.attachment = part2

    return body


def insect_v2() -> BodyV2:
    """
    Get the insect modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.right.attachment = (
        ActiveHingeV2(0.0)
    )
    body.core_v2.right_face.bottom.attachment.attachment.left.attachment.right.attachment.attachment = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def linkin_v2() -> BodyV2:
    """
    Get the linkin modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = BrickV2(0.0)

    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment
    part2.front = BrickV2(0.0)

    part2.left = ActiveHingeV2(0.0)
    part2.left.attachment = ActiveHingeV2(0.0)

    part2.right = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.right.attachment.attachment = ActiveHingeV2(0.0)
    part2.right.attachment.attachment.attachment = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment.attachment.attachment.attachment = ActiveHingeV2(0.0)

    return body


def longleg_v2() -> BodyV2:
    """
    Get the longleg modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.attachment.attachment.attachment.attachment = BrickV2(
        0.0
    )

    part2 = body.core_v2.left_face.bottom.attachment.attachment.attachment.attachment.attachment
    part2.right = ActiveHingeV2(0.0)
    part2.front = ActiveHingeV2(0.0)
    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.left.attachment.attachment = BrickV2(0.0)
    part2.left.attachment.attachment.right = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment.attachment.left.attachment = ActiveHingeV2(0.0)

    return body


def penguin_v2() -> BodyV2:
    """
    Get the penguin modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = BrickV2(0.0)
    body.core_v2.right_face.bottom.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.left.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.left.attachment.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.left.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment = ActiveHingeV2(
        -np.pi / 2.0
    )
    body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment.attachment = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment.attachment.attachment = BrickV2(
        -np.pi / 2.0
    )

    part2 = (
        body.core_v2.right_face.bottom.left.attachment.attachment.left.attachment.attachment.attachment
    )

    part2.front = ActiveHingeV2(np.pi / 2.0)
    part2.front.attachment = BrickV2(-np.pi / 2.0)

    part2.right = ActiveHingeV2(0.0)
    part2.right.attachment = ActiveHingeV2(0.0)
    part2.right.attachment.attachment = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment.attachment.attachment = BrickV2(-np.pi / 2.0)

    part2.right.attachment.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)

    part2.right.attachment.attachment.attachment.right = BrickV2(0.0)
    part2.right.attachment.attachment.attachment.right.front = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def pentapod_v2() -> BodyV2:
    """
    Get the pentapod modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment

    part2.left = ActiveHingeV2(0.0)
    part2.front = ActiveHingeV2(np.pi / 2.0)
    part2.front.attachment = BrickV2(-np.pi / 2.0)
    part2.front.attachment.left = BrickV2(0.0)
    part2.front.attachment.right = ActiveHingeV2(0.0)
    part2.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    part2.front.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    part2.front.attachment.front.attachment.left = ActiveHingeV2(0.0)
    part2.front.attachment.front.attachment.right = ActiveHingeV2(0.0)

    return body


def queen_v2() -> BodyV2:
    """
    Get the queen modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment

    part2.left = ActiveHingeV2(0.0)
    part2.right = BrickV2(0.0)
    part2.right.front = BrickV2(0.0)
    part2.right.front.left = ActiveHingeV2(0.0)
    part2.right.front.right = ActiveHingeV2(0.0)

    part2.right.right = BrickV2(0.0)
    part2.right.right.front = ActiveHingeV2(np.pi / 2.0)
    part2.right.right.front.attachment = ActiveHingeV2(0.0)

    return body


def squarish_v2() -> BodyV2:
    """
    Get the squarish modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment = BrickV2(0.0)
    body.core_v2.back_face.bottom.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.back_face.bottom.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.back_face.bottom.attachment.left.attachment.left = BrickV2(0.0)
    part2 = body.core_v2.back_face.bottom.attachment.left.attachment.left

    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.front = ActiveHingeV2(0.0)
    part2.right = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment = BrickV2(-np.pi / 2.0)
    part2.right.attachment.left = BrickV2(0.0)
    part2.right.attachment.left.left = BrickV2(0.0)

    return body


def snake_v2() -> BodyV2:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front.attachment = BrickV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment = BrickV2(0.0)
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment = (
        BrickV2(-np.pi / 2.0)
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickV2(
        -np.pi / 2.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickV2(
        0.0
    )
    body.core_v2.left_face.bottom.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = ActiveHingeV2(
        np.pi / 2.0
    )

    return body


def stingray_v2() -> BodyV2:
    """
    Get the stingray modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.right = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.right = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.front = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.left = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment.right = (
        ActiveHingeV2(np.pi / 2.0)
    )
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment.front = (
        ActiveHingeV2(0.0)
    )
    body.core_v2.right_face.bottom.attachment.attachment.front.left.attachment.front.attachment = (
        BrickV2(0.0)
    )

    return body


def tinlicker_v2() -> BodyV2:
    """
    Get the tinlicker modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment

    part2.left = BrickV2(0.0)
    part2.left.front = ActiveHingeV2(np.pi / 2.0)
    part2.left.right = BrickV2(0.0)
    part2.left.right.left = BrickV2(0.0)
    part2.left.right.front = ActiveHingeV2(0.0)
    part2.left.right.front.attachment = BrickV2(0.0)
    part2.left.right.front.attachment.front = ActiveHingeV2(np.pi / 2.0)
    part2.left.right.front.attachment.right = BrickV2(0.0)
    part2.left.right.front.attachment.right.right = ActiveHingeV2(np.pi / 2.0)

    return body


def turtle_v2() -> BodyV2:
    """
    Get the turtle modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.left_face.bottom = BrickV2(0.0)
    body.core_v2.left_face.bottom.right = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.left_face.bottom.left.attachment.attachment = BrickV2(0.0)

    body.core_v2.left_face.bottom.left.attachment.attachment.front = BrickV2(0.0)
    body.core_v2.left_face.bottom.left.attachment.attachment.left = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.left_face.bottom.left.attachment.attachment.right = ActiveHingeV2(0.0)
    body.core_v2.left_face.bottom.left.attachment.attachment.right.attachment = BrickV2(0.0)
    part2 = body.core_v2.left_face.bottom.left.attachment.attachment.right.attachment

    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.front = BrickV2(0.0)
    part2.right = ActiveHingeV2(0.0)
    part2.right.attachment = BrickV2(0.0)
    part2.right.attachment.right = ActiveHingeV2(0.0)
    part2.right.attachment.left = ActiveHingeV2(np.pi / 2.0)
    part2.right.attachment.left.attachment = ActiveHingeV2(-np.pi / 2.0)
    part2.right.attachment.left.attachment.attachment = ActiveHingeV2(0.0)
    part2.right.attachment.left.attachment.attachment.attachment = ActiveHingeV2(0.0)

    return body


def ww_v2() -> BodyV2:
    """
    Get the ww modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = BrickV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.left = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.left.attachment = BrickV2(0.0)
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.left.attachment

    part2.left = ActiveHingeV2(0.0)
    part2.front = BrickV2(0.0)
    part2.front.right = ActiveHingeV2(np.pi / 2.0)
    part2.front.right.attachment = BrickV2(-np.pi / 2.0)
    part2.front.right.attachment.left = ActiveHingeV2(np.pi / 2.0)
    part2.front.right.attachment.left.attachment = ActiveHingeV2(0.0)
    part2.front.right.attachment.left.attachment.attachment = ActiveHingeV2(
        -np.pi / 2.0
    )

    return body


def zappa_v2() -> BodyV2:
    """
    Get the zappa modular robot.

    :returns: the robot.
    """
    body = BodyV2()

    body.core_v2.back_face.bottom = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom = ActiveHingeV2(np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment = ActiveHingeV2(-np.pi / 2.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment = ActiveHingeV2(0.0)
    body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment.attachment = BrickV2(
        0.0
    )
    part2 = body.core_v2.right_face.bottom.attachment.attachment.attachment.attachment.attachment

    part2.front = ActiveHingeV2(0.0)
    part2.front.attachment = ActiveHingeV2(0.0)
    part2.left = ActiveHingeV2(np.pi / 2.0)
    part2.left.attachment = BrickV2(-np.pi / 2.0)
    part2.left.attachment.left = ActiveHingeV2(0.0)
    part2.left.attachment.left.attachment = BrickV2(0.0)
    part2.left.attachment.front = ActiveHingeV2(0.0)

    return body


# --- The pentagon spider --------------------------------------------------
#
# A spider with a PENTAGONAL core: one face for the head and one for each leg.
# The standard core is a cube with four faces, so a spider with four legs has no
# free face for a head. This core has five faces, 72 degrees apart: face 0 is
# the head (the robot's front, along +x), faces 1-4 carry the legs at 72, 144,
# 216 and 288 degrees, so two legs sit toward the front and two toward the
# back. Each leg is built exactly like a leg of spider_v2.
#
# The simulator only has box shapes, so the core is five box slabs, one under
# each face, whose union is exactly a regular pentagonal prism, with the
# standard core's total mass shared between them. It needs the engine's
# CoreBuilder hook (make_geometries, see engine/PATCHES.md).
#
# It is not a *_v2 body and not in all_bodies(); tests/test_head.py checks its
# structure and its built core geometry.

NUM_FACES = 5
FACE_WIDTH = 0.15  # metres, as wide as one face of the standard cube core
CORE_HEIGHT = 0.15  # metres, as tall as the standard cube core
APOTHEM = FACE_WIDTH / (2.0 * math.tan(math.pi / NUM_FACES))  # centre to face
CIRCUMRADIUS = APOTHEM / math.cos(math.pi / NUM_FACES)  # centre to corner
HEAD_FACE = 0
HEAD_COLOR = Color(255, 140, 0, 255)


class AttachmentFacePentagon(AttachmentFaceCoreV2):
    """
    One face of the pentagonal core, with the same 3x3 grid of slots as a cube face.

    The slots are given in the face's own frame (x out of the face, y along it,
    z up), so this works for a face at any angle. The core turns the face into place.
    """

    def __init__(
        self,
        apothem: float,
        horizontal_offset: float = 0.029,
        vertical_offset: float = 0.032,
    ) -> None:
        """
        Initialize this object.

        :param apothem: Distance from the core's centre to this face.
        :param horizontal_offset: Slot spacing along the face.
        :param vertical_offset: Slot spacing up the face.
        """
        super().__init__(
            face_rotation=0.0,
            horizontal_offset=horizontal_offset,
            vertical_offset=vertical_offset,
        )
        # Replace the cube face's slots, whose offsets only work for faces at
        # multiples of 90 degrees, with slots that work at any angle.
        self._attachment_points = {
            slot: AttachmentPoint(
                orientation=Quaternion(),
                offset=Vector3(
                    [
                        apothem,
                        (slot % 3 - 1) * horizontal_offset,
                        -(slot // 3 - 1) * vertical_offset,
                    ]
                ),
            )
            for slot in range(9)
        }


class CorePentagon(Core):
    """A core with five faces 72 degrees apart (see the module docstring)."""

    _BATTERY_MASS = 0.39712  # kg, as the standard core
    _FRAME_MASS = 1.0644  # kg, as the standard core

    _faces: list[AttachmentFacePentagon]

    def __init__(self, rotation: float = 0.0, num_batteries: int = 1) -> None:
        """
        Initialize this object.

        :param rotation: The module's rotation.
        :param num_batteries: The number of batteries.
        """
        super().__init__(
            rotation=rotation,
            mass=num_batteries * self._BATTERY_MASS + self._FRAME_MASS,
            bounding_box=Vector3([2 * CIRCUMRADIUS, 2 * CIRCUMRADIUS, CORE_HEIGHT]),
            child_offset=0.0,
            sensors=[],
        )
        # Five faces instead of the base class's four.
        self._attachment_points = {
            index: AttachmentPoint(
                offset=Vector3([0.0, 0.0, 0.0]),
                orientation=Quaternion.from_eulers(
                    [0.0, 0.0, 2.0 * math.pi * index / NUM_FACES]
                ),
            )
            for index in range(NUM_FACES)
        }
        self._faces = []
        for index in range(NUM_FACES):
            face = AttachmentFacePentagon(APOTHEM)
            self.set_child(face, index)
            self._faces.append(face)

    @property
    def front_face(self) -> AttachmentFacePentagon:
        """
        Get the face the head is on, which is the robot's front (+x).

        :returns: The face.
        """
        return self._faces[HEAD_FACE]

    @property
    def leg_faces(self) -> list[AttachmentFacePentagon]:
        """
        Get the four faces for legs.

        :returns: The faces, counter-clockwise from the front.
        """
        return [
            face for index, face in enumerate(self._faces) if index != HEAD_FACE
        ]

    def make_geometries(self, pose: Pose, texture: object) -> list[GeometryBox]:
        """
        Build the core's shape: five box slabs whose union is a regular pentagon.

        Slab ``i`` reaches from the centre out to face ``i``, as wide as the face,
        so its outer side is exactly that face and its corners are exactly the
        pentagon's corners. Together they cover the pentagon and nothing more.

        :param pose: The core's pose.
        :param texture: The core's texture.
        :returns: The geometries to put in the core's rigid body.
        """
        geometries = []
        for index in range(NUM_FACES):
            turn = Quaternion.from_eulers(
                [0.0, 0.0, 2.0 * math.pi * index / NUM_FACES]
            )
            geometries.append(
                GeometryBox(
                    pose=Pose(
                        pose.position
                        + pose.orientation * (turn * Vector3([APOTHEM / 2.0, 0.0, 0.0])),
                        pose.orientation * turn,
                    ),
                    mass=self.mass / NUM_FACES,
                    texture=texture,
                    aabb=AABB(Vector3([APOTHEM, FACE_WIDTH, CORE_HEIGHT])),
                )
            )
        return geometries


class BodyPentagon(Body):
    """A body whose core is a pentagon."""

    _core: CorePentagon

    def __init__(self) -> None:
        """Initialize the body."""
        super().__init__(CorePentagon(0.0))

    @property
    def core_pentagon(self) -> CorePentagon:
        """
        Get the pentagonal core.

        :returns: The core.
        """
        return self._core


def spider_pentagon() -> BodyPentagon:
    """
    Get the spider with a pentagonal core: four legs and a head, one per face.

    Each leg is built exactly like a leg of ``spider_v2``. The head is one small
    orange block in the middle of the front face, on the +x axis.

    :returns: The robot.
    """
    body = BodyPentagon()
    for face in body.core_pentagon.leg_faces:
        face.bottom = ActiveHingeV2(math.pi / 2.0)
        face.bottom.attachment = BrickV2(-math.pi / 2.0)
        face.bottom.attachment.front = ActiveHingeV2(0.0)
        face.bottom.attachment.front.attachment = BrickV2(0.0)
    head = BrickV2(0.0)
    head.color = HEAD_COLOR
    body.core_pentagon.front_face.middle = head
    return body


def get(name: str) -> BodyV2:
    """
    Get a robot by name.

    :param name: The name of the robot to get.
    :returns: The robot with that name.
    :raises ValueError: When a robot with that name does not exist.
    """
    match name:
        case "gecko":
            return gecko_v2()
        case "spider":
            return spider_v2()
        case "babya":
            return babya_v2()
        case "ant":
            return ant_v2()
        case "salamander":
            return salamander_v2()
        case "blokky":
            return blokky_v2()
        case "park":
            return park_v2()
        case "babyb":
            return babyb_v2()
        case "garrix":
            return garrix_v2()
        case "insect":
            return insect_v2()
        case "linkin":
            return linkin_v2()
        case "longleg":
            return longleg_v2()
        case "penguin":
            return penguin_v2()
        case "pentapod":
            return pentapod_v2()
        case "queen":
            return queen_v2()
        case "squarish":
            return squarish_v2()
        case "snake":
            return snake_v2()
        case "stingray":
            return stingray_v2()
        case "tinlicker":
            return tinlicker_v2()
        case "turtle":
            return turtle_v2()
        case "ww":
            return ww_v2()
        case "zappa":
            return zappa_v2()
        case "simple":
            return simple_v2()
        case "spider_pentagon":
            return spider_pentagon()
        case _:
            raise ValueError(f"Robot does not exist: {name}")
