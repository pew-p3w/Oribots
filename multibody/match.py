"""Build one match: the walled arena, K robots and one ball in one scene (the Match_Builder, R8).

- One ``Ball`` is created per match, added to the scene once, and handed to every
  robot's brain, so all K brains read the same ball (R8.2).
- K distinct ``ModularRobot`` objects, each with its own brain built fresh for the
  match (independent CPG state, R8.1, R8.3).
- By default the K robots share one body object: scene conversion gives each
  robot its own body-to-simulation mapping, and hinges are looked up by their
  uuid, so each robot drives its own joints (checked by the engine probe, R21.2).
  ``per_robot_body_copy=True`` (the R21.7 fallback) gives each robot a deep copy
  and rebuilds the CPG wiring from that copy, refusing if the hinge order differs.
- Robots are placed at their layout pose with the body resting on the floor;
  see ``arena.spawn_orientation`` for how the yaw reaches the engine (R7.9, R7.11).
- No stop condition: every match runs its full length (R8.6).
- Standard batch parameters: 0.001 s step, 20 Hz control, 5 Hz sampling (R8.8).
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()
from pyrr import Vector3

import arena
from config_loader import ContestSettings, body_wiring, settings_values
from layered_brain import LayeredBallAwareBrain
from revolve2.ci_group.interactive_objects import Ball
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene
from revolve2.simulation.scene import Pose

ROLES = ("Learner", "Opponent", "self")


class _SpawnOrientationNotice(logging.Filter):
    """Drop the engine's per-robot notice about lifting a rotated robot.

    The engine logs it for every robot spawned with a yaw. A yaw does not change
    the vertical lift (the probe checks the spawn height, R21.5), so the notice
    is noise: hundreds of lines per generation.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return not str(record.msg).startswith("translate_z_aabb does not yet support")


logging.getLogger().addFilter(_SpawnOrientationNotice())


@dataclass(frozen=True)
class SlotController:
    """What one slot runs: its Layer_Parameters and its role."""

    layer_params: np.ndarray
    role: str


def batch_timing(simulation_time: float) -> dict[str, Any]:
    """
    The standard batch timing for a match of ``simulation_time`` seconds (R8.7-8.9).

    :param simulation_time: Match length in seconds.
    :returns: control_step, sample_step, sampling_frequency, simulation_time, simulation_timestep.
    """
    parameters = make_standard_batch_parameters()
    return {
        "control_step": 1.0 / parameters.control_frequency,
        "sample_step": 1.0 / parameters.sampling_frequency,
        "sampling_frequency": parameters.sampling_frequency,
        "simulation_time": float(simulation_time),
        "simulation_timestep": parameters.simulation_timestep,
    }


def num_samples(simulation_time: float, sampling_frequency: float) -> int:
    """
    N = floor(simulation_time x sampling_frequency) (R8.9).

    :returns: N.
    """
    return int(math.floor(float(simulation_time) * float(sampling_frequency) + 1e-9))


def build_match_scene(
    settings: ContestSettings,
    layout: arena.Layout,
    slot_controllers: list[SlotController],
    cpg_network_structure: Any,
    output_mapping: list[Any],
    per_robot_body_copy: bool = False,
) -> tuple[ModularRobotScene, list[ModularRobot], Ball]:
    """
    Build the scene for one match (frozen signature, design.md, plus the R21.7 switch).

    :param settings: The contest settings.
    :param layout: Where the K robots start.
    :param slot_controllers: One per slot, in slot order.
    :param cpg_network_structure: The body's CPG structure (``body_wiring``).
    :param output_mapping: The body's hinge output mapping (``body_wiring``).
    :param per_robot_body_copy: Give each robot its own body copy (R21.7 fallback).
    :returns: The scene, the K robots in slot order, and the ball.
    :raises ValueError: If the slot count does not match the layout.
    """
    if len(slot_controllers) != len(layout.starts):
        raise ValueError(
            f"{len(slot_controllers)} slot controllers for a layout of {len(layout.starts)} robots"
        )
    ball = Ball(
        radius=settings.ball_radius,
        mass=settings.ball_mass,
        pose=Pose(Vector3(list(arena.ball_start_position(settings.ball_radius)))),
    )
    scene = ModularRobotScene(
        terrain=arena.build_walled_terrain(
            settings.terrain_size, settings.wall_height, settings.wall_thickness
        )
    )
    robots = []
    for start, controller in zip(layout.starts, slot_controllers):
        body = settings.body
        structure, mapping = cpg_network_structure, output_mapping
        if per_robot_body_copy:
            body = copy.deepcopy(settings.body)
            structure, mapping = body_wiring(body)
            _require_same_hinge_order(output_mapping, mapping)
        brain = LayeredBallAwareBrain.from_params(
            settings.trained_weights,
            controller.layer_params,
            cpg_network_structure=structure,
            output_mapping=mapping,
            ball=ball,
            feedback_num_inputs=settings.feedback_num_inputs,
            steering_output_scale=settings.feedback_output_scale,
            distance_scale=settings.feedback_distance_scale,
            reference_offset=settings.head_offset,
            hidden_size=settings.hidden_size,
            layer_output_scale=settings.layer_output_scale,
        )
        robot = ModularRobot(body=body, brain=brain)
        scene.add_robot(robot, pose=arena.start_pose(start))
        robots.append(robot)
    scene.add_interactive_object(ball)
    return scene, robots, ball


def _require_same_hinge_order(original: list[Any], rebuilt: list[Any]) -> None:
    """
    A body copy's hinge mapping must list the hinges in the original order, so the
    trained steering rows and the W2 rows still drive the same hinges (R21.7).

    :raises ValueError: If the order differs.
    """
    before = [(index, hinge.uuid) for index, hinge in original]
    after = [(index, hinge.uuid) for index, hinge in rebuilt]
    if before != after:
        raise ValueError("A per-robot body copy changed the hinge order; trained weights would misalign.")


def slot_controllers_for(
    settings: ContestSettings,
    candidate: np.ndarray,
    layout_index: int,
    reference_layer: np.ndarray,
) -> list[SlotController]:
    """
    Who runs what in one match (R14.1, R14.5, R15.4).

    ``"reference"``: the Learner (the candidate) in slot ``layout_index mod K``,
    the Reference_Layer in every other slot. ``"self"``: the candidate in all K.

    :returns: K slot controllers.
    """
    if settings.opponents == "self":
        return [SlotController(candidate, "self") for _ in range(settings.k)]
    learner_slot = layout_index % settings.k
    return [
        SlotController(candidate, "Learner") if slot == learner_slot else SlotController(reference_layer, "Opponent")
        for slot in range(settings.k)
    ]


def zero_output_layer(settings: ContestSettings, w1: np.ndarray | None = None) -> np.ndarray:
    """
    A Zero_Output_Layer: W2 = 0 (and the given W1, or zeros), so the robot runs exactly
    its trained controller.

    :returns: The Layer_Parameters.
    """
    w1_values = np.zeros(4 * settings.hidden_size) if w1 is None else np.asarray(w1, dtype=np.float64)
    return np.concatenate([w1_values, np.zeros(settings.num_active_hinges * settings.hidden_size)])


def make_match_task(
    settings: ContestSettings,
    layout: arena.Layout,
    layout_index: int,
    slot_controllers: list[SlotController],
    cpg_network_structure: Any,
    output_mapping: list[Any],
    scene_id: int,
    simulation_time: float | None = None,
    per_robot_body_copy: bool = False,
) -> dict[str, Any]:
    """
    Everything a worker needs to simulate and score one match (the match-task dict).

    The scene is built and converted here, in the main process, so results do not
    depend on which worker runs it (R15.5).

    :returns: The task dict.
    """
    scene, robots, ball = build_match_scene(
        settings, layout, slot_controllers, cpg_network_structure, output_mapping, per_robot_body_copy
    )
    simulation_scene, mapping = scene.to_simulation_scene()
    timing = batch_timing(settings.simulation_time if simulation_time is None else simulation_time)
    return {
        "scene_id": scene_id,
        "scene": simulation_scene,
        "mapping": mapping,
        "robots": robots,
        "ball": ball,
        "roles": [controller.role for controller in slot_controllers],
        "layout_index": layout_index,
        "base_config_source": settings.base_config_source,
        "base_config_name": settings.base_config_name,
        "multibody_config_source": settings.multibody_config_source,
        "config_dir": str(contest_paths.CONFIG),
        "settings_values": settings_values(settings),
        "cast_shadows": False,
        "fast_sim": False,
        "viewer_type": None,
        **timing,
    }
