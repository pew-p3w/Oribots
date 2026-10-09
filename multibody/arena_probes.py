"""Engine probes: confirm the engine behaviours the contest relies on (R21, phase 1 gate).

Each probe reads MuJoCo's own data (geoms, joint positions, contacts and contact
forces) rather than the engine's state readings, and returns a list of
failures, each naming the criterion of requirement 21, the measured value and
the expected bound. :func:`run_all` runs the six probes; ``python
multibody/arena_probes.py`` prints the results.

1. The arena compiles to exactly four static wall boxes of the configured size
   with their inner faces on the floor edges, and a ball launched at a wall at
   ``PROBE_BALL_SPEED`` touches that wall and stays inside (R21.1).
2. K robots built from ONE shared body each obey their own hinge commands, for
   K = 2..5 (R21.2); if not, the check is repeated with a body copy per robot (R21.7).
3. K robots added at K start poses read back K separate poses at t = 0, through
   the real Match_Builder (R21.3).
4. A scene with no stop condition simulates its full length (R21.4).
5. Every configs/ base body spawns resting on the floor at 8 yaws (R21.5).
6. Robot-robot, robot-ball, robot-wall and ball-wall contacts all produce a
   normal force (R21.6).
"""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import contest_paths

contest_paths.install()

import mujoco  # noqa: E402
from pyrr import Vector3  # noqa: E402

import arena  # noqa: E402
import config_loader  # noqa: E402
import match  # noqa: E402
from revolve2.ci_group.interactive_objects import Ball  # noqa: E402
from revolve2.modular_robot import ModularRobot  # noqa: E402
from revolve2.modular_robot.body.base import ActiveHinge  # noqa: E402
from revolve2.modular_robot.brain import Brain, BrainInstance  # noqa: E402
from revolve2.modular_robot_simulation import ModularRobotScene  # noqa: E402
from revolve2.modular_robot_simulation._modular_robot_scene import physical_orientation  # noqa: E402
from revolve2.simulation.scene import Pose, UUIDKey  # noqa: E402

PROBE_BALL_SPEED = 2.0  # m/s (R21.1 default)
TIMESTEP = 0.001
CONTROL_STEP = 1.0 / 20.0
TINY_CONFIG = contest_paths.MULTIBODY / "configs" / "_tiny.py"


@dataclass
class ProbeFailure:
    """One failed check."""

    criterion: str
    detail: str

    def __str__(self) -> str:
        return f"R{self.criterion}: {self.detail}"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


class _ConstantTargets(Brain):
    """A brain that holds every hinge at a fixed target (probe 2)."""

    def __init__(self, targets: dict[Any, float]) -> None:
        self._targets = targets

    def make_instance(self) -> BrainInstance:
        return _ConstantTargetsInstance(self._targets)


class _ConstantTargetsInstance(BrainInstance):
    def __init__(self, targets: dict[Any, float]) -> None:
        self._targets = targets

    def control(self, dt: float, sensor_state: Any, control_interface: Any) -> None:
        for hinge, target in self._targets.items():
            control_interface.set_active_hinge_target(hinge, target)


def _compile(scene: ModularRobotScene) -> tuple[Any, Any, Any, Any]:
    """Convert and compile a scene; returns (simulation scene, robot mapping, model, mujoco mapping)."""
    from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model

    simulation_scene, robot_mapping = scene.to_simulation_scene()
    model, mujoco_mapping = scene_to_model(simulation_scene, TIMESTEP, cast_shadows=False, fast_sim=True)
    return simulation_scene, robot_mapping, model, mujoco_mapping


def _step(
    model: Any,
    data: Any,
    simulation_scene: Any,
    mujoco_mapping: Any,
    seconds: float,
    on_step: Callable[[Any, Any], None] | None = None,
) -> None:
    """The engine's simulation loop (control at 20 Hz), with a callback after every step."""
    from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
    from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl

    control = ControlInterfaceImpl(data=data, abstraction_to_mujoco_mapping=mujoco_mapping)
    last_control = 0.0
    while data.time < seconds:
        if data.time >= last_control + CONTROL_STEP:
            last_control = math.floor(data.time / CONTROL_STEP) * CONTROL_STEP
            state = SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mujoco_mapping, camera_views={})
            simulation_scene.handler.handle(state, control, CONTROL_STEP)
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(model, data)


def _wall_geoms(model: Any, simulation_scene: Any, mujoco_mapping: Any) -> list[int]:
    """The box geoms of the static terrain body (the walls), in model order."""
    terrain_body = mujoco_mapping.multi_body_system[UUIDKey(simulation_scene.multi_body_systems[0])].id
    return [
        geom
        for geom in range(model.ngeom)
        if model.geom_bodyid[geom] == terrain_body and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
    ]


def _subtree_geoms(model: Any, root_body: int) -> set[int]:
    """Every geom of the bodies whose root is ``root_body``."""
    return {geom for geom in range(model.ngeom) if model.body_rootid[model.geom_bodyid[geom]] == root_body}


def _root_body(mujoco_mapping: Any, multi_body_system: Any) -> int:
    return mujoco_mapping.multi_body_system[UUIDKey(multi_body_system)].id


def _settings() -> config_loader.ContestSettings:
    return config_loader.load_contest(TINY_CONFIG)


# --------------------------------------------------------------------------- #
# Probe 1: walls
# --------------------------------------------------------------------------- #


def probe_walls(settings: config_loader.ContestSettings) -> list[ProbeFailure]:
    """R21.1: four wall boxes of the right size and place; a launched ball stays in."""
    failures = []
    size_x, size_y = settings.terrain_size
    height, thickness = settings.wall_height, settings.wall_thickness
    expected = arena.wall_boxes(size_x, size_y, height, thickness)

    scene = ModularRobotScene(terrain=arena.build_walled_terrain(settings.terrain_size, height, thickness))
    simulation_scene, _, model, mujoco_mapping = _compile(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    walls = _wall_geoms(model, simulation_scene, mujoco_mapping)
    if len(walls) != 4:
        return [ProbeFailure("21.1", f"terrain has {len(walls)} box geoms, expected 4")]
    found = sorted((tuple(np.round(data.geom_xpos[g], 9)), tuple(np.round(2 * model.geom_size[g], 9))) for g in walls)
    for centre, extent in expected:
        match_found = any(
            np.allclose(c, centre, atol=1e-6) and np.allclose(e, extent, atol=1e-6) for c, e in found
        )
        if not match_found:
            failures.append(ProbeFailure("21.1", f"no wall box with centre {centre} and size {extent}; found {found}"))
    # Inner faces on the floor edges (D-5).
    faces = []
    for geom in walls:
        centre, half = data.geom_xpos[geom], model.geom_size[geom]
        if half[0] < half[1]:  # thin in x: an x-wall
            faces.append(abs(centre[0]) - half[0] - size_x / 2.0)
        else:
            faces.append(abs(centre[1]) - half[1] - size_y / 2.0)
    worst = max(abs(f) for f in faces)
    if worst > 1e-6:
        failures.append(ProbeFailure("21.1", f"a wall's inner face is {worst:.3e} m off the floor edge (bound 1e-6)"))

    # Launch a ball at each wall.
    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    for direction in directions:
        failures.extend(_launch_ball_at_wall(settings, direction))
    return failures


def _launch_ball_at_wall(settings: config_loader.ContestSettings, direction: tuple[float, float]) -> list[ProbeFailure]:
    size_x, size_y = settings.terrain_size
    radius = settings.ball_radius
    half = size_x / 2.0 if direction[0] else size_y / 2.0
    start = ((half - 1.0) * direction[0], (half - 1.0) * direction[1], radius)
    ball = Ball(radius=radius, mass=settings.ball_mass, pose=Pose(Vector3(list(start))))
    scene = ModularRobotScene(
        terrain=arena.build_walled_terrain(settings.terrain_size, settings.wall_height, settings.wall_thickness)
    )
    scene.add_interactive_object(ball)
    simulation_scene, _, model, mujoco_mapping = _compile(scene)
    data = mujoco.MjData(model)
    ball_body = _root_body(mujoco_mapping, ball)
    joint = model.body_jntadr[ball_body]
    dof = model.jnt_dofadr[joint]
    data.qvel[dof : dof + 3] = [PROBE_BALL_SPEED * direction[0], PROBE_BALL_SPEED * direction[1], 0.0]
    mujoco.mj_forward(model, data)
    walls = _wall_geoms(model, simulation_scene, mujoco_mapping)
    target = _facing_wall(data, model, walls, direction)
    ball_geoms = _subtree_geoms(model, ball_body)
    state = {"touched": False, "outside_at": None}

    def watch(model: Any, data: Any) -> None:
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = {contact.geom1, contact.geom2}
            if target in pair and pair & ball_geoms:
                state["touched"] = True
        x, y = data.xpos[ball_body][0], data.xpos[ball_body][1]
        if state["outside_at"] is None and not (abs(x) < size_x / 2.0 and abs(y) < size_y / 2.0):
            state["outside_at"] = float(data.time)

    _step(model, data, simulation_scene, mujoco_mapping, 5.0, watch)
    failures = []
    if not state["touched"]:
        failures.append(ProbeFailure("21.1", f"ball launched {direction} never touched that wall in 5 s"))
    if state["outside_at"] is not None:
        failures.append(ProbeFailure("21.1", f"ball launched {direction} left the walls at t={state['outside_at']:.3f} s"))
    return failures


def _facing_wall(data: Any, model: Any, walls: list[int], direction: tuple[float, float]) -> int:
    """The wall geom the direction points at."""
    axis = 0 if direction[0] else 1
    sign = direction[axis]
    return max(walls, key=lambda g: sign * data.geom_xpos[g][axis] if model.geom_size[g][axis] < model.geom_size[g][1 - axis] else -1e9)


# --------------------------------------------------------------------------- #
# Probe 2: per-robot hinge commands
# --------------------------------------------------------------------------- #


def probe_hinge_commands(
    settings: config_loader.ContestSettings, per_robot_body_copy: bool = False, _send_robot0_targets: bool = False
) -> list[ProbeFailure]:
    """
    R21.2: K robots from one body obey their own constant hinge targets (K = 2..5).

    ``_send_robot0_targets`` is for the check suite only: every robot is sent robot
    0's targets while each is still judged against its own, so the probe must fail.
    """
    failures = []
    for k in (2, 3, 4, 5):
        failures.extend(_hinge_commands_for(settings, k, per_robot_body_copy, _send_robot0_targets))
    return failures


def _hinge_signs(robot: int, num_hinges: int) -> list[float]:
    """Robot r's target signs: hinges 0-2 carry r in binary, so any two robots differ on one of them."""
    return [(-1.0 if (robot >> hinge) & 1 else 1.0) if hinge < 3 else (1.0 if (robot + hinge) % 2 else -1.0) for hinge in range(num_hinges)]


def _hinge_commands_for(
    settings: config_loader.ContestSettings, k: int, per_robot_body_copy: bool, send_robot0_targets: bool = False
) -> list[ProbeFailure]:
    scene = ModularRobotScene(
        terrain=arena.build_walled_terrain((12.0, 12.0), settings.wall_height, settings.wall_thickness)
    )
    expected = []
    for robot_index in range(k):
        body = copy.deepcopy(settings.body) if per_robot_body_copy else settings.body
        hinges = body.find_modules_of_type(ActiveHinge)
        signs = _hinge_signs(robot_index, len(hinges))
        sent = _hinge_signs(0, len(hinges)) if send_robot0_targets else signs
        targets = {hinge: sign * 0.5 * hinge.range for hinge, sign in zip(hinges, sent)}
        angle = 2.0 * math.pi * robot_index / k
        scene.add_robot(
            ModularRobot(body=body, brain=_ConstantTargets(targets)),
            pose=arena.start_pose(arena.RobotStart(xy=(3.0 * math.cos(angle), 3.0 * math.sin(angle)), yaw=0.0)),
        )
        expected.append([(hinge, sign * 0.5 * hinge.range) for hinge, sign in zip(hinges, signs)])
    simulation_scene, _, model, mujoco_mapping = _compile(scene)
    # This probe checks command ROUTING (does each robot obey its own targets?),
    # not strength: a spider standing on all legs cannot lift its body to 0.5 x
    # range in 2 s, whichever robot the command reaches. Without gravity the
    # hinges move freely, so the sign and magnitude bound isolates the routing.
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _step(model, data, simulation_scene, mujoco_mapping, 2.0)
    failures = []
    label = "with a body copy per robot" if per_robot_body_copy else "sharing one body"
    for robot_index, (_, body_mapping) in enumerate(simulation_scene.handler._brains):
        for hinge, target in expected[robot_index]:
            joint = body_mapping.active_hinge_to_joint_hinge[UUIDKey(hinge)]
            joint_id = mujoco_mapping.hinge_joint[UUIDKey(joint)].id
            position = float(data.qpos[model.jnt_qposadr[joint_id]])
            if not (np.sign(position) == np.sign(target) and abs(position) >= 0.25 * hinge.range):
                failures.append(
                    ProbeFailure(
                        "21.2",
                        f"K={k} {label}: robot {robot_index} hinge at {position:+.4f} rad after 2 s, "
                        f"target {target:+.4f} (needs the same sign and |pos| >= {0.25 * hinge.range:.4f})",
                    )
                )
    return failures


# --------------------------------------------------------------------------- #
# Probe 3: K readable poses (through the Match_Builder)
# --------------------------------------------------------------------------- #


def probe_readable_poses(settings: config_loader.ContestSettings) -> list[ProbeFailure]:
    """R21.3: K robots read back their own start poses at t = 0 (K = 2..5)."""
    from revolve2.modular_robot_simulation._scene_simulation_state import SceneSimulationState
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

    failures = []
    structure, mapping = config_loader.body_wiring(settings.body)
    for k in (2, 3, 4, 5):
        starts = tuple(
            arena.RobotStart(
                xy=(1.6 * math.cos(2.0 * math.pi * i / k), 1.6 * math.sin(2.0 * math.pi * i / k)),
                yaw=-math.pi + 2.0 * math.pi * (i + 0.5) / k,
            )
            for i in range(k)
        )
        layout = arena.Layout(starts=starts)
        controllers = [match.SlotController(match.zero_output_layer(settings), "self") for _ in range(k)]
        scene, robots, _ = match.build_match_scene(settings, layout, controllers, structure, mapping)
        simulation_scene, robot_mapping = scene.to_simulation_scene()
        states = simulate_scene(0, simulation_scene, True, None, False, CONTROL_STEP, 0.2, 0.2, TIMESTEP, False, True, None)
        first = SceneSimulationState(states[0], robot_mapping)
        for start, robot in zip(starts, robots):
            pose = first.get_modular_robot_simulation_state(robot).get_pose()
            heading = physical_orientation(pose) * Vector3([1.0, 0.0, 0.0])
            yaw_error = abs((math.atan2(heading.y, heading.x) - start.yaw + math.pi) % (2.0 * math.pi) - math.pi)
            xy_error = math.hypot(pose.position.x - start.xy[0], pose.position.y - start.xy[1])
            if xy_error > 0.01 or yaw_error > 0.01:
                failures.append(
                    ProbeFailure(
                        "21.3",
                        f"K={k}: robot read back {xy_error:.4f} m / {yaw_error:.4f} rad from its own "
                        "start pose (bounds 0.01 m, 0.01 rad)",
                    )
                )
    return failures


# --------------------------------------------------------------------------- #
# Probe 4: full length without a stop condition
# --------------------------------------------------------------------------- #


def probe_full_length(settings: config_loader.ContestSettings, seconds: float = 30.0) -> list[ProbeFailure]:
    """R21.4: no stop condition => at least seconds x 5 Hz states after t = 0."""
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

    structure, mapping = config_loader.body_wiring(settings.body)
    layout = arena.Layout(starts=(arena.RobotStart(xy=(1.5, 0.0), yaw=math.pi),))
    one = config_loader.ContestSettings(**{**settings.__dict__, "k": 1})
    scene, _, _ = match.build_match_scene(one, layout, [match.SlotController(match.zero_output_layer(one), "self")], structure, mapping)
    simulation_scene, _ = scene.to_simulation_scene()
    states = simulate_scene(0, simulation_scene, True, None, False, CONTROL_STEP, 0.2, seconds, TIMESTEP, False, True, None)
    needed = int(seconds * 5)
    if len(states) - 1 < needed:
        return [ProbeFailure("21.4", f"{len(states) - 1} states after t=0 for {seconds} s, expected at least {needed}")]
    return []


# --------------------------------------------------------------------------- #
# Probe 5: spawn height
# --------------------------------------------------------------------------- #


def _lowest_point(model: Any, data: Any, geoms: set[int]) -> float:
    """The lowest corner of the geoms' (oriented) bounding boxes, in world z."""
    lowest = math.inf
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    for geom in geoms:
        centre_local = model.geom_aabb[geom][:3]
        half = model.geom_aabb[geom][3:]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        points = data.geom_xpos[geom] + (centre_local + corners * half) @ rotation.T
        lowest = min(lowest, float(points[:, 2].min()))
    return lowest


def probe_spawn_height(tolerance: float | None = None) -> list[ProbeFailure]:
    """R21.5: every configs/ base body rests on the floor at 8 yaws."""
    failures = []
    for base_name in sorted(_configured_base_names()):
        source = (contest_paths.CONFIG / f"{base_name}.py").read_text(encoding="utf-8")
        module = config_loader.load_base_module(base_name, source, str(contest_paths.CONFIG / f"{base_name}.py"))
        bound = 0.01 if tolerance is None else tolerance
        for m in range(8):
            yaw = -math.pi + m * math.pi / 4.0
            scene = ModularRobotScene(terrain=arena.build_walled_terrain((6.0, 6.0), 1.2, 0.2))
            robot = ModularRobot(body=module.BODY, brain=_ConstantTargets({}))
            scene.add_robot(robot, pose=arena.start_pose(arena.RobotStart(xy=(0.0, 0.0), yaw=yaw)))
            simulation_scene, robot_mapping, model, mujoco_mapping = _compile(scene)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            root = _root_body(mujoco_mapping, robot_mapping[UUIDKey(robot)])
            low = _lowest_point(model, data, _subtree_geoms(model, root))
            if not (-1e-9 <= low <= bound):
                failures.append(
                    ProbeFailure("21.5", f"{base_name} at yaw {yaw:+.4f}: lowest point {low:+.6f} m (bounds 0 .. {bound})")
                )
    return failures


def _configured_base_names() -> set[str]:
    """The BASE_CONFIG of every config in multibody/configs/."""
    import ast

    names = set()
    for path in (contest_paths.MULTIBODY / "configs").glob("*.py"):
        if path.name == "__init__.py":
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BASE_CONFIG" for t in node.targets):
                names.add(ast.literal_eval(node.value))
    return names


# --------------------------------------------------------------------------- #
# Probe 6: the four contact types
# --------------------------------------------------------------------------- #


def probe_contacts(settings: config_loader.ContestSettings) -> list[ProbeFailure]:
    """
    R21.6: robot-robot, robot-ball, robot-wall and ball-wall contacts all produce force.

    Arena 5 x 5 m. Robot A is spawned pressed against the +x wall (its far corner
    2 mm into it); robot B is dropped 2 cm onto robot A; the ball starts touching
    the -y wall and is rolled along it, at 3 m/s, into robot A.
    """
    size = (5.0, 5.0)
    half = size[0] / 2.0
    centre, aabb = arena._t_pose_aabb(settings.body)
    max_x = centre.x + aabb.size.x / 2.0
    min_y = centre.y - aabb.size.y / 2.0
    a_xy = (half - max_x + 0.002, -half - min_y + 0.05)
    radius = settings.ball_radius

    scene = ModularRobotScene(terrain=arena.build_walled_terrain(size, settings.wall_height, settings.wall_thickness))
    robot_a = ModularRobot(body=settings.body, brain=_ConstantTargets({}))
    robot_b = ModularRobot(body=settings.body, brain=_ConstantTargets({}))
    scene.add_robot(robot_a, pose=arena.start_pose(arena.RobotStart(xy=a_xy, yaw=0.0)))
    drop = Pose(Vector3([a_xy[0], a_xy[1], settings.standing_height + 0.02]), arena.spawn_orientation(0.0))
    scene.add_robot(robot_b, pose=drop)
    ball = Ball(radius=radius, mass=settings.ball_mass, pose=Pose(Vector3([-1.5, -half + radius - 0.002, radius])))
    scene.add_interactive_object(ball)
    simulation_scene, robot_mapping, model, mujoco_mapping = _compile(scene)
    data = mujoco.MjData(model)
    ball_body = _root_body(mujoco_mapping, ball)
    dof = model.jnt_dofadr[model.body_jntadr[ball_body]]
    data.qvel[dof : dof + 3] = [3.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    walls = set(_wall_geoms(model, simulation_scene, mujoco_mapping))
    ball_geoms = _subtree_geoms(model, ball_body)
    robot_geoms = [
        _subtree_geoms(model, _root_body(mujoco_mapping, robot_mapping[UUIDKey(robot)])) for robot in (robot_a, robot_b)
    ]
    seen = {"robot-robot": 0.0, "robot-ball": 0.0, "robot-wall": 0.0, "ball-wall": 0.0}
    force = np.zeros(6)

    def party(geom: int) -> str | None:
        if geom in walls:
            return "wall"
        if geom in ball_geoms:
            return "ball"
        for index, geoms in enumerate(robot_geoms):
            if geom in geoms:
                return f"robot{index}"
        return None

    def watch(model: Any, data: Any) -> None:
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = party(contact.geom1), party(contact.geom2)
            if a is None or b is None:
                continue
            kinds = sorted("robot" if p.startswith("robot") else p for p in (a, b))
            if kinds == ["robot", "robot"] and a == b:
                continue  # a robot touching itself
            key = "-".join(kinds) if kinds[0] != kinds[1] else "robot-robot"
            key = {"ball-robot": "robot-ball", "robot-wall": "robot-wall", "ball-wall": "ball-wall", "robot-robot": "robot-robot"}.get(key)
            if key is None:
                continue
            mujoco.mj_contactForce(model, data, index, force)
            seen[key] = max(seen[key], float(force[0]))

    _step(model, data, simulation_scene, mujoco_mapping, 5.0, watch)
    return [
        ProbeFailure("21.6", f"{kind}: no contact with a normal force > 0 N in 5 s (max {value:.3g} N)")
        for kind, value in seen.items()
        if not value > 0.0
    ]


# --------------------------------------------------------------------------- #
# All probes
# --------------------------------------------------------------------------- #


def run_all() -> dict[str, list[ProbeFailure]]:
    """
    Run the six probes; probe 2 is repeated with per-robot body copies if sharing fails (R21.7).

    :returns: Probe name to its failures (empty = passed). Key ``hinge_commands``
        holds the judged result; ``hinge_commands_shared`` the shared-body result.
    """
    settings = _settings()
    config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    results = {"walls": probe_walls(settings)}
    shared = probe_hinge_commands(settings)
    results["hinge_commands_shared"] = shared
    results["hinge_commands"] = shared if not shared else probe_hinge_commands(settings, per_robot_body_copy=True)
    results["readable_poses"] = probe_readable_poses(settings)
    results["full_length"] = probe_full_length(settings)
    results["spawn_height"] = probe_spawn_height()
    results["contacts"] = probe_contacts(settings)
    return results


def main() -> None:
    """Print the probe results; exit non-zero if any judged probe failed."""
    results = run_all()
    failed = False
    for name, failures in results.items():
        if name == "hinge_commands_shared":
            print(f"hinge_commands (shared body): {'ok' if not failures else 'FAILED, judged on per-robot copies'}")
            continue
        print(f"{name}: {'ok' if not failures else 'FAILED'}")
        for failure in failures:
            print(f"  {failure}")
        failed = failed or bool(failures)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
