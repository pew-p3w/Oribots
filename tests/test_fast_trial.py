"""Check that the fast trial (src/fast_trial.py) gives bit-identical results.

"Identical" here means the same IEEE 754 bits, compared with ``tobytes()``, so
even ``0.0`` against ``-0.0`` counts as a difference. Checks:

In-process, on the tiny config:

1.  pyrr's quaternion rotation, rewritten in ``_rotate``, matches pyrr for
    random, unnormalised and signed-zero quaternions;
2.  ``_reference_point`` matches ``robot_frame.reference_position`` (core and head);
3.  ``_rk45_exact`` matches the brain's ``_rk45`` (including saturation);
4.  one control tick: the fast brain step and the original brain, driven
    through the original state, sensor and control-interface objects, write the
    same MuJoCo controls and keep the same CPG state, over many random robot
    and ball poses, for core and head mode, with saturating weights;
5.  the stop condition agrees with the original at random distances;
6.  ``supports`` accepts training tasks and refuses everything else;
7.  the end-to-end evaluator gives the same answer with the fast trial off and
    on, with one worker and with a kept pool of two (which is reused, and not
    reused when turned off).

In a fresh process per config (each process can hold one config):

8.  every shipped config: fitness and movement record identical, original
    against fast;
9.  the spider (core mode) and the 8-ball spider (head mode) across all four
    integrators and several timesteps, with a trial length that is not a
    whole number of samples: the sampled trajectories (robot position and
    orientation, ball position, at every sample), the fitness and the movement
    record are identical;
10. "ball reached": trials that stop at the first tick and part-way through,
    identical.

Run with `python tests/test_fast_trial.py` (or `tests\\run_tests.bat fast_trial`).
Takes a few minutes; `--quick` runs checks 1-7 and a reduced 8-10.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402

TINY = ROOT / "config" / "_tiny_cmaes.py"
PYTHON = sys.executable


def _bits(value) -> bytes:
    """The exact bits of a float or float array."""
    import numpy as np

    return np.asarray(value, dtype=np.float64).tobytes()


def same(a, b) -> bool:
    """Bit-identical floats (or float arrays, or (fitness, movement) results)."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if a is None or b is None:
        return a is b
    return _bits(a) == _bits(b)


# --------------------------------------------------------------------------
# 1-7: in-process checks on the tiny config
# --------------------------------------------------------------------------


def load_tiny_config():
    """Load the tiny config as the module named ``config``."""
    paths.install(TINY.parent)
    spec = importlib.util.spec_from_file_location("config", TINY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    return module


def check_rotate() -> list[str]:
    """1. ``_rotate`` is pyrr's apply_to_vector, bit for bit."""
    import numpy as np
    from pyrr import Quaternion, Vector3

    import fast_trial

    problems = []
    rng = np.random.default_rng(1)
    quats = [rng.normal(size=4) for _ in range(3000)]
    quats += [q / np.linalg.norm(q) for q in quats[:1000]]
    quats += [
        np.array(q, dtype=float)
        for q in (
            (0.0, 0.0, 0.0, 1.0), (-0.0, 0.0, -0.0, 1.0), (1.0, 0.0, 0.0, 0.0),
            (0.0, -0.0, 1.0, -0.0), (0.0, 0.0, 0.0, 0.0), (-0.0, -0.0, -0.0, -0.0),
            (0.5, -0.5, 0.5, -0.5), (1e-300, -1e-300, 1e300, 1.0),
        )
    ]
    vectors = [(1.0, 0.0, 0.0), (0.12, -0.0, 0.035), (0.0, 0.0, 0.0), (-0.0, 1.0, -2.5)]
    vectors += [tuple(rng.normal(size=3)) for _ in range(5)]
    for q in quats:
        for v in vectors:
            with np.errstate(over="ignore", invalid="ignore"):  # the 1e300 case overflows in both
                expected = Quaternion(q) * Vector3(v)
                got = fast_trial._rotate(*(float(x) for x in q), *v)
            if _bits(expected) != _bits(got):
                problems.append(f"rotate({q}, {v}): pyrr {list(expected)}, fast {list(got)}")
                if len(problems) > 5:
                    return problems
    return problems


def check_reference_point() -> list[str]:
    """2. ``_reference_point`` is robot_frame.reference_position."""
    import numpy as np
    from pyrr import Quaternion, Vector3

    import fast_trial
    import robot_frame
    from revolve2.simulation.scene import Pose

    problems = []
    rng = np.random.default_rng(2)
    for _ in range(2000):
        position = rng.normal(size=3)
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        pose = Pose(Vector3(position), Quaternion(quat))  # as the state reports it
        for offset in (fast_trial.ZERO, (0.105, 0.0, 0.01), (0.2, -0.0004, -0.03)):
            expected = robot_frame.reference_position(pose, offset)
            got = fast_trial._reference_point(position[0], position[1], position[2], quat, offset)
            if _bits(expected) != _bits(got):
                problems.append(f"reference point {offset}: {list(expected)} vs {got}")
                return problems
    return problems


def check_rk45() -> list[str]:
    """3. ``_rk45_exact`` is the brain's ``_rk45``."""
    import numpy as np

    import fast_trial
    from ball_aware_brain import _rk45

    problems = []
    rng = np.random.default_rng(3)
    for size in (2, 8, 16, 40):
        for scale in (0.1, 1.0, 30.0):
            weights = rng.normal(scale=scale, size=(size, size))
            state = rng.uniform(-1, 1, size)
            other = state.copy()
            for dt in (0.05, 0.02, 0.1):
                for _ in range(50):
                    state = _rk45(state, weights, dt)
                    other = fast_trial._rk45_exact(other, weights, dt)
                    if _bits(state) != _bits(other):
                        problems.append(f"rk45 differs (size {size}, scale {scale}, dt {dt})")
                        return problems
    nan_state = np.array([np.nan, 0.5, -2.0, 2.0])
    weights = np.eye(4)
    a, b = _rk45(nan_state, weights, 0.05), fast_trial._rk45_exact(nan_state, weights, 0.05)
    if not (np.array_equal(np.isnan(a), np.isnan(b)) and _bits(a[~np.isnan(a)]) == _bits(b[~np.isnan(b)])):
        problems.append(f"rk45 with NaN: {a} vs {b}")
    return problems


def _tiny_task(evaluator_module, instance, controller, ball_pose, with_movement=False):
    common = instance.worker_task_common(with_movement)
    return instance.worker_task(common, controller, ball_pose, 0)


def _random_free_pose(model, data, rng, ball_id, robot_id) -> None:
    """Scatter the robot and ball: random free-joint poses and hinge angles."""
    import mujoco
    import numpy as np

    for joint in range(model.njnt):
        adr = model.jnt_qposadr[joint]
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[adr:adr + 3] = rng.uniform(-3, 3, 3)
            quat = rng.normal(size=4)
            data.qpos[adr + 3:adr + 7] = quat / np.linalg.norm(quat)
        elif model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE:
            data.qpos[adr] = rng.uniform(-1.5, 1.5)
    mujoco.mj_forward(model, data)


def check_control_tick(config, evaluator_module) -> list[str]:
    """4. One control tick, fast against the original objects, many times."""
    import mujoco
    import numpy as np

    import fast_trial
    from revolve2.modular_robot_simulation._modular_robot_control_interface_impl import (
        ModularRobotControlInterfaceImpl,
    )
    from revolve2.modular_robot_simulation._sensor_state_impl import ModularRobotSensorStateImpl
    from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
    from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl

    problems = []
    rng = np.random.default_rng(4)
    instance = evaluator_module.Evaluator()
    ball_pose = instance.sample_ball_poses(1)[0]
    controllers = [
        np.clip(rng.normal(0.0, 0.3, instance.num_parameters), -1, 1),
        rng.choice([-1.0, 1.0], instance.num_parameters),  # saturating
        rng.normal(0.0, 5.0, instance.num_parameters),  # far outside the bounds
    ]
    dt = 0.05
    for controller in controllers:
        for offset in (fast_trial.ZERO, (0.105, 0.0, 0.01)):
            task = _tiny_task(evaluator_module, instance, controller, ball_pose)
            trial = fast_trial._Trial(task)
            brain, body_mapping = task["scene"].handler._brains[0]
            brain._reference_offset = offset
            trial.brain_offset = offset
            data_original = mujoco.MjData(trial.model)
            original_interface = ModularRobotControlInterfaceImpl(
                ControlInterfaceImpl(data_original, trial.mapping), body_mapping
            )
            for tick in range(300):
                _random_free_pose(trial.model, trial.data, rng, trial.ball_id, trial.robot_id)
                data_original.qpos[:] = trial.data.qpos
                mujoco.mj_forward(trial.model, data_original)
                state = SimulationStateImpl(data_original, trial.mapping, {})
                sensor = ModularRobotSensorStateImpl(
                    simulation_state=state, body_to_multi_body_system_mapping=body_mapping
                )
                brain.control(dt=dt, sensor_state=sensor, control_interface=original_interface)
                trial.control(dt)
                if _bits(data_original.ctrl) != _bits(trial.data.ctrl) or _bits(brain._state) != _bits(trial.state):
                    problems.append(
                        f"control differs at tick {tick} (offset {offset}): "
                        f"{data_original.ctrl.tolist()} vs {trial.data.ctrl.tolist()}"
                    )
                    return problems
    return problems


def check_stop_condition(config, evaluator_module) -> list[str]:
    """5. The stop condition, fast against the original, at random distances."""
    import mujoco
    import numpy as np

    import fast_trial
    from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl

    problems = []
    rng = np.random.default_rng(5)
    instance = evaluator_module.Evaluator()
    ball_pose = instance.sample_ball_poses(1)[0]
    controller = np.zeros(instance.num_parameters)
    task = _tiny_task(evaluator_module, instance, controller, ball_pose)
    trial = fast_trial._Trial(task)
    stop = task["scene"].handler._stop_conditions[0]
    seen = set()
    for offset in (fast_trial.ZERO, (0.105, 0.0, 0.01)):
        object.__setattr__(stop, "reference_offset", offset)
        trial.stops = [(trial.stops[0][0], trial.stops[0][1], trial.stops[0][2], offset)]
        for _ in range(2000):
            _random_free_pose(trial.model, trial.data, rng, trial.ball_id, trial.robot_id)
            # Put the ball near the robot, so both answers happen.
            adr = trial.model.jnt_qposadr[trial.model.body_jntadr[trial.ball_id]]
            trial.data.qpos[adr:adr + 2] = trial.data.xpos[trial.robot_id][:2] + rng.uniform(-0.6, 0.6, 2)
            mujoco.mj_forward(trial.model, trial.data)
            expected = stop(SimulationStateImpl(trial.data, trial.mapping, {}))
            got = trial.should_stop()
            seen.add(expected)
            if expected != got:
                problems.append(f"stop condition: original {expected}, fast {got}")
                return problems
    if seen != {True, False}:
        problems.append(f"the stop check only saw {seen}; the test did not exercise both outcomes")
    return problems


def check_supports(config, evaluator_module) -> list[str]:
    """6. Training tasks qualify; anything else goes to the original code."""
    import copy

    import numpy as np

    import fast_trial

    problems = []
    instance = evaluator_module.Evaluator()
    task = _tiny_task(evaluator_module, instance, np.zeros(instance.num_parameters), instance.sample_ball_poses(1)[0])
    if not fast_trial.supports(task):
        problems.append("a training task is not supported")
    if fast_trial.supports({**task, "sample_step": None}):
        problems.append("a task without sampling was accepted")
    if fast_trial.supports({**task, "scene": None}):
        problems.append("a task without a scene was accepted")
    two_brains = copy.deepcopy(task)
    two_brains["scene"].handler._brains.append(two_brains["scene"].handler._brains[0])
    if fast_trial.supports(two_brains):
        problems.append("a scene with two robots was accepted")
    other_stop = copy.deepcopy(task)
    other_stop["scene"].handler._stop_conditions.append(lambda state: False)
    if fast_trial.supports(other_stop):
        problems.append("a scene with an unknown stop condition was accepted")

    class OtherBrain:
        pass

    other_brain = copy.deepcopy(task)
    mapping = other_brain["scene"].handler._brains[0][1]
    other_brain["scene"].handler._brains[0] = (OtherBrain(), mapping)
    if fast_trial.supports(other_brain):
        problems.append("a scene with another brain was accepted")

    saved = os.environ.get("REVOLVE2_FAST_TRIAL")
    try:
        os.environ["REVOLVE2_FAST_TRIAL"] = "0"
        if fast_trial.enabled():
            problems.append("REVOLVE2_FAST_TRIAL=0 did not turn it off")
        os.environ.pop("REVOLVE2_FAST_TRIAL")
        config.FAST_TRIAL = False
        if fast_trial.enabled():
            problems.append("FAST_TRIAL = False did not turn it off")
        del config.FAST_TRIAL
        if not fast_trial.enabled():
            problems.append("the fast trial is not on by default")
    finally:
        if saved is not None:
            os.environ["REVOLVE2_FAST_TRIAL"] = saved
    return problems


def check_evaluator(config, evaluator_module) -> list[str]:
    """7. The evaluator: fast off/on, one worker and a kept pool, all identical."""
    import numpy as np

    problems = []
    saved = {name: getattr(config, name) for name in ("NUM_SIMULATORS", "SIMULATION_TIME")}
    saved_env = {name: os.environ.get(name) for name in ("REVOLVE2_FAST_TRIAL", "REVOLVE2_PERSISTENT_WORKERS")}
    rng = np.random.default_rng(7)

    def setenv(name, value):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    try:
        config.SIMULATION_TIME = 3
        config.NUM_SIMULATORS = 1
        instance = evaluator_module.Evaluator()
        population = [np.clip(rng.normal(0.0, 0.5, instance.num_parameters), -1, 1) for _ in range(3)]
        poses = instance.sample_ball_poses(2)

        def run():
            return instance.evaluate_on_ball_poses_full(population, poses)

        setenv("REVOLVE2_FAST_TRIAL", "0")
        reference = run()
        setenv("REVOLVE2_FAST_TRIAL", None)
        if not _same_result(reference, run()):
            problems.append("one worker: the fast trial changed the result")

        # Workers load the config from its file, so they see SIMULATION_TIME 20;
        # compare like with like by using the file's value here too.
        config.SIMULATION_TIME = saved["SIMULATION_TIME"]
        config.NUM_SIMULATORS = 2
        setenv("REVOLVE2_FAST_TRIAL", "0")
        setenv("REVOLVE2_PERSISTENT_WORKERS", "0")
        fresh_original = run()
        if getattr(instance, "_pool", None) is not None:
            problems.append("REVOLVE2_PERSISTENT_WORKERS=0 kept the pool")
        setenv("REVOLVE2_FAST_TRIAL", None)
        setenv("REVOLVE2_PERSISTENT_WORKERS", None)
        first = run()
        pool = getattr(instance, "_pool", None)
        second = run()
        if pool is None or getattr(instance, "_pool", None) is not pool:
            problems.append("the worker pool was not kept between evaluations")
        instance.close()
        if not (_same_result(fresh_original, first) and _same_result(first, second)):
            problems.append("two workers: fresh/original, kept/fast and reused/fast pools disagree")
        config.NUM_SIMULATORS = 1
        if not _same_result(first, run()):
            problems.append("two workers and one worker disagree")
    finally:
        for name, value in saved.items():
            setattr(config, name, value)
        for name, value in saved_env.items():
            setenv(name, value)
    return problems


def _same_result(a, b) -> bool:
    means_a, trials_a, moves_a = a
    means_b, trials_b, moves_b = b
    return (
        same(means_a, means_b)
        and same(trials_a, trials_b)
        and len(moves_a) == len(moves_b)
        and all(
            len(x) == len(y) and all(same(m, n) for m, n in zip(x, y))
            for x, y in zip(moves_a, moves_b)
        )
    )


# --------------------------------------------------------------------------
# 8-10: one config per process
# --------------------------------------------------------------------------


def one_config(args: list[str]) -> None:
    """
    Child process: compare original and fast trials for one config.

    Prints one JSON line: trials compared, mismatches (with details), and how
    many trials stopped early.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--integrator", default=None)
    parser.add_argument("--timestep", type=float, default=None)
    parser.add_argument("--sim-time", type=float, required=True)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--balls", type=int, default=1)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--reach", type=float, default=None,
                        help="override BALL_REACHED_DISTANCE and put balls just around it")
    options = parser.parse_args(args)

    import bench

    config_path = (ROOT / options.config).resolve()
    overrides: dict = {"SIMULATION_TIME": options.sim_time, "NUM_SIMULATORS": 1}
    if options.integrator:
        overrides["SIM_INTEGRATOR"] = options.integrator
    if options.timestep:
        overrides["SIM_TIMESTEP"] = options.timestep
    if options.reach:
        overrides["BALL_REACHED_DISTANCE"] = options.reach
    label = f"fasttest_{config_path.stem}_{options.integrator}_{options.timestep}_{options.reach}"
    generated = bench.write_generated_config(config_path, label, overrides)
    paths.install(config_path.parent)
    config = bench.load_config(generated)

    import numpy as np
    from pyrr import Vector3

    import evaluator
    import fast_trial
    from revolve2.simulation.scene import Pose
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

    instance = evaluator.Evaluator()
    rng = np.random.default_rng(10)
    controllers = bench.make_candidates(instance.num_parameters, options.candidates, 11, config)
    controllers[-1] = rng.choice([-1.0, 1.0], instance.num_parameters)  # saturating
    if options.reach:
        poses = []
        for k in range(options.balls):
            angle = 2.0 * math.pi * k / options.balls
            radius = options.reach * (0.9 if k == 0 else 1.02 + 0.02 * k)
            poses.append(Pose(Vector3([radius * math.cos(angle), radius * math.sin(angle), config.BALL_RADIUS])))
    else:
        count = options.balls
        if getattr(config, "make_training_ball_poses", None) is not None:
            count = 4 * math.ceil(options.balls / 4)
        poses = bench.make_ball_poses(count, 12, config)[: options.balls]

    common = instance.worker_task_common(True)
    compared, mismatches, stopped_early = 0, [], 0
    full_samples = None
    for c, controller in enumerate(controllers):
        for b, pose in enumerate(poses):
            task = instance.worker_task(common, controller, pose, compared)
            if not fast_trial.supports(task):
                mismatches.append(f"trial {c}/{b}: not supported by the fast trial")
                continue
            original = evaluator.simulate_and_score_scene_original(task)
            fast = fast_trial.simulate_and_score(instance.worker_task(common, controller, pose, compared))
            compared += 1
            if not same(original, fast):
                mismatches.append(f"trial {c}/{b}: original {original} vs fast {fast}")
            if options.trajectory:
                task = instance.worker_task(common, controller, pose, compared)
                trial = fast_trial._Trial(task)
                robot_pos, robot_quat, ball_pos = fast_trial._run(trial, task)
                task = instance.worker_task(common, controller, pose, compared)
                states = simulate_scene(
                    0, task["scene"], True, None, False, task["control_step"], task["sample_step"],
                    task["simulation_time"], task["simulation_timestep"], task["cast_shadows"],
                    task["fast_sim"], task["viewer_type"], integrator=task["integrator"],
                )
                expected_robot = np.array([s._xpos[trial.robot_id] for s in states])
                expected_quat = np.array([s._xquat[trial.robot_id] for s in states])
                expected_ball = np.array([s._xpos[trial.ball_id] for s in states])
                if len(states) != len(robot_pos):
                    mismatches.append(f"trial {c}/{b}: {len(states)} samples vs {len(robot_pos)}")
                elif not (same(expected_robot, robot_pos) and same(expected_quat, robot_quat) and same(expected_ball, ball_pos)):
                    first = next(i for i in range(len(states)) if not (
                        same(expected_robot[i], robot_pos[i]) and same(expected_quat[i], robot_quat[i])
                        and same(expected_ball[i], ball_pos[i])))
                    mismatches.append(f"trial {c}/{b}: trajectories differ from sample {first}")
                full = int(options.sim_time * task["sampling_frequency"]) + 2
                full_samples = full
                if len(states) < full - 1:
                    stopped_early += 1
    print(json.dumps({
        "compared": compared, "mismatches": mismatches[:5], "num_mismatches": len(mismatches),
        "stopped_early": stopped_early, "full_samples": full_samples,
    }))


def run_child(*args: str) -> dict:
    """Run ``one_config`` in a fresh process and return its JSON."""
    completed = subprocess.run(
        [PYTHON, str(Path(__file__).resolve()), "--one", *args],
        capture_output=True, text=True, cwd=ROOT,
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if completed.returncode != 0 or not lines:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-6:]
        return {"error": " | ".join(tail)}
    return json.loads(lines[-1])


def check_child(label: str, *args: str, expect_stops: bool = False) -> list[str]:
    result = run_child(*args)
    if "error" in result:
        return [f"{label}: crashed: {result['error']}"]
    problems = [f"{label}: {m}" for m in result["mismatches"]]
    if result["num_mismatches"] > len(result["mismatches"]):
        problems.append(f"{label}: ... {result['num_mismatches']} mismatches in total")
    if result["compared"] == 0:
        problems.append(f"{label}: nothing was compared")
    if expect_stops and result["stopped_early"] < 2:
        problems.append(f"{label}: only {result['stopped_early']} trial(s) stopped early; the stop path was not exercised")
    if problems:
        print(f"    {label}: FAILED", flush=True)
    else:
        early = f", {result['stopped_early']} stopped early" if "--trajectory" in args else ""
        print(f"    {label}: {result['compared']} trials identical{early}", flush=True)
    return problems


def check_all_configs(quick: bool) -> list[str]:
    """8. Every shipped config: fitness and movement identical."""
    problems = []
    configs = sorted(p for p in (ROOT / "config").glob("*.py") if p.name not in ("bodies.py", "__init__.py"))
    configs = [p for p in configs if "BODY" in p.read_text(encoding="utf-8")]
    if quick:
        configs = [p for p in configs if p.stem in ("spider_cmaes", "spider_8ball_cmaes", "_tiny")]
    for path in configs:
        problems += check_child(
            path.name, "--config", f"config/{path.name}", "--integrator", "implicitfast",
            "--timestep", "0.005", "--sim-time", "4", "--candidates", "2",
        )
    return problems


def check_integrators(quick: bool) -> list[str]:
    """9. Trajectories, fitness and movement across integrators and timesteps."""
    problems = []
    settings = [("RK4", "0.001"), ("Euler", "0.001"), ("implicit", "0.002"),
                ("implicitfast", "0.0025"), ("implicitfast", "0.005"), ("RK4", "0.01")]
    if quick:
        settings = [("RK4", "0.001"), ("implicitfast", "0.005")]
    for config_name in ("spider_cmaes.py", "spider_8ball_cmaes.py"):
        for integrator, timestep in settings:
            sim_time = "3.37" if float(timestep) < 0.002 else "9.37"
            problems += check_child(
                f"{config_name} {integrator} {float(timestep) * 1000:g} ms",
                "--config", f"config/{config_name}", "--integrator", integrator, "--timestep", timestep,
                "--sim-time", sim_time, "--candidates", "3", "--balls", "2", "--trajectory",
            )
    return problems


def check_stops(quick: bool) -> list[str]:
    """10. Trials that reach the ball: at the first tick and part-way."""
    problems = []
    for config_name in ("spider_cmaes.py", "spider_8ball_cmaes.py"):
        problems += check_child(
            f"{config_name} ball reached",
            "--config", f"config/{config_name}", "--integrator", "implicitfast", "--timestep", "0.005",
            "--sim-time", "8" if quick else "20", "--candidates", "4", "--balls", "4",
            "--reach", "0.9", "--trajectory", expect_stops=True,
        )
    return problems


def main() -> None:
    """Run every check and report."""
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one_config(sys.argv[2:])
        return
    quick = "--quick" in sys.argv[1:]

    config = load_tiny_config()
    import evaluator

    sections = [
        ("rotate", check_rotate),
        ("reference point", check_reference_point),
        ("rk45", check_rk45),
        ("control tick", lambda: check_control_tick(config, evaluator)),
        ("stop condition", lambda: check_stop_condition(config, evaluator)),
        ("supports", lambda: check_supports(config, evaluator)),
        ("evaluator", lambda: check_evaluator(config, evaluator)),
        ("every config", lambda: check_all_configs(quick)),
        ("integrators", lambda: check_integrators(quick)),
        ("ball reached", lambda: check_stops(quick)),
    ]
    problems = []
    for name, check in sections:
        print(f"  {name} ...", flush=True)
        try:
            found = check()
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            found = [f"crashed: {type(error).__name__}: {error}"]
        problems.extend(f"[{name}] {problem}" for problem in found)
    if problems:
        print("FAST TRIAL CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("FAST TRIAL CHECK PASSED")
    print("  the fast trial is bit-identical to the original: rotation, CPG, control tick,")
    print("  stop condition, trajectories, fitness and movement, for every config, all")
    print("  integrators, head mode and early stops; the kept worker pool changes nothing")


if __name__ == "__main__":
    main()
