"""Check the configurable physics (option 1: integrator and timestep).

Covers, in order:

1. defaults: a config without SIM_INTEGRATOR / SIM_TIMESTEP gets RK4 at 1 ms;
2. every supported integrator, and timesteps that divide control and sampling;
3. invalid integrators and timesteps are refused with a clear message;
4. `steps_per` counts whole steps and absorbs float noise (0.05 / 0.001);
5. `batch_parameters` carries the integrator, timestep and simulation time;
6. the engine keeps RK4 as its default everywhere (old callers are unchanged);
7. the setting really reaches the compiled MuJoCo model (`model.opt`);
8. the worker path passes the integrator on, and an old task without one
   still gets RK4;
9. `run.py -t` replays with the shared physics settings;
10. every shipped config has valid physics settings;
11. a real simulation: the default physics and an explicit RK4/1 ms give
    identical fitness, and a different integrator runs and gives a finite,
    different result.

Run with `python tests/test_physics.py` (or `tests\\run_tests.bat physics`).
"""

from __future__ import annotations

import importlib.util
import inspect
import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

ROOT = paths.ROOT
TINY = ROOT / "config" / "_tiny_cmaes.py"


def _stub(**values) -> types.SimpleNamespace:
    """A config stand-in with just the given attributes."""
    return types.SimpleNamespace(**values)


def _refused(function, *args) -> str | None:
    """
    Call a function that should raise ValueError.

    :returns: None if it raised ValueError, otherwise a description of what happened.
    """
    try:
        function(*args)
    except ValueError:
        return None
    except Exception as error:  # noqa: BLE001
        return f"raised {type(error).__name__} instead of ValueError: {error}"
    return "was accepted"


def check_defaults(physics) -> list[str]:
    """1. Missing settings give the original physics."""
    problems = []
    chosen = physics.settings(_stub())
    if chosen.integrator != "RK4" or chosen.timestep != 0.001:
        problems.append(f"defaults are {chosen}, expected RK4 at 0.001 s")
    if not chosen.is_default:
        problems.append("the default settings do not report is_default")
    if physics.settings(_stub(SIM_INTEGRATOR="implicitfast")).is_default:
        problems.append("implicitfast reports is_default")
    return problems


def check_valid_settings(physics) -> list[str]:
    """2. Every integrator and every timestep that divides 50 ms and 200 ms."""
    problems = []
    for integrator in physics.INTEGRATORS:
        for timestep in (0.001, 0.002, 0.0025, 0.005, 0.01):
            try:
                chosen = physics.settings(_stub(SIM_INTEGRATOR=integrator, SIM_TIMESTEP=timestep))
            except ValueError as error:
                problems.append(f"{integrator} at {timestep} was refused: {error}")
                continue
            if chosen.integrator != integrator or chosen.timestep != timestep:
                problems.append(f"{integrator} at {timestep} came back as {chosen}")
    # An integer timestep in seconds is unusual but numerically valid input type.
    if _refused(physics.settings, _stub(SIM_TIMESTEP=1)) is None:
        pass  # 1 s does not divide 50 ms, so refusing it is correct
    else:
        problems.append("a 1 s timestep (does not divide 50 ms) was not refused")
    return problems


def check_invalid_settings(physics) -> list[str]:
    """3. Bad integrators and timesteps are refused."""
    problems = []
    for integrator in ("rk4", "Verlet", "", None, 4):
        outcome = _refused(physics.settings, _stub(SIM_INTEGRATOR=integrator))
        if outcome is not None:
            problems.append(f"integrator {integrator!r} {outcome}")
    for timestep in (0, -0.001, 0.003, 0.0007, 0.03, "0.002", True, None, float("nan")):
        outcome = _refused(physics.settings, _stub(SIM_TIMESTEP=timestep))
        if outcome is not None:
            problems.append(f"timestep {timestep!r} {outcome}")
    # The message must name the setting, so a user knows what to fix.
    try:
        physics.settings(_stub(SIM_TIMESTEP=0.003))
    except ValueError as error:
        if "SIM_TIMESTEP" not in str(error):
            problems.append(f"timestep error does not name SIM_TIMESTEP: {error}")
    try:
        physics.settings(_stub(SIM_INTEGRATOR="Verlet"))
    except ValueError as error:
        if "SIM_INTEGRATOR" not in str(error) or "implicitfast" not in str(error):
            problems.append(f"integrator error does not list the choices: {error}")
    return problems


def check_steps_per(physics) -> list[str]:
    """4. Whole-step counting."""
    problems = []
    cases = [
        (0.05, 0.001, 50),
        (0.05, 0.002, 25),
        (0.05, 0.0025, 20),
        (0.05, 0.005, 10),
        (0.2, 0.001, 200),
        (0.2, 0.0025, 80),
        (1.0 / 20, 0.001, 50),  # 0.05 / 0.001 is 50.000000000000007 in floats
        (0.05, 0.05, 1),
    ]
    for interval, timestep, expected in cases:
        try:
            got = physics.steps_per(interval, timestep)
        except ValueError as error:
            problems.append(f"steps_per({interval}, {timestep}) refused: {error}")
            continue
        if got != expected or not isinstance(got, int):
            problems.append(f"steps_per({interval}, {timestep}) = {got!r}, expected {expected}")
    for interval, timestep in ((0.05, 0.003), (0.05, 0.1), (0.2, 0.03)):
        if _refused(physics.steps_per, interval, timestep) is not None:
            problems.append(f"steps_per({interval}, {timestep}) was accepted")
    return problems


def check_batch_parameters(physics) -> list[str]:
    """5. Batch parameters carry the chosen physics."""
    problems = []
    parameters = physics.batch_parameters(_stub(SIMULATION_TIME=42))
    if (parameters.integrator, parameters.simulation_timestep, parameters.simulation_time) != ("RK4", 0.001, 42):
        problems.append(
            f"default batch parameters are {parameters.integrator}, "
            f"{parameters.simulation_timestep}, {parameters.simulation_time}"
        )
    parameters = physics.batch_parameters(
        _stub(SIMULATION_TIME=42, SIM_INTEGRATOR="implicitfast", SIM_TIMESTEP=0.005), simulation_time=7
    )
    if (parameters.integrator, parameters.simulation_timestep, parameters.simulation_time) != ("implicitfast", 0.005, 7):
        problems.append(
            f"override batch parameters are {parameters.integrator}, "
            f"{parameters.simulation_timestep}, {parameters.simulation_time}"
        )
    # Control and sampling rates must not change with the physics.
    if parameters.control_frequency != 20 or parameters.sampling_frequency != 5:
        problems.append(
            f"control/sampling changed to {parameters.control_frequency}/{parameters.sampling_frequency} Hz"
        )
    # Without an explicit config it reads the run's `config` module.
    saved = sys.modules.get("config")
    sys.modules["config"] = _stub(SIMULATION_TIME=3, SIM_INTEGRATOR="Euler")
    try:
        parameters = physics.batch_parameters()
        if parameters.integrator != "Euler" or parameters.simulation_time != 3:
            problems.append("batch_parameters() did not read the loaded config module")
    finally:
        if saved is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = saved
    return problems


def check_engine_defaults() -> list[str]:
    """6. The engine's defaults are still RK4, so unchanged callers keep their physics."""
    from revolve2.simulation.simulator import BatchParameters
    from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

    problems = []
    plain = BatchParameters(
        simulation_time=1, sampling_frequency=5, simulation_timestep=0.001, control_frequency=20
    )
    if plain.integrator != "RK4":
        problems.append(f"BatchParameters defaults to {plain.integrator}, expected RK4")
    for function in (scene_to_model, simulate_scene):
        default = inspect.signature(function).parameters.get("integrator")
        if default is None:
            problems.append(f"{function.__name__} has no integrator parameter")
        elif default.default != "RK4":
            problems.append(f"{function.__name__} defaults to {default.default!r}, expected 'RK4'")
    return problems


def _tiny_simulation_scene():
    """A real scene (simple body, ball, CPG brain) from the tiny config."""
    import numpy as np

    from ball_aware_brain import BallAwareCpgBrain, steering_parameter_count
    from revolve2.ci_group.interactive_objects import Ball
    from revolve2.modular_robot import ModularRobot
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import active_hinges_to_cpg_network_structure_neighbor
    from revolve2.modular_robot_simulation import ModularRobotScene
    from revolve2.simulation.scene import Pose
    from pyrr import Vector3

    config = sys.modules["config"]
    structure, mapping = active_hinges_to_cpg_network_structure_neighbor(
        config.BODY.find_modules_of_type(ActiveHinge)
    )
    count = structure.num_connections + steering_parameter_count(mapping, config.FEEDBACK_NUM_INPUTS)
    ball = Ball(radius=config.BALL_RADIUS, mass=config.BALL_MASS, pose=Pose(Vector3([3.0, 0.0, 0.3])))
    brain = BallAwareCpgBrain.from_params(
        params=np.full(count, 0.1),
        cpg_network_structure=structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=mapping,
        ball=ball,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
        steering_output_scale=config.FEEDBACK_OUTPUT_SCALE,
        distance_scale=config.FEEDBACK_DISTANCE_SCALE,
    )
    scene = ModularRobotScene(terrain=config.make_terrain())
    scene.add_robot(ModularRobot(body=config.BODY, brain=brain))
    scene.add_interactive_object(ball)
    simulation_scene, _ = scene.to_simulation_scene()
    return simulation_scene


def check_compiled_model() -> list[str]:
    """7. The integrator and timestep reach model.opt."""
    import mujoco

    from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model

    problems = []
    expected = {
        "RK4": mujoco.mjtIntegrator.mjINT_RK4,
        "Euler": mujoco.mjtIntegrator.mjINT_EULER,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }
    for integrator, enum in expected.items():
        model, _ = scene_to_model(
            _tiny_simulation_scene(), 0.0025, cast_shadows=False, fast_sim=False, integrator=integrator
        )
        if model.opt.integrator != enum:
            problems.append(f"{integrator}: model.opt.integrator is {model.opt.integrator}, expected {enum}")
        if abs(model.opt.timestep - 0.0025) > 1e-15:
            problems.append(f"{integrator}: model.opt.timestep is {model.opt.timestep}, expected 0.0025")
    # No integrator argument: still RK4.
    model, _ = scene_to_model(_tiny_simulation_scene(), 0.001, cast_shadows=False, fast_sim=False)
    if model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4:
        problems.append("scene_to_model without an integrator no longer compiles RK4")
    return problems


def check_worker_passes_integrator(evaluator) -> list[str]:
    """8. The worker forwards the task's integrator, and defaults old tasks to RK4."""
    import revolve2.simulators.mujoco_simulator._simulate_scene as simulate_module

    problems = []
    seen: list[str] = []

    class _Stop(Exception):
        pass

    def fake_simulate_scene(*args, integrator="MISSING", **kwargs):
        seen.append(integrator)
        raise _Stop

    original = simulate_module.simulate_scene
    simulate_module.simulate_scene = fake_simulate_scene
    try:
        base = {
            "scene_id": 0, "scene": None, "control_step": 0.05, "sample_step": 0.2,
            "simulation_time": 1, "simulation_timestep": 0.001, "cast_shadows": False,
            "fast_sim": False, "viewer_type": None,
        }
        for task in ({**base, "integrator": "implicitfast"}, base):
            try:
                evaluator.simulate_and_score_scene(task)
            except _Stop:
                pass
    finally:
        simulate_module.simulate_scene = original
    if seen != ["implicitfast", "RK4"]:
        problems.append(f"worker passed integrators {seen}, expected ['implicitfast', 'RK4']")
    source = (ROOT / "src" / "evaluator.py").read_text(encoding="utf-8")
    if '"integrator": parameters.integrator' not in source:
        problems.append("the evaluator's worker task does not carry the integrator")
    if "make_standard_batch_parameters" in source:
        problems.append("the evaluator still builds batch parameters itself instead of via physics")
    return problems


def check_replay_uses_physics() -> list[str]:
    """9. run.py -t replays with the trained physics."""
    source = (ROOT / "run.py").read_text(encoding="utf-8")
    problems = []
    if "physics.batch_parameters(config, simulation_time=simulation_time)" not in source:
        problems.append("run.py does not build its replay parameters with physics.batch_parameters")
    if "make_standard_batch_parameters()" in source:
        problems.append("run.py still builds default batch parameters (would replay with the wrong physics)")
    return problems


def check_shipped_configs(physics) -> list[str]:
    """10. Every config in config/ has valid physics settings."""
    problems = []
    saved = sys.modules.get("config")
    try:
        for path in sorted((ROOT / "config").glob("*.py")):
            if path.name == "bodies.py":
                continue
            spec = importlib.util.spec_from_file_location(f"_physics_check_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                physics.settings(module)
            except Exception as error:  # noqa: BLE001
                problems.append(f"{path.name}: {type(error).__name__}: {error}")
    finally:
        if saved is not None:
            sys.modules["config"] = saved
    return problems


def check_real_simulation(evaluator, config) -> list[str]:
    """11. Default == explicit RK4/1 ms exactly; another integrator differs but is finite."""
    import numpy as np

    problems = []
    saved = {name: getattr(config, name, None) for name in ("SIM_INTEGRATOR", "SIM_TIMESTEP", "NUM_SIMULATORS", "SIMULATION_TIME")}
    config.NUM_SIMULATORS = 1  # in-process, deterministic, and sees the attribute changes below
    config.SIMULATION_TIME = 5
    try:
        instance = evaluator.Evaluator()
        rng = np.random.default_rng(7)
        population = [np.clip(rng.normal(0.0, 0.3, instance.num_parameters), -1, 1) for _ in range(2)]
        poses = instance.sample_ball_poses(1)

        def run(**settings):
            for name in ("SIM_INTEGRATOR", "SIM_TIMESTEP"):
                if hasattr(config, name):
                    delattr(config, name)
            for name, value in settings.items():
                setattr(config, name, value)
            return instance.evaluate_on_ball_poses(population, poses)

        default = run()
        explicit = run(SIM_INTEGRATOR="RK4", SIM_TIMESTEP=0.001)
        fast = run(SIM_INTEGRATOR="implicitfast", SIM_TIMESTEP=0.002)
        if default != explicit:
            problems.append(f"default physics {default} differs from explicit RK4/1 ms {explicit}")
        if not all(math.isfinite(value) for value in fast):
            problems.append(f"implicitfast at 2 ms gave non-finite fitness {fast}")
        if fast == default:
            problems.append("implicitfast at 2 ms gave exactly the RK4 result; the setting is not applied")
    finally:
        for name, value in saved.items():
            if value is None:
                if hasattr(config, name):
                    delattr(config, name)
            else:
                setattr(config, name, value)
    return problems


def load_tiny_config():
    """Load the tiny CMA-ES config as the module named `config`."""
    paths.install(TINY.parent)
    spec = importlib.util.spec_from_file_location("config", TINY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run every check and report."""
    config = load_tiny_config()
    import evaluator
    import physics

    sections = [
        ("defaults", lambda: check_defaults(physics)),
        ("valid settings", lambda: check_valid_settings(physics)),
        ("invalid settings", lambda: check_invalid_settings(physics)),
        ("whole steps", lambda: check_steps_per(physics)),
        ("batch parameters", lambda: check_batch_parameters(physics)),
        ("engine defaults", check_engine_defaults),
        ("compiled model", check_compiled_model),
        ("worker task", lambda: check_worker_passes_integrator(evaluator)),
        ("replay", check_replay_uses_physics),
        ("shipped configs", lambda: check_shipped_configs(physics)),
        ("real simulation", lambda: check_real_simulation(evaluator, config)),
    ]
    problems = []
    for name, check in sections:
        try:
            found = check()
        except Exception as error:  # noqa: BLE001
            found = [f"crashed: {type(error).__name__}: {error}"]
        problems.extend(f"[{name}] {problem}" for problem in found)

    if problems:
        print("PHYSICS CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("PHYSICS CHECK PASSED")
    print(f"  {len(sections)} groups: defaults stay RK4 at 1 ms, every integrator and valid timestep")
    print("  is accepted and reaches the compiled model, bad values are refused clearly,")
    print("  workers and run.py -t use the same settings, and the default physics is unchanged")


if __name__ == "__main__":
    main()
