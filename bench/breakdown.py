"""Where does one trial's time go? Physics, the Python loop, or the brain.

Builds one real trial scene (robot, ball, brain) and times, for the same number
of physics steps:

1. ``full``      the real trial: ``simulate_scene`` with control, sampling and
                 the stop condition (what training runs today);
2. ``py_loop``   a bare Python loop calling ``mj_step`` once per step, with no
                 control: physics plus the per-step Python call overhead;
3. ``c_nstep``   ``mj_step(model, data, nstep=k)``: physics only, stepped in C;
4. ``brain``     the controller's ``control`` call alone, as many times as the
                 trial calls it.

Then ``full - py_loop`` is the cost of control, sampling and states,
``py_loop - c_nstep`` is the per-step Python overhead, and ``c_nstep`` is the
floor for this integrator and timestep. Run with ``--integrator`` and
``--timestep`` to see how much the physics settings move the floor.

Usage:

    .venv/Scripts/python bench/breakdown.py [--config config/spider_cmaes.py] [--sim-time 20]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "src"))
import bench as bench_module  # noqa: E402
import paths  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    """
    Time the parts of one trial and print them.

    :param argv: Arguments (defaults to sys.argv).
    :returns: Seconds for each part.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config/spider_cmaes.py")
    parser.add_argument("--sim-time", type=float, default=20.0)
    parser.add_argument("--integrator", default=None, help="override SIM_INTEGRATOR")
    parser.add_argument("--timestep", type=float, default=None, help="override SIM_TIMESTEP")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)

    config_path = (ROOT / args.config).resolve()
    overrides: dict[str, object] = {"SIMULATION_TIME": args.sim_time, "NUM_SIMULATORS": 1}
    if args.integrator is not None:
        overrides["SIM_INTEGRATOR"] = args.integrator
    if args.timestep is not None:
        overrides["SIM_TIMESTEP"] = args.timestep
    generated = bench_module.write_generated_config(config_path, "breakdown", overrides)
    paths.install(config_path.parent)
    config = bench_module.load_config(generated)

    import mujoco

    import physics
    import robot_frame
    from evaluator import Evaluator
    from revolve2.modular_robot_simulation._scene_simulation_state import SceneSimulationState  # noqa: F401
    from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene
    from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl
    from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
    from revolve2.simulators.mujoco_simulator.viewers._viewer_type import ViewerType

    evaluator = Evaluator()
    candidate = bench_module.make_candidates(evaluator.num_parameters, 1, args.seed, config)[0]
    ball_pose = bench_module.make_ball_poses(1, args.seed, config)[0]
    # The same settings training would use (validated by physics.settings).
    parameters = physics.batch_parameters(config)
    timestep = parameters.simulation_timestep
    integrator = parameters.integrator
    control_step = 1.0 / parameters.control_frequency
    num_steps = int(round(args.sim_time / timestep))

    def fresh():
        scene, robot, ball = evaluator._make_scene(candidate, ball_pose)
        simulation_scene, _ = scene.to_simulation_scene()
        return simulation_scene, robot, ball

    # 1. The real trial.
    simulation_scene, _, _ = fresh()
    start = time.perf_counter()
    states = simulate_scene(
        0, simulation_scene, True, None, False, control_step,
        1.0 / parameters.sampling_frequency, args.sim_time, timestep, False, False, ViewerType.CUSTOM,
        integrator=integrator,
    )
    full = time.perf_counter() - start
    reached_early = len(states) < int(args.sim_time * parameters.sampling_frequency)

    # Model for the bare loops (compiled the way simulate_scene compiles it).
    simulation_scene, _, _ = fresh()
    start = time.perf_counter()
    model, mapping = scene_to_model(
        simulation_scene, timestep, cast_shadows=False, fast_sim=False, integrator=integrator
    )
    compile_seconds = time.perf_counter() - start

    # 2. Bare Python loop.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = time.perf_counter()
    for _ in range(num_steps):
        mujoco.mj_step(model, data)
    py_loop = time.perf_counter() - start

    # 3. Physics stepped in C.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    chunk = max(1, int(round(control_step / timestep)))
    start = time.perf_counter()
    done = 0
    while done < num_steps:
        n = min(chunk, num_steps - done)
        mujoco.mj_step(model, data, nstep=n)
        done += n
    c_nstep = time.perf_counter() - start

    # 4. The brain alone.
    simulation_scene, _, _ = fresh()
    model, mapping = scene_to_model(
        simulation_scene, timestep, cast_shadows=False, fast_sim=False, integrator=integrator
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    control_interface = ControlInterfaceImpl(data=data, abstraction_to_mujoco_mapping=mapping)
    num_controls = int(args.sim_time / control_step)
    state = SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping, camera_views={})
    start = time.perf_counter()
    for _ in range(num_controls):
        simulation_scene.handler.handle(state, control_interface, control_step)
    brain = time.perf_counter() - start

    del robot_frame  # imported for its side effect on the config-dependent modules
    result = {
        "integrator": integrator,
        "timestep": timestep,
        "steps": num_steps,
        "full": full,
        "py_loop": py_loop,
        "c_nstep": c_nstep,
        "brain": brain,
        "compile": compile_seconds,
        "reached_early": reached_early,
    }
    us = 1e6 / num_steps
    print(f"one trial: {args.sim_time:g} s simulated, {num_steps} steps of {timestep * 1000:g} ms, "
          f"integrator {result['integrator']}")
    if reached_early:
        print("  NOTE: this trial reached the ball early, so 'full' covers fewer steps than the rest")
    print(f"  full trial (today)       {full:7.3f} s   {full * us:6.1f} us/step")
    print(f"  bare python mj_step loop {py_loop:7.3f} s   {py_loop * us:6.1f} us/step")
    print(f"  physics in C (nstep)     {c_nstep:7.3f} s   {c_nstep * us:6.1f} us/step")
    print(f"  brain control calls      {brain:7.3f} s   ({num_controls} calls, "
          f"{brain / num_controls * 1e6:.0f} us/call)")
    print(f"  model compile            {compile_seconds:7.3f} s   (once per trial today)")
    print("  => share of the full trial:")
    print(f"     physics floor          {c_nstep / full:6.1%}")
    print(f"     python per-step loop   {max(0.0, py_loop - c_nstep) / full:6.1%}")
    print(f"     control+sampling+state {max(0.0, full - py_loop) / full:6.1%}  (brain alone {brain / full:.1%})")
    return result


if __name__ == "__main__":
    main()
