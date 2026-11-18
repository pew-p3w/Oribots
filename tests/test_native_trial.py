"""The C++ trial loop (src/native_trial.py) must match the Python loop bit for bit.

For every shipped config, and for the spider configs under every integrator,
the same trials are run through ``fast_trial._run`` (Python) and
``native_trial.run`` (C++), and every sampled position and orientation, the
final CPG state, and the fitness are compared exactly (including the sign of
zero). Trials that stop early on reaching the ball are covered too.

Skipped (and passing) when the library is not built, since training then uses
the Python loop: build it with ``python src/native/build_native.py``.

    python tests/test_native_trial.py            all checks
    python tests/test_native_trial.py --quick    a reduced set
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402


def one(args: argparse.Namespace) -> None:
    """Child mode: compare both loops on one config."""
    import bench

    overrides = {"SIMULATION_TIME": args.sim_time, "SIM_INTEGRATOR": args.integrator, "SIM_TIMESTEP": args.timestep}
    if args.reach is not None:
        overrides["BALL_REACHED_DISTANCE"] = args.reach
    config_path = (ROOT / args.config).resolve()
    generated = bench.write_generated_config(config_path, f"nativetest_{config_path.stem}_{args.integrator}_{args.timestep}_{args.reach}", overrides)
    paths.install(config_path.parent)
    os.environ["REVOLVE2_RUN_CONFIG_PATH"] = str(generated)
    spec = importlib.util.spec_from_file_location("config", generated)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)

    import numpy as np

    import evaluator
    import fast_trial
    import native_trial

    if not native_trial.available():
        print("SKIP native library not available")
        return
    ev = evaluator.Evaluator()
    rng = np.random.default_rng(args.seed)
    # Some configs (spider_8ball) only sample balls in multiples of 4.
    poses = ev.sample_ball_poses(4 * ((args.balls + 3) // 4))[: args.balls] if args.reach is None else []
    if args.reach is not None:
        # Balls close to the start, so some trials stop on reaching them.
        from pyrr import Vector3
        from revolve2.simulation.scene import Pose

        poses = [Pose(Vector3([d, 0.0, config.BALL_RADIUS])) for d in (0.6, 0.9, 1.4)][: args.balls]
    common = ev.worker_task_common(True)
    problems, trials, stopped = [], 0, 0
    for c in range(args.candidates):
        scale = 3.0 if c % 3 == 2 else 1.0  # some saturating controllers
        controller = np.clip(rng.normal(0.0, config.CMA_INITIAL_STD * scale if hasattr(config, "CMA_INITIAL_STD") else 0.5 * scale, ev.num_parameters), -1.0, 1.0)
        for b, pose in enumerate(poses):
            task = ev.worker_task(dict(common), controller, pose, 0)
            py_trial = fast_trial._Trial(task)
            py = fast_trial._run(py_trial, task)
            nat_trial = fast_trial._Trial(task)
            nat = native_trial.run(nat_trial, task)
            trials += 1
            if nat is None:
                problems.append(f"candidate {c} ball {b}: native loop declined the trial")
                continue
            for name, a, n in zip(("robot_pos", "robot_quat", "ball_pos"), py, nat):
                if a.shape != n.shape or a.tobytes() != n.tobytes():
                    diff = "shape" if a.shape != n.shape else f"max |diff| {np.max(np.abs(a - n)):.3e}"
                    problems.append(f"candidate {c} ball {b}: {name} differs ({a.shape} vs {n.shape}, {diff})")
            if py_trial.state.tobytes() != nat_trial.state.tobytes():
                problems.append(f"candidate {c} ball {b}: final CPG state differs")
            f_py = fast_trial._score(*py, task)
            f_nat = fast_trial._score(*nat, task)
            if repr(f_py) != repr(f_nat):
                problems.append(f"candidate {c} ball {b}: fitness {f_py[0]!r} vs {f_nat[0]!r}")
            end_time = config.SIMULATION_TIME
            if len(py[0]) < int(end_time * 5) + 1:
                stopped += 1
    ev.close()
    if problems:
        print("FAIL " + "; ".join(problems[:5]))
        sys.exit(1)
    print(f"OK {trials} trials identical ({stopped} stopped early)")


def run_one(config: str, integrator: str, timestep: float, sim_time: float, candidates: int,
            balls: int, reach: float | None = None) -> tuple[bool, str]:
    command = [sys.executable, __file__, "--one", "--config", config, "--integrator", integrator,
               "--timestep", str(timestep), "--sim-time", str(sim_time),
               "--candidates", str(candidates), "--balls", str(balls)]
    if reach is not None:
        command += ["--reach", str(reach)]
    out = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    lines = [line for line in out.stdout.splitlines() if line.startswith(("OK", "FAIL", "SKIP"))]
    last = lines[-1] if lines else (out.stderr.strip().splitlines() or ["(no output)"])[-1]
    return out.returncode == 0, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--integrator", default="implicitfast")
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument("--sim-time", type=float, default=4.0)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--balls", type=int, default=2)
    parser.add_argument("--reach", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.one:
        one(args)
        return

    sys.path.insert(0, str(ROOT / "src"))
    paths.install(ROOT / "config")
    import native_trial  # noqa: F401  (import check only; availability is checked per child)

    failures = []

    def check(label: str, *a, **k) -> None:
        ok, message = run_one(*a, **k)
        print(f"  {'ok  ' if ok else 'FAIL'} {label:48} {message}")
        if message.startswith("SKIP"):
            print("NATIVE_TRIAL CHECK SKIPPED: the library is not built or does not match this "
                  "MuJoCo (python src/native/build_native.py); training uses the Python loop.")
            sys.exit(0)
        if not ok:
            failures.append(label)

    configs = sorted(p for p in (ROOT / "config").glob("*.py") if p.name not in ("bodies.py", "__init__.py"))
    if args.quick:
        configs = [p for p in configs if p.stem in ("spider_cmaes", "spider_8ball_cmaes", "_tiny")]
    print("Every config, implicitfast 5 ms:")
    for path in configs:
        check(path.name, f"config/{path.name}", "implicitfast", 0.005, 4.0, 3, 2)
    print("Spider configs (core and head mode), every integrator:")
    settings = [("RK4", 0.001), ("Euler", 0.001), ("implicit", 0.002), ("implicitfast", 0.0025)]
    if args.quick:
        settings = settings[:1]
    for name in ("spider_cmaes.py", "spider_8ball_cmaes.py"):
        for integrator, timestep in settings:
            check(f"{name} {integrator} {timestep}", f"config/{name}", integrator, timestep, 3.37, 2, 2)
    print("Trials that stop on reaching the ball:")
    for name in ("spider_cmaes.py", "spider_8ball_cmaes.py"):
        check(f"{name} reach 0.9", f"config/{name}", "implicitfast", 0.005, 9.37, 4, 3, reach=0.9)

    if failures:
        print(f"NATIVE_TRIAL CHECK FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("NATIVE_TRIAL CHECK PASSED")


if __name__ == "__main__":
    main()
