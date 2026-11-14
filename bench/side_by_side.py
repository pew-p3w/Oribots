"""Render one controller under the baseline and the fast physics, side by side.

The same weights, body and ball are simulated twice, once with RK4 at 1 ms and
once with implicitfast at 5 ms, through the fast trial loop (which matches the
original loop exactly), and written to one video, left and right.

Usage (from the repository root):

    .venv/Scripts/python bench/side_by_side.py output/speed_compare_spider_arena/baseline/gen8.pkl
    .venv/Scripts/python bench/side_by_side.py <gen.pkl> --ball 1 --seconds 200 --out video.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402

SETTINGS = [
    ("baseline: RK4, 1 ms", "RK4", 0.001),
    ("fast: implicitfast, 5 ms", "implicitfast", 0.005),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("snapshot", help="gen<N>.pkl whose best-ever controller to show")
    parser.add_argument("--config", default="config/spider_arena_cmaes.py")
    parser.add_argument("--ball", type=int, default=None, help="training ball index (default: the best one)")
    parser.add_argument("--seconds", type=float, default=None, help="simulated seconds (default: config)")
    parser.add_argument("--every", type=float, default=0.2, help="simulated seconds between frames")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", default=None, help="video path (default: next to the snapshot)")
    args = parser.parse_args()

    cfg = Path(args.config).resolve()
    paths.install(cfg.parent)
    os.environ["REVOLVE2_RUN_CONFIG_PATH"] = str(cfg)
    spec = importlib.util.spec_from_file_location("config", cfg)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)

    import cv2
    import mujoco
    import numpy as np

    import evaluator
    import fast_trial

    snap_path = Path(args.snapshot).resolve()
    with open(snap_path, "rb") as handle:
        snap = pickle.load(handle)
    weights = np.asarray(snap["best_ever_parameters"])
    balls = snap["training_ball_poses"]
    seconds = float(args.seconds if args.seconds is not None else config.SIMULATION_TIME)

    ev = evaluator.Evaluator()
    common = ev.worker_task_common(False)
    common["simulation_time"] = seconds

    def task_for(ball_index: int, integrator: str, timestep: float) -> dict:
        task = ev.worker_task(dict(common), weights, balls[ball_index], 0)
        task["integrator"] = integrator
        task["simulation_timestep"] = timestep
        return task

    # Score every training ball under both physics; show the best ball by default.
    scores = {}
    for b in range(len(balls)):
        for name, integrator, timestep in SETTINGS:
            fitness, _ = fast_trial.simulate_and_score(task_for(b, integrator, timestep))
            scores[(b, name)] = float(fitness)
    ball = args.ball if args.ball is not None else max(
        range(len(balls)), key=lambda b: scores[(b, SETTINGS[0][0])]
    )
    for b in range(len(balls)):
        print(f"ball {b} at ({balls[b].position.x:+.2f}, {balls[b].position.y:+.2f}): "
              + "   ".join(f"{name} {scores[(b, name)]:.4f}" for name, _, _ in SETTINGS))
    print(f"showing ball {ball}")

    trials = [fast_trial._Trial(task_for(ball, i, t)) for _, i, t in SETTINGS]
    width, height = 640, 480
    renderers = []
    for trial in trials:
        mujoco.mj_forward(trial.model, trial.data)
        renderers.append(mujoco.Renderer(trial.model, height, width))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance = 11.5
    cam.elevation = -55
    cam.azimuth = 90

    control_step = common["control_step"]
    steps_per_control = [int(round(control_step / t.model.opt.timestep)) for t in trials]
    controls_per_frame = max(1, int(round(args.every / control_step)))
    stopped_at = [None, None]

    out = Path(args.out) if args.out else snap_path.with_name(f"side_by_side_{snap_path.stem}_ball{ball}.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * width, height))

    def label(img, lines):
        y = 28
        for text in lines:
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28

    total_controls = int(round(seconds / control_step))
    k = 0
    while True:
        t_sim = k * control_step
        if k % controls_per_frame == 0 or k == total_controls:
            halves = []
            for i, (trial, renderer) in enumerate(zip(trials, renderers)):
                renderer.update_scene(trial.data, cam)
                img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                status = (f"reached ball at {stopped_at[i]:.1f} s" if stopped_at[i] is not None
                          else f"t = {t_sim:6.1f} s")
                label(img, [SETTINGS[i][0], status, f"trial fitness {scores[(ball, SETTINGS[i][0])]:.4f}"])
                halves.append(img)
            writer.write(np.concatenate(halves, axis=1))
        if k >= total_controls or all(s is not None for s in stopped_at):
            break
        for i, trial in enumerate(trials):
            if stopped_at[i] is not None:
                continue
            trial.control(control_step)
            if trial.should_stop():
                stopped_at[i] = t_sim
                continue
            mujoco.mj_step(trial.model, trial.data, nstep=steps_per_control[i])
        k += 1
    writer.release()
    ev.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
