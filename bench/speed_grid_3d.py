"""Illustration video: a grid of rendered robots at the original speed next to
the same grid at the sped-up speed.

An ILLUSTRATION, not a measurement. N x N real trials (the latest generations'
controllers from a training run, on its training balls) are simulated once
with the fast physics and rendered, each cell with a camera following its
robot. Both grids replay those same renders; only the playback speed differs,
by --speedup. While the slow grid finishes one batch, the fast grid finishes
about --speedup batches (shuffled between cells so they look different).

Usage (from the repository root):

    .venv/Scripts/python bench/speed_grid_3d.py output/speed_compare_spider_arena/baseline
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import paths  # noqa: E402

CONFIG = ROOT / "config" / "spider_arena_cmaes.py"


def _load_config():
    paths.install(CONFIG.parent)
    os.environ["REVOLVE2_RUN_CONFIG_PATH"] = str(CONFIG)
    spec = importlib.util.spec_from_file_location("config", CONFIG)
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    return module


def _worker_init() -> None:
    _load_config()


def _render_trial(job: tuple) -> list[bytes]:
    """Simulate one trial and return JPEG frames every `every` simulated seconds."""
    task, size, every = job
    import cv2
    import mujoco

    import fast_trial

    trial = fast_trial._Trial(task)
    m, d = trial.model, trial.data
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, size, size)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 3.2, -50.0, 90.0
    robot_body = trial.robot_id

    control_step = task["control_step"]
    steps = int(round(control_step / m.opt.timestep))
    per_frame = max(1, int(round(every / control_step)))
    total = int(round(task["simulation_time"] / control_step))
    frames, stopped = [], False
    for k in range(total + 1):
        if k % per_frame == 0:
            cam.lookat[:] = [d.xpos[robot_body][0], d.xpos[robot_body][1], 0.1]
            renderer.update_scene(d, cam)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            frames.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
        if stopped or k == total:
            continue
        trial.control(control_step)
        if trial.should_stop():
            stopped = True
            continue
        mujoco.mj_step(m, d, nstep=steps)
    renderer.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", help="a training run folder with gen*.pkl")
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--cell", type=int, default=180, help="pixels per cell")
    parser.add_argument("--speedup", type=float, default=22.7)
    parser.add_argument("--slow-seconds", type=float, default=12.0,
                        help="video seconds the slow grid takes for one batch")
    parser.add_argument("--every", type=float, default=0.5, help="simulated seconds between rendered frames")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = _load_config()
    import cv2
    import numpy as np

    import evaluator

    run_dir = Path(args.run_dir).resolve()
    snaps = sorted(run_dir.glob("gen*.pkl"), key=lambda p: int(p.stem[3:]))
    controllers, balls = [], None
    for path in snaps:
        with open(path, "rb") as handle:
            snap = pickle.load(handle)
        controllers.extend(snap["population_parameters"])
        balls = list(snap["training_ball_poses"])
    controllers.reverse()  # latest generations first
    n = args.grid * args.grid
    pairs = [(c, b) for b in range(len(balls)) for c in range(len(controllers))][:n]

    ev = evaluator.Evaluator()
    common = ev.worker_task_common(False)
    common["integrator"], common["simulation_timestep"] = "implicitfast", 0.005
    jobs = [(ev.worker_task(dict(common), controllers[c], balls[b], i), args.cell, args.every)
            for i, (c, b) in enumerate(pairs)]
    print(f"simulating and rendering {n} trials...", flush=True)
    with ProcessPoolExecutor(max_workers=config.NUM_SIMULATORS, initializer=_worker_init) as pool:
        encoded = list(pool.map(_render_trial, jobs))
    ev.close()
    frames = [[cv2.imdecode(np.frombuffer(f, np.uint8), cv2.IMREAD_COLOR) for f in trial] for trial in encoded]
    length = max(len(f) for f in frames)

    g, cell = args.grid, args.cell
    side = g * cell
    gap, header, footer = 50, 100, 30
    width, height = 2 * side + 3 * gap, side + header + footer
    width += width % 2
    height += height % 2
    origins = [(gap, header), (2 * gap + side, header)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    background = np.full((height, width, 3), 245, np.uint8)
    cv2.putText(background, "Original: RK4 1 ms, original loop", (origins[0][0], 40), font, 0.85, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(background, "Rewritten: implicitfast 5 ms, fast loop", (origins[1][0], 40), font, 0.85, (40, 40, 40), 2, cv2.LINE_AA)
    corner = f"{args.speedup:g}x faster (measured)"
    (tw, _), _ = cv2.getTextSize(corner, font, 0.85, 2)
    cv2.putText(background, corner, (origins[1][0] + side - tw, 78), font, 0.85, (0, 0, 170), 2, cv2.LINE_AA)
    cv2.putText(background, "Illustration: the same trials replayed at each setup's speed",
                (gap, height - 10), font, 0.5, (110, 110, 110), 1, cv2.LINE_AA)

    def draw(img, origin, progress, order):
        idx = min(length - 1, int(progress * (length - 1)))
        for c in range(n):
            trial = frames[order[c]]
            tile = trial[min(idx, len(trial) - 1)]
            row, col = divmod(c, g)
            x0, y0 = origin[0] + col * cell, origin[1] + row * cell
            img[y0:y0 + cell, x0:x0 + cell] = tile
            cv2.rectangle(img, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (245, 245, 245), 2)

    slow_frames = int(round(args.slow_seconds * args.fps))
    fast_frames = max(2, int(round(slow_frames / args.speedup)))
    hold = args.fps * 2
    identity = np.arange(n)
    shuffles = [identity] + [np.random.default_rng(100 + k).permutation(n) for k in range(int(args.speedup) + 2)]
    out = Path(args.out) if args.out else run_dir.parent / f"speed_grid_3d_{g}x{g}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    print("writing video...", flush=True)
    for f in range(slow_frames + hold):
        img = background.copy()
        slow_p = min(1.0, f / slow_frames)
        draw(img, origins[0], slow_p, identity)
        fast_f = min(f, slow_frames)
        batch, within = divmod(fast_f, fast_frames)
        done_all = f >= slow_frames
        draw(img, origins[1], 1.0 if done_all else within / fast_frames,
             shuffles[batch % len(shuffles)])
        slow_done = n if slow_p >= 1.0 else 0
        fast_done = int(round(n * args.speedup)) if done_all else n * batch
        cv2.putText(img, f"trials done: {slow_done}", (origins[0][0], 78), font, 0.75, (0, 110, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"trials done: {fast_done}", (origins[1][0], 78), font, 0.75, (0, 110, 0), 2, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
