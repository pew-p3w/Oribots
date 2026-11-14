"""Train the same config twice, baseline and fast, and compare the results.

    baseline   the original code and physics: RK4 at 1 ms, the original trial
               loop, a fresh worker pool every generation
    fast       both speed-ups: implicitfast at 5 ms, the fast trial loop, kept
               workers

Both runs use the config's SEED, so they start from the same CMA-ES state and
train on the same balls; any difference comes from the physics change (the
fast trial loop alone gives identical results, see tests/test_fast_trial.py).

Afterwards both runs' best controllers are scored again under the *baseline*
physics, on the training balls and on unseen balls. That is the real test: a
controller trained with the cheaper physics should do as well when judged with
the original physics.

Usage (from the repository root):

    .venv/Scripts/python bench/compare_training.py --config config/spider_arena_cmaes.py
    .venv/Scripts/python bench/compare_training.py --report output/speed_compare_<stamp>

Output: output/speed_compare_<stamp>/{baseline,fast}/ (normal training runs,
watchable with `run.py -t .../gen30.pkl`), report.txt and report.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402

RUNS = {
    "baseline": {
        "FAST_TRIAL": False, "PERSISTENT_WORKERS": False,
        "SIM_INTEGRATOR": "RK4", "SIM_TIMESTEP": 0.001,
    },
    "fast": {
        "FAST_TRIAL": True, "PERSISTENT_WORKERS": True,
        "SIM_INTEGRATOR": "implicitfast", "SIM_TIMESTEP": 0.005,
    },
}
BASELINE_PHYSICS = {"SIM_INTEGRATOR": "RK4", "SIM_TIMESTEP": 0.001}
UNSEEN_BALL_SEED = 777


def generated_config(config: Path, label: str, overrides: dict) -> Path:
    """The config with overrides, as a file the worker processes can load."""
    import bench

    return bench.write_generated_config(config, label, overrides)


def train(config: Path, out_dir: Path, label: str, generations: int | None = None) -> float:
    """Run one training run in a child process; return its wall-clock seconds."""
    overrides = dict(RUNS[label])
    if generations is not None:
        # CMA-ES's maxiter only stops the run; the first N generations are the
        # same as in a longer run.
        overrides["NUM_GENERATIONS"] = generations
    gen = generated_config(config, f"train_{label}", overrides)
    env = paths.child_environment(config_dir=config.parent)
    env["REVOLVE2_RUN_CONFIG_PATH"] = str(gen)
    env["REVOLVE2_RUN_MAIN_PATH"] = str(ROOT / "src" / "cmaes" / "main.py")
    env["REVOLVE2_RUN_OUTPUT_DIR"] = str(out_dir / label)
    (out_dir / label).mkdir(parents=True, exist_ok=False)
    print(f"\n=== training {label}: {RUNS[label]} ===", flush=True)
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "cmaes" / "main.py")],
        cwd=str(ROOT), env=env, check=True,
    )
    seconds = time.perf_counter() - start
    (out_dir / label / "wall_seconds.txt").write_text(f"{seconds:.3f}\n")
    print(f"=== {label} finished in {seconds / 60:.1f} min ===", flush=True)
    return seconds


def snapshots(run_dir: Path) -> list[Path]:
    """The run's gen<N>.pkl files in generation order."""
    return sorted(run_dir.glob("gen*.pkl"), key=lambda p: int(p.stem[3:]))


def load(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def evaluate_main(argv: list[str]) -> None:
    """Child mode: score weight vectors under the config this process loaded."""
    request, result_path = Path(argv[0]), Path(argv[1])
    with open(request, "rb") as handle:
        job = pickle.load(handle)
    import config
    from evaluator import Evaluator

    evaluator = Evaluator()
    poses = list(job["training_poses"])
    import numpy as np

    rng = np.random.default_rng(UNSEEN_BALL_SEED)
    unseen = [config.make_training_ball_pose(rng) for _ in range(job["num_unseen"])]
    _, trials, _ = evaluator.evaluate_on_ball_poses_full(job["weights"], poses + unseen)
    evaluator.close()
    # trials is indexed [ball][controller].
    n_train = len(poses)
    out = []
    for c in range(len(job["weights"])):
        per_ball = [float(trials[b][c]) for b in range(len(trials))]
        out.append({
            "training_mean": float(np.mean(per_ball[:n_train])),
            "unseen_mean": float(np.mean(per_ball[n_train:])),
            "trials": per_ball,
        })
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump({"results": out}, handle, indent=1)


def evaluate(config: Path, out_dir: Path, weights: list, training_poses: list, num_unseen: int) -> list[dict]:
    """Score controllers under the baseline physics in a fresh process."""
    gen = generated_config(config, "evaluate_baseline_physics", {**BASELINE_PHYSICS, "FAST_TRIAL": True})
    request = out_dir / "evaluate_request.pkl"
    result = out_dir / "evaluate_result.json"
    with open(request, "wb") as handle:
        pickle.dump({"weights": weights, "training_poses": training_poses, "num_unseen": num_unseen}, handle)
    env = paths.child_environment(config_dir=config.parent)
    env["REVOLVE2_RUN_CONFIG_PATH"] = str(gen)
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('config', {str(gen)!r}); "
        "m = importlib.util.module_from_spec(spec); sys.modules['config'] = m; spec.loader.exec_module(m); "
        f"sys.path.insert(0, {str(ROOT / 'bench')!r}); "
        "import compare_training; "
        f"compare_training.evaluate_main([{str(request)!r}, {str(result)!r}])"
    )
    print("\n=== scoring both best controllers under the baseline physics ===", flush=True)
    subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, check=True)
    return json.loads(result.read_text())["results"]


def report(config: Path, out_dir: Path, num_unseen: int) -> None:
    """Compare the two finished runs and write report.txt / report.json."""
    runs = {}
    # Compare only the generations both runs completed (a run may have been
    # stopped early). Time to generation N is measured from the run folder's
    # creation to gen<N>.pkl being written, so it includes start-up.
    common = min(len(snapshots(out_dir / label)) for label in RUNS)
    if common == 0:
        raise SystemExit(f"{out_dir}: a run has no gen*.pkl; did it start?")
    for label in RUNS:
        run_dir = out_dir / label
        snaps = snapshots(run_dir)[:common]
        rows = [load(p) for p in snaps]
        runs[label] = {
            "dir": run_dir,
            "wall": snaps[-1].stat().st_mtime - run_dir.stat().st_ctime,
            "best": [r["best_fitness"] for r in rows],
            "mean": [float(sum(r["population_fitnesses"]) / len(r["population_fitnesses"])) for r in rows],
            "best_ever": [r["best_ever_fitness"] for r in rows],
            "final": rows[-1],
            "last_snapshot": snaps[-1],
        }
    final_b, final_f = runs["baseline"]["final"], runs["fast"]["final"]
    training_poses = final_b["training_ball_poses"]
    same_balls = [(p.position.x, p.position.y) for p in training_poses] == [
        (p.position.x, p.position.y) for p in final_f["training_ball_poses"]
    ]
    scores = evaluate(
        config, out_dir,
        [final_b["best_ever_parameters"], final_f["best_ever_parameters"]],
        training_poses, num_unseen,
    )

    b, f = runs["baseline"], runs["fast"]
    lines = []
    add = lines.append
    add(f"Config: {config}")
    add(f"Runs:   {out_dir}")
    add(f"Same training balls in both runs: {same_balls}")
    add("")
    add(f"Wall-clock time for {common} generations")
    add(f"  baseline {b['wall'] / 60:7.1f} min  ({b['wall'] / common:.0f} s/generation)")
    add(f"  fast     {f['wall'] / 60:7.1f} min  ({f['wall'] / common:.0f} s/generation)   "
        f"{b['wall'] / f['wall']:.2f}x faster")
    add("")
    add("Fitness during training (each run judged by its own physics)")
    add(f"  {'gen':>4} {'baseline best':>14} {'fast best':>10} {'baseline mean':>14} {'fast mean':>10}")
    for i in range(max(len(b["best"]), len(f["best"]))):
        def cell(xs, w):
            return f"{xs[i]:{w}.4f}" if i < len(xs) else " " * w
        add(f"  {i + 1:>4} {cell(b['best'], 14)} {cell(f['best'], 10)} {cell(b['mean'], 14)} {cell(f['mean'], 10)}")
    add("")
    add("Best-ever controllers scored again under the BASELINE physics (RK4, 1 ms)")
    add(f"  {'':10} {'trained fitness':>16} {'training balls':>15} {f'{num_unseen} unseen balls':>16}")
    for label, s in zip(RUNS, scores):
        add(f"  {label:10} {runs[label]['best_ever'][-1]:16.4f} {s['training_mean']:15.4f} {s['unseen_mean']:16.4f}")
    add("")
    add("Watch them:")
    for label in RUNS:
        add(f"  .venv\\Scripts\\python run.py -t {runs[label]['last_snapshot']}")
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps({
        "config": str(config),
        "same_training_balls": same_balls,
        "wall_seconds": {k: runs[k]["wall"] for k in RUNS},
        "best_per_generation": {k: runs[k]["best"] for k in RUNS},
        "mean_per_generation": {k: runs[k]["mean"] for k in RUNS},
        "best_ever_per_generation": {k: runs[k]["best_ever"] for k in RUNS},
        "rescored_under_baseline_physics": dict(zip(RUNS, scores)),
    }, indent=1), encoding="utf-8")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        evaluate_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config/spider_arena_cmaes.py")
    parser.add_argument("--report", metavar="OUT_DIR", help="only (re)write the report for finished runs")
    parser.add_argument("--only", choices=list(RUNS), help="train only this run")
    parser.add_argument("--out", help="output folder (default output/speed_compare_<stamp>)")
    parser.add_argument("--unseen", type=int, default=8, help="unseen balls in the final scoring")
    parser.add_argument("--generations", type=int, help="override NUM_GENERATIONS for the runs")
    args = parser.parse_args()
    config = Path(args.config).resolve()
    paths.install(config.parent)
    if args.report:
        report(config, Path(args.report).resolve(), args.unseen)
        return
    out_dir = Path(args.out).resolve() if args.out else (
        ROOT / "output" / f"speed_compare_{datetime.now():%Y%m%d_%H%M%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}", flush=True)
    for label in RUNS:
        if args.only and label != args.only:
            continue
        train(config, out_dir, label, args.generations)
    if not args.only:
        report(config, out_dir, args.unseen)


if __name__ == "__main__":
    main()
