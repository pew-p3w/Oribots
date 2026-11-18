"""Check the benchmark tools, so their numbers can be trusted.

No simulation is run. Checks:

1. `--set NAME=VALUE` parsing: literals, bare strings, and bad input;
2. the generated config runs the real config, then applies the overrides, and
   is what a fresh worker process would load (a file, not an in-memory change);
3. candidates and ball poses are reproducible from the seed, and differ for a
   different seed;
4. `compare`: identical results give speedup 1, zero difference and rank
   correlation 1; a reversed ranking gives -1; results from different trials
   are refused; ties and constant fitness are handled.

Run with `python tests/test_bench.py` (or `tests\\run_tests.bat bench`).
"""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import paths  # noqa: E402

TINY = ROOT / "config" / "_tiny_cmaes.py"


def check_overrides(bench) -> list[str]:
    """1. NAME=VALUE parsing."""
    problems = []
    parsed = bench.parse_overrides(
        ["SIM_TIMESTEP=0.002", "SIM_INTEGRATOR=implicitfast", "FLAG=True", "QUOTED='RK4'", "N=3", "EXPR=a=b"]
    )
    expected = {
        "SIM_TIMESTEP": 0.002,
        "SIM_INTEGRATOR": "implicitfast",
        "FLAG": True,
        "QUOTED": "RK4",
        "N": 3,
        "EXPR": "a=b",
    }
    if parsed != expected:
        problems.append(f"parsed {parsed}, expected {expected}")
    for bad in ("NOEQUALS", "1BAD=2", "has space=1", "=3"):
        try:
            bench.parse_overrides([bad])
        except SystemExit:
            continue
        problems.append(f"--set {bad!r} was accepted")
    return problems


def check_generated_config(bench) -> list[str]:
    """2. The generated config is the real one plus overrides, loadable from its file."""
    problems = []
    generated = bench.write_generated_config(
        TINY, "test bench/label", {"SIMULATION_TIME": 3.5, "SIM_INTEGRATOR": "Euler"}
    )
    if generated.parent != bench.GENERATED_DIR:
        problems.append(f"generated config written to {generated.parent}, not {bench.GENERATED_DIR}")
    if any(ch in generated.name for ch in " /\\"):
        problems.append(f"label was not made filename-safe: {generated.name}")

    paths.install(TINY.parent)
    spec = importlib.util.spec_from_file_location("_bench_generated_check", generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_spec = importlib.util.spec_from_file_location("_bench_real_check", TINY)
    real = importlib.util.module_from_spec(real_spec)
    real_spec.loader.exec_module(real)

    if module.SIMULATION_TIME != 3.5 or module.SIM_INTEGRATOR != "Euler":
        problems.append("overrides were not applied in the generated config")
    for name in ("TEST_FILE", "BALL_RADIUS", "CMA_POPULATION_SIZE", "FEEDBACK_NUM_INPUTS"):
        if getattr(module, name) != getattr(real, name):
            problems.append(f"{name} differs from the real config")
    for name in ("make_terrain", "make_training_ball_pose", "parameter_filename"):
        if not callable(getattr(module, name, None)):
            problems.append(f"generated config lacks {name}()")
    # Functions from the real config must see the generated config's values
    # (they share one namespace), and still work.
    if module.parameter_filename() != real.parameter_filename():
        problems.append("parameter_filename() differs in the generated config")
    return problems


def check_reproducible(bench) -> list[str]:
    """3. Same seed, same generation; another seed, another generation."""
    import numpy as np

    problems = []
    paths.install(TINY.parent)
    spec = importlib.util.spec_from_file_location("_bench_repro_config", TINY)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    first = bench.make_candidates(20, 6, 99, config)
    second = bench.make_candidates(20, 6, 99, config)
    other = bench.make_candidates(20, 6, 100, config)
    if not all(np.array_equal(a, b) for a, b in zip(first, second)):
        problems.append("the same seed gave different candidates")
    if all(np.array_equal(a, b) for a, b in zip(first, other)):
        problems.append("a different seed gave the same candidates")
    low, high = config.CMA_BOUNDS
    if any(c.min() < low or c.max() > high for c in first):
        problems.append("candidates fall outside CMA_BOUNDS")
    if len(first) != 6 or any(len(c) != 20 for c in first):
        problems.append("wrong number or length of candidates")

    poses = bench.make_ball_poses(4, 99, config)
    again = bench.make_ball_poses(4, 99, config)
    if [tuple(p.position) for p in poses] != [tuple(p.position) for p in again]:
        problems.append("the same seed gave different ball poses")
    min_distance = config.MIN_TRAINING_BALL_DISTANCE_FRACTION * min(config.TERRAIN_SIZE.x, config.TERRAIN_SIZE.y)
    for pose in poses:
        if math.hypot(pose.position.x, pose.position.y) < min_distance - 1e-9:
            problems.append("a benchmark ball is closer than the config's training minimum")
    return problems


def _result(label: str, wall: float, means: list[float], trials: list[list[float]]) -> dict:
    """A minimal benchmark result."""
    return {
        "label": label,
        "config": "config/_tiny_cmaes.py",
        "candidates": len(means),
        "balls": len(trials),
        "seed": 1,
        "sim_time": 10.0,
        "trials": len(means) * len(trials),
        "workers": 1,
        "wall_seconds": wall,
        "seconds_per_trial": wall / (len(means) * len(trials)),
        "mean_fitness": means,
        "trial_fitness": trials,
    }


def check_compare() -> list[str]:
    """4. The comparison figures."""
    import compare

    problems = []
    means = [0.1, 0.5, 0.3, 0.9, 0.2, 0.7]
    trials = [means, means]
    base = _result("base", 10.0, means, trials)

    same = compare.compare(base, _result("same", 5.0, list(means), [list(means), list(means)]))
    if abs(same["speedup"] - 2.0) > 1e-12:
        problems.append(f"half the time gave speedup {same['speedup']}, expected 2")
    if same["max_abs_diff"] != 0.0 or abs(same["rank_corr"] - 1.0) > 1e-12:
        problems.append(f"identical fitness gave diff {same['max_abs_diff']}, corr {same['rank_corr']}")
    if same["top_k_kept"] != same["top_k"]:
        problems.append("identical fitness did not keep the top k")

    reversed_means = [-m for m in means]
    flipped = compare.compare(base, _result("flip", 10.0, reversed_means, [reversed_means, reversed_means]))
    if abs(flipped["rank_corr"] + 1.0) > 1e-12:
        problems.append(f"a reversed ranking gave corr {flipped['rank_corr']}, expected -1")

    shifted = [m + 0.25 for m in means]
    moved = compare.compare(base, _result("shift", 10.0, shifted, [shifted, shifted]))
    if abs(moved["max_abs_diff"] - 0.25) > 1e-12 or abs(moved["rank_corr"] - 1.0) > 1e-12:
        problems.append("a constant shift should change diff by 0.25 but keep the ranking")

    constant = [0.5] * len(means)
    flat = compare.compare(base, _result("flat", 10.0, constant, [constant, constant]))
    if not math.isnan(flat["rank_corr"]):
        problems.append(f"constant fitness gave corr {flat['rank_corr']}, expected nan")

    for key, value in (("seed", 2), ("sim_time", 20.0), ("config", "config/other.py")):
        other = copy.deepcopy(base)
        other["label"] = "mismatch"
        other[key] = value
        try:
            compare.compare(base, other)
        except ValueError:
            continue
        problems.append(f"results with a different {key} were compared instead of refused")

    try:
        compare.main(["only_one.json"])
    except SystemExit:
        pass
    else:
        problems.append("compare.py with one result did not stop with usage")
    return problems


def main() -> None:
    """Run the checks and report."""
    import bench

    sections = [
        ("overrides", lambda: check_overrides(bench)),
        ("generated config", lambda: check_generated_config(bench)),
        ("reproducible", lambda: check_reproducible(bench)),
        ("compare", check_compare),
    ]
    problems = []
    for name, check in sections:
        try:
            found = check()
        except Exception as error:  # noqa: BLE001
            found = [f"crashed: {type(error).__name__}: {error}"]
        problems.extend(f"[{name}] {problem}" for problem in found)
    if problems:
        print("BENCH CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("BENCH CHECK PASSED")
    print("  overrides parse, generated configs equal the real config plus overrides,")
    print("  generations are reproducible from the seed, and comparisons are correct")


if __name__ == "__main__":
    main()
