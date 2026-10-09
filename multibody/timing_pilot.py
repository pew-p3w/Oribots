"""Timing and possession pilot (R24, phase 5 gate): what a contest run would cost, and how the
trained controller behaves around the ball, measured before any cluster time is spent.

For each K (default 2, 3, 4, 5) the pilot samples the config's layouts for that
K and plays one match per layout with EVERY robot running the frozen trained
controller (a Zero_Output_Layer). It reports:

- seconds per match (scene build to scores; mean, min, max) and how many matches
  ran in parallel while measuring (R24.1);
- hours per generation and total core-hours on UCT and on CHPC Lengau, for
  M = population x NUM_LAYOUTS matches per generation (the cost of both the
  "reference" and the "self" mode) and, projected, for a future
  candidates-vs-candidates "population" mode with ceil(M / K) matches (R24.2);
- the controller's possession behaviour: mean possession share, the fraction of
  matches with any holder, the closest any head came to the ball, and the mean
  closest approach (R24.3);
- its mean approach speed toward the ball, the data source for V_REF (R24.4).

Usage, from ``Oribots/``::

    python multibody/timing_pilot.py multibody/configs/spider_k4.py output/pilot_spider \\
        [--ks 2 3 4 5] [--workers 8] [--uct-cores 30] [--uct-factor 1.0] [--lengau-factor 2.5]

The report goes to ``<output>/timing_pilot_report.json`` and ``.txt``; the folder
must not hold a training run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contest_paths  # noqa: E402

contest_paths.install()
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import arena  # noqa: E402
import config_loader  # noqa: E402
import match  # noqa: E402
import match_recording  # noqa: E402
import match_worker  # noqa: E402
from match_scoring import MatchFailure  # noqa: E402

LENGAU_CORES = 24
DEFAULT_UCT_CORES = 24
REPORT_NAME = "timing_pilot_report"


def _pilot_match(job: dict[str, Any]) -> dict[str, Any]:
    """Build, simulate and score one all-frozen match; runs in a worker (or inline)."""
    start = time.perf_counter()
    settings = job["settings"]
    config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    structure, mapping = config_loader.body_wiring(settings.body)
    zero = match.zero_output_layer(settings)
    controllers = [match.SlotController(zero, "self") for _ in range(settings.k)]
    task = match.make_match_task(settings, job["layout"], job["layout_index"], controllers, structure, mapping, scene_id=job["layout_index"])
    states = match_worker.simulate_match_states(task)
    n = match.num_samples(task["simulation_time"], task["sampling_frequency"])
    result: dict[str, Any] = {"k": settings.k, "layout_index": job["layout_index"]}
    try:
        tracks, ball_xy = match_worker.extract_tracks(states, task["robots"], task["ball"], settings.head_offset, n, task["sampling_frequency"])
        outcome = match_worker.score_states(task, states)
    except MatchFailure as failure:
        tracks, outcome = None, match_worker.MatchOutcome(None, str(failure))
    result["seconds"] = time.perf_counter() - start
    result["failure"] = outcome.failure
    if tracks is not None:
        result["robots"] = [_robot_summary(track, settings.possession_distance) for track in tracks]
    if outcome.metrics is not None:
        result["possession_shares"] = [m.possession_share for m in outcome.metrics]
    return result


def _robot_summary(track: Any, possession_distance: float) -> dict[str, float]:
    """d0, closest approach, and approach speed (R24.4) of one robot."""
    distances, times = track.distances, track.sample_times
    within = np.flatnonzero(distances <= possession_distance)
    index = int(within[0]) if within.size else int(np.argmin(distances))
    closest = float(distances.min())
    speed = 0.0 if closest >= track.d0 else float((track.d0 - distances[index]) / times[index])
    return {"d0": float(track.d0), "min_distance": closest, "speed": max(speed, 0.0)}


def run_pilot(
    config_path: Path,
    output_dir: Path,
    ks: list[int],
    workers: int,
    uct_cores: int,
    uct_factor: float,
    lengau_factor: float,
) -> dict[str, Any]:
    """
    Run the pilot and write its report.

    :returns: The report.
    :raises config_loader.ConfigRefused: If the config is refused for any K (no match is simulated).
    """
    existing = match_recording.existing_run_files(output_dir)
    if existing:
        raise config_loader.ConfigRefused(f"{output_dir} holds a training run ({', '.join(existing)}); use a separate folder")
    base = config_loader.load_contest(config_path)
    per_k_settings = {}
    for k in ks:
        if k not in config_loader.ALLOWED_K:
            raise config_loader.ConfigRefused(f"K = {k}: must be one of {config_loader.ALLOWED_K}")
        problems = arena.feasibility_problems(
            k=k, terrain_size=base.terrain_size, start_radius=base.start_radius, min_arc_gap=base.min_arc_gap,
            footprint=base.footprint_radius, ball_radius=base.ball_radius, head_offset=base.head_offset,
            possession_distance=base.possession_distance,
        )
        if problems:
            raise config_loader.ConfigRefused(f"K = {k}: the config is refused:\n  " + "\n  ".join(problems))
        per_k_settings[k] = dataclasses.replace(base, k=k)

    jobs = []
    for k, settings in per_k_settings.items():
        layouts = arena.sample_layouts(
            k=k, num_layouts=settings.num_layouts, seed=settings.layout_seed, start_radius=settings.start_radius,
            min_arc_gap=settings.min_arc_gap, footprint=settings.footprint_radius, head_offset=settings.head_offset,
            possession_distance=settings.possession_distance,
        )
        jobs.extend({"settings": settings, "layout": layout, "layout_index": index} for index, layout in enumerate(layouts))

    workers = max(1, int(workers))
    if workers == 1:
        results = [_pilot_match(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_pilot_match, jobs))

    population, layouts_per_run, generations = base.cma_population_size, base.num_layouts, base.num_generations
    clusters = {"UCT": (uct_cores, uct_factor), "Lengau": (LENGAU_CORES, lengau_factor)}
    report: dict[str, Any] = {
        "settings": {
            "config": str(config_path),
            "simulation_time": base.simulation_time,
            "possession_distance": base.possession_distance,
            "num_layouts": layouts_per_run,
            "layout_seed": base.layout_seed,
            "population_size": population,
            "num_generations": generations,
            "trained_weights_source": base.trained_weights_source,
            "workers_in_parallel": workers,
            "clusters": {name: {"cores_per_job": cores, "speed_factor": factor} for name, (cores, factor) in clusters.items()},
        },
        "per_k": {},
    }
    all_speeds = []
    for k in ks:
        mine = [r for r in results if r["k"] == k]
        seconds = [r["seconds"] for r in mine]
        mean_seconds = float(np.mean(seconds))
        robots = [robot for r in mine for robot in r.get("robots", [])]
        shares = [share for r in mine for share in r.get("possession_shares", [])]
        speeds = [robot["speed"] for robot in robots]
        all_speeds.extend(speeds)
        matches_per_generation = population * layouts_per_run
        population_mode = math.ceil(matches_per_generation / k)
        cost = {}
        for name, (cores, factor) in clusters.items():
            cost[name] = {
                "hours_per_generation": math.ceil(matches_per_generation / cores) * mean_seconds * factor / 3600.0,
                "total_core_hours": generations * matches_per_generation * mean_seconds * factor / 3600.0,
                "population_mode_hours_per_generation": math.ceil(population_mode / cores) * mean_seconds * factor / 3600.0,
                "population_mode_total_core_hours": generations * population_mode * mean_seconds * factor / 3600.0,
            }
        report["per_k"][str(k)] = {
            "matches": len(mine),
            "failed_matches": sum(1 for r in mine if r["failure"] is not None),
            "seconds_per_match": {"mean": mean_seconds, "min": float(min(seconds)), "max": float(max(seconds))},
            "matches_per_generation": matches_per_generation,
            "population_mode_matches_per_generation": population_mode,
            "cost": cost,
            "mean_possession_share": float(np.mean(shares)) if shares else 0.0,
            "fraction_of_matches_with_a_holder": float(np.mean([any(s > 0 for s in r.get("possession_shares", [])) for r in mine])),
            "smallest_head_distance": float(min(robot["min_distance"] for robot in robots)) if robots else math.nan,
            "mean_closest_approach": float(np.mean([robot["min_distance"] for robot in robots])) if robots else math.nan,
            "mean_approach_speed": float(np.mean(speeds)) if speeds else 0.0,
        }
    report["mean_approach_speed_all_k"] = float(np.mean(all_speeds)) if all_speeds else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{REPORT_NAME}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / f"{REPORT_NAME}.txt").write_text(format_report(report), encoding="utf-8")
    return report


def format_report(report: dict[str, Any]) -> str:
    """A readable version of the report."""
    s = report["settings"]
    lines = [
        f"Timing and possession pilot: {s['config']}",
        f"Matches of {s['simulation_time']:g} s, {s['num_layouts']} layouts (seed {s['layout_seed']}), "
        f"{s['workers_in_parallel']} matches in parallel, POSSESSION_DISTANCE {s['possession_distance']:g} m",
        f"Planned run: population {s['population_size']} x {s['num_layouts']} layouts, {s['num_generations']} generations",
        f"Trained weights: {s['trained_weights_source']}",
        "",
    ]
    for k, row in report["per_k"].items():
        sec = row["seconds_per_match"]
        lines.append(f"K = {k}: {sec['mean']:.1f} s per match (min {sec['min']:.1f}, max {sec['max']:.1f}), "
                     f"{row['failed_matches']} failed of {row['matches']}")
        for name, cost in row["cost"].items():
            lines.append(
                f"  {name}: {cost['hours_per_generation']:.2f} h per generation, {cost['total_core_hours']:.0f} core-hours "
                f"total; population mode would be {cost['population_mode_hours_per_generation']:.2f} h / "
                f"{cost['population_mode_total_core_hours']:.0f} core-hours"
            )
        lines.append(
            f"  possession share {row['mean_possession_share']:.4f}, matches with a holder "
            f"{row['fraction_of_matches_with_a_holder']:.2f}, closest head {row['smallest_head_distance']:.3f} m, "
            f"mean closest {row['mean_closest_approach']:.3f} m, approach speed {row['mean_approach_speed']:.5f} m/s"
        )
    lines.append(f"Mean approach speed over all K (V_REF source): {report['mean_approach_speed_all_k']:.5f} m/s")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Command line."""
    parser = argparse.ArgumentParser(description="Measure a contest's cost and the trained controller's possession.")
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=None, help="matches in parallel (default: the config's worker count)")
    parser.add_argument("--uct-cores", type=int, default=DEFAULT_UCT_CORES)
    parser.add_argument("--uct-factor", type=float, default=1.0, help="UCT core-seconds per Mac second")
    parser.add_argument("--lengau-factor", type=float, default=1.0, help="Lengau core-seconds per Mac second")
    args = parser.parse_args()
    workers = args.workers
    if workers is None:
        workers = config_loader.default_worker_count()
    try:
        report = run_pilot(args.config, args.output, args.ks, workers, args.uct_cores, args.uct_factor, args.lengau_factor)
    except config_loader.ConfigRefused as error:
        print(f"Refused: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(format_report(report), end="")


if __name__ == "__main__":
    main()
