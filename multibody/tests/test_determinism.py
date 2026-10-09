"""Evaluation reproducibility (Property 11, R15.5, R15.6).

The same candidates on the same layouts, evaluated inline (one worker) and in
pools of 2 and 3 worker processes, and evaluated twice, must give bit-identical
candidate scores and per-robot metrics. Uses the fixture trained weights, in
both opponent modes.
"""

import dataclasses

import numpy as np
from _check import check_main, require

import arena
import config_loader
import match
from arena_evaluator import MatchEvaluator


def evaluate(settings, layouts, candidates):
    structure, mapping = config_loader.body_wiring(settings.body)
    evaluator = MatchEvaluator(settings, structure, mapping, layouts)
    scores, records = evaluator.score_candidates(candidates, match.zero_output_layer(settings))
    flat = [(r.candidate_index, r.layout_index, r.failure, [dataclasses.astuple(m) for m in (r.metrics or [])]) for r in records.matches]
    return scores, flat


def run() -> None:
    base = config_loader.load_contest(config_loader.contest_paths.MULTIBODY / "configs" / "_tiny.py")
    config_loader.install_base_config_as_config(base.base_config_source, base.base_config_name)
    base = dataclasses.replace(base, simulation_time=10.0)
    layouts = arena.sample_layouts(
        k=base.k, num_layouts=base.num_layouts, seed=base.layout_seed, start_radius=base.start_radius,
        min_arc_gap=base.min_arc_gap, footprint=base.footprint_radius, head_offset=base.head_offset,
        possession_distance=base.possession_distance,
    )
    rng = np.random.default_rng(11)
    candidates = [rng.normal(0.0, 0.7, base.layer_param_count) for _ in range(3)]
    for mode in ("reference", "self"):
        results = {}
        for workers in (1, 2, 3, 2):
            settings = dataclasses.replace(base, worker_count=workers, opponents=mode)
            results.setdefault(workers, []).append(evaluate(settings, layouts, candidates))
        inline = results[1][0]
        for workers, runs in results.items():
            for index, result in enumerate(runs):
                require(result[0] == inline[0], f"{mode}: candidate scores with {workers} worker(s) (run {index}) differ from inline")
                require(result[1] == inline[1], f"{mode}: per-robot metrics with {workers} worker(s) (run {index}) differ from inline")


if __name__ == "__main__":
    check_main("DETERMINISM", run)
