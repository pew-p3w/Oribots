"""Phase 5 gate (code side): the pilot writes a complete report (R24.7, R24.5, R24.6).

Runs the pilot on a small, short config (fixture weights, 10 s matches, one
layout) for K = 2..5 and checks that the report holds every value of R24.1-24.4
for each K, all finite and non-negative. Also checks that a K the config cannot
fit is refused before any match and without a report. (The real pilot uses the
real trained weights (the UCT run's gen30) and the full match length; deciding the weights and the
budget from it is a human step.)
"""

import json
import math

from _check import check_main, require, temp_workspace

import _runs
import config_loader
import timing_pilot


def numbers(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numbers(item, f"{path}.{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value


def run() -> None:
    with temp_workspace() as workspace:
        config = _runs.tiny_config(workspace, "pilot", SIMULATION_TIME=10, NUM_LAYOUTS=1, START_RADIUS=1.6, MIN_ARC_GAP=1.15)
        output = workspace / "pilot"
        report = timing_pilot.run_pilot(config, output, [2, 3, 4, 5], workers=2, uct_cores=30, uct_factor=1.0, lengau_factor=2.5)
        stored = json.loads((output / "timing_pilot_report.json").read_text())
        require(stored == json.loads(json.dumps(report)), "the written report differs from the returned one")
        require((output / "timing_pilot_report.txt").is_file(), "no readable report")
        for k in ("2", "3", "4", "5"):
            row = report["per_k"].get(k)
            require(row is not None, f"no report for K = {k}")
            for key in ("seconds_per_match", "cost", "mean_possession_share", "fraction_of_matches_with_a_holder",
                        "smallest_head_distance", "mean_closest_approach", "mean_approach_speed"):
                require(key in row, f"K = {k}: missing {key}")
            for name in ("UCT", "Lengau"):
                for key in ("hours_per_generation", "total_core_hours", "population_mode_hours_per_generation"):
                    require(key in row["cost"][name], f"K = {k}: {name} cost lacks {key}")
        for path, value in numbers(report):
            require(math.isfinite(value) and value >= 0, f"report value {path} = {value} is not finite and non-negative")
        require("mean_approach_speed_all_k" in report, "no overall approach speed")
        for key in ("simulation_time", "possession_distance", "num_layouts", "layout_seed", "population_size",
                    "num_generations", "trained_weights_source", "clusters"):
            require(key in report["settings"], f"report settings lack {key}")

        tight = _runs.tiny_config(workspace, "tight", NUM_LAYOUTS=1, START_RADIUS=1.6, MIN_ARC_GAP=2.1)
        refused_dir = workspace / "refused"
        try:
            timing_pilot.run_pilot(tight, refused_dir, [2, 5], workers=1, uct_cores=30, uct_factor=1.0, lengau_factor=1.0)
        except config_loader.ConfigRefused as error:
            require("K = 5" in str(error), f"refusal does not name K = 5: {error}")
        else:
            raise AssertionError("a K whose layouts cannot fit was not refused")
        require(not refused_dir.exists(), "a refused pilot wrote a report")


if __name__ == "__main__":
    check_main("TIMING_PILOT", run)
