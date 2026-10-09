"""Phase 4 gate (a): a training run of the ``_tiny`` fixture completes all its generations (R23.4).

Also checks what the run leaves behind: one snapshot per generation, the two
record files with the right row counts (R19.4), every candidate inside the
CMA-ES bounds (R16.6), the initial W1 stored as drawn from its seed (R16.2),
and nothing written outside the run folder (no ``outcmaes/``, nothing under
``Oribots/output/``).
"""

import json

import numpy as np
from _check import check_main, require, require_output_unchanged, snapshot_output, temp_workspace

import _runs
import config_loader


def run() -> None:
    before = snapshot_output()
    stray_before = set(p.name for p in _runs.ROOT.iterdir())
    with temp_workspace() as workspace:
        config = _runs.tiny_config(workspace)
        output = workspace / "run"
        status = _runs.run(["-r", str(config), str(output)], workspace / "log.txt")
        log = (workspace / "log.txt").read_text()
        require(status == 0, f"training exited with {status}:\n{log[-3000:]}")
        snapshots = sorted(p.name for p in output.glob("gen*.pkl"))
        require(snapshots == ["gen1.pkl", "gen2.pkl", "gen3.pkl"], f"snapshots {snapshots}")
        rows = _runs.record_rows(output)
        require(len(rows["candidates_"]) - 1 == 3 * 4, f"{len(rows['candidates_']) - 1} candidate rows, expected 12")
        require(len(rows["per_robot_"]) - 1 == 3 * 4 * 2 * 2, f"{len(rows['per_robot_']) - 1} per-robot rows, expected 48")

        settings = config_loader.load_contest(config)
        lower, upper = settings.cma_bounds
        for line in rows["candidates_"][1:]:
            parameters = json.loads(line.split('"')[1])
            require(len(parameters) == settings.layer_param_count, "layer_params length")
            require(all(lower <= v <= upper for v in parameters), f"a candidate left the bounds {settings.cma_bounds}")
        last = _runs.load(output / "gen3.pkl")
        expected_w1 = config_loader.initial_w1(settings.hidden_size, settings.w1_init_std, settings.w1_init_seed)
        require(np.array_equal(last["w1_init"], expected_w1), "stored initial W1 differs from its seed's draw")
        require(np.array_equal(last["trained_weights"], settings.trained_weights), "trained weights changed")
        require(last["completed_generations"] == 3, "completed generations")
    require_output_unchanged(before)
    stray = set(p.name for p in _runs.ROOT.iterdir()) - stray_before
    require(not stray, f"the run wrote into the Oribots root: {sorted(stray)}")


if __name__ == "__main__":
    check_main("TINY_TRAINING", run)
