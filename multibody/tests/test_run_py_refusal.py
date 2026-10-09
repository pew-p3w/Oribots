"""The single-robot ``run.py -c`` refuses a multibody snapshot (R17.6).

The refusal must come from ``run.py``'s existing algorithm check, exit non-zero,
start no training and write nothing into the snapshot's folder.
"""

import subprocess
import sys

from _check import check_main, require, temp_workspace

import _runs


def run() -> None:
    with temp_workspace() as workspace:
        config = _runs.tiny_config(workspace, NUM_GENERATIONS=1)
        output = workspace / "run"
        status = _runs.run(["-r", str(config), str(output)], workspace / "train.log")
        require(status == 0, f"training exited with {status}")
        listing = sorted((p.name, p.stat().st_mtime_ns) for p in output.iterdir())
        result = subprocess.run(
            [sys.executable, str(_runs.ROOT / "run.py"), "-c", str(output / "gen1.pkl")],
            cwd=str(_runs.ROOT), capture_output=True, text=True, timeout=300,
        )
        text = result.stdout + result.stderr
        require(result.returncode != 0, "run.py -c accepted a multibody snapshot")
        require("does not name a known algorithm" in text and "multibody_cmaes" in text,
                f"run.py did not refuse through its algorithm check: {text[-1500:]}")
        require(sorted((p.name, p.stat().st_mtime_ns) for p in output.iterdir()) == listing,
                "run.py changed the snapshot's folder")


if __name__ == "__main__":
    check_main("RUN_PY_REFUSAL", run)
