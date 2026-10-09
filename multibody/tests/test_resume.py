"""Phase 4 gate (b)-(d): resume after a real SIGTERM and a real SIGKILL (Property 12, R18, R23.4).

Run A trains the ``_tiny`` contest uninterrupted. Run B gets SIGTERM (sent to the
runner) during generation 2: the runner must stop the training child and all
its workers within 10 s (R18.7), leaving gen1.pkl as a complete resume point.
Run C gets SIGKILL (sent to the whole training process tree) during generation
2; then a stray generation-2 row and a half-written line are appended to its
record files, as a kill during a write would leave them. C is resumed, killed
again during generation 3, and resumed again.

After resuming, B and C must equal A: every snapshot bit-identical apart from the
record file names (CMA-ES state compared by the candidates it draws next), and
every record row identical, with the row counts of R19.4.
"""

import signal
import time

from _check import check_main, require, temp_workspace

import _runs

GENERATIONS = 3


def latest(folder):
    return max(folder.glob("gen*.pkl"), key=lambda p: int(p.stem[3:]))


def interrupt(workspace, config, name, signum, plant_partial_rows=False, cycles=1):
    output = workspace / name
    process = _runs.start(["-r", str(config), str(output)], workspace / f"{name}.log")
    for cycle in range(cycles):
        _runs.wait_for(output / f"gen{cycle + 1}.pkl", process)
        time.sleep(2.0)  # well into the next generation (one generation takes ~10 s)
        tree = [process.pid] + _runs.descendants(process.pid)
        sent = time.monotonic()
        if signum == signal.SIGTERM:
            process.send_signal(signal.SIGTERM)
        else:
            _runs.kill_training_tree(process, signal.SIGKILL)
        process.wait(timeout=30)
        deadline = sent + 10.0
        while _runs.alive(tree) and time.monotonic() < deadline:
            time.sleep(0.1)
        remaining = _runs.alive(tree)
        require(not remaining, f"{name}: processes still running 10 s after the signal: {remaining} (R18.7)")
        resume_from = latest(output)
        require(resume_from.name == f"gen{cycle + 1}.pkl", f"{name}: latest snapshot {resume_from.name} after the kill")
        _runs.load(resume_from)  # complete and loadable
        if plant_partial_rows:
            for path in output.glob("candidates_*_run_*.csv"):
                with open(path, "a") as handle:
                    handle.write(f'{cycle + 2},0,0.5,"[0.1, 0.2]"\n{cycle + 2},1,0.25,"[0.1, 0.')
            for path in output.glob("per_robot_*_run_*.csv"):
                with open(path, "a") as handle:
                    handle.write(f"{cycle + 2},0,0,0,Learner,1,1,1,0,0,0,0,0,0.1,False\n{cycle + 2},0,0,1,Opp")
        process = _runs.start(["-c", str(resume_from)], workspace / f"{name}.resume{cycle}.log")
    status = process.wait(timeout=1200)
    require(status == 0, f"{name}: the final resume exited with {status}")
    return output


def run() -> None:
    with temp_workspace() as workspace:
        config = _runs.tiny_config(workspace)
        reference = workspace / "uninterrupted"
        status = _runs.run(["-r", str(config), str(reference)], workspace / "a.log")
        require(status == 0, f"uninterrupted run exited with {status}")

        runs = {
            "sigterm": interrupt(workspace, config, "sigterm", signal.SIGTERM),
            "sigkill": interrupt(workspace, config, "sigkill", signal.SIGKILL, plant_partial_rows=True, cycles=2),
        }
        expected_rows = _runs.record_rows(reference)
        for name, output in runs.items():
            for generation in range(1, GENERATIONS + 1):
                a = _runs.load(reference / f"gen{generation}.pkl")
                b = _runs.load(output / f"gen{generation}.pkl")
                differ = _runs.snapshot_differences(a, b)
                require(not differ, f"{name}: gen{generation}.pkl differs from the uninterrupted run in {differ}")
            rows = _runs.record_rows(output)
            for kind, lines in rows.items():
                require(lines == expected_rows[kind], f"{name}: {kind} records differ from the uninterrupted run")
            require(len(rows["candidates_"]) - 1 == GENERATIONS * 4, f"{name}: candidate row count")
            require(len(rows["per_robot_"]) - 1 == GENERATIONS * 4 * 2 * 2, f"{name}: per-robot row count")


if __name__ == "__main__":
    check_main("RESUME", run)
