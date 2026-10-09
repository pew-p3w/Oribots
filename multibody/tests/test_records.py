"""Record completeness (Property 14, R19.4, R18.4, R19.2, R19.3, R19.7).

The Recorder is driven directly with synthetic generations (no simulation), for
100 random cases: population size, layouts, K, number of generations, failed
matches, and stop-and-resume cycles at random points, including a kill that
leaves a stray later-generation row and a half-written line. After the run the
files must hold exactly G x popsize candidate rows and G x popsize x NUM_LAYOUTS
x K per-robot rows, each key once, in the documented order, with every float
read back to the identical value.
"""

import math
import random

import numpy as np
from _check import check_main, run_property, temp_workspace

import match_recording
from arena_evaluator import CandidateRecords, MatchRecord
from match_scoring import RobotMetrics

FAILED_SCORE = -4.0


def generation_data(rng, popsize, layouts, k):
    candidates = [rng.normal(0.0, 1.0, 6) for _ in range(popsize)]
    scores = [float(rng.choice([rng.normal(), 0.5, -0.25])) for _ in range(popsize)]
    matches = []
    for c in range(popsize):
        for j in range(layouts):
            roles = ["Learner" if s == j % k else "Opponent" for s in range(k)]
            if rng.random() < 0.1:
                matches.append(MatchRecord(c, j, roles, None, "escaped", FAILED_SCORE))
            else:
                metrics = [RobotMetrics(*[float(x) for x in rng.normal(0.0, 1.0, 9)]) for _ in range(k)]
                matches.append(MatchRecord(c, j, roles, metrics, None, metrics[j % k].score))
    return candidates, scores, CandidateRecords(matches=matches)


def sample(rng):
    return {
        "popsize": int(rng.integers(2, 6)),
        "layouts": int(rng.integers(1, 4)),
        "k": int(rng.integers(2, 6)),
        "generations": int(rng.integers(1, 6)),
        "stops": sorted(set(int(x) for x in rng.integers(1, 6, size=int(rng.integers(0, 3))))),
        "seed": int(rng.integers(0, 2**31)),
    }


def check(case):
    rng = np.random.default_rng(case["seed"])
    data = [generation_data(rng, case["popsize"], case["layouts"], case["k"]) for _ in range(case["generations"])]
    with temp_workspace() as folder:
        names = match_recording.new_record_names("prop")
        recorder = match_recording.Recorder(folder, names)
        generation = 1
        stops = [s for s in case["stops"] if s < case["generations"]]
        while generation <= case["generations"]:
            recorder.write_generation(generation, *data[generation - 1], FAILED_SCORE)
            if stops and generation == stops[0]:
                stops.pop(0)
                # Killed during the next generation's write: a stray later row and a cut line.
                # Also a truncated row of a kept generation that does end in a newline: a kill
                # cannot leave one, but the trim must still drop any row that is not complete.
                with open(recorder.candidates_path, "a") as handle:
                    handle.write(f'{generation},0,0.5\n{generation + 1},0,0.5,"[1.0]"\n{generation + 1},1,0.2')
                with open(recorder.per_robot_path, "a") as handle:
                    handle.write(f"{generation},0,0,0,Learner,1\n{generation + 1},0,0,0,Learner,1,1,1,0,0,0,0,0,0.1,False\n{generation + 1},0,0")
                recorder = match_recording.Recorder(folder, names)
                recorder.trim_after(generation)
            generation += 1

        candidates = match_recording.read_rows(recorder.candidates_path)
        robots = match_recording.read_rows(recorder.per_robot_path)
        g, p, layouts, k = case["generations"], case["popsize"], case["layouts"], case["k"]
        assert len(candidates) == g * p, f"{len(candidates)} candidate rows, expected {g * p}"
        assert len(robots) == g * p * layouts * k, f"{len(robots)} per-robot rows, expected {g * p * layouts * k}"
        assert len({(r["generation"], r["candidate_index"]) for r in candidates}) == g * p, "duplicate candidate key"
        keys = {(r["generation"], r["candidate_index"], r["layout_index"], r["slot"]) for r in robots}
        assert len(keys) == len(robots), "duplicate per-robot key"
        for gen in range(1, g + 1):
            candidate_vectors, scores, records = data[gen - 1]
            expected_order = sorted(range(p), key=lambda i: (scores[i], i))
            got = [int(r["candidate_index"]) for r in candidates if int(r["generation"]) == gen]
            assert got == expected_order, f"gen {gen}: candidate order {got} != {expected_order}"
            for r in candidates:
                if int(r["generation"]) == gen:
                    assert float(r["score"]) == scores[int(r["candidate_index"])], "score did not round-trip"
            robot_order = [(int(r["candidate_index"]), int(r["layout_index"]), int(r["slot"])) for r in robots if int(r["generation"]) == gen]
            assert robot_order == [(c, j, s) for c in expected_order for j in range(layouts) for s in range(k)], "per-robot row order"
            for r in robots:
                if int(r["generation"]) != gen:
                    continue
                record = next(m for m in records.matches if m.candidate_index == int(r["candidate_index"]) and m.layout_index == int(r["layout_index"]))
                if record.metrics is None:
                    assert r["failed"] == "True" and float(r["score"]) == FAILED_SCORE and math.isnan(float(r["d0"]))
                else:
                    m = record.metrics[int(r["slot"])]
                    assert r["failed"] == "False"
                    for column in match_recording.METRIC_COLUMNS:
                        assert float(r[column]) == getattr(m, column), f"{column} did not round-trip"


def run() -> None:
    run_property("record completeness", sample, check, seed=14)


if __name__ == "__main__":
    check_main("RECORDS", run)
