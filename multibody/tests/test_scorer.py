"""Phase 3 gate, part 1: the Match_Scorer on hand-built tracks and as properties (R9-R13).

Hand-built cases cover each scoring rule and the tie / no-holder cases; the
"FOR ALL" criteria run as seeded 100-case loops (Properties 4-10). The
deterministic-failure path is checked too: each failure raises ``MatchFailure``,
which the evaluator turns into FAILED_MATCH_SCORE.
"""

import math
from types import SimpleNamespace

import numpy as np
from _check import check_main, require, require_close, run_property

from match_scoring import MatchFailure, RobotTrack, score_match, time_term

SETTINGS = SimpleNamespace(
    terrain_size=(12.0, 12.0),
    possession_distance=0.33,
    v_ref=0.01,
    time_bonus_cap=2.0,
    possession_weight=3.0,
    possession_penalty=1.0,
    first_reach_bonus=0.5,
    most_possession_bonus=2.0,
)
FREQUENCY = 5.0


def track(d0, distances, core=(1.0, 1.0)):
    distances = np.asarray(distances, dtype=float)
    n = distances.size
    return RobotTrack(
        d0=float(d0),
        distances=distances,
        sample_times=np.arange(1, n + 1) / FREQUENCY,
        core_xy=np.tile(np.asarray(core, dtype=float), (n + 1, 1)),
    )


def ball(n, xy=(0.0, 0.0)):
    return np.tile(np.asarray(xy, dtype=float), (n + 1, 1))


def score(tracks, settings=SETTINGS):
    n = tracks[0].distances.size
    return score_match(settings, tracks, ball(n), ["self"] * len(tracks), 0)


def exact_time_term(d0, distances, times, v_ref):
    """The R9.2 formula in high precision (Python floats via math.fsum, no shift)."""
    tau = d0 / v_ref
    weights = [math.exp(-t / tau) for t in times]
    close = [max(-1.0, min(1.0, 1.0 - d / d0)) for d in distances]
    total = math.fsum(weights)
    return math.fsum(w * c for w, c in zip(weights, close)) / total if total > 0 else close[0]


def hand_built() -> None:
    # Closeness clip and time term: a robot that stays at its start distance scores 0;
    # one that sits on the ball from the first sample scores 1; one twice as far scores -1.
    n = 10
    m = score([track(4.0, [4.0] * n), track(4.0, [0.0] * n), track(4.0, [8.0] * n)])
    require(m[0].time_bonus == 0.0 and m[1].time_bonus == 1.0 and m[2].time_bonus == -1.0,
            f"time terms {[x.time_bonus for x in m]} != [0, 1, -1]")
    require(m[1].score == 2.0 * 1.0 + 3.0 * 1.0 + 0.5 + 2.0, f"holder's F {m[1].score}")

    # Earlier closeness is worth more (decaying weight).
    early = time_term(4.0, np.array([1.0] * 5 + [4.0] * 5), np.arange(1, 11) / 5.0, 0.01)
    late = time_term(4.0, np.array([4.0] * 5 + [1.0] * 5), np.arange(1, 11) / 5.0, 0.01)
    require(early > late, f"early closeness {early} not worth more than late {late}")

    # Holder: strictly nearest within the possession distance, inclusive (R10.1, R10.6).
    m = score([track(3.0, [0.33, 0.5]), track(3.0, [0.4, 0.34])])
    require(m[0].possession_share == 0.5 and m[1].possession_share == 0.0, "inclusive boundary / beyond distance")
    require(m[1].opposed_share == 0.5 and m[0].opposed_share == 0.0, "opposed shares")
    # Exact tie => no holder (R10.3, D-1); first reach skips it (R11.4).
    m = score([track(3.0, [0.2, 0.2, 0.3]), track(3.0, [0.2, 0.1, 0.25])])
    require(m[0].possession_share == 0.0 and m[1].possession_share == 2 / 3, "tie sample must have no holder")
    require(m[1].first_reach_bonus == 0.5 and m[0].first_reach_bonus == 0.0, "first reach must skip the tie sample")
    # First reach is kept after losing the ball (R11.5).
    m = score([track(3.0, [0.1, 2.0, 2.0, 2.0]), track(3.0, [2.0, 0.1, 0.1, 0.1])])
    require(m[0].first_reach_bonus == 0.5 and m[1].first_reach_bonus == 0.0, "first reach after losing the ball")
    require(m[1].most_possession_bonus == 2.0 and m[0].most_possession_bonus == 0.0, "most possession winner")
    # No holder at all (R11.3, R12.3).
    m = score([track(3.0, [1.0] * 4), track(3.0, [2.0] * 4)])
    require(all(x.first_reach_bonus == 0.0 and x.most_possession_bonus == 0.0 for x in m), "no holder => no bonuses")
    require(all(x.possession_share == 0.0 and x.opposed_share == 0.0 for x in m), "no holder => no shares")
    # Most-possession tie splits on integer counts (R12.2).
    m = score([track(3.0, [0.1, 2.0, 0.1, 2.0]), track(3.0, [2.0, 0.1, 2.0, 0.1]), track(3.0, [3.0] * 4)])
    require([x.most_possession_bonus for x in m] == [1.0, 1.0, 0.0], f"split {[x.most_possession_bonus for x in m]}")
    # Raw signed score, no clamp (R13.2).
    m = score([track(1.0, [2.0] * 5), track(1.0, [0.1] * 5)])
    require(m[0].score == 2.0 * -1.0 - 1.0 * 1.0, f"negative F not raw: {m[0].score}")

    # Underflow (R9.7): tiny d0 makes every weight after the first underflow.
    times = np.arange(1, 6) / FREQUENCY
    value = time_term(1e-6, np.array([5e-7, 1e-6, 2e-6, 0.0, 1.0]), times, 0.01)
    require(math.isfinite(value) and abs(value - 0.5) <= 1e-9, f"underflow case gives {value}, expected c(t_1) = 0.5")

    # Deterministic failures (R6.9, R9.6, R9.8, R13.7).
    cases = {
        "d0 = 0": [track(0.0, [1.0, 1.0]), track(2.0, [1.0, 1.0])],
        "non-finite d0": [track(float("nan"), [1.0, 1.0]), track(2.0, [1.0, 1.0])],
        "non-finite distance": [track(2.0, [1.0, float("inf")]), track(2.0, [1.0, 1.0])],
        "core outside": [track(2.0, [1.0, 1.0], core=(6.0, 0.0)), track(2.0, [1.0, 1.0])],
    }
    for name, tracks in cases.items():
        try:
            score(tracks)
        except MatchFailure:
            continue
        raise AssertionError(f"{name} did not raise MatchFailure")
    try:
        score_match(SETTINGS, [track(2.0, [1.0, 1.0])], ball(2, (0.0, 6.5)), ["self"], 3)
    except MatchFailure as failure:
        require("layout 3" in str(failure) and "ball" in str(failure), f"ball escape message: {failure}")
    else:
        raise AssertionError("ball outside the walls did not raise MatchFailure")
    try:
        score_match(SETTINGS, [track(2.0, [])], ball(0), ["self"], 0)
    except MatchFailure:
        pass
    else:
        raise AssertionError("a match with no samples did not raise MatchFailure")


def random_tracks(rng, k=None, n=None):
    k = int(rng.integers(2, 6)) if k is None else k
    n = int(rng.integers(1, 40)) if n is None else n
    tracks = []
    for _ in range(k):
        d0 = float(rng.uniform(0.4, 6.0))
        # Mix of near-ball, far and repeated values so ties and holders occur.
        choices = np.array([0.1, 0.2, 0.33, 0.34, 1.0, d0, 2 * d0])
        distances = np.where(rng.random(n) < 0.5, rng.choice(choices, n), rng.uniform(0.0, 3 * d0, n))
        tracks.append(track(d0, distances))
    return tracks


def properties() -> None:
    def p4_sample(rng):
        tracks = random_tracks(rng, k=1)
        better = tracks[0].distances * rng.uniform(0.0, 1.0, tracks[0].distances.size)
        return tracks[0], better

    def p4_check(case):
        base, better = case
        t_base = time_term(base.d0, base.distances, base.sample_times, 0.01)
        t_better = time_term(base.d0, better, base.sample_times, 0.01)
        assert -1.0 <= t_base <= 1.0, f"T out of [-1, 1]: {t_base}"
        assert t_better >= t_base, f"closer everywhere lowered T: {t_better} < {t_base}"

    run_property("time-bonus range and monotonicity", p4_sample, p4_check, seed=4)

    def p5_sample(rng):
        d0 = float(10 ** rng.uniform(-7, 1))
        n = int(rng.integers(1, 30))
        return d0, rng.uniform(0.0, 3 * d0, n), np.arange(1, n + 1) / FREQUENCY

    def p5_check(case):
        d0, distances, times = case
        value = time_term(d0, distances, times, 0.01)
        assert math.isfinite(value), "T not finite"
        expected = exact_time_term(d0, distances, times, 0.01)
        assert abs(value - expected) <= 1e-9, f"T {value} differs from the exact formula {expected}"

    run_property("time-bonus numerical safety", p5_sample, p5_check, seed=5)

    def p6_check(tracks):
        m = score(tracks)
        n = tracks[0].distances.size
        d = np.array([t.distances for t in tracks])
        nearest = d.min(axis=0)
        held = int(np.count_nonzero((nearest <= 0.33) & ((d == nearest).sum(axis=0) == 1)))
        shares = [x.possession_share for x in m]
        assert all(0 <= x.possession_share <= 1 and 0 <= x.opposed_share <= 1 for x in m), "share out of [0, 1]"
        assert sum(shares) <= 1 + 1e-12, "possession shares sum above 1"
        for x in m:
            assert x.possession_share + x.opposed_share <= 1 + 1e-12, "P + O above 1"
            assert abs(x.possession_share + x.opposed_share - held / n) <= 1e-9, "P + O != H / N"
        assert abs(sum(shares) - held / n) <= 1e-9, "sum P != H / N"

    run_property("possession conservation", random_tracks, p6_check, seed=6)

    def p7_check(tracks):
        m = score(tracks)
        awarded = [x for x in m if x.first_reach_bonus != 0.0]
        assert len(awarded) <= 1, "first reach awarded more than once"
        held = sum(x.possession_share for x in m) > 0
        assert (len(awarded) == 1) == held, "first reach awarded iff someone held the ball"
        assert all(x.possession_share > 0 for x in awarded), "first reach to a robot that never held"

    run_property("first-reach uniqueness", random_tracks, p7_check, seed=7)

    def p8_check(tracks):
        m = score(tracks)
        total = sum(x.most_possession_bonus for x in m)
        held = sum(x.possession_share for x in m) > 0
        if held:
            assert abs(total - 2.0) <= 1e-9, f"most-possession total {total}"
        else:
            assert total == 0.0, f"most-possession total {total} with no holder"

    run_property("most-possession conservation", random_tracks, p8_check, seed=8)

    def p9_sample(rng):
        weights = SimpleNamespace(
            terrain_size=(12.0, 12.0), possession_distance=0.33, v_ref=float(rng.uniform(0.001, 1.0)),
            time_bonus_cap=float(rng.uniform(0, 5)), possession_weight=float(rng.uniform(0, 5)),
            possession_penalty=float(rng.uniform(0, 5)), first_reach_bonus=float(rng.uniform(0, 5)),
            most_possession_bonus=float(rng.uniform(0, 5)),
        )
        return weights, random_tracks(rng)

    def p9_check(case):
        s, tracks = case
        m = score(tracks, s)
        lower = -(s.time_bonus_cap + s.possession_penalty)
        upper = s.time_bonus_cap + s.possession_weight + s.first_reach_bonus + s.most_possession_bonus
        k = len(tracks)
        for x in m:
            assert math.isfinite(x.score), "F not finite"
            assert lower - 1e-12 <= x.score <= upper + 1e-12, f"F {x.score} outside [{lower}, {upper}]"
            assert x.first_reach_bonus in (0.0, s.first_reach_bonus)
            assert x.most_possession_bonus == 0.0 or any(abs(x.most_possession_bonus - s.most_possession_bonus / j) <= 1e-12 for j in range(1, k + 1))
            poss = s.possession_weight * x.possession_share - s.possession_penalty * x.opposed_share
            assert -s.possession_penalty - 1e-12 <= poss <= s.possession_weight + 1e-12

    run_property("score finiteness and bounds", p9_sample, p9_check, seed=9)

    def p10_sample(rng):
        tracks = random_tracks(rng)
        return tracks, rng.permutation(len(tracks))

    def p10_check(case):
        tracks, permutation = case
        before = score(tracks)
        moved = [None] * len(tracks)
        for slot, target in enumerate(permutation):
            moved[target] = tracks[slot]
        after = score(moved)
        for slot, target in enumerate(permutation):
            assert after[target].score == before[slot].score, f"slot {slot} -> {target}: {after[target].score} != {before[slot].score}"

    run_property("slot permutation invariance", p10_sample, p10_check, seed=10)


def run() -> None:
    hand_built()
    properties()


if __name__ == "__main__":
    check_main("SCORER", run)
