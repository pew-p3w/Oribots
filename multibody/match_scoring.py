"""Score one match: K Match_Scores from the robots' head-to-ball distances (R9-R13).

Inputs are compact per-robot series, not simulation states, so a worker returns
little data:

- ``RobotTrack``: d0 (head-to-ball xy distance at t = 0), the N distances at the
  samples t_1..t_N, the sample times, and the core's xy at t = 0 and every
  sample (for the escape check);
- ``ball_xy``: the ball centre's xy at t = 0 and every sample.

Terms, for robot i (glossary of requirements.md):

- closeness c_i(t) = clip(1 - d_i(t) / d0_i, -1, 1);
- T_i = sum_k w_i(t_k) c_i(t_k) / sum_k w_i(t_k), w_i(t) = exp(-t / tau_i),
  tau_i = d0_i / V_REF; TIME_i = TIME_BONUS_CAP * T_i. The weights are shifted by
  their largest value before summing (a softmax-style shift), so the sum is at
  least 1 and an all-underflow case gives T_i = c_i(t_1) (R9.7);
- holder at a sample: the one robot whose head is strictly nearest the ball and
  within POSSESSION_DISTANCE (inclusive); an exact tie means no holder (D-1);
  P_i, O_i and POSS_i = POSSESSION_WEIGHT * P_i - POSSESSION_PENALTY * O_i;
- FIRST_i: FIRST_REACH_BONUS to the holder of the earliest sample with a holder;
- MOST_i: MOST_POSSESSION_BONUS split equally among the robots with the largest
  holder count, when that count is at least 1;
- F_i = TIME_i + POSS_i + FIRST_i + MOST_i, raw and signed.

A match that cannot be scored deterministically (ball or core outside the walls,
d0 <= 0, no samples, a non-finite distance) raises :class:`MatchFailure`; the
evaluator then gives it FAILED_MATCH_SCORE and the run continues (D-11, D-13).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class MatchFailure(Exception):
    """A deterministic reason a match cannot be scored (R6.9, R9.6, R9.8, R13.7)."""


@dataclass
class RobotTrack:
    """One robot's compact match record (frozen interface, design.md)."""

    d0: float
    distances: np.ndarray  # (N,) head-to-ball xy distance at t_1..t_N
    sample_times: np.ndarray  # (N,) t_k in seconds
    core_xy: np.ndarray  # (N+1, 2) core xy at t = 0 then each t_k


@dataclass
class RobotMetrics:
    """One robot's scored result (frozen interface, design.md; the per-robot record row).

    ``time_bonus`` holds T_i, the normalised time term in [-1, 1] (R19.3); the
    score uses TIME_i = TIME_BONUS_CAP * T_i.
    """

    d0: float
    final_distance: float
    min_distance: float
    time_bonus: float
    possession_share: float
    opposed_share: float
    first_reach_bonus: float
    most_possession_bonus: float
    score: float


def time_term(d0: float, distances: np.ndarray, sample_times: np.ndarray, v_ref: float) -> float:
    """
    T_i for one robot (R9.1-9.2, R9.7).

    :param d0: Distance at t = 0 (> 0).
    :param distances: Distances at t_1..t_N.
    :param sample_times: t_1..t_N.
    :param v_ref: V_REF.
    :returns: T_i in [-1, 1].
    """
    closeness = np.clip(1.0 - distances / d0, -1.0, 1.0)
    tau = d0 / v_ref
    exponents = -(sample_times / tau)
    weights = np.exp(exponents - np.max(exponents))  # the largest weight is exactly 1
    return float(np.sum(weights * closeness) / np.sum(weights))


def score_match(
    settings: Any,
    tracks: list[RobotTrack],
    ball_xy: np.ndarray,
    roles: list[str],
    layout_index: int,
) -> list[RobotMetrics]:
    """
    Score one match (frozen signature, design.md).

    :param settings: Anything with the contest's fitness settings as attributes
        (``terrain_size``, ``possession_distance``, ``v_ref``, ``time_bonus_cap``,
        ``possession_weight``, ``possession_penalty``, ``first_reach_bonus``,
        ``most_possession_bonus``).
    :param tracks: One per slot.
    :param ball_xy: (N+1, 2) ball centre xy at t = 0 then each sample.
    :param roles: One role per slot (for messages).
    :param layout_index: The layout (for messages).
    :returns: One RobotMetrics per slot, in slot order.
    :raises MatchFailure: For a deterministic failure (D-11).
    """
    k = len(tracks)
    if k == 0:
        raise MatchFailure(f"layout {layout_index}: no robots")
    n = int(np.asarray(tracks[0].distances).size)
    if n == 0:
        raise MatchFailure(f"layout {layout_index}: the match has no samples")
    times = np.asarray(tracks[0].sample_times, dtype=np.float64)

    _escape_check(settings, tracks, np.asarray(ball_xy, dtype=np.float64), times, layout_index)

    distances = np.empty((k, n), dtype=np.float64)
    d0 = np.empty(k, dtype=np.float64)
    for slot, track in enumerate(tracks):
        value = float(track.d0)
        if not math.isfinite(value):
            raise MatchFailure(f"layout {layout_index}, slot {slot}: d0 is not finite ({value}) at t=0")
        if value <= 0.0:
            raise MatchFailure(f"layout {layout_index}, slot {slot}: d0 = {value} is not greater than 0")
        row = np.asarray(track.distances, dtype=np.float64)
        if row.size != n:
            raise MatchFailure(f"layout {layout_index}, slot {slot}: {row.size} samples, expected {n}")
        bad = np.flatnonzero(~np.isfinite(row))
        if bad.size:
            index = int(bad[0])
            raise MatchFailure(
                f"layout {layout_index}, slot {slot}: non-finite head distance at t={times[index]:.3f} s"
            )
        distances[slot] = row
        d0[slot] = value

    # Time term (R9).
    t_terms = np.array([time_term(d0[i], distances[i], times, settings.v_ref) for i in range(k)])

    # Possession (R10): strictly nearest head within the possession distance.
    nearest = distances.min(axis=0)
    tie_count = (distances == nearest).sum(axis=0)
    has_holder = (nearest <= settings.possession_distance) & (tie_count == 1)
    holder = np.where(has_holder, distances.argmin(axis=0), -1)
    holder_counts = np.array([int(np.count_nonzero(holder == i)) for i in range(k)])
    total_held = int(np.count_nonzero(has_holder))
    possession = holder_counts / n
    opposed = (total_held - holder_counts) / n

    # First reach (R11).
    first = np.zeros(k)
    held_samples = np.flatnonzero(has_holder)
    if held_samples.size:
        first[int(holder[held_samples[0]])] = settings.first_reach_bonus

    # Most possession (R12): integer counts, ties split.
    most = np.zeros(k)
    top = int(holder_counts.max())
    if top >= 1:
        winners = holder_counts == top
        most[winners] = settings.most_possession_bonus / int(np.count_nonzero(winners))

    metrics = []
    for i in range(k):
        time_bonus = settings.time_bonus_cap * t_terms[i]
        poss = settings.possession_weight * possession[i] - settings.possession_penalty * opposed[i]
        score = time_bonus + poss + first[i] + most[i]
        metrics.append(
            RobotMetrics(
                d0=float(d0[i]),
                final_distance=float(distances[i, -1]),
                min_distance=float(distances[i].min()),
                time_bonus=float(t_terms[i]),
                possession_share=float(possession[i]),
                opposed_share=float(opposed[i]),
                first_reach_bonus=float(first[i]),
                most_possession_bonus=float(most[i]),
                score=float(score),
            )
        )
    return metrics


def _escape_check(
    settings: Any,
    tracks: list[RobotTrack],
    ball_xy: np.ndarray,
    times: np.ndarray,
    layout_index: int,
) -> None:
    """
    The ball centre and every core must stay strictly inside the walls (R6.9).

    :raises MatchFailure: Naming the layout, the party and the first time it was outside.
    """
    half_x = float(settings.terrain_size[0]) / 2.0
    half_y = float(settings.terrain_size[1]) / 2.0
    all_times = np.concatenate([[0.0], times])

    def first_outside(xy: np.ndarray) -> float | None:
        xy = np.asarray(xy, dtype=np.float64)
        inside = (np.abs(xy[:, 0]) < half_x) & (np.abs(xy[:, 1]) < half_y)
        outside = np.flatnonzero(~inside)  # NaN counts as outside
        return None if outside.size == 0 else float(all_times[min(int(outside[0]), all_times.size - 1)])

    when = first_outside(ball_xy)
    if when is not None:
        raise MatchFailure(f"layout {layout_index}: the ball left the walls at t={when:.3f} s")
    for slot, track in enumerate(tracks):
        when = first_outside(track.core_xy)
        if when is not None:
            raise MatchFailure(f"layout {layout_index}, slot {slot}: the robot core left the walls at t={when:.3f} s")
