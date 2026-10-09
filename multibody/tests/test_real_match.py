"""Phase 3 gate, part 2: one real 2-robot match, and the brain's view of it (R23.3, R3.3, R3.4).

A real match of the ``_tiny`` contest (fixture trained weights, a random
candidate against the reference) must give each robot a finite score whose
terms satisfy R9.4, R10.5, R11.2 and R12.4.

From the same simulated states the brain's feedback inputs are checked:

- R3.3: at t = 0, sin/cos of the heading error equal sin/cos(alpha - psi) within
  1e-6, where alpha is the direction from the head to the ball and psi the yaw the
  layout gave the robot (so the spawn yaw and the heading the brain sees agree);
- R3.4: at every state, the distance input equals min(d / FEEDBACK_DISTANCE_SCALE, 1)
  within 1e-9, with d the head distance the scorer uses.
"""

import math

import numpy as np
from _check import check_main, require

import arena
import config_loader
import match
import match_worker
from ball_aware_brain import _ball_relative_inputs
from revolve2.modular_robot_simulation._sensor_state_impl import ModularRobotSensorStateImpl


def run() -> None:
    settings = config_loader.load_contest(config_loader.contest_paths.MULTIBODY / "configs" / "_tiny.py")
    config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    structure, mapping = config_loader.body_wiring(settings.body)
    layouts = arena.sample_layouts(
        k=settings.k, num_layouts=settings.num_layouts, seed=settings.layout_seed,
        start_radius=settings.start_radius, min_arc_gap=settings.min_arc_gap,
        footprint=settings.footprint_radius, head_offset=settings.head_offset,
        possession_distance=settings.possession_distance,
    )
    candidate = np.random.default_rng(42).normal(0.0, 0.5, settings.layer_param_count)
    reference = match.zero_output_layer(settings)
    layout_index = 1
    task = match.make_match_task(
        settings, layouts[layout_index], layout_index,
        match.slot_controllers_for(settings, candidate, layout_index, reference),
        structure, mapping, scene_id=0,
    )
    states = match_worker.simulate_match_states(task)
    outcome = match_worker.score_states(task, states)
    require(outcome.failure is None, f"the real match failed: {outcome.failure}")
    metrics = outcome.metrics
    require(len(metrics) == 2, f"{len(metrics)} metrics for 2 robots")
    require(sorted(task["roles"]) == ["Learner", "Opponent"] and task["roles"][layout_index % 2] == "Learner",
            f"roles {task['roles']} (the Learner belongs in slot {layout_index % 2})")
    for slot, m in enumerate(metrics):
        require(math.isfinite(m.score), f"slot {slot}: score {m.score} not finite")
        require(-1.0 <= m.time_bonus <= 1.0, f"slot {slot}: T {m.time_bonus} outside [-1, 1] (R9.4)")
        require(0 <= m.possession_share <= 1 and 0 <= m.opposed_share <= 1
                and m.possession_share + m.opposed_share <= 1, f"slot {slot}: shares (R10.5)")
    require(sum(m.possession_share for m in metrics) <= 1, "shares sum above 1 (R10.5)")
    require(sum(1 for m in metrics if m.first_reach_bonus) <= 1, "first reach awarded twice (R11.2)")
    held = any(m.possession_share > 0 for m in metrics)
    total_most = sum(m.most_possession_bonus for m in metrics)
    require(abs(total_most - (settings.most_possession_bonus if held else 0.0)) <= 1e-9, "most-possession total (R12.4)")

    # The brain's inputs, read from the same states.
    tracks, ball_xy = match_worker.extract_tracks(
        states, task["robots"], task["ball"], settings.head_offset,
        match.num_samples(task["simulation_time"], task["sampling_frequency"]), task["sampling_frequency"],
    )
    brains = task["scene"].handler._brains
    for slot, start in enumerate(layouts[layout_index].starts):
        body_mapping = brains[slot][1]
        head = arena.head_xy(start, settings.head_offset)
        alpha = math.atan2(0.0 - head[1], 0.0 - head[0])
        error = alpha - start.yaw
        for index, state in enumerate(states):
            sensor = ModularRobotSensorStateImpl(
                simulation_state=state._simulation_state, body_to_multi_body_system_mapping=body_mapping
            )
            inputs = _ball_relative_inputs(sensor, task["ball"], settings.feedback_distance_scale, settings.head_offset)
            if index == 0:
                require(abs(inputs[0] - math.sin(error)) <= 1e-6 and abs(inputs[1] - math.cos(error)) <= 1e-6,
                        f"slot {slot}: t=0 heading inputs ({inputs[0]:.8f}, {inputs[1]:.8f}) != "
                        f"sin/cos(alpha - psi) ({math.sin(error):.8f}, {math.cos(error):.8f}) (R3.3)")
                distance = tracks[slot].d0
            else:
                distance = tracks[slot].distances[index - 1]
            expected = min(distance / settings.feedback_distance_scale, 1.0)
            require(abs(inputs[2] - expected) <= 1e-9,
                    f"slot {slot}, state {index}: distance input {inputs[2]!r} != {expected!r} (R3.4)")


if __name__ == "__main__":
    check_main("REAL_MATCH", run)
