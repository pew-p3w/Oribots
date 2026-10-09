"""Phase 2 gate: the layered brain (R5, R16.3, R16.4, R23.8).

Property 1 (frozen-controller equivalence): for 100 random cases (trained
weights, W1, hidden size, yaw, ball position) a Zero_Output_Layer brain gives
hinge targets bit-identical to ``BallAwareCpgBrain`` at every one of 1000+
control steps. Both brains run in the same single-robot simulation: the layered
brain drives the robot and, at each control step, the trained brain receives the
same sensor state and its targets are recorded for comparison.

Also: Property 2 (layer round trip), Property 3 (first-order W2 effect), and the
refusal of a wrong-length or non-finite Layer_Parameters vector.
"""

import math

import numpy as np
from _check import check_main, require, run_property
from pyrr import Vector3

import arena
import config_loader
import layered_brain
from ball_aware_brain import BallAwareCpgBrain
from revolve2.ci_group.interactive_objects import Ball
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot_simulation import ModularRobotScene
from revolve2.simulation.scene import Pose

CONTROL_STEPS = 1000
SIMULATED_SECONDS = CONTROL_STEPS / 20.0 + 0.1


class _Capture:
    """A control interface that only records targets."""

    def __init__(self) -> None:
        self.targets = []

    def set_active_hinge_target(self, hinge, target) -> None:
        self.targets.append((hinge, target))


class _Forward(_Capture):
    """Records targets and passes them on to the real control interface."""

    def __init__(self, real) -> None:
        super().__init__()
        self._real = real

    def set_active_hinge_target(self, hinge, target) -> None:
        super().set_active_hinge_target(hinge, target)
        self._real.set_active_hinge_target(hinge, target)


class _Twin(Brain):
    def __init__(self, layered, frozen, log) -> None:
        self._layered, self._frozen, self._log = layered, frozen, log

    def make_instance(self) -> BrainInstance:
        return _TwinInstance(self._layered.make_instance(), self._frozen.make_instance(), self._log)


class _TwinInstance(BrainInstance):
    def __init__(self, layered, frozen, log) -> None:
        self._layered, self._frozen, self._log = layered, frozen, log

    def control(self, dt, sensor_state, control_interface) -> None:
        frozen = _Capture()
        self._frozen.control(dt, sensor_state, frozen)
        layered = _Forward(control_interface)
        self._layered.control(dt, sensor_state, layered)
        self._log.append((frozen.targets, layered.targets))


def _equivalence_case(settings, structure, mapping):
    def sample(rng):
        hidden = int(rng.integers(6, 11))
        return {
            "weights": rng.uniform(-1.0, 1.0, settings.cpg_plus_steering_count),
            "w1": rng.normal(0.0, 1.0, 4 * hidden),
            "hidden": hidden,
            "yaw": float(rng.uniform(-math.pi, math.pi)),
            "ball_angle": float(rng.uniform(-math.pi, math.pi)),
            "ball_distance": float(rng.uniform(0.8, 4.0)),
            "scale": float(rng.uniform(0.05, 1.0)),
        }

    def check(case):
        from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

        ball_xy = (case["ball_distance"] * math.cos(case["ball_angle"]), case["ball_distance"] * math.sin(case["ball_angle"]))
        ball = Ball(radius=settings.ball_radius, mass=settings.ball_mass, pose=Pose(Vector3([ball_xy[0], ball_xy[1], settings.ball_radius])))
        common = dict(
            cpg_network_structure=structure,
            output_mapping=mapping,
            ball=ball,
        )
        layer = np.concatenate([case["w1"], np.zeros(settings.num_active_hinges * case["hidden"])])
        layered = layered_brain.LayeredBallAwareBrain.from_params(
            case["weights"], layer,
            feedback_num_inputs=4,
            steering_output_scale=settings.feedback_output_scale,
            distance_scale=settings.feedback_distance_scale,
            reference_offset=settings.head_offset,
            hidden_size=case["hidden"],
            layer_output_scale=case["scale"],
            **common,
        )
        frozen = BallAwareCpgBrain.from_params(
            params=case["weights"],
            cpg_network_structure=structure,
            initial_state_uniform=math.sqrt(2) * 0.5,
            output_mapping=mapping,
            ball=ball,
            num_steering_inputs=4,
            steering_output_scale=settings.feedback_output_scale,
            distance_scale=settings.feedback_distance_scale,
            reference_offset=settings.head_offset,
        )
        log = []
        scene = ModularRobotScene(terrain=arena.build_walled_terrain((12.0, 12.0), 1.2, 0.2))
        scene.add_robot(ModularRobot(body=settings.body, brain=_Twin(layered, frozen, log)),
                        pose=arena.start_pose(arena.RobotStart(xy=(0.0, 0.0), yaw=case["yaw"])))
        scene.add_interactive_object(ball)
        simulation_scene, _ = scene.to_simulation_scene()
        simulate_scene(0, simulation_scene, True, None, False, 1 / 20, None, SIMULATED_SECONDS, 0.001, False, True, None)
        assert len(log) >= CONTROL_STEPS, f"only {len(log)} control steps"
        for step, (expected, actual) in enumerate(log):
            assert len(expected) == len(actual), f"step {step}: {len(actual)} targets, expected {len(expected)}"
            for (hinge_a, target_a), (hinge_b, target_b) in zip(expected, actual):
                assert hinge_a is hinge_b, f"step {step}: hinge order differs"
                assert target_a == target_b, f"step {step}: target {target_b!r} != trained controller's {target_a!r}"

    return sample, check


def run() -> None:
    settings = config_loader.load_contest(config_loader.contest_paths.MULTIBODY / "configs" / "_tiny.py")
    config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    structure, mapping = config_loader.body_wiring(settings.body)
    n = settings.num_active_hinges

    # Lengths (R5.3) and refusals (R5.4, R5.8).
    require(layered_brain.layer_param_count(8, n) == 96, "4L + nL for L=8, n=8 is not 96")
    for bad, what in ((np.zeros(95), "length"), (np.r_[np.zeros(95), np.nan], "NaN"), (np.r_[np.zeros(95), np.inf], "inf")):
        try:
            layered_brain.validate_layer_params(bad, 8, n)
        except ValueError as error:
            text = str(error)
            if what == "length":
                require("expected 96" in text and "got 95" in text, f"wrong-length message: {text}")
            else:
                require("index 95" in text, f"non-finite message: {text}")
        else:
            raise AssertionError(f"a {what} Layer_Parameters vector was accepted")

    # Property 2: round trip.
    def round_trip_sample(rng):
        hidden = int(rng.integers(6, 11))
        return hidden, rng.normal(0.0, 10.0, layered_brain.layer_param_count(hidden, n))

    def round_trip_check(case):
        hidden, vector = case
        w1, w2 = layered_brain.split_layer_params(vector, hidden, n)
        assert w1.shape == (hidden, 4) and w2.shape == (n, hidden)
        assert layered_brain.pack_layer_params(w1, w2).tobytes() == vector.tobytes()

    run_property("layer round trip", round_trip_sample, round_trip_check, seed=1)

    # Property 3: first-order W2 effect.
    def first_order_sample(rng):
        hidden = int(rng.integers(6, 11))
        x = np.array([math.sin(a := rng.uniform(-math.pi, math.pi)), math.cos(a), rng.uniform(0, 1), 1.0])
        return hidden, rng.normal(0.0, 1.0, (hidden, 4)), x, int(rng.integers(0, n)), int(rng.integers(0, hidden)), float(rng.normal(0.0, 3.0)) or 0.5, float(rng.uniform(0.05, 1.0))

    def first_order_check(case):
        hidden, w1, x, i, j, delta, scale = case
        h = np.tanh(w1 @ x)
        if h[j] == 0.0:
            return
        w2 = np.zeros((n, hidden))
        w2[i, j] = delta
        extra = layered_brain.layer_output(w1, w2, x, scale)
        expected = np.tanh(delta * h[j]) * scale
        assert extra[i] == expected, f"extra_i {extra[i]!r} != {expected!r}"
        assert all(extra[k] == 0.0 for k in range(n) if k != i), "another hinge got a non-zero output"

    run_property("first-order W2 effect", first_order_sample, first_order_check, seed=2)

    # Property 1: frozen-controller equivalence (the gate's main check).
    sample, check = _equivalence_case(settings, structure, mapping)
    run_property("frozen-controller equivalence", sample, check, seed=3)


if __name__ == "__main__":
    check_main("LAYERED_BRAIN", run)
