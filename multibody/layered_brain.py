"""The contest brain: the frozen trained controller plus a small residual tanh layer.

The trained controller is the single-robot ``BallAwareCpgBrain`` (CPG gait plus
the evolved steering layer), built from the frozen Trained_Weights. On top of it
sits the Residual_Layer, which reads the same four feedback inputs::

    h     = tanh(W1 x)                         W1: L x 4 (bias through the constant input)
    extra = tanh(W2 h) * LAYER_OUTPUT_SCALE    W2: n x L, row i drives hinge i
    target_i = clip(cpg_i + steering_i + extra_i * range_i, -range_i, range_i)

Only W1 and W2 are trained. With W2 = 0 the layer adds exactly zero, so the
hinge targets equal the trained controller's (R5.5). The CPG step and the
feedback inputs are the single-robot functions themselves, imported from
``brains/ball_aware_brain.py`` (left unchanged), and the per-hinge arithmetic
repeats that brain's loop term by term.

Layer_Parameters order (R5.7): W1 row-major (L x 4), then W2 row-major (n x L).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()
import numpy.typing as npt

from ball_aware_brain import (
    BallAwareCpgBrain,
    _ball_relative_inputs,
    _rk45,
)
from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.brain.cpg import CpgNetworkStructure
from revolve2.modular_robot.sensor_state import ModularRobotSensorState
from revolve2.simulation.scene import MultiBodySystem

LAYER_PARAM_ORDER = "W1 row-major (L x 4), then W2 row-major (n x L)"
NUM_FEEDBACK_INPUTS = 4
INITIAL_STATE_UNIFORM = math.sqrt(2) * 0.5  # the single-robot evaluator's initial CPG state


def layer_param_count(hidden_size: int, num_hinges: int) -> int:
    """
    4L + nL.

    :param hidden_size: L.
    :param num_hinges: n.
    :returns: The Layer_Parameters length.
    """
    return NUM_FEEDBACK_INPUTS * hidden_size + num_hinges * hidden_size


def split_layer_params(
    vector: npt.ArrayLike, hidden_size: int, num_hinges: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split Layer_Parameters into W1 (L x 4) and W2 (n x L).

    :param vector: The flat vector.
    :param hidden_size: L.
    :param num_hinges: n.
    :returns: (W1, W2), copies.
    """
    values = np.asarray(vector, dtype=np.float64)
    split = NUM_FEEDBACK_INPUTS * hidden_size
    w1 = values[:split].reshape(hidden_size, NUM_FEEDBACK_INPUTS).copy()
    w2 = values[split:].reshape(num_hinges, hidden_size).copy()
    return w1, w2


def pack_layer_params(w1: npt.ArrayLike, w2: npt.ArrayLike) -> np.ndarray:
    """
    The exact inverse of :func:`split_layer_params`.

    :param w1: L x 4.
    :param w2: n x L.
    :returns: The flat vector.
    """
    return np.concatenate(
        [np.asarray(w1, dtype=np.float64).reshape(-1), np.asarray(w2, dtype=np.float64).reshape(-1)]
    )


def validate_layer_params(vector: npt.ArrayLike, hidden_size: int, num_hinges: int) -> np.ndarray:
    """
    Refuse a Layer_Parameters vector of the wrong length or with a non-finite value (R5.4, R5.8).

    :returns: The vector as float64.
    :raises ValueError: Naming the expected and actual length, or the first bad index.
    """
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    expected = layer_param_count(hidden_size, num_hinges)
    if values.size != expected:
        raise ValueError(
            f"Layer_Parameters: expected {expected} values (4L + nL with L={hidden_size}, "
            f"n={num_hinges}), got {values.size}"
        )
    bad = np.flatnonzero(~np.isfinite(values))
    if bad.size:
        raise ValueError(f"Layer_Parameters: non-finite value at index {int(bad[0])}")
    return values


def layer_output(w1: np.ndarray, w2: np.ndarray, inputs: np.ndarray, scale: float) -> np.ndarray:
    """
    The Residual_Layer's output before hinge-range scaling: tanh(W2 tanh(W1 x)) * scale.

    :returns: One value per hinge.
    """
    hidden = np.tanh(w1 @ inputs)
    return np.tanh(w2 @ hidden) * scale


class LayeredBallAwareBrain(Brain):
    """The frozen trained controller plus the residual tanh layer."""

    def __init__(
        self,
        frozen: BallAwareCpgBrain,
        w1: np.ndarray,
        w2: np.ndarray,
        layer_output_scale: float,
    ) -> None:
        """
        :param frozen: The trained controller, built from the Trained_Weights.
        :param w1: L x 4.
        :param w2: n x L.
        :param layer_output_scale: LAYER_OUTPUT_SCALE.
        """
        self._frozen = frozen
        self._w1 = w1
        self._w2 = w2
        self._layer_output_scale = float(layer_output_scale)

    @classmethod
    def from_params(
        cls,
        trained_weights: npt.ArrayLike,
        layer_params: npt.ArrayLike,
        *,
        cpg_network_structure: CpgNetworkStructure,
        output_mapping: list[tuple[int, ActiveHinge]],
        ball: MultiBodySystem,
        feedback_num_inputs: int,
        steering_output_scale: float,
        distance_scale: float,
        reference_offset: tuple[float, float, float],
        hidden_size: int,
        layer_output_scale: float,
    ) -> "LayeredBallAwareBrain":
        """
        Build the brain. Refuses bad Layer_Parameters before anything is built (R5.4, R5.8).

        :returns: The brain.
        """
        num_hinges = len(output_mapping)
        values = validate_layer_params(layer_params, hidden_size, num_hinges)
        frozen = BallAwareCpgBrain.from_params(
            params=np.asarray(trained_weights, dtype=np.float64),
            cpg_network_structure=cpg_network_structure,
            initial_state_uniform=INITIAL_STATE_UNIFORM,
            output_mapping=output_mapping,
            ball=ball,
            num_steering_inputs=feedback_num_inputs,
            steering_output_scale=steering_output_scale,
            distance_scale=distance_scale,
            reference_offset=reference_offset,
        )
        w1, w2 = split_layer_params(values, hidden_size, num_hinges)
        return cls(frozen, w1, w2, layer_output_scale)

    def make_instance(self) -> BrainInstance:
        """
        A fresh instance with its own CPG state (R8.3).

        :returns: The instance.
        """
        frozen = self._frozen
        return LayeredBallAwareBrainInstance(
            initial_state=frozen._initial_state.copy(),
            weight_matrix=frozen._weight_matrix.copy(),
            output_mapping=frozen._output_mapping,
            steering_matrix=frozen._steering_matrix.copy(),
            ball=frozen._ball,
            steering_output_scale=frozen._steering_output_scale,
            distance_scale=frozen._distance_scale,
            reference_offset=frozen._reference_offset,
            w1=self._w1.copy(),
            w2=self._w2.copy(),
            layer_output_scale=self._layer_output_scale,
        )


class LayeredBallAwareBrainInstance(BrainInstance):
    """Stateful instance: the single-robot control step with the layer inserted."""

    def __init__(
        self,
        *,
        initial_state: np.ndarray,
        weight_matrix: np.ndarray,
        output_mapping: list[tuple[int, ActiveHinge]],
        steering_matrix: np.ndarray,
        ball: MultiBodySystem,
        steering_output_scale: float,
        distance_scale: float,
        reference_offset: tuple[float, float, float],
        w1: np.ndarray,
        w2: np.ndarray,
        layer_output_scale: float,
    ) -> None:
        self._state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping
        self._steering_matrix = steering_matrix
        self._ball = ball
        self._steering_output_scale = steering_output_scale
        self._distance_scale = distance_scale
        self._reference_offset = reference_offset
        self._w1 = w1
        self._w2 = w2
        self._layer_output_scale = layer_output_scale

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        One control step (R5.1, R5.2).

        :param dt: Seconds since the last control step.
        :param sensor_state: Sensor state.
        :param control_interface: Hinge targets go here.
        """
        self._state = _rk45(self._state, self._weight_matrix, dt)
        steering_inputs = _ball_relative_inputs(
            sensor_state=sensor_state,
            ball=self._ball,
            distance_scale=self._distance_scale,
            reference_offset=self._reference_offset,
        )
        extra = layer_output(self._w1, self._w2, steering_inputs, self._layer_output_scale)

        for output_index, (state_index, active_hinge) in enumerate(self._output_mapping):
            cpg_target = float(self._state[state_index]) * active_hinge.range
            steering_fraction = (
                np.tanh(
                    float(
                        np.dot(
                            self._steering_matrix[output_index],
                            steering_inputs,
                        )
                    )
                )
                * self._steering_output_scale
            )
            steering_offset = steering_fraction * active_hinge.range
            layer_offset = float(extra[output_index]) * active_hinge.range
            target = np.clip(
                cpg_target + steering_offset + layer_offset,
                -active_hinge.range,
                active_hinge.range,
            )
            control_interface.set_active_hinge_target(active_hinge, float(target))
