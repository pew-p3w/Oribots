"""Reference-value fingerprints of the fitness calculation.

The evaluator has two halves. One runs MuJoCo; the other turns the resulting
simulation states into a fitness number. Only the second half decides what
evolution rewards, and it is a pure function of the sampled states, so it can be
driven with scripted robot and ball positions and recorded exactly.

The scenarios below deliberately include the case that scoring the final state
gets wrong: a robot that touches the ball and knocks it away, where the last
sampled state is far from the ball even though the robot reached it. Scoring the
final state instead of the closest approach turns a success into a near-miss.

`tests/evaluator_reference.json` holds the reference values;
`tests/test_evaluator.py` checks the current module against them.
"""

import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

REFERENCE_PATH = Path(__file__).resolve().parent / "evaluator_reference.json"

DECIMALS = 12
SAMPLING_FREQUENCY = 5.0
SIMULATION_TIME = 20.0

# The fitness weights the experiments run with. Defined here so the recorded
# values do not depend on whichever config happens to be present.
CONFIG_VALUES = {
    "BALL_RADIUS": 0.3,
    "BALL_REACHED_DISTANCE": 0.33,
    "NO_PROGRESS_PENALTY": 0.01,
    "NO_PROGRESS_EPSILON": 1e-6,
    "FITNESS_PROGRESS_WEIGHT": 1.0,
    "FITNESS_REACHED_BONUS_WEIGHT": 0.20,
    "FITNESS_TIME_TO_REACH_WEIGHT": 0.10,
    "FITNESS_ALIGNMENT_WEIGHT": 0.05,
    "SIMULATION_TIME": SIMULATION_TIME,
}

def make_stub_config() -> ModuleType:
    """
    Build the `config` module the evaluator reads its fitness weights from.

    :returns: A module carrying the fixed fitness settings.
    """
    module = ModuleType("config")
    for name, value in CONFIG_VALUES.items():
        setattr(module, name, value)
    return module

def _round(value: Any) -> float:
    """Round a number so float noise cannot change a fingerprint."""
    rounded = round(float(value), DECIMALS)
    return 0.0 if rounded == 0.0 else rounded

class _Position:
    """A stand-in for a simulated position."""

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z

class _Pose:
    """A stand-in for a simulated pose."""

    def __init__(self, position: _Position) -> None:
        self.position = position

class _RobotState:
    """A stand-in for a robot's simulation state."""

    def __init__(self, position: _Position) -> None:
        self._pose = _Pose(position)

    def get_pose(self) -> _Pose:
        """
        Get the robot's pose.

        :returns: The scripted pose.
        """
        return self._pose

class _InnerSimulationState:
    """A stand-in exposing ball poses the way the evaluator reads them."""

    def __init__(self, ball_position: _Position) -> None:
        self._ball_pose = _Pose(ball_position)

    def get_multi_body_system_pose(self, _ball: Any) -> _Pose:
        """
        Get the ball's pose.

        :param _ball: The ball, ignored: one ball per scripted state.
        :returns: The scripted ball pose.
        """
        return self._ball_pose

class _SceneState:
    """A stand-in for one sampled simulation state."""

    def __init__(self, robot_position: _Position, ball_position: _Position) -> None:
        self._robot_state = _RobotState(robot_position)
        self._simulation_state = _InnerSimulationState(ball_position)

    def get_modular_robot_simulation_state(self, _robot: Any) -> _RobotState:
        """
        Get the robot's state.

        :param _robot: The robot, ignored: one robot per scripted state.
        :returns: The scripted robot state.
        """
        return self._robot_state

def _states(path: list[tuple[float, float]], ball: tuple[float, float]) -> list[_SceneState]:
    """
    Build scene states from a robot path and a fixed ball position.

    :param path: Robot xy positions, one per sampled state.
    :param ball: The ball's xy position.
    :returns: Scripted scene states.
    """
    return [
        _SceneState(_Position(x, y), _Position(ball[0], ball[1], 0.3))
        for x, y in path
    ]

def _moving_ball_states(
    path: list[tuple[float, float]], ball_path: list[tuple[float, float]]
) -> list[_SceneState]:
    """
    Build scene states where the ball moves too, as it does when knocked.

    :param path: Robot xy positions.
    :param ball_path: Ball xy positions, same length as the robot path.
    :returns: Scripted scene states.
    """
    return [
        _SceneState(_Position(x, y), _Position(bx, by, 0.3))
        for (x, y), (bx, by) in zip(path, ball_path)
    ]

def scenarios() -> dict[str, list[_SceneState]]:
    """
    Build the scripted trials the fitness function is recorded against.

    :returns: Named scene-state sequences.
    """
    straight_in = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (4.8, 0.0)]
    return {
        # Walks straight at the ball and arrives.
        "reaches_ball_late": _states(straight_in + [(5.0, 0.0)], ball=(5.0, 0.0)),
        # Same, but arrives in the second sample: the time bonus should be larger.
        "reaches_ball_early": _states(
            [(0.0, 0.0), (5.0, 0.0), (5.0, 0.0), (5.0, 0.0), (5.0, 0.0), (5.0, 0.0), (5.0, 0.0)],
            ball=(5.0, 0.0),
        ),
        # Closes most of the gap but never arrives.
        "approaches_without_reaching": _states(straight_in, ball=(8.0, 0.0)),
        # Walks away from the ball: progress must be negative.
        "moves_away": _states(
            [(0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0), (-3.0, 0.0)], ball=(5.0, 0.0)
        ),
        # Never moves: the no-progress penalty applies.
        "stationary": _states([(0.0, 0.0)] * 5, ball=(5.0, 0.0)),
        # Moves at a right angle: alignment should be roughly neutral.
        "sideways": _states(
            [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)], ball=(5.0, 0.0)
        ),
        # THE REGRESSION CASE. The robot reaches the ball and knocks it away, so
        # the final sampled state is far from the ball. Scoring the final state
        # scores this as a near-miss; scoring the closest approach scores it as
        # the success it was.
        "reaches_then_ball_knocked_away": _moving_ball_states(
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (5.0, 0.0), (5.2, 0.0)],
            [(5.0, 0.0), (5.0, 0.0), (5.0, 0.0), (5.1, 0.0), (9.0, 0.0)],
        ),
        # Gets very close, then drifts away again without ever reaching.
        "closes_then_drifts_away": _states(
            [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (2.0, 0.0), (0.5, 0.0)], ball=(5.0, 0.0)
        ),
    }

def fingerprint(evaluator_module: Any) -> dict[str, Any]:
    """
    Record the fitness values the module produces for every scripted trial.

    :param evaluator_module: The module exposing the fitness helpers.
    :returns: A JSON-serializable fingerprint.
    """
    robot, ball = object(), object()
    results: dict[str, Any] = {}

    for name, scene_states in scenarios().items():
        initial = evaluator_module._robot_to_ball_distance(scene_states[0], robot, ball)
        final = evaluator_module._robot_to_ball_distance(scene_states[-1], robot, ball)
        closest = min(
            evaluator_module._robot_to_ball_distance(state, robot, ball) for state in scene_states
        )
        reached, time_bonus = evaluator_module._reach_bonuses(
            scene_states=scene_states,
            robot=robot,
            ball=ball,
            sampling_frequency=SAMPLING_FREQUENCY,
            simulation_time=SIMULATION_TIME,
        )
        results[name] = {
            "initial_distance": _round(initial),
            "final_distance": _round(final),
            "closest_distance": _round(closest),
            "progress_from_closest": _round(
                evaluator_module._signed_distance_progress(initial, closest)
            ),
            "progress_from_final": _round(
                evaluator_module._signed_distance_progress(initial, final)
            ),
            "reached_bonus": _round(reached),
            "time_to_reach_bonus": _round(time_bonus),
            "movement_alignment": _round(
                evaluator_module._movement_alignment(scene_states, robot, ball)
            ),
            "trial_fitness": _round(
                evaluator_module._trial_fitness(
                    scene_states=scene_states,
                    robot=robot,
                    ball=ball,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    simulation_time=SIMULATION_TIME,
                )
            ),
        }

    progress_cases = {
        f"{initial}->{final}": _round(evaluator_module._signed_distance_progress(initial, final))
        for initial, final in (
            (10.0, 0.0), (10.0, 5.0), (10.0, 10.0), (10.0, 15.0), (10.0, 20.0), (0.0, 0.0), (5.0, 5.0)
        )
    }

    time_cases = {
        f"index={index},n={count}": _round(
            evaluator_module._sample_index_to_time(
                index=index,
                num_samples=count,
                sampling_frequency=frequency,
                simulation_time=SIMULATION_TIME,
            )
        )
        for index, count, frequency in (
            (0, 10, SAMPLING_FREQUENCY), (5, 10, SAMPLING_FREQUENCY), (9, 10, SAMPLING_FREQUENCY),
            (0, 10, None), (5, 10, None), (9, 10, None), (0, 1, None),
        )
    }

    return {
        "settings": {
            "sampling_frequency": SAMPLING_FREQUENCY,
            "simulation_time": SIMULATION_TIME,
            "config_values": {name: _round(value) for name, value in CONFIG_VALUES.items()},
            "decimals": DECIMALS,
        },
        "trials": results,
        "signed_distance_progress": progress_cases,
        "sample_index_to_time": time_cases,
    }

def load_reference() -> dict[str, Any]:
    """
    Load the stored reference fingerprint.

    :returns: The reference values of the evaluator.
    """
    return json.loads(REFERENCE_PATH.read_text())
