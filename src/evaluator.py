"""Evaluate controllers: simulate each one and score how well it approached the ball.

This is the only evaluator in the project. Both the EA and CMA-ES use it, so
the two algorithms always score controllers identically.

It takes plain parameter vectors rather than genotypes, so neither algorithm's
representation leaks in here.

Fitness for one trial combines four terms, weighted by the config:

- progress, how much of the starting distance to the ball was closed;
- a bonus for reaching the ball at all;
- a bonus for reaching it sooner;
- how well the robot's movement pointed at the ball.

Two design decisions matter most:

Progress is measured from the **closest approach**, not the final position.
When a robot reaches the ball it usually knocks it away, and the simulator
samples one more state after the stop condition fires, so the final state can
sit far from the ball in exactly the runs that went best. Scoring the final
state would turn those successes into near-misses.

Fitness is returned **raw and signed**, and is never clipped here. Progress is
measured at the closest approach, which includes the start, so it is never
negative; a trial goes below zero only through the no-progress penalty and a
negative alignment (moving away from the ball), about -0.06 at worst with the
shipped weights. CMA-ES uses and records these raw values. The EA records the
mean clamped to `[0, 1]` but, by default, selects on the unclamped mean
(`SELECT_ON_RAW_FITNESS`), because a strong population ties at the clamp.

Head mode (a config with `USE_HEAD_COORDINATES = True` and `HEAD_OFFSET`)
measures the fitness, the "ball reached" stop condition and the movement report
from the robot's head instead of its core; see `src/robot_frame.py`.

Every trial can also be returned individually, with a report of how the robot
moved relative to its heading (`evaluate_on_ball_poses_full`), which is what the
per-individual recording needs, at no extra simulation cost. Trials are
simulated and scored inside the worker processes by default
(`src/scoring_worker.py`); `REVOLVE2_SCORE_IN_WORKERS=0` switches to one
simulation batch per ball instead, with identical results.

`tests/test_evaluator.py` checks the scoring against the reference values in
`tests/evaluator_reference.json`, and `tests/test_head.py` checks head mode.
"""

import logging
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

import numpy as np
import numpy.typing as npt

import config
import robot_frame
import scoring_worker
from ball_aware_brain import BallAwareCpgBrain, steering_parameter_count

from revolve2.ci_group.interactive_objects import Ball
from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain.cpg import (
    CpgNetworkStructure,
    active_hinges_to_cpg_network_structure_neighbor,
)
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    simulate_scenes,
)
from revolve2.simulation.scene import Pose


class Evaluator:
    """
    Evaluates a population of controllers by running simulations.

    A "controller" is either a plain parameter vector or any object with a
    ``parameters`` attribute (such as the EA's ``Genotype``).
    """

    _simulator: Any
    _rng: np.random.Generator
    _cpg_network_structure: CpgNetworkStructure
    _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(self) -> None:
        """Initialize the evaluator using settings from config."""
        logging.info("Importing LocalSimulator.")
        from revolve2.simulators.mujoco_simulator import LocalSimulator

        logging.info("Creating LocalSimulator.")
        self._simulator = LocalSimulator(
            headless=True,
            num_simulators=config.NUM_SIMULATORS,
        )
        self._rng = make_rng_time_seed()
        active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
        (
            self._cpg_network_structure,
            self._output_mapping,
        ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

    @property
    def num_parameters(self) -> int:
        """
        Number of evolved CPG and steering parameters for this robot body.

        :returns: The number of parameters.
        """
        return self._cpg_network_structure.num_connections + steering_parameter_count(
            output_mapping=self._output_mapping,
            num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
        )

    def sample_ball_pose(self) -> Pose:
        """
        Sample a training ball pose using this evaluator's random number generator.

        :returns: A random training ball pose.
        """
        training_pose_factory = getattr(
            config,
            "make_training_ball_pose",
            config.make_random_ball_pose,
        )
        return training_pose_factory(self._rng)

    def sample_ball_poses(self, num_poses: int) -> list[Pose]:
        """
        Sample the fixed training ball poses for a run.

        A config may define ``make_training_ball_poses(rng, num_poses)`` to
        choose the whole set at once (for example to spread the balls evenly
        around the robot). Otherwise each pose is drawn independently.

        :param num_poses: Number of poses to sample.
        :returns: Training ball poses.
        :raises ValueError: If the config's factory returns the wrong number.
        """
        set_factory = getattr(config, "make_training_ball_poses", None)
        if set_factory is None:
            return [self.sample_ball_pose() for _ in range(num_poses)]
        poses = list(set_factory(self._rng, num_poses))
        if len(poses) != num_poses:
            raise ValueError(
                f"make_training_ball_poses returned {len(poses)} poses, but "
                f"{num_poses} were requested."
            )
        return poses

    def evaluate_on_ball_poses(
        self, population: list[Any], ball_poses: list[Pose]
    ) -> list[float]:
        """
        Evaluate controllers on fixed ball poses and average signed progress.

        :param population: Controllers to evaluate.
        :param ball_poses: Shared ball poses to test every controller on.
        :returns: Raw signed mean fitness per controller.
        :raises ValueError: If no ball poses are provided.
        """
        mean_fitnesses, _ = self.evaluate_on_ball_poses_detailed(
            population,
            ball_poses,
        )
        return mean_fitnesses

    def evaluate_on_ball_poses_detailed(
        self, population: list[Any], ball_poses: list[Pose]
    ) -> tuple[list[float], list[list[float]]]:
        """
        Evaluate controllers on fixed ball poses, keeping every trial score.

        Same as ``evaluate_on_ball_poses_full`` without the movement records.

        :param population: Controllers to evaluate.
        :param ball_poses: Shared ball poses to test every controller on.
        :returns: Tuple of (mean fitness per controller, per-trial fitnesses).
        """
        mean_fitnesses, trial_fitnesses, _ = self._evaluate_on_poses(
            population, ball_poses, with_movement=False
        )
        return mean_fitnesses, trial_fitnesses

    def evaluate_on_ball_poses_full(
        self, population: list[Any], ball_poses: list[Pose]
    ) -> tuple[list[float], list[list[float]], list[list[dict[str, float]]]]:
        """
        Evaluate controllers on fixed ball poses, keeping every trial score and
        how each trial moved relative to the robot's heading.

        The movement of every trial comes from the states already simulated for
        scoring, so recording it costs no extra simulations.

        :param population: Controllers to evaluate.
        :param ball_poses: Shared ball poses to test every controller on.
        :returns: Tuple of (mean fitness per controller, per-trial fitnesses,
            per-trial movement records), the last two indexed [trial][individual].
        """
        mean_fitnesses, trial_fitnesses, movements = self._evaluate_on_poses(
            population, ball_poses, with_movement=True
        )
        assert movements is not None
        return mean_fitnesses, trial_fitnesses, movements

    def _evaluate_on_poses(
        self,
        population: list[Any],
        ball_poses: list[Pose],
        with_movement: bool,
    ) -> tuple[list[float], list[list[float]], list[list[dict[str, float]]] | None]:
        """
        Evaluate controllers on fixed ball poses (shared implementation).

        The trial matrix is indexed ``trial_fitnesses[trial_index][individual]``
        and is always the raw signed score. The mean returned alongside it is
        calculated from exactly this matrix, so recording costs no extra
        simulations.

        :param population: Controllers to evaluate.
        :param ball_poses: Shared ball poses to test every controller on.
        :param with_movement: Also collect the movement record of every trial.
        :returns: Tuple of (raw mean fitness per controller, per-trial fitnesses,
            per-trial movement records or None).
        :raises ValueError: If no ball poses are provided.
        """
        if len(ball_poses) == 0:
            raise ValueError("At least one ball pose is required.")

        fitnesses_by_pose = []
        movements_by_pose: list[list[dict[str, float]]] = []
        # On by default. To use one simulation batch per ball instead, set the
        # environment variable REVOLVE2_SCORE_IN_WORKERS=0 (this also works for a
        # resumed run, whose config is frozen inside its snapshot), or set
        # SCORE_IN_WORKERS = False in the config. Both give identical results.
        if (
            os.environ.get("REVOLVE2_SCORE_IN_WORKERS", "1") != "0"
            and getattr(config, "SCORE_IN_WORKERS", True)
        ):
            fitnesses_by_pose, movements_by_pose = self._evaluate_poses_in_workers(
                population, ball_poses, with_movement
            )
        else:
            for index, ball_pose in enumerate(ball_poses):
                logging.info(
                    f"Evaluating {len(population)} controllers on training ball "
                    f"{index} at x={ball_pose.position.x:.4f}, "
                    f"y={ball_pose.position.y:.4f}."
                )
                fitnesses, movements = self._evaluate_scenes(
                    population, ball_pose, with_movement=with_movement
                )
                fitnesses_by_pose.append(fitnesses)
                if with_movement:
                    movements_by_pose.append(movements)

        fitness_array = np.asarray(fitnesses_by_pose, dtype=float)
        mean_fitnesses = np.mean(fitness_array, axis=0)

        return (
            [float(value) for value in mean_fitnesses],
            [
                [float(value) for value in trial_values]
                for trial_values in fitness_array.tolist()
            ],
            movements_by_pose if with_movement else None,
        )

    def evaluate(
        self, population: list[Any], ball_pose: Pose | None = None
    ) -> list[float]:
        """
        Evaluate a list of controllers on one ball pose.

        :param population: Controllers to evaluate (parameter vectors or
            objects with a ``parameters`` attribute).
        :param ball_pose: Optional shared ball pose for this evaluation batch.
        :returns: Raw signed trial fitness, one per controller.
        """
        fitnesses, _ = self._evaluate_scenes(population, ball_pose, with_movement=False)
        return fitnesses

    def _make_scene(
        self, controller: Any, ball_pose: Pose
    ) -> tuple[ModularRobotScene, ModularRobot, Ball]:
        """
        Build the scene for one controller and one ball pose.

        :param controller: A parameter vector or an object with ``parameters``.
        :param ball_pose: Where the ball starts.
        :returns: The scene, the robot and the ball in it.
        """
        ball = Ball(
            radius=config.BALL_RADIUS,
            mass=config.BALL_MASS,
            pose=_copy_pose(ball_pose),
        )
        brain = BallAwareCpgBrain.from_params(
            params=np.asarray(getattr(controller, "parameters", controller)),
            cpg_network_structure=self._cpg_network_structure,
            initial_state_uniform=math.sqrt(2) * 0.5,
            output_mapping=self._output_mapping,
            ball=ball,
            num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
            steering_output_scale=config.FEEDBACK_OUTPUT_SCALE,
            distance_scale=config.FEEDBACK_DISTANCE_SCALE,
            reference_offset=robot_frame.reference_offset(),
        )
        robot = ModularRobot(body=config.BODY, brain=brain)
        scene = ModularRobotScene(terrain=config.make_terrain())
        scene.add_robot(robot)
        scene.add_interactive_object(ball)
        # Stops measuring from the same point as the fitness below (the head
        # if the config uses head coordinates), so they agree on "reached".
        scene.add_stop_condition(robot_frame.make_stop_condition(robot, ball))
        return scene, robot, ball

    def _evaluate_poses_in_workers(
        self,
        population: list[Any],
        ball_poses: list[Pose],
        with_movement: bool,
    ) -> tuple[list[list[float]], list[list[dict[str, float]]]]:
        """
        Simulate and score every (ball pose, controller) trial at once.

        All trials of all controllers go into one pool of worker processes, and
        each worker scores its own simulation, returning only a fitness and a
        small movement record instead of thousands of simulated states. The main
        process therefore never waits to score a whole batch, and its memory
        stays flat. Nothing is shared between workers:
        each simulation runs in its own process, and results are put back by
        trial index, so the answers do not depend on which worker finishes first.

        At most a couple of scenes per worker are built ahead of time, so memory
        does not grow with the number of trials.

        :param population: Controllers to evaluate.
        :param ball_poses: Ball poses to test every controller on.
        :param with_movement: Also collect each trial's movement record.
        :returns: Fitness and movement records, indexed ``[pose][controller]``.
        """
        num_controllers, num_poses = len(population), len(ball_poses)
        total = num_controllers * num_poses
        parameters = make_standard_batch_parameters()
        parameters.simulation_time = config.SIMULATION_TIME
        common = {
            "config_path": config.__file__,
            "control_step": 1.0 / parameters.control_frequency,
            "sample_step": (
                None
                if parameters.sampling_frequency is None
                else 1.0 / parameters.sampling_frequency
            ),
            "sampling_frequency": parameters.sampling_frequency,
            "simulation_time": parameters.simulation_time,
            "simulation_timestep": parameters.simulation_timestep,
            "cast_shadows": getattr(self._simulator, "_cast_shadows", False),
            "fast_sim": getattr(self._simulator, "_fast_sim", False),
            "viewer_type": getattr(self._simulator, "_viewer_type"),
            "with_movement": with_movement,
        }

        def make_task(index: int) -> dict[str, Any]:
            pose_index, controller_index = divmod(index, num_controllers)
            scene, robot, ball = self._make_scene(
                population[controller_index], ball_poses[pose_index]
            )
            simulation_scene, mapping = scene.to_simulation_scene()
            return {
                **common,
                "scene_id": index,
                "scene": simulation_scene,
                "mapping": mapping,
                "robot": robot,
                "ball": ball,
            }

        for pose_index, ball_pose in enumerate(ball_poses):
            logging.info(
                f"Evaluating {num_controllers} controllers on training ball "
                f"{pose_index} at x={ball_pose.position.x:.4f}, "
                f"y={ball_pose.position.y:.4f}."
            )
        workers = max(1, int(config.NUM_SIMULATORS))
        logging.info(
            f"Simulating {total} trials ({num_controllers} controllers x "
            f"{num_poses} balls) at once on {workers} worker process(es); each "
            "worker scores its own simulation."
        )

        results: list[tuple[float, dict[str, float] | None] | None] = [None] * total
        if workers == 1:
            for index in range(total):
                results[index] = scoring_worker.simulate_and_score(make_task(index))
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            pending: dict[Any, int] = {}
            next_index = done = 0
            report_every = max(1, total // 20)
            try:
                while done < total:
                    while next_index < total and len(pending) < 2 * workers:
                        future = executor.submit(
                            scoring_worker.simulate_and_score, make_task(next_index)
                        )
                        pending[future] = next_index
                        next_index += 1
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        results[pending.pop(future)] = future.result()
                        done += 1
                        if done % report_every == 0 or done == total:
                            logging.info(f"Finished {done} / {total} trials.")
            except BaseException:
                # Something failed (or the run was interrupted): stop everything now
                # instead of letting in-flight simulations run to their end. The
                # workers share nothing, so stopping them is safe.
                workers_running = list(getattr(executor, "_processes", {}).values())
                executor.shutdown(wait=False, cancel_futures=True)
                for process in workers_running:
                    process.terminate()
                raise
            executor.shutdown(wait=True)

        fitnesses = [
            [results[p * num_controllers + c][0] for c in range(num_controllers)]  # type: ignore[index]
            for p in range(num_poses)
        ]
        movements = [
            [results[p * num_controllers + c][1] for c in range(num_controllers)]  # type: ignore[index]
            for p in range(num_poses)
        ]
        return fitnesses, movements

    def _evaluate_scenes(
        self,
        population: list[Any],
        ball_pose: Pose | None,
        with_movement: bool,
    ) -> tuple[list[float], list[dict[str, float]]]:
        """
        Simulate controllers on one ball pose and score them.

        :param population: Controllers to evaluate.
        :param ball_pose: Optional shared ball pose for this evaluation batch.
        :param with_movement: Also describe how each trial moved relative to the
            robot's heading (see robot_frame.movement_report).
        :returns: Raw signed trial fitness and (if requested) the movement
            record, one of each per controller.
        """
        robots = []
        balls = []
        scenes = []
        if ball_pose is None:
            ball_pose = self.sample_ball_pose()
        for controller in population:
            scene, robot, ball = self._make_scene(controller, ball_pose)
            robots.append(robot)
            balls.append(ball)
            scenes.append(scene)

        batch_parameters = make_standard_batch_parameters()
        batch_parameters.simulation_time = config.SIMULATION_TIME

        logging.info(f"Starting simulation batch with {len(scenes)} scenes.")
        all_scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=batch_parameters,
            scenes=scenes,
        )
        logging.info("Simulation batch returned to evaluator.")

        fitnesses = []
        movements = []
        for robot, ball, scene_states in zip(robots, balls, all_scene_states):
            fitnesses.append(
                _trial_fitness(
                    scene_states=scene_states,
                    robot=robot,
                    ball=ball,
                    sampling_frequency=batch_parameters.sampling_frequency,
                    simulation_time=config.SIMULATION_TIME,
                )
            )
            if with_movement:
                movements.append(
                    robot_frame.movement_report(scene_states, robot, ball)
                )

        return fitnesses, movements


def simulate_and_score_scene(task: dict[str, Any]) -> tuple[float, dict[str, float] | None]:
    """
    Simulate one trial and score it (this runs inside a worker process).

    Does exactly what the one-batch-per-ball path does for each scene in the
    main process: run the simulation, wrap each state the way
    ``simulate_scenes`` does, and score it with the same functions, so the
    numbers are identical.

    :param task: Everything needed for one trial, built by the Evaluator.
    :returns: The trial fitness and, if requested, its movement record.
    """
    from revolve2.modular_robot_simulation._scene_simulation_state import (
        SceneSimulationState,
    )
    from revolve2.simulators.mujoco_simulator._simulate_scene import simulate_scene

    states = simulate_scene(
        task["scene_id"],
        task["scene"],
        True,  # headless
        None,  # no recording
        False,  # not paused
        task["control_step"],
        task["sample_step"],
        task["simulation_time"],
        task["simulation_timestep"],
        task["cast_shadows"],
        task["fast_sim"],
        task["viewer_type"],
    )
    scene_states = [SceneSimulationState(state, task["mapping"]) for state in states]
    fitness = _trial_fitness(
        scene_states=scene_states,
        robot=task["robot"],
        ball=task["ball"],
        sampling_frequency=task["sampling_frequency"],
        simulation_time=config.SIMULATION_TIME,
    )
    movement = (
        robot_frame.movement_report(scene_states, task["robot"], task["ball"])
        if task["with_movement"]
        else None
    )
    return fitness, movement


def _trial_fitness(
    scene_states: list[Any],
    robot: ModularRobot,
    ball: Ball,
    sampling_frequency: float | None,
    simulation_time: float,
) -> float:
    """
    Calculate weighted trial fitness from progress, reach, speed, and alignment.

    :param scene_states: Sampled scene states for one simulated trial.
    :param robot: The robot.
    :param ball: The target ball.
    :param sampling_frequency: Scene sampling frequency in Hz, if available.
    :param simulation_time: Maximum simulation time in seconds.
    :returns: Signed trial fitness before cross-pose averaging and clipping.
    """
    initial_dist = _robot_to_ball_distance(scene_states[0], robot, ball)
    # Use the minimum distance across all sampled states rather than the final
    # state alone. When the robot contacts the ball and knocks it away, the
    # simulator appends one extra state after the stop-condition break, making
    # scene_states[-1] show a large robot-to-ball distance even though the robot
    # clearly reached the ball. min_dist correctly captures the closest approach.
    min_dist = min(
        _robot_to_ball_distance(state, robot, ball) for state in scene_states
    )
    progress = _signed_distance_progress(initial_dist, min_dist)
    reached_bonus, time_to_reach_bonus = _reach_bonuses(
        scene_states=scene_states,
        robot=robot,
        ball=ball,
        sampling_frequency=sampling_frequency,
        simulation_time=simulation_time,
    )
    alignment = _movement_alignment(scene_states, robot, ball)

    return (
        getattr(config, "FITNESS_PROGRESS_WEIGHT", 1.0) * progress
        + getattr(config, "FITNESS_REACHED_BONUS_WEIGHT", 0.20) * reached_bonus
        + getattr(config, "FITNESS_TIME_TO_REACH_WEIGHT", 0.10) * time_to_reach_bonus
        + getattr(config, "FITNESS_ALIGNMENT_WEIGHT", 0.05) * alignment
    )


def _signed_distance_progress(initial_dist: float, final_dist: float) -> float:
    """
    Calculate signed normalized distance progress for one trial.

    :param initial_dist: Initial robot-to-ball distance.
    :param final_dist: Final robot-to-ball distance.
    :returns: Positive if closer, zero if unchanged, negative if farther away.
    """
    if initial_dist <= 0.0:
        return 0.0
    progress = (initial_dist - final_dist) / initial_dist
    if abs(progress) <= getattr(config, "NO_PROGRESS_EPSILON", 1e-6):
        return -getattr(config, "NO_PROGRESS_PENALTY", 0.01)
    return progress


def _reach_bonuses(
    scene_states: list[Any],
    robot: ModularRobot,
    ball: Ball,
    sampling_frequency: float | None,
    simulation_time: float,
) -> tuple[float, float]:
    """
    Calculate binary reached-ball and speed-to-reach bonuses.

    :param scene_states: Sampled scene states for one simulated trial.
    :param robot: The robot.
    :param ball: The target ball.
    :param sampling_frequency: Scene sampling frequency in Hz, if available.
    :param simulation_time: Maximum simulation time in seconds.
    :returns: A tuple of reached bonus and time-to-reach bonus.
    """
    for index, scene_state in enumerate(scene_states):
        distance = _robot_to_ball_distance(scene_state, robot, ball)
        if distance <= config.BALL_REACHED_DISTANCE:
            reach_time = _sample_index_to_time(
                index=index,
                num_samples=len(scene_states),
                sampling_frequency=sampling_frequency,
                simulation_time=simulation_time,
            )
            if simulation_time <= 0.0:
                return 1.0, 1.0
            return 1.0, max(0.0, min(1.0, 1.0 - reach_time / simulation_time))
    return 0.0, 0.0


def _sample_index_to_time(
    index: int,
    num_samples: int,
    sampling_frequency: float | None,
    simulation_time: float,
) -> float:
    """
    Approximate sample index as simulation time.

    :param index: Sample index.
    :param num_samples: Total number of samples.
    :param sampling_frequency: Scene sampling frequency in Hz, if available.
    :param simulation_time: Maximum simulation time in seconds.
    :returns: Approximate simulation time in seconds.
    """
    if sampling_frequency is not None and sampling_frequency > 0.0:
        return index / sampling_frequency
    if num_samples <= 1:
        return 0.0
    return simulation_time * index / (num_samples - 1)


def _movement_alignment(
    scene_states: list[Any],
    robot: ModularRobot,
    ball: Ball,
) -> float:
    """
    Calculate average movement alignment toward the ball.

    :param scene_states: Sampled scene states for one simulated trial.
    :param robot: The robot.
    :param ball: The target ball.
    :returns: Mean signed cosine alignment in [-1.0, 1.0].
    """
    alignments = []
    epsilon = getattr(config, "NO_PROGRESS_EPSILON", 1e-6)
    for previous_state, current_state in zip(scene_states, scene_states[1:]):
        previous_robot_pos = _robot_position(previous_state, robot)
        current_robot_pos = _robot_position(current_state, robot)
        previous_ball_pos = _ball_position(previous_state, ball)

        move_x = current_robot_pos.x - previous_robot_pos.x
        move_y = current_robot_pos.y - previous_robot_pos.y
        target_x = previous_ball_pos.x - previous_robot_pos.x
        target_y = previous_ball_pos.y - previous_robot_pos.y

        move_norm = math.sqrt(move_x**2 + move_y**2)
        target_norm = math.sqrt(target_x**2 + target_y**2)
        if move_norm <= epsilon or target_norm <= epsilon:
            continue

        alignments.append(
            (move_x * target_x + move_y * target_y) / (move_norm * target_norm)
        )

    if len(alignments) == 0:
        return 0.0
    return float(np.mean(alignments))


def _robot_to_ball_distance(scene_state, robot: ModularRobot, ball: Ball) -> float:
    """
    Compute xy-plane distance between robot and ball at a given simulation state.

    :param scene_state: A SceneSimulationState snapshot.
    :param robot: The robot.
    :param ball: The ball.
    :returns: Euclidean distance on the xy-plane in metres.
    """
    robot_pos = _robot_position(scene_state, robot)
    ball_pos = _ball_position(scene_state, ball)
    return math.sqrt(
        (robot_pos.x - ball_pos.x) ** 2 + (robot_pos.y - ball_pos.y) ** 2
    )


def _robot_position(scene_state, robot: ModularRobot):
    """
    Get the position the task measures from, from a scene state.

    That is the robot's head when the config uses head coordinates, otherwise
    the core (the robot's own position).

    :param scene_state: A SceneSimulationState snapshot.
    :param robot: The robot.
    :returns: The measuring point's position.
    """
    pose = scene_state.get_modular_robot_simulation_state(robot).get_pose()
    return robot_frame.reference_position(pose, robot_frame.reference_offset())


def _ball_position(scene_state, ball: Ball):
    """
    Get ball position from a scene state.

    :param scene_state: A SceneSimulationState snapshot.
    :param ball: The ball.
    :returns: Ball pose position.
    """
    return scene_state._simulation_state.get_multi_body_system_pose(ball).position


def _copy_pose(pose: Pose) -> Pose:
    """
    Copy a pose so scenes do not share mutable pose objects.

    :param pose: The pose to copy.
    :returns: A copy of the pose.
    """
    return Pose(position=pose.position.copy(), orientation=pose.orientation.copy())
