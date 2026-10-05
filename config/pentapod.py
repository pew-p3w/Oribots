"""Training settings for the pentapod robot, evolved with the evolutionary algorithm.

Everything one run needs is in this file: which body to evolve, how big the
arena is, where the ball may appear, what the brain senses, how a trial is
scored, and how the search is configured. Configs are deliberately standalone
and share nothing, so this file alone explains a run's behaviour.

Used as:

    python run.py -r config/pentapod.py src/ea/main.py output/pentapod
"""

import math
import os

from numpy.random import Generator
from pyrr import Vector3

from bodies import pentapod_v2
from revolve2.ci_group import terrains
from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene.vector2 import Vector2

# The robot being evolved. TEST_FILE names this experiment's output files and
# always matches this config's filename.
BODY = pentapod_v2()
TEST_FILE = "pentapod"

# The arena and the ball the robot has to reach.
TERRAIN_SIZE = Vector2([30.0, 30.0])
BALL_RADIUS = 0.3
BALL_MASS = 0.1
BALL_REACHED_DISTANCE = BALL_RADIUS * (1.0 + 0.1)
BALL_SPAWN_MARGIN = 0.4
# Training balls never spawn nearer than this fraction of the arena, so the
# robot has to travel rather than start on top of the ball.
MIN_TRAINING_BALL_DISTANCE_FRACTION = 0.3
NUM_TRAINING_BALL_POSES = 5

# What the steering layer senses each control step: sin and cos of the heading
# error, the distance to the ball, and a bias.
FEEDBACK_NUM_INPUTS = 4
FEEDBACK_OUTPUT_SCALE = 0.5
# Distances are normalized by the furthest a ball can ever spawn, so the input
# uses its full range instead of saturating on a fixed guess.
FEEDBACK_DISTANCE_SCALE = math.sqrt(
    (TERRAIN_SIZE.x / 2.0 - max(BALL_SPAWN_MARGIN, BALL_RADIUS)) ** 2
    + (TERRAIN_SIZE.y / 2.0 - max(BALL_SPAWN_MARGIN, BALL_RADIUS)) ** 2
)

# How a single trial is scored. Progress dominates; the rest shape the search.
NO_PROGRESS_PENALTY = 0.01
NO_PROGRESS_EPSILON = 1e-6
FITNESS_PROGRESS_WEIGHT = 1.0
FITNESS_REACHED_BONUS_WEIGHT = 0.20
FITNESS_TIME_TO_REACH_WEIGHT = 0.10
FITNESS_ALIGNMENT_WEIGHT = 0.05

# The evolutionary search. Each generation replaces the whole population with
# offspring of tournament winners; mutation supplies nearly all the variation.
POPULATION_SIZE = 200
TOURNAMENT_SIZE = 2
NUM_GENERATIONS = 100
MUTATE_STD = 0.15
MUTATION_PROBABILITY = 0.01

# How long each trial runs, and how many run at once. On the HPC the simulator
# count follows the Slurm task count; on a laptop it stays modest, since every
# simulator is a separate process.
SIMULATION_TIME = 1000
NUM_SIMULATORS = int(
    os.environ.get("SLURM_NTASKS")  # Slurm (UCT HPC)
    or os.environ.get("NCPUS")  # PBS Pro (CHPC Lengau)
    or min(os.cpu_count() or 1, 8)  # a laptop-safe default
)
HEADLESS = True


def parameter_filename() -> str:
    """
    Build the filename the best-ever controller weights are saved to.

    :returns: Parameter filename.
    """
    name = TEST_FILE.removesuffix(".npy")
    name = name.removeprefix("best_weights_")
    name = name.removeprefix("parameter_")
    return f"parameters_{name}.npy"


def make_terrain() -> Terrain:
    """
    Create the flat arena the robot walks on.

    :returns: The terrain.
    """
    return terrains.flat(size=TERRAIN_SIZE)


def make_random_ball_pose(rng: Generator) -> Pose:
    """
    Place the ball anywhere inside the arena, clear of the edges.

    Used for visual testing, where the ball should be able to appear anywhere.

    :param rng: Random number generator.
    :returns: A random ball pose.
    """
    spawn_margin = max(BALL_SPAWN_MARGIN, BALL_RADIUS)
    half_x = TERRAIN_SIZE.x / 2.0 - spawn_margin
    half_y = TERRAIN_SIZE.y / 2.0 - spawn_margin
    return Pose(
        Vector3(
            [
                rng.uniform(-half_x, half_x),
                rng.uniform(-half_y, half_y),
                BALL_RADIUS,
            ]
        )
    )


def make_training_ball_pose(rng: Generator) -> Pose:
    """
    Place a training ball far enough away that reaching it takes real walking.

    :param rng: Random number generator.
    :returns: A random training ball pose.
    :raises ValueError: If the configured minimum distance cannot fit.
    :raises RuntimeError: If no valid pose is sampled after many attempts.
    """
    spawn_margin = max(BALL_SPAWN_MARGIN, BALL_RADIUS)
    half_x = TERRAIN_SIZE.x / 2.0 - spawn_margin
    half_y = TERRAIN_SIZE.y / 2.0 - spawn_margin
    min_distance = MIN_TRAINING_BALL_DISTANCE_FRACTION * min(
        TERRAIN_SIZE.x,
        TERRAIN_SIZE.y,
    )
    max_distance = math.sqrt(half_x**2 + half_y**2)
    if min_distance > max_distance:
        raise ValueError(
            "MIN_TRAINING_BALL_DISTANCE_FRACTION places the minimum "
            "training distance outside the terrain bounds."
        )

    for _ in range(10000):
        pose = make_random_ball_pose(rng)
        distance = math.sqrt(pose.position.x**2 + pose.position.y**2)
        if distance >= min_distance:
            return pose

    raise RuntimeError("Could not sample a valid training ball pose.")
