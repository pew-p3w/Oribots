"""Training settings for a deliberately tiny CMA-ES run used by the automated tests.

The CMA-ES counterpart of `_tiny.py`: the simple body, a short run, small
population. Standalone like every config. Used by the CMA-ES end-to-end and
resume checks so they finish in seconds without a real experiment's size.

Used as:

    python run.py -r config/_tiny_cmaes.py src/cmaes/main.py output/_tiny_cmaes
"""

import math
import os

from numpy.random import Generator
from pyrr import Vector3

from bodies import simple_v2
from revolve2.ci_group import terrains
from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene.vector2 import Vector2

# The robot being evolved. TEST_FILE names this experiment's output files and
# always matches this config's filename.
BODY = simple_v2()
TEST_FILE = "_tiny_cmaes"

# The arena and the ball the robot has to reach.
TERRAIN_SIZE = Vector2([30.0, 30.0])
BALL_RADIUS = 0.3
BALL_MASS = 0.1
BALL_REACHED_DISTANCE = BALL_RADIUS * (1.0 + 0.1)
BALL_SPAWN_MARGIN = 0.4
MIN_TRAINING_BALL_DISTANCE_FRACTION = 0.3
NUM_TRAINING_BALL_POSES = 1

# What the steering layer senses each control step.
FEEDBACK_NUM_INPUTS = 4
FEEDBACK_OUTPUT_SCALE = 0.5
FEEDBACK_DISTANCE_SCALE = math.sqrt(
    (TERRAIN_SIZE.x / 2.0 - max(BALL_SPAWN_MARGIN, BALL_RADIUS)) ** 2
    + (TERRAIN_SIZE.y / 2.0 - max(BALL_SPAWN_MARGIN, BALL_RADIUS)) ** 2
)

# How a single trial is scored.
NO_PROGRESS_PENALTY = 0.01
NO_PROGRESS_EPSILON = 1e-6
FITNESS_PROGRESS_WEIGHT = 1.0
FITNESS_REACHED_BONUS_WEIGHT = 0.20
FITNESS_TIME_TO_REACH_WEIGHT = 0.10
FITNESS_ALIGNMENT_WEIGHT = 0.05

# The CMA-ES search: a short run with a small explicit population.
NUM_GENERATIONS = 2
CMA_INITIAL_STD = 0.3
CMA_INITIAL_MEAN = 0.0
CMA_BOUNDS = [-1.0, 1.0]
CMA_POPULATION_SIZE = 4

# How long each trial runs, and how many run at once.
SIMULATION_TIME = 20
NUM_SIMULATORS = int(
    os.environ.get("SLURM_NTASKS")  # Slurm (UCT HPC)
    or os.environ.get("NCPUS")  # PBS Pro (CHPC Lengau)
    or min(os.cpu_count() or 1, 4)  # a laptop-safe default
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
