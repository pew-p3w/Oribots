"""Training settings for the pentagon spider on 8 balanced balls, optimized with CMA-ES.

Everything one run needs is in this file. The body is the pentagon spider
(config/bodies.py): four legs and a head, a small orange block on the front
face. The head makes the robot's front visible, and it is the point everything
is measured from (src/robot_frame.py): the brain's view of the ball, the
fitness, the "ball reached" stop condition and the movement report. Set
USE_HEAD_COORDINATES = False to keep the head as a marker only and measure from
the core instead.

The 8 training balls sit two per quadrant, all about the same distance from the
robot's start, so a run cannot favour one direction (make_training_ball_poses).

This is the CMA-ES counterpart of spider_8ball.py: the same body, head, arena,
ball layout, fitness and trial length, so the two algorithms can be compared.

Used as:

    python run.py -r config/spider_8ball_cmaes.py src/cmaes/main.py output/spider_8ball_cmaes
"""

import math
import os

import robot_frame
from numpy.random import Generator
from pyrr import Vector3

from bodies import spider_pentagon
from revolve2.ci_group import terrains
from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import Pose
from revolve2.simulation.scene.vector2 import Vector2

# The robot being evolved. TEST_FILE names this experiment's output files and
# always matches this config's filename.
BODY = spider_pentagon()
TEST_FILE = "spider_8ball_cmaes"

# The head, and where it sits: read from the real built geometry. It must sit
# on the +x axis the controller steers by, or this raises.
HEAD = BODY.core_pentagon.front_face.middle
HEAD_OFFSET = robot_frame.head_offset_of(BODY, HEAD)
USE_HEAD_COORDINATES = True

# The arena and the ball the robot has to reach.
TERRAIN_SIZE = Vector2([30.0, 30.0])
BALL_RADIUS = 0.3
BALL_MASS = 0.1
BALL_REACHED_DISTANCE = BALL_RADIUS * (1.0 + 0.1)
BALL_SPAWN_MARGIN = 0.4
# Only used by make_training_ball_pose() (a single random training ball); the
# run's fixed training balls come from make_training_ball_poses() below.
MIN_TRAINING_BALL_DISTANCE_FRACTION = 0.3

# The training balls: two per quadrant, roughly equidistant from the centre.
NUM_TRAINING_BALL_POSES = 8
TRAINING_BALL_DISTANCE = 10.0  # metres from the robot's spawn point (terrain centre)
TRAINING_BALL_DISTANCE_JITTER = 0.05  # +-5% => 9.5 m to 10.5 m
TRAINING_BALL_ANGLE_MARGIN = 0.10  # keep balls this fraction of a sector off its edges

# What the steering layer senses each control step: sin and cos of the heading
# error, the distance to the ball, and a bias.
FEEDBACK_NUM_INPUTS = 4
FEEDBACK_OUTPUT_SCALE = 0.5
# Distances are normalized by the furthest a ball can ever spawn.
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

# The CMA-ES search. It adapts its own step size and covariance, so there is
# no population, tournament or mutation rate to set.
NUM_GENERATIONS = 100
CMA_INITIAL_STD = 0.3
CMA_INITIAL_MEAN = 0.0
CMA_BOUNDS = [-1.0, 1.0]
CMA_POPULATION_SIZE = 500

# How long each trial runs, and how many run at once. On a cluster the
# simulator count follows the scheduler's core count; on a laptop it stays
# modest, since every simulator is a separate process.
SIMULATION_TIME = 1500
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
    Place one random training ball far enough away that reaching it takes walking.

    Not used for the run's fixed training balls (see make_training_ball_poses).

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


def make_training_ball_poses(rng: Generator, num_poses: int) -> list[Pose]:
    """
    Create the fixed training balls: an equal number in each quadrant.

    Each quadrant is split into ``num_poses / 4`` equal angular sectors and one
    ball is placed at a random angle inside each sector (kept off the sector's
    edges, so balls in a quadrant never sit on top of each other or on an axis).
    Every ball is TRAINING_BALL_DISTANCE from the terrain centre, give or take
    TRAINING_BALL_DISTANCE_JITTER. Poses are ordered quadrant by quadrant
    (x>0,y>0 first, then counter-clockwise), and within a quadrant by angle.

    :param rng: Random number generator.
    :param num_poses: Number of balls. Must be a multiple of 4.
    :returns: The training ball poses.
    :raises ValueError: If num_poses is not a multiple of 4, or the balls would
        not fit inside the terrain.
    """
    if num_poses <= 0 or num_poses % 4 != 0:
        raise ValueError(
            f"Need a positive multiple of 4 training balls (an equal number per "
            f"quadrant), got {num_poses}."
        )
    spawn_margin = max(BALL_SPAWN_MARGIN, BALL_RADIUS)
    max_radius = TRAINING_BALL_DISTANCE * (1.0 + TRAINING_BALL_DISTANCE_JITTER)
    if max_radius > min(TERRAIN_SIZE.x, TERRAIN_SIZE.y) / 2.0 - spawn_margin:
        raise ValueError(
            f"TRAINING_BALL_DISTANCE {TRAINING_BALL_DISTANCE} m does not fit in "
            f"the terrain."
        )

    per_quadrant = num_poses // 4
    sector_width = (math.pi / 2.0) / per_quadrant
    poses = []
    for quadrant in range(4):
        for sector in range(per_quadrant):
            sector_start = quadrant * (math.pi / 2.0) + sector * sector_width
            angle = rng.uniform(
                sector_start + TRAINING_BALL_ANGLE_MARGIN * sector_width,
                sector_start + (1.0 - TRAINING_BALL_ANGLE_MARGIN) * sector_width,
            )
            distance = TRAINING_BALL_DISTANCE * (
                1.0
                + rng.uniform(-TRAINING_BALL_DISTANCE_JITTER, TRAINING_BALL_DISTANCE_JITTER)
            )
            poses.append(
                Pose(
                    Vector3(
                        [
                            distance * math.cos(angle),
                            distance * math.sin(angle),
                            BALL_RADIUS,
                        ]
                    )
                )
            )
    return poses
