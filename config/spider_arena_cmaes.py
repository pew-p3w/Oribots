"""Spider in a walled square arena, optimized with CMA-ES: the speed-up comparison.

The same robot, brain and scoring as spider_cmaes, in a smaller square arena
with four walls, sized so a short run trains in reasonable time. SEED is fixed
so two runs that differ only in physics start from the same CMA-ES state and
train on the same ball positions; bench/compare_training.bat runs it twice:

    baseline   RK4 at 1 ms, original trial loop, fresh workers each generation
    fast       implicitfast at 5 ms, fast trial loop, kept workers (about 9x)

Used as:

    python run.py -r config/spider_arena_cmaes.py src/cmaes/main.py output/spider_arena
"""

import math
import os

from numpy.random import Generator
from pyrr import Vector3

from bodies import spider_v2
from revolve2.modular_robot_simulation import Terrain
from revolve2.simulation.scene import AABB, Color, Pose
from revolve2.simulation.scene.geometry import GeometryBox, GeometryPlane
from revolve2.simulation.scene.geometry.textures import MapType, Texture
from revolve2.simulation.scene.vector2 import Vector2

# The robot being evolved. TEST_FILE names this experiment's output files and
# always matches this config's filename.
BODY = spider_v2()
TEST_FILE = "spider_arena_cmaes"

# Fixes the CMA-ES seed and the training ball positions, so runs are
# comparable. Remove it (or set None) for a clock-seeded run.
SEED = 20261114

# The arena: a square floor inside four walls. TERRAIN_SIZE is the space
# inside the walls; balls spawn inside it.
TERRAIN_SIZE = Vector2([8.0, 8.0])
WALL_THICKNESS = 0.2
WALL_HEIGHT = 0.5
BALL_RADIUS = 0.3
BALL_MASS = 0.1
BALL_REACHED_DISTANCE = BALL_RADIUS * (1.0 + 0.1)
BALL_SPAWN_MARGIN = 0.4
# Training balls never spawn nearer than this fraction of the arena, so the
# robot has to travel rather than start on top of the ball.
MIN_TRAINING_BALL_DISTANCE_FRACTION = 0.3
NUM_TRAINING_BALL_POSES = 2

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

# The CMA-ES search, shortened for a comparison run.
NUM_GENERATIONS = 30
CMA_INITIAL_STD = 0.3
CMA_INITIAL_MEAN = 0.0
CMA_BOUNDS = [-1.0, 1.0]
CMA_POPULATION_SIZE = 24

# Physics. implicitfast at 5 ms is about 5x faster than the original RK4 at
# 1 ms and scores robots nearly the same (see CHANGES.md). For the original
# physics, set SIM_INTEGRATOR = "RK4" and SIM_TIMESTEP = 0.001.
SIM_INTEGRATOR = "implicitfast"
SIM_TIMESTEP = 0.005

# How long each trial runs, and how many run at once.
SIMULATION_TIME = 200
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
    Create the square arena: a floor with a wall along each edge.

    The walls stand just outside TERRAIN_SIZE, so the space inside them is
    exactly TERRAIN_SIZE.

    :returns: The terrain.
    """
    half_x = TERRAIN_SIZE.x / 2.0
    half_y = TERRAIN_SIZE.y / 2.0
    t = WALL_THICKNESS
    wall_texture = Texture(base_color=Color(70, 70, 80, 255))

    def wall(x: float, y: float, size_x: float, size_y: float) -> GeometryBox:
        return GeometryBox(
            pose=Pose(Vector3([x, y, WALL_HEIGHT / 2.0])),
            mass=0.0,
            texture=wall_texture,
            aabb=AABB(Vector3([size_x, size_y, WALL_HEIGHT])),
        )

    outer_x = TERRAIN_SIZE.x + 2.0 * t
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(),
                mass=0.0,
                size=Vector2([TERRAIN_SIZE.x + 4.0, TERRAIN_SIZE.y + 4.0]),
                texture=Texture(base_color=Color(100, 100, 100, 255), map_type=MapType.MAP2D),
            ),
            wall(half_x + t / 2.0, 0.0, t, TERRAIN_SIZE.y),
            wall(-half_x - t / 2.0, 0.0, t, TERRAIN_SIZE.y),
            wall(0.0, half_y + t / 2.0, outer_x, t),
            wall(0.0, -half_y - t / 2.0, outer_x, t),
        ]
    )


def make_random_ball_pose(rng: Generator) -> Pose:
    """
    Place the ball anywhere inside the arena, clear of the walls.

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
