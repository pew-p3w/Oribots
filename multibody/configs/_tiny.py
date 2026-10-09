"""A seconds-long contest for the test gate (phase 4): K = 2, 4 candidates, 2 layouts, 30 s.

The trained weights are the test fixture (a seeded random vector inside a fake
single-robot snapshot that carries a real stored config), so this runs without
the real trained spider. The arena is small and the robots start close to the
ball so a short match still has contact.
"""

BASE_CONFIG = "spider_8ball"
TRAINED_WEIGHTS_FILE = "../tests/fixtures/fake_single_robot_gen1.pkl"
TRAINED_WEIGHTS_FIELD = "best_parameters"

K = 2
HIDDEN_SIZE = 6
OPPONENTS = "reference"
LAYER_OUTPUT_SCALE = 0.5

TERRAIN_SIZE = (5.0, 5.0)
START_RADIUS = 1.5
MIN_ARC_GAP = 1.2
NUM_LAYOUTS = 2
LAYOUT_SEED = 7

SIMULATION_TIME = 30
PLAYBACK_SIMULATION_TIME = 30

W1_INIT_STD = 0.5
W1_INIT_SEED = 11
CMA_POPULATION_SIZE = 4
CMA_INITIAL_STD = 0.3
CMA_BOUNDS = (-3.0, 3.0)
NUM_GENERATIONS = 3
NUM_WORKERS = 2
