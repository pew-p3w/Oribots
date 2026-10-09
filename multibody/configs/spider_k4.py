"""First contest: four pentagon spiders compete for one ball in a 12 x 12 m walled arena.

Each spider runs the trained single-robot controller of the UCT spider_8ball run
(the EA run, raw selection, generation 30: individual g30_i275, unclamped mean
1.1987, the run's best so far) plus a residual tanh
layer of 8 nodes; only the layer is trained, with CMA-ES. The opponents run the
trained controller unchanged (a zero-output layer), and the learner moves round
the ring of start slots from layout to layout.

RUNTIME PREREQUISITE: the trained weights are not in this repository. They are
the old repository's pulled UCT output, revolve2-folding/output/spider_8ball/gen30.pkl
(the path below is relative to this file). Its best_parameters was checked to be
the top individual by unclamped mean in that run's per-individual CSV. On a
cluster, copy gen30.pkl next to the Oribots checkout and change the path.

V_REF, POSSESSION_DISTANCE, the fitness weights and NUM_GENERATIONS come from the
timing and possession pilot (output/multibody_pilot, 2026-10-05). The trained
controller approaches the ball at 0.0074 m/s, so V_REF is that speed. Its head
gets to 0.26 m of the ball (touching), so POSSESSION_DISTANCE keeps the base
config's BALL_REACHED_DISTANCE (0.33 m). The fitness weights keep their defaults.
At K = 4 a match took 475 s on one Mac core, about 1.06 h per generation on 24
cores, so 50 generations fit in one 100 h UCT job.
"""

BASE_CONFIG = "spider_8ball"
TRAINED_WEIGHTS_FILE = "../../../output/spider_8ball/gen30.pkl"
TRAINED_WEIGHTS_FIELD = "best_parameters"

K = 4
HIDDEN_SIZE = 8
OPPONENTS = "reference"
LAYER_OUTPUT_SCALE = 0.5

TERRAIN_SIZE = (12.0, 12.0)
START_RADIUS = 4.5
MIN_ARC_GAP = 3.0
NUM_LAYOUTS = 8
LAYOUT_SEED = 2026

SIMULATION_TIME = 1000
PLAYBACK_SIMULATION_TIME = 1000
V_REF = 0.0074

W1_INIT_STD = 0.5
W1_INIT_SEED = 1
CMA_POPULATION_SIZE = 24
CMA_INITIAL_STD = 0.3
CMA_BOUNDS = (-3.0, 3.0)
NUM_GENERATIONS = 50
