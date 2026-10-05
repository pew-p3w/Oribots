# Oribots

Evolving controllers for modular robots that walk to a ball, simulated in
MuJoCo. Bodies are built from Revolve2 modules; the controller is a CPG network
with a steering layer that sees the ball, and it is optimised with an
evolutionary algorithm (EA) or CMA-ES.

Everything needed to train, watch and analyse a run is in this repository:
the simulator library is vendored in `engine/`, and nothing is imported from
outside it.

## Quick start

```bash
make setup                                               # Python 3.11 venv + all packages
make check                                               # prove the install and the code
mjpython run.py --random-test config/spider.py           # look at a body (macOS: mjpython; Linux: python)
python run.py -r config/spider.py src/ea/main.py output/spider_1   # train
python run.py -o output/spider_1                         # all generations -> output/spider_1/generations.csv
```

Use `.venv.nosync/bin/python` in place of `python` unless the venv is active
(`source .venv.nosync/bin/activate`). On macOS, the two commands that open a
window (`--random-test`, `-t`) need `mjpython` instead of `python`; see
[Watching a robot](#watching-a-robot).

## Repository map

```
run.py              the single entry point: train, watch, export, resume, preview
config/
  bodies.py         the catalogue of robot bodies (spider_v2, gecko_v2, ..., spider_pentagon)
  <body>.py         one EA experiment per body: arena, ball, fitness weights, search settings
  <body>_cmaes.py   the CMA-ES experiments (simple, gecko, spider, spider_8ball)
  spider_8ball*.py  the pentagon spider with a head, 8 balanced balls (EA and CMA-ES)
  _tiny.py          a seconds-long fixture used by the tests; not an experiment
brains/
  ball_aware_brain.py   the controller: CPG gait + ball-steering layer
src/
  evaluator.py      simulates controllers and scores them (shared by both algorithms)
  scoring_worker.py runs one trial's simulation and scoring inside a worker process
  robot_frame.py    head mode: the robot's front, and the point the task measures from
  recording.py      writes every individual of every generation to CSV
  genotype.py       the EA's parameter-vector genotype (random, mutate, crossover)
  ea/main.py        the evolutionary algorithm
  cmaes/main.py     CMA-ES
  paths.py          puts the repository's directories on the import path
engine/             Revolve2 1.2.0, vendored; changes listed in engine/PATCHES.md
jobs/               one Slurm job per config, for the HPC
hpc/gate.job        the Slurm cluster gate: `make check` plus tiny real jobs, on a compute node
hpc/pbs/            PBS Pro jobs for a container-based cluster (train, resume, gate)
hpc/build_sif.sh    builds the Singularity image for the PBS route
hpc/CHPC-notes.md   a general guide to running on a CHPC Lengau (PBS Pro) cluster
docker/             the same gate in a Linux container, for use before the cluster
tests/              `make check`: environment, behaviour-equivalence and workflow checks
output/             training runs (git-ignored)
  analysis/         fitness-curve notebook and its helpers (tracked)
```

Configs are deliberately flat: each one is a complete, standalone description
of an experiment, so reading one file tells you exactly what a run did. A
config imports its body from `bodies` directly (`from bodies import spider_v2`).

## How it works

**Task.** A robot starts at the centre of a flat arena (30 m × 30 m in the
shipped configs), and a ball is placed somewhere around it. A trial ends when
the robot reaches the ball or the time runs out. Each controller is tried on the same set of training ball
positions (`NUM_TRAINING_BALL_POSES`, sampled once per run and at least
`MIN_TRAINING_BALL_DISTANCE_FRACTION` of the arena away), and its fitness is the
mean over those trials. A config may instead lay out the whole set itself with
`make_training_ball_poses(rng, n)`; `spider_8ball*.py` put two balls in each
quadrant, all about 10 m away, so a run cannot favour one direction.

**Head mode** (`src/robot_frame.py`). A symmetric body such as the spider has no
visible front. `spider_pentagon` (in `config/bodies.py`) has a five-sided core:
four legs and a small orange head block on the front face. A config that sets
`HEAD_OFFSET` and `USE_HEAD_COORDINATES = True` measures everything from the
head: the brain's angle and distance to the ball, the fitness, the "ball
reached" stop condition and the movement report. It uses the robot's true
orientation; the simulator reports orientations permuted, so without head mode
the brain's heading is always the world's +x axis (core mode keeps this
behaviour unchanged so that existing configs and controllers behave the same;
see `engine/PATCHES.md`).

**Controller** (`brains/ball_aware_brain.py`). Every hinge is driven by a
central pattern generator, which produces the gait. On top of it an evolved
steering layer reads, every control step, the sine and cosine of the angle to
the ball, the normalised distance to the ball and a bias, and adds a correction
to each hinge target. One parameter vector holds both parts, the CPG weights
first and then the steering weights; its length depends on the body.

**Fitness** (`src/evaluator.py`). Per trial, a weighted sum of:

| term | weight in the config | meaning |
|---|---|---|
| progress | `FITNESS_PROGRESS_WEIGHT` | fraction of the starting distance closed, measured at the **closest approach** |
| reached | `FITNESS_REACHED_BONUS_WEIGHT` | bonus for getting within `BALL_REACHED_DISTANCE` |
| speed | `FITNESS_TIME_TO_REACH_WEIGHT` | bonus for reaching it sooner |
| alignment | `FITNESS_ALIGNMENT_WEIGHT` | how well the robot's movement pointed at the ball |

Progress uses the closest approach rather than the final position because a
robot that reaches the ball usually knocks it away, so the final position
would penalise exactly the best runs. The evaluator returns the trial fitness
**raw and signed**. Because the closest approach includes the starting
position, progress itself is never negative. A trial goes below zero only
through two small terms: a penalty of `NO_PROGRESS_PENALTY` when the robot got
no closer at all, and a negative alignment when it moved away from the ball.
With the shipped weights (0.01 and 0.05) the lowest possible trial fitness is
about −0.06.

**Evolutionary algorithm** (`src/ea/main.py`). Each generation the whole
population is replaced. A tournament (`TOURNAMENT_SIZE`) picks a winner, the
winner is cloned twice, the clones may be crossed over, and one is kept and
mutated (Gaussian noise `MUTATE_STD`, per-gene probability
`MUTATION_PROBABILITY`). Because both parents are clones of the same winner,
nearly all variation comes from mutation; this is intentional. There is no
elitism. Each individual has two fitness values: the **recorded** one, the
mean clamped to [0, 1], which every CSV shows; and the **selection** one, the
same mean unclamped, which the tournament compares (`SELECT_ON_RAW_FITNESS`,
default on). With the clamp, a strong population ties at 1.0 and selection goes
blind; `SELECT_ON_RAW_FITNESS = False` restores the original clamped selection.
A resumed run keeps its rule; `REVOLVE2_SWITCH_TO_RAW_SELECTION=1` switches a
clamped run to the unclamped rule when it is resumed.

**CMA-ES** (`src/cmaes/main.py`). The same brain, evaluator and scenes, with
the `cma` package as the optimiser (`CMA_POPULATION_SIZE`, `CMA_INITIAL_STD`,
`CMA_BOUNDS`). It uses the raw signed fitness with no clip, and records it
unclipped too, because it needs to tell apart candidates that a clip would tie
at 0 (no progress, or moving away from the ball).

## Setup

Python **3.11** is required, on every platform. `requirements.txt` pins the
direct dependencies and `constraints.txt` locks the full resolved set,
including transitive packages, so every install gets the same versions. The
Revolve2 source in `engine/` is installed in editable mode without letting pip
resolve its outdated dependency lists. `make` does all of it.

### Laptop (macOS or Linux): virtual environment

```bash
make setup      # creates .venv.nosync with python3.11 and installs everything
make verify     # checks Revolve2 is imported from engine/ and the pins match
```

If your Python 3.11 is not called `python3.11`, use `make setup PYTHON=/path/to/python3.11`.
`noise` may be compiled during install, so a C compiler can be needed (on
macOS: `xcode-select --install`).

### HPC (Slurm): conda environment

From the repository root on the cluster, on a login node:

```bash
module load python/miniconda3-py3.12      # provides conda; the env itself is 3.11
make setup-conda                          # creates .conda inside the repository
make verify RUN_PYTHON=.conda/bin/python
```

The job scripts activate exactly this `.conda`, so do not build the environment
anywhere else. Keep the repository under `/scratch`, not `/home`. Do not reuse
an environment from another copy of Revolve2: its editable installs would be
imported instead of `engine/`.

### macOS: checkouts inside iCloud Drive

If the repository lives under `~/Documents` or `~/Desktop` with iCloud Drive
sync enabled (the macOS default), iCloud interferes with virtual environments
in two ways: it marks files hidden (Python then silently skips hidden `.pth`
files, which is how editable installs make `import revolve2` work), and it
creates duplicate `" 2".dylib` copies of libraries (MuJoCo loads every library
in its plugin directory and aborts when a plugin registers twice).

The venv is therefore named **`.venv.nosync`**: iCloud skips any path ending in
`.nosync`. This is automatic with `make setup`. As further safety nets,
`run.py` and the tests put the `engine/*` directories on `sys.path` themselves
instead of relying on `.pth` files, and `make verify` fails loudly if it finds
hidden `.pth` files or duplicated libraries. Keeping the checkout outside iCloud
Drive (for example `~/code/`) avoids the problem entirely.

### Why `engine/` and `--no-deps`

`engine/` is Revolve2 v1.2.0 with a handful of documented changes
(`engine/PATCHES.md`). Its packages declare dependency ranges that conflict with
the versions this project is tested on (for example `mujoco ^2.2` and
`opencv-python`), so they are installed with `pip install --no-deps -e` and
`requirements.txt` plus `constraints.txt` are the only dependency lists.

## Using `run.py`

Exactly one mode per call:

| command | what it does |
|---|---|
| `run.py -r CONFIG MAIN OUTPUT` | train a new run into the folder `OUTPUT` |
| `run.py -c OUTPUT/genN.pkl` | resume a run from its latest snapshot |
| `run.py -t OUTPUT/genN.pkl` | watch the best controller of generation N and print its scores |
| `run.py -o OUTPUT` | export every `gen*.pkl` in a run to `OUTPUT/generations.csv` |
| `run.py --random-test CONFIG` | preview a body with one random controller |

**Train.** `MAIN` is `src/ea/main.py` for `config/<body>.py` and
`src/cmaes/main.py` for `config/<body>_cmaes.py`; a mismatched pair is refused
before anything starts. `OUTPUT` must be a new or empty folder: `-r` refuses to
write into a folder that already holds snapshots, because snapshots cannot be
recovered once overwritten. Pick a new name for every run.

**Resume.** `-c` works out from the snapshot whether it was an EA or a CMA-ES
run, continues from the next generation, and appends to the same CSV. CMA-ES
snapshots include the optimiser state, so a resumed CMA-ES run continues its
covariance adaptation instead of restarting it.

**Watch.** `-t` rebuilds the body and brain from what the snapshot stored,
including a copy of the config it was trained with, and runs one trial with a
random ball. It prints the initial and closest distance, the improvement,
the signed progress and the weighted trial fitness. Snapshots written by earlier
versions of the code, whose configs import the body catalogue under its former
module name, also load.

`make train CONFIG=... OUTPUT=... [MAIN=...]`, `make export OUTPUT=...` and
`make test SNAPSHOT=...` are shortcuts for `-r`, `-o` and `-t` using the venv
(`make test` launches with `mjpython`).

## Watching a robot

**macOS.** Launch visual modes with `mjpython`, which is installed with `mujoco`:

```bash
.venv.nosync/bin/mjpython run.py -t output/<run>/gen<N>.pkl
.venv.nosync/bin/mjpython run.py --random-test config/spider.py
make test SNAPSHOT=output/<run>/gen<N>.pkl        # the same, via mjpython
```

An interactive MuJoCo window on macOS must own the main thread, and `mjpython`
is the only launcher that gives it one. `--viewer` defaults to `auto`, which
picks the viewer that matches the launcher: `native` under `mjpython`,
`custom` under plain `python`. On macOS the `custom` window usually never
appears, so a visual mode started with plain `python` prints a warning naming
the `mjpython` command and then runs without a visible window (the simulation
and the printed scores are still correct). Asking explicitly for a combination
that cannot work is refused with the correct command: `--viewer native` under
plain `python`, and `--viewer custom` under `mjpython`, which crashes.

`mjpython` only starts in a venv built from a framework Python, such as
Homebrew's `python@3.11` (`brew install python@3.11`); `make setup` uses it
automatically when it is installed. A venv built from a standalone Python (for
example one installed by `uv`) trains fine but cannot open the viewer, and
`make verify` says so.

***Linux.** `python run.py -t ...` on a machine with a display. The tested Linux
path is headless training; the visual modes are tested on macOS.

**Duration.** `--sim-seconds` sets how long a visual mode runs, in *simulated*
seconds. The viewer renders much faster than real time (about 34× on an
M-series Mac), so simulated seconds are not wall-clock seconds.
`--random-test` defaults to 1000 simulated seconds, roughly 30 seconds of
window; `-t` defaults to the config's `SIMULATION_TIME`.

## Output

A training run writes into its `OUTPUT` folder:

| file | contents |
|---|---|
| `gen<N>.pkl` | a snapshot of generation N: best and worst controllers, fitnesses, the config's source, and everything `-c` needs to resume |
| `parameters_<name>_run_<timestamp>.csv` | one row per generation, written live during training |
| `parameters_<name>.npy` | the best-ever controller's parameter vector |
| `individual_performance_<name>_run_<timestamp>.csv` | every individual of every generation, one row per training-ball trial: raw `trial_fitness`, the recorded `overall_fitness`, the ball, and how the robot moved relative to its heading (`forward_m`, `backward_m`, `sideways_m`, `end_heading_deg`, `end_angle_to_ball_deg`) |
| `weights.csv` | every individual's parameter vector, by `individual_id` (`g<generation>_i<index>`) |
| `generations.csv` | written by `run.py -o`, rebuilt from the snapshots |

The per-individual files are written in ascending fitness order within each
generation, so the end of a generation holds its best individuals. A resumed run
keeps writing into the same files, after dropping anything the interrupted run
had written past the resume point.

Both CSVs have the same eight columns, for both algorithms:

| column | meaning |
|---|---|
| `num_of_generation` | generation index (0 is the initial population) |
| `best_fitness` | best fitness in this generation |
| `worst_fitness` | worst fitness in this generation |
| `best_parent_fitness` | EA: best fitness of the generation being replaced. CMA-ES: blank |
| `best_offspring_fitness` | EA: best fitness among the new children. CMA-ES: blank |
| `best_ever_fitness` | best fitness seen so far in the run |
| `best_robot_weights` | parameter vector of this generation's best controller (JSON list) |
| `worst_robot_weights` | parameter vector of this generation's worst controller (JSON list) |

**The fitness values mean different things for the two algorithms.** In EA
runs every fitness column is the recorded value, clamped to [0, 1] (the
unclamped values selection used are in the snapshots, as
`population_selection_fitnesses`, and can be recomputed as the mean of an
individual's `trial_fitness` rows). In CMA-ES runs the fitness columns are the
raw signed values the optimiser used, which can be negative. Compare curves
within one algorithm; across algorithms, clip the CMA-ES values to [0, 1] first.

Run folders in `output/` are git-ignored; `output/analysis/` is tracked. Results
are never deleted or overwritten by the tooling; each run gets its own folder.

**Plotting.** `output/analysis/fitness_curves.ipynb` finds the run CSVs under
`output/` and plots their fitness curves, resolving paths from the repository
root so it works wherever Jupyter is started. Its extra packages are kept out
of the training install:

```bash
.venv.nosync/bin/python -m pip install -c constraints.txt -r requirements-analysis.txt
```

## Running on the HPC

This section covers a **Slurm** cluster, where the environment is a conda env
built on the cluster itself. A **PBS Pro** cluster whose nodes are too old for
the pinned stack (for example CHPC Lengau, glibc 2.17) is supported through a
Singularity container instead: build the image off-cluster, copy it in, and
submit the `hpc/pbs/*.pbs` jobs. That route is documented in
[`hpc/CHPC-notes.md`](hpc/CHPC-notes.md) and `hpc/build_sif.sh`; the rest of
this section is the Slurm path.

The Slurm jobs are set up for one particular cluster (the UCT HPC): the
`--account` and `--partition` lines in `jobs/*.job` and `hpc/gate.job`, and the
`module load` line there and in [HPC setup](#hpc-slurm-conda-environment), are
that cluster's. On another Slurm cluster, change them in all of those files and
in `REQUIRED` in `tests/test_jobs.py`, which checks them. The PBS jobs have
per-account lines in the same way; `hpc/CHPC-notes.md` lists them.

1. Copy the repository to `/scratch/$USER/` and create the environment (see
   [HPC setup](#hpc-slurm-conda-environment)). When syncing again later, never
   let the sync delete `.conda/` or `output/` on the cluster (with rsync:
   `--exclude .conda --exclude output`).
2. After creating or changing the environment, and after every sync, run the
   gate from the repository root and wait for it to pass:

   ```bash
   sbatch hpc/gate.job
   tail -1 jobs/logs/oribots_gate-<job id>.out   # GATE PASSED at commit <hash>
   ```

   It takes a few minutes on 4 cores (plus queueing) and trains nothing real.
3. Submit from the repository root:

   ```bash
   sbatch jobs/spider.job
   sbatch --mail-user=you@example.org --mail-type=END,FAIL jobs/gecko_cmaes.job   # with email
   ```

4. Every job starts a new run in `output/<config>_<job id>`, so submitting the
   same job twice gives two separate runs. Logs go to
   `jobs/logs/<config>-<job id>.out` and `.err`. **Training progress is in the
   `.err` file**: Python's logging writes to stderr, so the per-generation
   lines (for example `Generation 12 / 100` and, for the EA, `Best offspring
   unclamped fitness (used for selection)`) land there. The `.out` file only
   holds the job script's own `echo` lines (host, job ID, config).

Each job sets `--ntasks`, and the config's `NUM_SIMULATORS` follows it through
`SLURM_NTASKS`, so the number of parallel simulators always matches what Slurm
allocated. Jobs also set `MUJOCO_GL=egl` for headless rendering and limit BLAS
to one thread per process.

**Resuming on the cluster.** Jobs never resume, so that a submission can never
silently continue an old run. To resume, copy a job next to the run, replace its
last line with `python run.py -c output/<run>/gen<N>.pkl`, and submit that copy
from the repository root:

```bash
cp jobs/spider.job output/spider_123456/resume.job   # then edit its last line
sbatch output/spider_123456/resume.job
```

Keep such copies out of `jobs/`: `make check-jobs` requires every job there to
start a clean run.

## Adding an experiment

1. If the body is new, add a constructor to `config/bodies.py`.
2. Copy the closest existing config to `config/<name>.py`, import the body, set
   `TEST_FILE = "<name>"`, and adjust the settings.
3. Copy the matching job to `jobs/<name>.job` and replace the config name in it
   (plus `--ntasks` and `--time` if needed).
4. Run `make check-configs check-jobs`, then a short local run.

`make check-bodies` compares every body with a stored reference, so adding a
body needs its reference entry too (see `tests/body_fingerprint.py`).

## Verifying

`make check` runs every check; `make help` lists them individually. They prove
that the environment imports from `engine/`; that bodies, brain, genotype,
evaluator and both search loops reproduce their recorded reference values
exactly; that head mode and the pentagon spider reproduce reference values on
frozen real simulation states (`check-head`); that both algorithms record every
individual consistently with their snapshots; that the configs, Slurm jobs and
PBS jobs are consistent; and that a real `-r` / `-t` / `-o` / `-c` workflow runs
end to end for both algorithms.

## License

LGPL-3.0; see `LICENSE`. `engine/` is Revolve2 by the CI Group
(github.com/ci-group/revolve2), also LGPL-3.0, with changes listed in `engine/PATCHES.md`;
see `NOTICE`.
