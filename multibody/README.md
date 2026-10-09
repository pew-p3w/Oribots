# Multibody ball contest

K robots of one trained kind start on a ring around a ball in a walled arena and
compete to reach the ball first and keep it. Each robot runs the frozen
controller of an already-trained single-robot run (its CPG gait plus its
steering layer) with a small residual tanh layer on top. Only that layer is
trained, with CMA-ES.

Everything here lives in `multibody/`. It imports the single-robot code
(`src/`, `brains/`, `config/`, the vendored `engine/`) and never changes it.

## Contents

| File | What it does |
|---|---|
| `run_multibody.py` | The command line: train, resume, play back, replay, record a video |
| `match_video.py` | Records a match to an MP4 at a fixed speed |
| `config_loader.py` | Loads and validates a contest config, its base config and the trained weights |
| `arena.py` | The walled arena and the robots' start layouts |
| `layered_brain.py` | The frozen controller plus the residual layer |
| `match.py` | Builds one match: arena, K robots, one ball |
| `match_scoring.py` | Scores a match |
| `match_worker.py` | Simulates and scores one match (in a worker process) |
| `arena_evaluator.py` | Scores candidates over all layouts, in parallel |
| `layer_cmaes.py` | The training process: CMA-ES over the layer, snapshots, resume |
| `match_recording.py` | The two record files |
| `timing_pilot.py` | Measures match cost and the trained controller's possession behaviour |
| `arena_probes.py` | Checks the engine behaviours the contest relies on |
| `contest_paths.py` | Import paths for every multibody entry point |
| `configs/` | Contest configs (`spider_k4.py`, and `_tiny.py` for the tests) |
| `jobs/` | Cluster jobs: UCT (Slurm) and CHPC Lengau (PBS Pro) |
| `tests/` | The check suite |

## Before the first run

1. Set up Oribots as usual (`make setup` on a laptop, `make setup-conda` on UCT,
   the Singularity image on Lengau). Nothing extra is installed for multibody.
2. Put the trained weights where the config expects them. `configs/spider_k4.py`
   reads `../../../output/spider_8ball/gen30.pkl` relative to itself, field
   `best_parameters`: a single-robot snapshot of the trained run, kept outside this
   repository. Check first that this field is the run's best individual by
   unclamped mean in its per-individual CSV, and adjust the path on a cluster.

All commands below run from `Oribots/`. `python` means the project's interpreter
(`.venv.nosync/bin/python` on a laptop, the conda env on UCT).

## Train

```
python multibody/run_multibody.py -r multibody/configs/spider_k4.py output/<run>
```

- Reads: the config, the base config it names (`config/<BASE_CONFIG>.py`) and the
  trained-weights file.
- Writes, in `output/<run>/` (which must not already hold a run): `gen<N>.pkl`
  after each generation and the two record files (see Output).
- Training runs in a child process. Ctrl-C (or SIGTERM) stops it and all its
  worker processes; the latest snapshot stays complete.

## Resume

```
python multibody/run_multibody.py -c output/<run>/gen<N>.pkl
```

- Continues from generation N + 1 in the snapshot's own folder (a copied run
  folder works), repeating only the generation that was in progress. A resumed
  run gives exactly the results of an uninterrupted one.
- Reads: the snapshot only. It carries the config and base config texts, the
  trained weights, the layouts and the CMA-ES state, so the original files may
  have changed or gone.
- Writes: new snapshots and record rows in the snapshot's folder. Rows of later
  generations left by an interrupted run are removed first.
- A snapshot of the final generation reports the run complete and exits 0.

## Play back

```
.venv.nosync/bin/mjpython multibody/run_multibody.py -t output/<run>/gen<N>.pkl [--layout J] [--sim-seconds S]
python multibody/run_multibody.py -t output/<run>/gen<N>.pkl --headless
```

- Plays the best candidate of that generation on layout J (default 0) of the
  run, against opponents as in training, and prints one line per robot: slot,
  role, d0, closest approach, possession share, first-reach and most-possession
  bonus, score.
- Length: `PLAYBACK_SIMULATION_TIME` unless `--sim-seconds` overrides it; any
  length runs in full, even beyond the training length.
- On macOS the native viewer needs `mjpython`; under plain `python` the command
  is refused with the `mjpython` command line. `--headless` needs no display.
- The viewer shows simulated time only as fast as the computer can simulate it,
  so several robots look slow on screen. To watch at a fixed speed, record a
  video instead (no window, any interpreter):

  ```
  python multibody/run_multibody.py -t output/<run>/gen<N>.pkl --video match.mp4 [--video-speed 20]
  ```

  `--video-speed` is simulated seconds per video second (default 20). The printed
  results are the same as for any other playback. `--video` also works with
  `--replay`.
- Reads the snapshot; writes nothing (except the video, if asked).

## Replay

```
python multibody/run_multibody.py --replay output/<run>/gen<N>.pkl GEN CAND LAYOUT [--headless]
```

- Re-simulates the recorded match of candidate CAND of generation GEN on layout
  LAYOUT, and, when the playback length equals the training `SIMULATION_TIME`,
  checks every recorded per-robot value against it bit for bit (same machine and
  software). At another length it plays the full length and says the comparison
  was skipped.
- Reads the snapshot and the run's record files; writes nothing.

## Timing and possession pilot

```
python multibody/timing_pilot.py multibody/configs/spider_k4.py output/<pilot> \
    [--ks 2 3 4 5] [--workers W] [--uct-cores C] [--uct-factor F] [--lengau-factor F]
```

- For each K, plays one match per layout with every robot running the trained
  controller, at the config's `SIMULATION_TIME`.
- Reports seconds per match; hours per generation and total core-hours on UCT
  and Lengau, for the current opponent modes and projected for a future
  candidates-vs-candidates mode; how often the trained controller holds the
  ball; and its mean approach speed, the source for `V_REF`.
- `--uct-factor` and `--lengau-factor` are a cluster core's seconds per laptop
  second for the same match (1.0 if not given). `--uct-cores` is the job's
  `--ntasks`. Lengau uses one 24-core node.
- Writes `timing_pilot_report.json` and `timing_pilot_report.txt` to
  `output/<pilot>/`, a folder of its own.
- Set the fitness weights, `POSSESSION_DISTANCE`, `V_REF` and the generation
  budget from it before cluster training.

## Check suite

```
make -C multibody check          # every test, including the slow single-robot isolation check
make -C multibody check-fast     # the same without the isolation check
make -C multibody check RUN_PYTHON=/path/to/python
```

Each test script prints `<NAME> CHECK PASSED` or `<NAME> CHECK FAILED` with the
reason; the command exits 0 only if all pass. Tests write only to the system
temp directory. `make check-multibody` in `Oribots/` runs the same suite.

## Cluster jobs

### UCT (Slurm)

```
sbatch multibody/jobs/multibody_train.job
sbatch --export=ALL,CONFIG=multibody/configs/<config>.py multibody/jobs/multibody_train.job
sbatch --export=ALL,OUTPUT=output/multibody_<config>_<job id> multibody/jobs/multibody_resume.job
```

- `multibody_train.job` starts a new run of `CONFIG` (default
  `multibody/configs/spider_k4.py`) in `output/multibody_<config name>_<job id>`.
- `multibody_resume.job` continues the run in `OUTPUT` (required) from its latest
  snapshot. Submit it again after each time limit until the run is complete.
- The worker count follows `--ntasks` (`SLURM_NTASKS`). Logs:
  `jobs/logs/<job name>-<job id>.out` and `.err`; training progress is in `.err`.
- At the time limit a run loses at most the generation in progress.

### CHPC Lengau (PBS Pro, Singularity)

```
qsub -v OUTPUT=output/<run>,CONFIG=multibody/configs/spider_k4.py multibody/jobs/multibody.pbs
qsub -W depend=afterany:<previous job id> -v OUTPUT=output/<run> \
     -o /mnt/lustre/users/<user>/oribots_logs/multibody.2.out \
     -e /mnt/lustre/users/<user>/oribots_logs/multibody.2.err multibody/jobs/multibody.pbs
```

- One script does both: if `OUTPUT` holds snapshots it continues from the
  latest, otherwise it starts `CONFIG` there.
- Chain jobs with `-W depend=afterany:<id>`, and give every chained job its own
  `-o`/`-e`. The `#PBS -o/-e` lines are only a default, and each job would
  otherwise overwrite the previous job's logs.
- Edit the `#PBS -P <PROJECT>`, `-M` and `-o/-e` lines to your own account
  before the first submission; the log folder must exist.
- The worker count follows `NCPUS` (24).

## Contest config settings

A config is a Python file in `configs/`. Settings with no default must be set.
The body, head, ball (`BALL_RADIUS`, `BALL_MASS`, `BALL_REACHED_DISTANCE`) and
feedback settings (`FEEDBACK_NUM_INPUTS`, `FEEDBACK_OUTPUT_SCALE`,
`FEEDBACK_DISTANCE_SCALE`) always come from the base config, unchanged, so the
trained controller sees exactly the inputs it was trained on.

| Setting | Default | Unit | Allowed | Meaning |
|---|---|---|---|---|
| `BASE_CONFIG` | none | | a file in `config/` using head coordinates | The single-robot config the controller was trained with |
| `TRAINED_WEIGHTS_FILE` | none | | an existing `.npy` or single-robot `gen<N>.pkl` | Where the trained CPG + steering weights are (relative paths are from the config's folder) |
| `TRAINED_WEIGHTS_FIELD` | none | | a field of that snapshot | Which parameter field of a snapshot to use; not used for `.npy` |
| `K` | none | robots | 2, 3, 4, 5 | Robots per match |
| `HIDDEN_SIZE` | 8 | nodes | 6 to 10 | L, the residual layer's hidden size |
| `OPPONENTS` | `"reference"` | | `"reference"`, `"self"` (`"population"` is planned, refused for now) | Who the candidate plays against |
| `REFERENCE_LAYER` | `None` | | `None` or `{"snapshot": <multibody gen<N>.pkl>, "which": "best" or "best_ever"}` | The opponents' layer; `None` means a zero-output layer (the trained controller unchanged) |
| `LAYER_OUTPUT_SCALE` | none | hinge range | > 0 | Largest layer output, as a fraction of a hinge's range |
| `TERRAIN_SIZE` | none | m | two numbers > 0 | Floor size inside the walls (x, y) |
| `WALL_HEIGHT` | 4 x `BALL_RADIUS` | m | above the ball diameter and the robot's standing height | Wall height |
| `WALL_THICKNESS` | 0.2 | m | > 0 | Wall thickness, outward from the floor edge |
| `START_RADIUS` | none | m | the ring must fit inside the walls | Distance of every robot's core from the ball at the start |
| `MIN_ARC_GAP` | none | m | >= 0; K gaps must fit the ring | Smallest distance between two robots along the ring |
| `NUM_LAYOUTS` | none | | integer >= 1 | Start layouts per run; every candidate plays each once |
| `LAYOUT_SEED` | none | | integer >= 0 | Seed of the layouts |
| `SPAWN_HEIGHT_TOLERANCE` | 0.01 | m | > 0 | How far above the floor a spawned robot's lowest point may be |
| `SIMULATION_TIME` | 700 | s | > 0 | Length of a training match |
| `PLAYBACK_SIMULATION_TIME` | `SIMULATION_TIME` | s | > 0 | Length of a playback |
| `POSSESSION_DISTANCE` | the base config's `BALL_REACHED_DISTANCE` | m | > 0 | How close a head must be to the ball centre to hold it |
| `V_REF` | 0.01 | m/s | > 0 | Reference approach speed for the time bonus's decay |
| `TIME_BONUS_CAP` | 2.0 | | finite | Weight of the time term |
| `POSSESSION_WEIGHT` | 3.0 | | finite | Reward per unit possession share |
| `POSSESSION_PENALTY` | 1.0 | | finite | Penalty per unit share of time another robot holds the ball |
| `FIRST_REACH_BONUS` | 0.5 | | finite | Bonus for the first robot to hold the ball |
| `MOST_POSSESSION_BONUS` | 2.0 | | finite, >= 0 | Bonus for the robot(s) holding the ball the most |
| `FAILED_MATCH_SCORE` | -(`TIME_BONUS_CAP` + `POSSESSION_PENALTY`) - 1 | | finite | Score of every robot in a match that cannot be scored |
| `MAX_FAILED_MATCH_FRACTION` | 0.5 | | 0 to 1 | Stop the run if more of a generation's matches fail |
| `W1_INIT_STD` | none | | > 0 | Standard deviation of the initial W1 |
| `W1_INIT_SEED` | none | | integer >= 0 | Seed of the initial W1 (CMA-ES uses this seed + 1) |
| `CMA_POPULATION_SIZE` | 24 | | integer >= 2 | Candidates per generation |
| `CMA_INITIAL_STD` | none | | > 0 | CMA-ES initial step size |
| `CMA_BOUNDS` | none | | (lower, upper), lower < 0 < upper | Bounds on every layer parameter |
| `NUM_GENERATIONS` | none | | integer >= 1 | Generations in the run, across resumes |
| `NUM_WORKERS` | `SLURM_NTASKS`, else `NCPUS`, else min(CPUs, 8) | processes | integer >= 1 | Matches simulated in parallel |

A config that breaks any rule is refused before any match runs or any file is
written, with a message naming the setting. When `TRAINED_WEIGHTS_FILE` is a
single-robot snapshot that stores its config, that config is checked against
the base config (head, feedback and ball settings, and the body), and a
mismatch is refused. A `.npy` file stores no config, so it is checked only for
length and finite values.

## The layered brain

For the four inputs x the trained controller reads (sin and cos of the angle to
the ball, the scaled distance, a constant 1):

```
h       = tanh(W1 x)                         W1: L x 4
extra   = tanh(W2 h) * LAYER_OUTPUT_SCALE    W2: n x L, row i drives hinge i
target  = clip(cpg + steering + extra * range, -range, range)    per hinge
```

A layer's parameters are W1 row by row, then W2 row by row (4L + nL values).
Training starts from W2 = 0 and a seeded W1, so the first mean is exactly the
trained controller.

## How a match is scored

Samples are taken 5 times a second, t_1 .. t_N, N = SIMULATION_TIME x 5. d_i(t)
is the horizontal distance from robot i's head to the ball centre, and d0_i is
that distance at the start.

- Closeness: c_i(t) = clip(1 - d_i(t) / d0_i, -1, 1).
- Time term: tau_i = d0_i / `V_REF`, w_i(t) = exp(-t / tau_i),
  T_i = sum_k w_i(t_k) c_i(t_k) / sum_k w_i(t_k), and TIME_i = `TIME_BONUS_CAP` x T_i.
  Being close early counts most.
- Holder: at each sample, the one robot whose head is nearer the ball than every
  other robot's and within `POSSESSION_DISTANCE` (inclusive). If two or more are
  exactly equally near, nobody holds the ball at that sample.
- P_i = share of samples robot i holds the ball; O_i = share of samples another
  robot holds it. POSS_i = `POSSESSION_WEIGHT` x P_i - `POSSESSION_PENALTY` x O_i.
- FIRST_i = `FIRST_REACH_BONUS` for the holder at the earliest sample that has
  one, 0 for everyone else (and for everyone if nobody ever holds it).
- MOST_i = `MOST_POSSESSION_BONUS`, split equally between the robots with the
  most samples held (at least one); 0 for everyone if nobody ever holds it.
- F_i = TIME_i + POSS_i + FIRST_i + MOST_i, unclamped.

A candidate's match score is the Learner's F in `"reference"` mode (the
candidate plays one slot, which moves round the ring from layout to layout; the
others run the reference layer), or the mean of the K F values in `"self"` mode
(all K robots run the candidate). Its score is the mean over the layouts.

A match in which the ball or a robot's core leaves the arena, or a distance is
not finite, cannot be scored. Every robot in it gets `FAILED_MATCH_SCORE`, it is
flagged in the record and logged, and training continues. The run stops if more
than `MAX_FAILED_MATCH_FRACTION` of a generation's matches fail, or if a worker
process dies; the latest snapshot remains the resume point.

## Output

A training run's folder holds:

- `gen<N>.pkl`: one snapshot per generation, written atomically. It holds the
  CMA-ES state, the generation's and the run's best layer and score, the trained
  weights, the config and base config texts, the layouts and seeds, the initial
  W1, the reference layer, the record file names and all settings. It loads with
  only the pinned packages installed. The single-robot `run.py` refuses it.
- `candidates_<config>_run_<timestamp>.csv`: one row per candidate per generation.
- `per_robot_<config>_run_<timestamp>.csv`: one row per candidate, layout and robot.

Candidate record columns:

| Column | Meaning |
|---|---|
| `generation` | N of the generation's snapshot `gen<N>.pkl` |
| `candidate_index` | The candidate's position in that generation's CMA-ES population |
| `score` | The candidate's score (mean over layouts of its match scores, as above) |
| `layer_params` | The candidate's 4L + nL layer parameters, W1 then W2 |

Rows of a generation are ordered by ascending score, ties by candidate index.

Per-robot record columns:

| Column | Meaning |
|---|---|
| `generation`, `candidate_index` | As above |
| `layout_index` | The layout, in the order stored in the snapshots |
| `slot` | The robot's place in the layout, 0 to K - 1 |
| `role` | `Learner` or `Opponent` (`"reference"` mode), `self` (`"self"` mode) |
| `d0` | Head-to-ball distance at the start (m) |
| `final_distance` | Head-to-ball distance at the last sample (m) |
| `min_distance` | Closest head-to-ball distance over the samples (m) |
| `time_bonus` | T_i, the time term before `TIME_BONUS_CAP`, in [-1, 1] |
| `possession_share` | P_i |
| `opposed_share` | O_i |
| `first_reach_bonus` | FIRST_i |
| `most_possession_bonus` | MOST_i |
| `score` | F_i (`FAILED_MATCH_SCORE` for a failed match) |
| `failed` | `True` if the match could not be scored; its other values are then `nan` |

Rows follow the candidate order, then layout, then slot. Numbers are written
with 17 significant digits and read back exactly.

The pilot's folder holds `timing_pilot_report.json` and `timing_pilot_report.txt`.
