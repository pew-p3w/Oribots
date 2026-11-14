# Oribots training speed: rewrite options

This document describes five ways to make training faster in `Oribots-main`
(CMA-ES and the EA, which share one evaluator). For each option it covers what
changes, where in the code, the expected gain, the effect on the physics, the
effort involved, and how to check the result.

All speedups are **estimates from reading the code**, not measurements. Step 0
(benchmark) comes before any of them.

---

## Status: Python overhead removed, results unchanged (2026-11-14)

Measured on `spider_cmaes` (24 candidates × 2 balls × 60 s, 8 workers, 3
generations, `bench\run_bench.bat code`):

| | original code | new code | speedup | fitness difference |
|---|---|---|---|---|
| RK4, 1 ms (the shipped physics) | 0.368 s/trial | 0.354 s/trial | 1.04× | **0 (bit-identical)** |
| implicitfast, 5 ms | 0.076 s/trial | 0.039 s/trial | **1.96×** | **0 (bit-identical)** |

From the original code and physics to the new code with implicitfast at 5 ms
is 0.368 → 0.039 s/trial, **9.4×**. The code change alone is worth little at RK4
1 ms, because physics is about 90% of that trial; it matters once option 1 makes
the physics cheap.

What changed (options 2–4 done in Python, without a compiler):

| | |
|---|---|
| `src/fast_trial.py` | Runs a training trial without per-tick Python objects. Body and joint indices are looked up once; MuJoCo steps in C (`nstep`) between control ticks and samples, on exactly the original schedule (checked against MuJoCo's own clock after every chunk); the brain and the stop condition read MuJoCo's arrays directly; samples keep only what scoring reads. Every floating-point operation is the original's, in the original order, on the same types; calls whose rounding depends on the library (`np.dot`, `np.matmul`, `np.tanh`, `np.float64 ** 2`) are kept as they were. Control, sampling and state handling went from about 180 µs per control tick to about 22 µs. |
| Kept worker pool | `Evaluator` keeps its worker processes between generations instead of starting new ones (about 1.1 s saved per generation on Windows). |
| Switches | `FAST_TRIAL = False` / `REVOLVE2_FAST_TRIAL=0` and `PERSISTENT_WORKERS = False` / `REVOLVE2_PERSISTENT_WORKERS=0` bring back the original behaviour. Scenes the fast trial does not cover (cameras, other brains) go through the original code automatically. |
| Tests | `tests/test_fast_trial.py`: rotation, CPG step, control tick and stop condition against the original objects; trajectories, fitness and movement records compared bit for bit for **every shipped config**, all four integrators, head mode, and trials that stop on reaching the ball; the kept pool. A mutation check showed it catches a 1-ulp difference. All 16 checks pass. |

A seeded CMA-ES run (spider, 4 generations × 8 candidates × 2 balls) gave the
same best fitness every generation, the same saved weights, and the same CMA-ES
state, with only output paths and wall-clock timers differing: 6.3 s with the
original code, 3.3 s with the new.

**Where the time goes now** (implicitfast 5 ms): physics about 80%, the control
tick about 18%, model compile under 2% of a 1000 s trial. MuJoCo computes
nothing unused (no sensors), so **physics is the floor for bit-identical
results**. Remaining options:

- A C/C++ trial loop would remove most of the remaining ~18%: roughly 1.2×
  more, at the cost of a compiler (MSVC) and keeping a second copy of the
  brain and scoring in step with the Python one. Exact equality would also need
  the BLAS and `tanh` calls to match, which is not guaranteed across machines.
- Anything larger has to make the physics cheaper (option 1, already
  measured), change the search (population size, trial length), or move to the
  GPU (option 5); all of those change results.

---

## Status and first measurement (2026-11-13)

### Done

| | |
|---|---|
| Port | `Oribots-main` copied here; runs on Windows with Python 3.11 (`setup.bat`). `noise` is skipped on Windows (only the unused rugged terrain needs it; see `engine/PATCHES.md`). |
| Step 0 | `bench/bench.py` (one fixed generation), `bench/compare.py` (speedup, fitness difference, rank correlation), `bench/breakdown.py` (where one trial's time goes), all run by `bench/run_bench.bat`. |
| Option 1 | `SIM_INTEGRATOR` / `SIM_TIMESTEP` in a config, read in one place (`src/physics.py`) by the evaluator, its workers and `run.py -t`. Defaults are unchanged (RK4, 1 ms). |
| Tests | `tests/run_tests.bat` runs every check. New: `tests/test_physics.py`, `tests/test_bench.py`. `tests/test_jobs.py` now skips its bash/git sub-checks when those tools are missing (as on this Windows machine). |

Before any change, all 13 original checks passed on this machine, including
the 25-second end-to-end training workflow.

### What one trial actually costs

`bench/breakdown.py` on `spider_cmaes`, one 20-second trial, RK4 at 1 ms, on
this Windows machine:

| part | share of the trial |
|---|---|
| physics (MuJoCo, stepped in C) | **77%** (28.5 µs per step) |
| Python's own per-step loop overhead | 2.7% |
| control, sampling and state copies | 20% (the brain alone: 5%) |
| model compile | about 35 ms once per trial (negligible for long trials) |

A full 1000-second trial therefore takes about 37 s on one core.

### What this changes in the plan

The estimates for options 2–4 were wrong. **Physics dominates**, and options
2, 3 and 4 all remove the same non-physics ~23%:

- **Option 2** (less Python per step) can save at most about 3%. Not worth
  doing on its own, and it would be replaced by option 3 or 4 anyway.
- **Options 3 and 4** (batched CPU / C++) can save at most about 23% per trial,
  roughly **1.3×**, not 5–30×. They also do the same job, so at most one should
  be built, and only after option 1, which shrinks the physics share and so
  makes the remaining overhead a larger fraction.
- **Option 1** directly reduces the 77%, so it is the right first change. Its
  real gain is measured by `bench\run_bench.bat` (not yet run, so the machine
  stays free for other work).
- **Option 5** (GPU) is the only option that changes the scale of the
  problem, because it runs hundreds of physics simulations at once. This
  machine has an RTX 5060 Ti, but JAX has no native Windows GPU support (MJX
  would need WSL2, which is not installed). MuJoCo Warp may run natively on
  Windows; that needs checking.

The larger savings outside these five options are still the ones from the
first review: running all cores (`NUM_SIMULATORS` defaults to at most 8),
stopping stalled trials early, and smaller CMA-ES populations.

### Next step

Run `tests\run_tests.bat`, then `bench\run_bench.bat`, and pick the integrator
and timestep from `bench\results\summary.txt`. A setting is a good candidate
when its rank correlation with the baseline is above about 0.9 and the top
controllers mostly stay the same. Check it visually with `run.py -t` as well.

---

## Why training is slow today

For `config/spider_cmaes.py` one run is:

| quantity | value |
|---|---|
| generations | 100 |
| candidates per generation (`CMA_POPULATION_SIZE`) | 100 |
| training balls per candidate (`NUM_TRAINING_BALL_POSES`) | 5 |
| **trials per run** | **50,000** |
| simulated time per trial (`SIMULATION_TIME`) | up to 1000 s |
| physics timestep | 1 ms, RK4 integrator |
| **physics steps per trial (no early reach)** | **1,000,000** |

(`config/spider.py`, the EA, runs 800 × 5 × 200 = 800,000 trials.)

Each trial pays for three things:

1. **Physics.** RK4 (`_scene_to_model.py:66`) evaluates the dynamics four times
   per step, at a 1 ms timestep (`simulation_parameters.py:7`).
2. **Python on every physics step.** `_simulate_scene.py:150-186` runs a Python
   loop iteration per 1 ms step: time comparisons, building an empty
   camera-image dict, and one `mj_step` call.
3. **Per-trial setup.** Each trial builds an MJCF scene, compiles a new
   `MjModel`, pickles the scene to a worker process (`evaluator.py:368-411`),
   and runs the brain in pure Python, looping over hinges one at a time
   (`ball_aware_brain.py:236-257`).

Every trial in a run uses **the same body**. Only the controller weights and
the ball's starting position differ. Options 3–5 below rely on this.

---

## Step 0: benchmark (do this first)

Add `OrigbotsRewrite/bench/` with a script that:

- loads one config (`spider_cmaes`) and one fixed generation of candidates
  (fixed seed), and fixed ball poses;
- times one full generation through the current evaluator;
- saves every trial's fitness to `baseline_fitness.npy`.

Every option below is timed against this script and compared against
`baseline_fitness.npy`:

- options that keep the physics must match it **exactly** (or to rounding);
- options that change the physics are compared on **rank order** and on whether
  the gaits still look sensible (`run.py -t`).

---

## Option 1: Switch the physics integrator

| | |
|---|---|
| Expected gain | **~4–8×** |
| Physics | **Changes** |
| Effort | Low (two lines plus validation) |

**What changes.** Use MuJoCo's recommended `implicitfast` integrator (one
dynamics evaluation per step) instead of RK4 (four), and raise the timestep
from 1 ms to about 2–5 ms.

**Where.**
- `engine/simulators/mujoco_simulator/revolve2/simulators/mujoco_simulator/_scene_to_model.py:66`:
  `env_mjcf.option.integrator = "RK4"` → `"implicitfast"`.
- `engine/ci_group/revolve2/ci_group/simulation_parameters.py:7`:
  `STANDARD_SIMULATION_TIMESTEP = 0.001` → try `0.002`, then `0.004`.
- Make both overridable from the config (for example `SIM_INTEGRATOR`,
  `SIM_TIMESTEP`) so existing runs and old snapshots keep the old physics when
  replayed with `run.py -t`.

**Keep the control rate.** Control runs at 20 Hz (every 50 ms), which is a
whole number of steps at 1, 2, 2.5 and 5 ms. Pick a timestep that divides 50 ms
evenly.

**Optional extra savings** (also change the physics):
- turn off robot self-collision (`contype` / `conaffinity`), so only
  robot–floor and robot–ball contacts are computed;
- lower `option.iterations` for the solver.

**Risks.**
- At larger timesteps the stiff position actuators (`kp` from
  `pid_gain_p`) can become unstable: jitter, or robots launching themselves.
- Gaits shift, so results are not directly comparable with RK4 runs.
- The `make check` reference values (bodies are unaffected; evaluator, brain
  and workflow references are) must be regenerated.

**Validation.**
- With the benchmark generation, compare fitness rank order with the baseline
  (Spearman correlation; aim for > 0.9).
- Replay the best controllers from existing RK4 runs with the new physics and
  check they still walk to the ball.
- Run a short training run and compare its fitness curve with an RK4 run.

---

## Option 2: Reduce Python overhead in the simulation loop

| | |
|---|---|
| Expected gain | **~1.2–1.5×** (more once option 1 makes physics cheaper) |
| Physics | Unchanged (see the timing caveat) |
| Effort | Low |

**What changes.** The loop at `_simulate_scene.py:150-186` runs Python on every
1 ms physics step, but control happens only every 50 steps and sampling every
200. Step to the next event in one call instead:

```python
# sketch: advance straight to the next control/sample event
steps = steps_until_next_event(data.time, last_control_time, last_sample_time)
mujoco.mj_step(model, data, nstep=steps)
```

Also:
- skip the camera dict comprehension (`_simulate_scene.py:183-186`) when the
  robot has no cameras (true for every Oribots body);
- skip the `not headless or record_settings` checks inside the loop when
  training headless (hoist them out of the loop).

**Timing caveat.** The current loop compares accumulated float times
(`time >= last_control_time + control_step`). After 50 steps of 0.001, the
time can be `0.04999999…`, so control can fire one step later than an exact
count would. Two choices:

- **exact count** (control every exactly 50 steps): cleaner, but the
  `make check` reference values may change slightly;
- **reproduce the float behaviour**: compute the next event by the same float
  comparison, stepping one at a time only near the boundary. This keeps results
  bit-identical.

Recommend the exact count, with references regenerated in the same change.

**Validation.** Benchmark fitnesses match the baseline exactly (with the float
behaviour reproduced) or to within about 1e-6 (with the exact count).

---

## Option 3: Compile the robot once and step many trials together on the CPU

| | |
|---|---|
| Expected gain | **~5–20×** on top of options 1–2 |
| Physics | **Unchanged** (same CPU MuJoCo) |
| Effort | Moderate |

**Idea.** Every trial in a run uses the same body, so one compiled model can
serve all of them. Run a whole generation (for example 100 candidates × 5
balls = 500 trials) as one batch: Python runs once per **control step for the
whole batch**, instead of once per **physics step per trial**.

**Design.**

1. **One model per run.** Build the scene (robot + ball + floor) once with the
   existing `scene_to_model` and keep the `MjModel`. The ball has a free joint,
   so its start position is just entries in `qpos`, set per trial.
2. **One `MjData` per trial**, all from that model. Reset each one to the
   robot's start pose and that trial's ball position.
3. **Batched brain** (replaces `BallAwareCpgBrainInstance.control`):
   - CPG state as an `(N, n_states)` array; the RK4 CPG step becomes batched
     matmuls with an `(N, n, n)` weight stack (`np.einsum` or `np.matmul`);
   - steering as `tanh(einsum(steering_weights, inputs))`, shape `(N, n_hinges)`;
   - robot and ball positions read for all trials at once from `qpos`/`xpos`
     at fixed indices (looked up once from the mapping).
   - Write targets into each trial's `ctrl` at the position-actuator indices
     (the velocity actuators stay at 0, as in `_control_interface_impl.py:44-46`).
4. **Batched physics.** Between control steps, advance every active trial 50
   physics steps with `mujoco.rollout` (multi-threaded in C, one `MjData` per
   thread, persistent thread pool), passing `nstep=50` and holding `ctrl`
   constant. That matches what happens now, because the actuators are position
   servos whose targets are set once per control step.
5. **Early stopping per trial.** Keep an `active` mask; a trial that reaches
   the ball (or stalls) drops out of the batch.
6. **Fitness on arrays.** Record the robot and ball xy positions at 5 Hz into
   an `(N, T, 2)` array and compute progress, reach, time-to-reach and
   alignment vectorised. This replaces the per-state Python loops in
   `evaluator.py:545-695`.
7. **Keep the evaluator's API.** `Evaluator.evaluate_on_ball_poses_full` keeps
   its signature, so `src/cmaes/main.py`, `src/ea/main.py` and recording are
   unchanged. Choose the backend with a config flag (for example
   `EVAL_BACKEND = "batched"`), with `"legacy"` kept for comparison.

**What disappears.** Per-trial MJCF building and compiling, pickling scenes to
worker processes, the `ProcessPoolExecutor` per generation, and nearly all of
the Python work per physics step.

**Parallelism.** `mujoco.rollout` threads in C, so one process can use every
core. For multi-node HPC, split the batch across processes (one per node or
socket).

**Risks.**
- Integration details: reproducing the brain's exact orientation handling
  (core mode uses the permuted orientation, see `ball_aware_brain.py:299-315`
  and `robot_frame.py`), head mode, and the stop condition.
- Initial CPG state and the order in which control and sampling happen must
  match the current loop exactly to get identical results.
- Check the `mujoco.rollout` thread and state API against the pinned
  `mujoco==3.10.0`.

**Validation.** Benchmark fitnesses must match the baseline exactly, for both
core mode and head mode (`spider_8ball*`). The existing `tests/` fingerprints
are the acceptance test.

---

## Option 4: Rewrite the trial loop in C++

| | |
|---|---|
| Expected gain | **~10–30×** |
| Physics | **Unchanged** (same MuJoCo library) |
| Effort | Moderate to high |

**Idea.** Move the whole trial into compiled code. Python keeps CMA-ES / the
EA, recording and snapshots, and calls one function per generation:

```python
fitnesses, trial_fitnesses, movements = oribots_sim.evaluate(
    model_xml, params_matrix, ball_poses, sim_settings
)
```

**Design.**
- A pybind11 module (`oribots_sim`) linking the MuJoCo C library.
- Load the model once from the MJCF string the existing `scene_to_model`
  produces (Python still builds the scene, so `bodies.py` and the configs are
  unchanged).
- In C++: the CPG RK4 step, the steering layer, the head/core reference point,
  the stop condition, 5 Hz sampling and the four fitness terms.
- Parallel across trials with OpenMP, one `mjData` per thread.
- Return the same arrays the evaluator returns today, so the search loops and
  recording do not change.

**Compared with option 3.** Similar or better speed, with no Python at all
inside a trial. The costs are a C++ build on each platform (Windows MSVC,
Linux HPC, the Singularity image for PBS) and the brain and fitness logic
existing twice (Python for `run.py -t` replay, C++ for training). Those two
copies must be kept in step; the reference tests check that they agree.

**Recommendation.** Do option 3 first. Move to option 4 only if profiling
shows Python is still a large share of the time once option 3 is in place.

**Validation.** As option 3: benchmark fitnesses match the baseline exactly,
plus a test that runs the same trials through the Python and C++ brains and
compares them.

---

## Option 5: Run everything on the GPU (MJX or MuJoCo Warp)

| | |
|---|---|
| Expected gain | **~50–500×** with suitable hardware |
| Physics | **Changes** (different engine and contact handling) |
| Effort | High |

**Idea.** Simulate the whole generation as one array computation on the GPU,
with the brain and the optimizer on the GPU as well.

**Design.**
- **Physics:** MJX (MuJoCo in JAX) or MuJoCo Warp (NVIDIA Warp backend),
  loaded from the same MJCF, with all 500+ trials in one batch (`jax.vmap`).
- **Brain:** a few JAX lines: CPG RK4 (matmuls), steering (`tanh` of a dot
  product), clip, written to `ctrl`.
- **Loop:** `jax.lax.scan` over control steps, with 50 physics steps inside
  each, compiled as a single XLA program.
- **Fitness:** computed on the GPU from the recorded positions.
- **Optimizer:** `evosax` (JAX CMA-ES), so a generation never leaves the
  device. Alternatively keep `cma` on the CPU and move only the parameter
  arrays and the fitness across each generation.

**Required model changes.**
- **Collision shapes:** MJX handles box–box contacts poorly, and Revolve2
  modules are boxes. Use capsules or spheres for robot collision geometry, and
  keep only robot–floor and robot–ball contacts. MuJoCo Warp copes better with
  contacts and is the safer choice for this robot.
- **Integrator:** `implicitfast` or Euler (as option 1).
- **Early stopping** saves nothing: finished trials sit idle until the whole
  batch ends. The real cost is therefore the full `SIMULATION_TIME` for every
  trial, so shortening `SIMULATION_TIME` matters more here.

**Platform.**
- NVIDIA GPU with CUDA. On Windows, JAX on a GPU needs WSL2. On the HPC, a GPU
  node (Slurm or PBS GPU queue).
- New dependencies (`jax[cuda]`, `mujoco-mjx` or `mujoco-warp`, `evosax`) must
  stay out of the CPU install and the Singularity image, or have their own
  image.

**Risks.**
- Gaits found on the GPU may not behave the same in CPU MuJoCo (which
  `run.py -t` uses for replay). Check the best controllers on the CPU before
  trusting results.
- Results are not comparable with earlier runs.
- Largest rewrite of the five, and the hardest to debug (compiled JAX code).

**Validation.**
- Replay GPU-trained controllers in CPU MuJoCo and compare the scores.
- Run the same CMA-ES settings on CPU and GPU on a short run and compare the
  fitness curves.
- Visual check with `run.py -t`.

---

## Summary

| # | Option | Gain (est.) | Physics | Effort | Matches baseline exactly? |
|---|---|---|---|---|---|
| 1 | Switch integrator (RK4 → implicitfast, larger timestep) | 4–8× | changes | low | no (rank order only) |
| 2 | Less Python overhead in the sim loop (`nstep`) | 1.2–1.5× | same | low | yes, or to ~1e-6 |
| 3 | Compile once, batch trials on CPU (`mujoco.rollout`, batched brain) | 5–20× | same | moderate | yes |
| 4 | C++ trial loop (pybind11 + OpenMP) | 10–30× | same | moderate–high | yes |
| 5 | GPU (MJX / MuJoCo Warp + evosax) | 50–500× | changes | high | no |

The gains of options 1 and 2 multiply with those of 3 and 4. Options 3 and 4
are alternatives to each other. Option 5 replaces everything else.

## Suggested order

1. **Step 0:** benchmark script and baseline fitness file.
2. **Option 2:** quick, and keeps the physics; this confirms that the benchmark
   and comparison work.
3. **Option 3:** largest gain without changing the physics; the existing
   reference tests stay valid.
4. **Option 1:** as a deliberate, validated physics change, applied on top of
   option 3, with new reference values.
5. **Option 4** only if profiling after option 3 shows Python still dominates.
6. **Option 5** only if GPU hardware is available and the project accepts
   different physics.

## Where the work happens

`OrigbotsRewrite/` is kept separate from `Oribots-main/` so that the original
code stays untouched as the baseline. Suggested layout:

```
OrigbotsRewrite/
  PERFORMANCE_OPTIONS.md   this document
  bench/                   Step 0: benchmark script and baseline fitness files
  ...                      a copy of Oribots-main with the chosen options applied
```
