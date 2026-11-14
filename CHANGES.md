# What we changed to make training fast

Hell dear reader,
This is Reece.

I took a long time to train origami bots. I didnt like that.
Training an origami robot used to take weeks. This copy of Oribots-main does the same job, A LOT FASTER. Up to a whole damn 56x faster!
You can keep working the way you did before:
the commands, configs, snapshots and cluster jobs are all the same.
blah blah blah, all that matters is this is faster and I tried to change as little possible so its works well.

I had a bunch of random notes on changes and just asked AI to rewrite what you see below to be less jumbled and propper to read.

Reece out.

## First, where does the time go?

Before speeding anything up, we measured. A training trial has two parts:

- **the physics**: MuJoCo working out how the robot moves, written in C and
  already fast per step, but it takes a thousand steps per simulated second;
- **the Python around it**: every 0.05 s the brain reads the robot's state,
  decides what the joints should do, and the result is recorded for scoring.

With the original settings, the physics is about 80% of the time. So there is
no point polishing the Python first: the physics has to get cheaper, and only
then does the Python become worth fixing.

## Step 1: a cheaper physics setting (on by default, changes results slightly - but basically unnoticable. need to look into more with the triagle/larger bots to see if it falls apart (HIGHLY LIKELY I IMAGINE))

MuJoCo has several ways (integrators) to move the simulation forward one
step. The original used RK4 with 1 ms steps: very accurate, and expensive.
We let a config choose instead, and every config in `config/` now has:

```python
SIM_INTEGRATOR = "implicitfast"   # was always "RK4"
SIM_TIMESTEP = 0.005              # was always 0.001
```

`implicitfast` with 5 ms steps does five times fewer steps, and each step is
cheaper. On its own that is about **4.8x faster**.

The catch: different physics gives slightly different numbers. We checked
how much. The same robot scores within about 0.05 under both settings, and
which robots come out best stays the same. It is worth it for the speed, so
it is **on in every config**. Snapshots remember the setting, so replaying or
resuming a robot uses the physics it was trained with: a run from the
original code still gets RK4.

The two small test configs (`config/_tiny.py`, `config/_tiny_cmaes.py`) stay
on RK4 on purpose: the checks use them to prove the code still matches the
original exactly.

## Step 2: less Python per trial (on, results identical)

Once the physics is cheap, the Python is suddenly half the time. Every control
tick, the original code built a pile of small Python objects (states, poses,
quaternions) just to read a few numbers, then threw them away.

The new trial loop (`src/fast_trial.py`) does the same work without that:

- it looks up which numbers belong to which joint once per trial, not every tick;
- it lets MuJoCo run many steps in C at a time, up to the next control tick;
- the brain reads MuJoCo's arrays directly.

Every calculation is the same as before, in the same order, so the results
are **identical, bit for bit**. We tested that on every robot config, all
four integrators, and trials that end early. One control tick went from about
180 µs to about 22 µs, which makes the trial loop about **2x faster** on top
of step 1.

## Step 3: keep the workers alive (on, results identical)

Trials run in 8 worker processes at once. The original started 8 fresh
processes every generation, and each one had to load MuJoCo and the config
again: about a second lost per generation. Now the workers are kept between
generations. Nothing about the trials changes, so neither do the results.

## Putting it together

| | per generation* | speed-up |
|---|---|---|
| original | ~73 s | 1x |
| all three changes | ~4.7 s | **~15x** |

\* Spider in the walled arena, 24 robots x 2 balls x 200 s trials, 8 workers.

Steps 2 and 3 only pay off after step 1. With the original RK4 physics they
give just 4%, because the physics still dominates.

## Turning things off

| change | turn it off |
|---|---|
| cheaper physics | `SIM_INTEGRATOR = "RK4"` and `SIM_TIMESTEP = 0.001` in the config (or delete the two lines) |
| fast trial loop | `FAST_TRIAL = False` in the config |
| kept workers | `PERSISTENT_WORKERS = False` in the config |

## One thing to watch out for

The fast trial loop has its own copy of the brain step and the scoring. If
you change the brain (`BallAwareCpgBrain`), the fitness (`_trial_fitness`) or
the "reached the ball" check, change `src/fast_trial.py` the same way, or set
`FAST_TRIAL = False` until you do. Then run `make check-fast-trial` (on
Windows: `tests\run_tests.bat fast_trial`); it fails if the two disagree.

## Smaller extras

- `SEED = 123` in a config fixes the random start and the training balls, so
  two runs can be compared fairly.
- `config/spider_arena_cmaes.py`: the spider in a small square arena with walls.
- Windows works now: `setup.bat` to install, `tests\run_tests.bat` to test.
- `bench/` has the tools we used to measure all this, and the videos.
- Full measurements and the options we did not take (a C++ rewrite, the
  GPU): `PERFORMANCE_OPTIONS.md`. Engine patches: `engine/PATCHES.md`.
