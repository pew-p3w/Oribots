# Running on CHPC Lengau — a practical reference

CHPC (the Centre for High Performance Computing, South Africa) runs **Lengau**,
a cluster that schedules work with **PBS Pro**. This note collects the basics of
using it, plus the specifics of running this project (Oribots) there.

It is written for anyone who has to run work on Lengau, whether or not they know
this project. If you have never used a PBS cluster before, read section 1 first;
if you just need the Oribots recipe, skip to section 6.

Nothing here is secret configuration: usernames, project codes and paths are
per-account and are shown as placeholders (`<user>`, `<PROJECT>`). Fill in your
own. Official documentation always wins over this note; see section 9.

---

## 1. The mental model

A cluster is not one computer you log into and run things on. It is:

- **Login nodes** — where you land when you SSH in. Shared by everyone. Use them
  only to edit files, move data, and submit jobs. **Do not run real compute
  here**; heavy work on a login node is killed and slows the node down for
  every other user.
- **Compute nodes** — where real work runs. You never SSH to them directly. You
  describe the work in a *job script* and hand it to the scheduler, which finds
  a free node and runs it for you, possibly after a wait in a queue.
- **The scheduler (PBS Pro)** — decides when and where your job runs, based on
  what you asked for (cores, memory, walltime) and your allocation.
- **Shared storage (lustre)** — a large parallel filesystem visible from both
  login and compute nodes. Your job's code, inputs and outputs live here.

The basic loop is: **write a job script → `qsub` it → wait in the queue → it runs
on a compute node → it writes output to lustre → you collect the output.**

---

## 2. Logging in and moving data

```bash
# Log in (from your own machine):
ssh <user>@lengau.chpc.ac.za

# Copy data TO the cluster (run this on your own machine, NOT while SSH'd in):
scp -r ./myproject <user>@lengau.chpc.ac.za:~/lustre/myproject

# Copy results BACK (again, from your own machine):
scp -r <user>@lengau.chpc.ac.za:~/lustre/myproject/output ./output
```

`~/lustre` is a symlink to your space on the shared filesystem, typically
`/mnt/lustre/users/<user>`. Put code and data there, not in your home directory,
which is small. For large or many files prefer `rsync -a` over `scp` so an
interrupted transfer can resume.

There is **no internet on the compute nodes** (and often not on login nodes
either), so you cannot `pip install`, `git clone`, or `docker pull` from inside
a job. Everything a job needs must already be on lustre before you submit it.

---

## 3. Software: modules

Lengau provides software through **environment modules**. You load what you need
at the top of a job (or in an interactive session):

```bash
module purge                 # start from a clean environment
module avail                 # list everything available (long)
module load <name>           # add a package to your environment
module list                  # show what is currently loaded
```

If `module` itself is not found inside a job script, run
`source /etc/profile.d/modules.sh` before the first `module` command; CHPC's own
example job scripts do this.

Common patterns seen on Lengau:

- A Python/conda stack via a module (for example an `anaconda` module), then a
  conda environment. Inside a **non-interactive PBS job** you must run
  `eval "$(conda shell.bash hook)"` **before** `conda activate <env>`; without
  it `conda activate` fails, because the job's shell is not set up for conda.
- A container runtime via a module, for example `chpc/singularity/3.5.3`. This
  is what Oribots uses (section 6).

Because there is no internet on the nodes, you cannot install new modules
yourself. You either use what is provided, or you bring your own environment in
a container image (section 6).

---

## 4. Anatomy of a PBS job script

A job script is a normal bash script with `#PBS` directive lines near the top.
The scheduler reads those lines; the shell ignores them (they look like
comments). A minimal, well-formed job:

```bash
#!/bin/bash
#PBS -N my_job                       # a short name (shows up in qstat)
#PBS -P <PROJECT>                    # the allocation to charge (REQUIRED)
#PBS -q smp                          # which queue (see section 5)
#PBS -l select=1:ncpus=24            # 1 node (chunk), 24 CPU cores on it
#PBS -l walltime=96:00:00            # max wall-clock time, HH:MM:SS
#PBS -m abe                          # email on (a)bort, (b)egin, (e)nd
#PBS -M you@example.com              # where to email
#PBS -o /mnt/lustre/users/<user>/logs/my_job.out   # stdout (absolute path)
#PBS -e /mnt/lustre/users/<user>/logs/my_job.err   # stderr (absolute path)

set -euo pipefail                    # stop on the first error
ulimit -s unlimited                  # large stack; avoids segfaults in native code

# PBS drops you in your home directory. Move to where you submitted from:
cd "$PBS_O_WORKDIR"

module purge
module load <whatever your work needs>

# ... run the actual work here ...
```

Key points people trip on:

- **`-P <PROJECT>` is mandatory** — jobs with no valid project are rejected. The
  project is your allocation; time is accounted against it.
- **`-o`/`-e` must be writable paths on lustre**, and the directory must already
  exist. PBS normally copies a job's output there when the job ends and does not
  create the folder, so if it is missing the log is not where you look for it.
  Make the folder first (`mkdir -p /mnt/lustre/users/<user>/logs`).
- **`#PBS` lines are read by the scheduler, not by the shell**, so shell
  variables such as `$USER` or `$HOME` are not expanded in them. Write literal
  values there, or pass the option on the `qsub` command line instead.
- **A job starts in `$HOME`, not where you ran `qsub`.** Use
  `cd "$PBS_O_WORKDIR"` (the directory you submitted from) or an absolute path.
- **`-l select=1:ncpus=N`** asks for one chunk (node) with N cores. Ask for the
  cores your work can actually use; asking for more only makes you wait longer.
- **`walltime`** is a hard cap. When it is reached the job is killed immediately,
  finished or not. Choose it a bit above your real need, within the queue limit.

### Useful PBS environment variables (set for you inside a job)
- `$PBS_O_WORKDIR` — the directory you ran `qsub` from.
- `$PBS_JOBID` — the full job id (for example `1234567.sched01`). Use
  `${PBS_JOBID%%.*}` to get just the number, handy for unique output folders.
- `$PBS_NODEFILE` — a file listing the nodes/cores assigned; `wc -l < $PBS_NODEFILE`
  gives the core count.
- `$NCPUS` — the number of cores requested (from your `select` line). Good for
  telling your program how many workers to start.

---

## 5. Queues, and picking one

PBS routes jobs to **queues**, each with its own limits (max cores, max walltime,
how many jobs you may run at once). Queue names and limits are cluster policy and
change over time, so check the current list rather than trusting any note.

Two that this project uses:

- **`smp`** — a shared-memory queue for single-node jobs (up to the node's core
  count, commonly 24 on Lengau) with a long walltime ceiling (up to ~96 hours).
  Good for one long multi-core run.
- **`serial`** — for single-node jobs that need fewer cores than a full node
  (1 to 23 on a 24-core node). Because it asks for only part of a node it often
  starts sooner than `smp`, which makes it a good place for a quick pre-flight
  check before committing a long run.

Jobs that span several nodes use other queues, such as `normal`. Match the queue
to the job: a short test does not belong in a long-walltime queue, and a
full-node run does not belong in `serial`.

---

## 6. Running this project (Oribots) on Lengau

Oribots pins a modern scientific stack (MuJoCo and friends) that needs a newer
system library (glibc ≥ 2.28) than Lengau's compute nodes provide. The fix is to
run inside a **Singularity container** that carries the right environment, while
the code itself is a normal checkout on lustre, bind-mounted into the container.
So: **the environment lives in the image; the code lives on lustre.** A code
change never needs a new image — only a dependency change does.

Because the nodes have no internet, the image is **built off-cluster and copied
in**. The steps:

1. **Build the image tarball** on any machine with Docker (off the cluster):
   ```bash
   bash hpc/build_sif.sh
   ```
   This builds the environment image and saves it to `hpc/dist/oribots-<commit>.tar`,
   then prints the exact copy and conversion commands with real values (run it
   as `LENGAU_USER=<user> bash hpc/build_sif.sh` to have the username filled in).

2. **Copy the tarball to the cluster** and convert it to a `.sif` there:
   ```bash
   scp hpc/dist/oribots-<commit>.tar <user>@lengau.chpc.ac.za:~/lustre/
   # then on a Lengau login node:
   module load chpc/singularity/3.5.3
   cd ~/lustre
   singularity build oribots.sif docker-archive://oribots-<commit>.tar
   ```
   If an unprivileged `singularity build` is not allowed on the login node, build
   the `.sif` off-cluster with Apptainer instead and copy the `.sif` over — but
   first confirm it opens in the cluster's (older) Singularity with
   `singularity inspect oribots.sif`.

3. **Set the per-account values, then put a checkout on lustre.** In each
   `hpc/pbs/*.pbs`, change the `-P` project code, the `-M` email address and the
   `-o`/`-e` log paths to your own. `make check-pbs` checks the project code
   against `PROJECT` in `tests/test_pbs.py`, so change it there as well. Then
   copy the repository to lustre (`scp`/`rsync`, or a clone from a machine that
   has internet) and create the log folder those paths point to, for example:
   ```bash
   mkdir -p /mnt/lustre/users/<user>/oribots_logs
   ```

4. **Run the pre-flight gate first.** It runs a short check on a real compute
   node — the image loads, the stack imports, offscreen rendering works, and a
   tiny train/resume completes — before you spend a long run finding a problem:
   ```bash
   cd ~/lustre/Oribots        # your checkout
   qsub hpc/pbs/gate.pbs
   ```
   Wait for it to finish and confirm the log ends in `PBS GATE PASSED`.

5. **Start a real run, and queue its first resume straight away**:
   ```bash
   qsub hpc/pbs/train.pbs                      # prints e.g. 7506242.sched01
   # The run writes to output/<NAME>_<job number>, e.g. output/spider_8ball_cmaes_7506242.
   qsub -W depend=afterany:7506242.sched01 \
        -o /mnt/lustre/users/<user>/oribots_logs/resume_1.out \
        -e /mnt/lustre/users/<user>/oribots_logs/resume_1.err \
        -v OUTPUT=output/spider_8ball_cmaes_7506242 hpc/pbs/resume.pbs
   ```
   - `afterany` holds the resume (state `H` in `qstat`) until the previous job
     ends for any reason, including its walltime, then it continues from the
     latest `gen*.pkl` by itself.
   - Once a resume starts running, queue the next one the same way, depending on
     *that* resume's job id and with the next log number (`resume_2`, ...).
   - **Give every resume its own `-o`/`-e`.** `resume.pbs` names fixed log files,
     so without these options each resume overwrites the previous one's log.
     Options on the `qsub` line win over the `#PBS` lines in the script.
   - **Plan the number of jobs from the measured speed.** After a few
     generations, `ls -l --time-style=full-iso output/<run>/gen*.pkl` gives the
     time per generation. For example, `spider_8ball_cmaes` (population 500,
     8 balls, 1500 s) took about 11 h per generation on 24 cores. That is about
     8 generations per 96 h job, so 100 generations need about 13 jobs.
     A walltime kill loses the generation in progress, so budget for that too.

The job scripts under `hpc/pbs/` carry the project code, queue, resources and log
paths; read the comments at the top of each for how to override the config, name
and output folder. `hpc/build_sif.sh` documents the image/bind-mount design in
detail.

Long runs outlive a single walltime window, so training **checkpoints every
generation** and `resume.pbs` continues from the latest snapshot. A walltime kill
therefore loses at most the generation in progress. Chain `resume.pbs` jobs until
the run reaches the target number of generations.

---

## 7. Everyday PBS commands

```bash
qsub jobs/my_job.pbs                 # submit; prints the job id
qsub -v NAME=x,CONFIG=y jobs/j.pbs   # submit, passing variables into the script
qstat -u <user>                      # your jobs and their state (Q=queued, R=running)
qstat -f <jobid>                     # everything about one job
qdel <jobid>                         # cancel a queued or running job
qsig -s SIGTERM <jobid>              # send a running job a signal

# An interactive shell on a compute node, for testing or compiling (login
# nodes are not for either):
qsub -I -P <PROJECT> -q serial -l select=1:ncpus=4 -l walltime=01:00:00
```

**Why did a job stop?** Its exit status (`Exit_status` in `qstat -xf <jobid>`
once it has finished, and in the end-of-job email) says. Below 128 it is the
script's own exit code. From 128 up the job was killed by a signal: the
remainder after dividing by 128 (or 256) is the signal number, so 137 means
signal 9. CHPC documents 271 as a job killed for exceeding what it requested,
such as its walltime.

While a job runs, its stdout/stderr may be buffered and only appear in the `-o`/`-e`
files at the end. For live progress, have the job write to its own log file on
lustre and `tail -f` that. Check `qstat` to see whether a job is still queued (`Q`),
held behind a dependency (`H`) or actually running (`R`).

**Oribots on Lengau:** the job logs (`oribots_logs/*.out`/`.err`) arrive only
when a job ends, and the training log is in the `.err` file (Python logging
writes to stderr). While a run is going, follow it through its output folder
instead, which is written live:
```bash
qstat -u <user>                                           # elapsed time, state
ls -l --time-style=full-iso output/<run>/gen*.pkl         # one snapshot per finished generation
cut -d, -f1-3 output/<run>/parameters_*_run_*.csv         # generation, best, worst
```

---

## 8. Gotchas and habits that save time

- **No internet on the nodes.** Stage everything on lustre first. This is the
  single most common surprise.
- **Test small before you run big.** A short job in a fast queue (or the gate in
  section 6) catches most problems for a fraction of the cost of a long run.
- **`ulimit -s unlimited`** near the top of a job avoids stack-overflow segfaults
  in native/simulation code. Cheap insurance.
- **Absolute, pre-created log paths.** `#PBS -o`/`-e` must point at a directory
  that already exists on lustre, or the log does not arrive where you look for it.
- **`cd "$PBS_O_WORKDIR"`** — never assume the job starts where you submitted it.
- **Headless rendering.** Compute nodes have no display. Graphics-using code must
  render offscreen (this project sets `MUJOCO_GL=egl`; other stacks use their own
  headless flag, e.g. an SDL dummy driver).
- **Right-size the request.** Fewer cores and shorter walltime usually mean a
  shorter queue wait. Do not pad "just in case".
- **Watch your file count and quota.** Lustre dislikes millions of tiny files;
  runs that write per-generation artefacts add up. Clean up old runs.
- **Lustre is workspace, not an archive.** It is built for speed rather than
  safekeeping, and CHPC's policy allows removing data that has not been used for
  90 days. Copy results you want to keep off the cluster.
- **Walltime is a hard kill.** Design long work to checkpoint and resume so a kill
  costs one unit of progress, not the whole run.
- **Singularity 3.5.3 is old.** It has no `singularity exec --env` (that arrived
  in 3.6; 3.5.3 fails with `unknown flag: --env`). Pass a variable into the
  container as `SINGULARITYENV_<NAME>=value` on the host instead; `--cleanenv`
  still lets those through. `hpc/pbs/_run_in_container.sh` does this.
- **A harmless Singularity warning.** Running `singularity` from inside `~/lustre`
  prints `Bind mount '/home/<user> => /home/<user>' overlaps container CWD`.
  It does not affect the run: the jobs bind `/mnt/lustre` and set the working
  directory explicitly.

---

## 9. Where to look next

- The CHPC wiki (https://wiki.chpc.ac.za) and the welcome email for your account
  are authoritative for queue names, limits, project codes and current policy.
  Its PBS Pro quick reference (https://wiki.chpc.ac.za/quick:pbspro) has example
  job scripts and the exit-code rules above.
- `hpc/build_sif.sh` — the container image build and the bind-mount design.
- `hpc/pbs/train.pbs`, `hpc/pbs/resume.pbs`, `hpc/pbs/gate.pbs` — the actual job
  scripts, with usage notes in their header comments.
- `tests/test_pbs.py` (`make check-pbs`) — checks the PBS scripts honour the
  cluster contract (project, queue, resources, log paths) before you submit.
