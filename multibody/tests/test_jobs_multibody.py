"""The multibody job scripts, checked without a scheduler (R25.6, R25.9, R25.8).

For every script in ``multibody/jobs/``: it parses (``bash -n``); its Entry_Point
command starts ``-r`` on a config or continues ``-c`` on the latest snapshot; the
latest-snapshot selection picks gen10 over gen9 and gen2; the Slurm jobs carry
the account, partition, time, logs, conda env, ``MUJOCO_GL`` and single-thread
settings of the existing UCT jobs; the PBS job carries the queue, cores,
walltime, project, absolute default log paths without shell variables, the
container launch and the documented ``qsub -o/-e`` override for chains. The
pre-flight refusals (R25.8) run for real, stopping before any module load.

The default config (``spider_k4``) must be accepted by the Config_Loader. Its
real trained weights (the UCT run's gen30, in the old repository) are not in this repository, so when that file
is absent the config is checked with the fixture weights in its place.
"""

import os
import re
import subprocess
from pathlib import Path

from _check import check_main, require, temp_workspace

import _runs
import config_loader

JOBS = _runs.MULTIBODY / "jobs"
ROOT = _runs.ROOT


def bash(script: str, env: dict, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", script], cwd=str(cwd), env={**os.environ, **env}, capture_output=True, text=True, timeout=60)


def run() -> None:
    scripts = sorted(JOBS.glob("*.job")) + sorted(JOBS.glob("*.pbs"))
    require(len(scripts) >= 3, f"expected the UCT start/resume jobs and the Lengau job, found {[s.name for s in scripts]}")
    existing_slurm = (ROOT / "jobs" / "spider_8ball.job").read_text()
    account = re.search(r"^#SBATCH --account=.*$", existing_slurm, re.M).group(0)
    partition = re.search(r"^#SBATCH --partition=.*$", existing_slurm, re.M).group(0)
    require((ROOT / "jobs" / "logs" / ".gitkeep").is_file(), "jobs/logs/ is not present in a fresh clone")

    for script in scripts:
        text = script.read_text()
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        require(result.returncode == 0, f"{script.name} does not parse: {result.stderr}")
        commands = re.findall(r"python multibody/run_multibody\.py (-[rc]) (\S+)(?: (\S+))?", text)
        require(commands, f"{script.name}: no run_multibody.py command")
        for flag, first, second in commands:
            require((flag == "-r" and first == '"$CONFIG"' and second == '"$OUTPUT"') or (flag == "-c" and first == '"$LATEST"'),
                    f"{script.name}: unexpected entry command {flag} {first} {second}")
        if script.suffix == ".job":
            for needed in (account, partition, "#SBATCH --output=jobs/logs/%x-%j.out", "#SBATCH --error=jobs/logs/%x-%j.err",
                           'CONDA_ENV="$REPO_DIR/.conda"', "export MUJOCO_GL=egl", "export OMP_NUM_THREADS=1",
                           "export OPENBLAS_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "export NUMEXPR_NUM_THREADS=1", "SLURM_NTASKS"):
                require(needed in text, f"{script.name}: missing '{needed}'")
            hours = int(re.search(r"^#SBATCH --time=(\d+):", text, re.M).group(1))
            require(hours <= 100, f"{script.name}: --time {hours} h exceeds 100 h")
        else:
            for needed in ("#PBS -q smp", "#PBS -l select=1:ncpus=24", 'cd "$PBS_O_WORKDIR"', "source hpc/pbs/_run_in_container.sh",
                           "run_in_container python multibody/run_multibody.py", "NCPUS"):
                require(needed in text, f"{script.name}: missing '{needed}'")
            pbs_lines = [line for line in text.splitlines() if line.startswith("#PBS")]
            require(not any("$" in line for line in pbs_lines), f"{script.name}: a #PBS line uses a shell variable")
            require(any(line.startswith("#PBS -P ") for line in pbs_lines), f"{script.name}: no #PBS -P")
            for flag in ("-o", "-e"):
                line = next((l for l in pbs_lines if l.startswith(f"#PBS {flag} ")), "")
                require(line.split()[-1].startswith("/mnt/lustre/"), f"{script.name}: #PBS {flag} is not an absolute /mnt/lustre path")
            walltime = re.search(r"walltime=(\d+):(\d+):(\d+)", text)
            require(walltime and int(walltime.group(1)) * 3600 + int(walltime.group(2)) * 60 + int(walltime.group(3)) <= 96 * 3600,
                    f"{script.name}: walltime above 96 h")
            require(re.search(r"qsub -W depend=afterany:\S+.*\n.*-o \S+.*\n.*-e \S+", text) is not None,
                    f"{script.name}: the per-job qsub -o/-e override for chains is not documented")

    # Latest-snapshot selection.
    with temp_workspace() as workspace:
        for name in ("gen2.pkl", "gen9.pkl", "gen10.pkl", "genx.pkl", "notes.txt"):
            (workspace / name).write_text("")
        result = subprocess.run(["bash", "-c", f'source "{JOBS / "_latest_snapshot.sh"}"; latest_snapshot "{workspace}"'],
                                capture_output=True, text=True)
        require(result.stdout == str(workspace / "gen10.pkl"), f"latest snapshot picked {result.stdout!r}")

        # Pre-flight refusals (R25.8): each must stop before anything runs.
        empty = workspace / "empty_run"
        empty.mkdir()
        cases = [
            ("multibody_train.job", {"SLURM_SUBMIT_DIR": str(workspace), "SLURM_NTASKS": "4", "SLURM_JOB_ID": "1"}, "repository root"),
            ("multibody_train.job", {"SLURM_SUBMIT_DIR": str(ROOT), "SLURM_NTASKS": "", "SLURM_JOB_ID": "1"}, "SLURM_NTASKS"),
            ("multibody_train.job", {"SLURM_SUBMIT_DIR": str(ROOT), "SLURM_NTASKS": "4", "SLURM_JOB_ID": "1", "CONFIG": "multibody/configs/nope.py"}, "not found"),
            ("multibody_resume.job", {"SLURM_SUBMIT_DIR": str(ROOT), "SLURM_NTASKS": "4", "OUTPUT": ""}, "OUTPUT is required"),
            ("multibody_resume.job", {"SLURM_SUBMIT_DIR": str(ROOT), "SLURM_NTASKS": "4", "OUTPUT": str(empty)}, "No gen<N>.pkl"),
            ("multibody_resume.job", {"SLURM_SUBMIT_DIR": str(ROOT), "SLURM_NTASKS": "4", "OUTPUT": str(workspace / "missing")}, "not found"),
            ("multibody.pbs", {"PBS_O_WORKDIR": str(workspace), "NCPUS": "24", "OUTPUT": str(empty)}, "repository root"),
            ("multibody.pbs", {"PBS_O_WORKDIR": str(ROOT), "NCPUS": "24", "OUTPUT": ""}, "OUTPUT is required"),
            ("multibody.pbs", {"PBS_O_WORKDIR": str(ROOT), "NCPUS": "", "OUTPUT": str(empty)}, "NCPUS"),
            ("multibody.pbs", {"PBS_O_WORKDIR": str(ROOT), "NCPUS": "24", "OUTPUT": str(empty), "CONFIG": ""}, "no existing CONFIG"),
        ]
        for name, env, words in cases:
            result = bash(str(JOBS / name), env, ROOT)
            require(result.returncode != 0 and words in result.stderr,
                    f"{name} with {env}: expected a refusal mentioning '{words}', got {result.returncode}: {result.stderr[-400:]}")
        require(not any(empty.iterdir()), "a refused job wrote into the run folder")

        # The default config is accepted (with the fixture weights while the real ones are absent).
        config = _runs.MULTIBODY / "configs" / "spider_k4.py"
        weights = (config.parent / "../../../output/spider_8ball/gen30.pkl").resolve()
        if weights.is_file():
            config_loader.load_contest(config)
        else:
            text = config.read_text().replace("../../../output/spider_8ball/gen30.pkl", str(_runs.FIXTURES / "fake_single_robot_gen1.pkl"))
            stand_in = workspace / "spider_k4.py"
            stand_in.write_text(text)
            settings = config_loader.load_contest(stand_in)
            require(settings.k == 4 and settings.terrain_size == (12.0, 12.0) and settings.simulation_time == 1000,
                    "spider_k4 settings")


if __name__ == "__main__":
    check_main("JOBS", run)
