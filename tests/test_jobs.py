"""Check the Slurm job scripts in jobs/ before anything is submitted.

Cluster time costs money and a job that fails only fails after it has waited in
the queue, so every job is checked here instead. Each trainable config has
exactly one job, every job starts a clean `run.py -r` into a folder named after
its Slurm job id, and all jobs are the same script apart from the config, the
main, `--ntasks` and `--time`. `jobs/spider.job` is the canonical one.

Run with `make check-jobs`, or `python tests/test_jobs.py`.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

JOBS_DIR = paths.ROOT / "jobs"
CANONICAL = "spider"
FINAL_COMMAND = 'python run.py -r "$CONFIG" "$MAIN" "$OUTPUT"'
MAINS = {"ea": "src/ea/main.py", "cmaes": "src/cmaes/main.py"}

# Anything that points outside the repo, resumes or replays an earlier run, or
# builds its own import path instead of leaving that to run.py.
FORBIDDEN = {
    "revolve2-folding": "refers to a directory outside this repository",
    "/scratch": "refers to a fixed cluster path",
    "run.py -c": "resumes an earlier run instead of starting a clean one",
    "run.py -t": "replays a snapshot instead of training",
    "PYTHONPATH": "sets PYTHONPATH itself; run.py builds it",
}

# The cluster setup every job needs. Checked directly, not only through the
# drift check, so dropping a line from the canonical job cannot pass unnoticed.
REQUIRED = (
    "set -euo pipefail",
    "#SBATCH --account=compsci",
    "#SBATCH --partition=ada",
    "#SBATCH --output=jobs/logs/%x-%j.out",
    "#SBATCH --error=jobs/logs/%x-%j.err",
    'CONDA_ENV="$REPO_DIR/.conda"',
    'conda activate "$CONDA_ENV"',
    "export MUJOCO_GL=egl",
    "export OMP_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
    "export MKL_NUM_THREADS=1",
    "export NUMEXPR_NUM_THREADS=1",
)


def load_run_module():
    """
    Import run.py without executing a workflow.

    :returns: The loaded run module.
    """
    spec = importlib.util.spec_from_file_location("oribots_run", paths.ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["oribots_run"] = module
    spec.loader.exec_module(module)
    return module


def trainable_configs() -> list[str]:
    """
    Names of the configs that should have a job.

    :returns: Config file stems, excluding the body catalogue and test fixtures.
    """
    return sorted(
        path.stem
        for path in (paths.ROOT / "config").glob("*.py")
        if path.stem != "bodies" and not path.stem.startswith("_")
    )


def assignment(text: str, name: str) -> str | None:
    """
    The value of a top-level `NAME=value` line in a job, without quotes.

    :param text: The job script.
    :param name: The shell variable.
    :returns: The value, or None if the job does not set it.
    """
    match = re.search(rf"^{name}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip('"') if match else None


def sbatch_option(text: str, option: str) -> str | None:
    """
    The value of an `#SBATCH --option=value` directive.

    :param text: The job script.
    :param option: The option name without dashes.
    :returns: The value, or None if the job does not set it.
    """
    match = re.search(rf"^#SBATCH .*--{option}=(\S+)", text, re.MULTILINE)
    return match.group(1) if match else None


def normalized(text: str, stem: str) -> str:
    """
    A job with the parts that may differ between jobs blanked out.

    :param text: The job script.
    :param stem: The config the job trains.
    :returns: The job text with the config name, main, ntasks and time replaced.
    """
    text = re.sub(r"--ntasks=\d+", "--ntasks=N", text)
    text = re.sub(r"--time=[\d:-]+", "--time=T", text)
    for main in MAINS.values():
        text = text.replace(main, "MAIN_PATH")
    return re.sub(rf"(?<![A-Za-z0-9]){re.escape(stem)}(?![A-Za-z0-9])", "STEM", text)


def check_job(path: Path, run, canonical: str) -> list[str]:
    """
    Check one job script.

    :param path: The job file.
    :param run: The loaded run module.
    :param canonical: The normalized canonical job.
    :returns: A list of problems; empty when the job is fine.
    """
    problems = []
    stem = path.stem
    text = path.read_text(encoding="utf-8")

    parsed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    if parsed.returncode != 0:
        problems.append(f"{path.name}: bash -n failed: {parsed.stderr.strip()}")

    if sbatch_option(text, "job-name") != stem:
        problems.append(f"{path.name}: --job-name is not {stem}")
    ntasks = sbatch_option(text, "ntasks")
    if ntasks is None or not ntasks.isdigit() or int(ntasks) < 1:
        problems.append(f"{path.name}: --ntasks is missing or not a positive number")
    if sbatch_option(text, "time") is None:
        problems.append(f"{path.name}: --time is missing")

    config = assignment(text, "CONFIG")
    main = assignment(text, "MAIN")
    output = assignment(text, "OUTPUT")
    if config != f"config/{stem}.py":
        problems.append(f"{path.name}: CONFIG is {config}, expected config/{stem}.py")
    elif not (paths.ROOT / config).is_file():
        problems.append(f"{path.name}: {config} does not exist")
    expected_main = MAINS["cmaes" if stem.endswith("_cmaes") else "ea"]
    if main != expected_main:
        problems.append(f"{path.name}: MAIN is {main}, expected {expected_main}")
    elif not (paths.ROOT / main).is_file():
        problems.append(f"{path.name}: {main} does not exist")
    if config and main and (paths.ROOT / config).is_file() and (paths.ROOT / main).is_file():
        try:
            run._require_config_fits_main(paths.ROOT / config, paths.ROOT / main)
        except SystemExit as error:
            problems.append(f"{path.name}: run.py would refuse this pairing: {error}")
    if output != f"output/{stem}_${{SLURM_JOB_ID}}":
        problems.append(f"{path.name}: OUTPUT is {output}, expected a fresh folder per job id")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines or lines[-1] != FINAL_COMMAND:
        problems.append(f"{path.name}: the last command is not {FINAL_COMMAND}")

    present = set(text.splitlines())
    for line in REQUIRED:
        if line not in present:
            problems.append(f"{path.name}: missing {line!r}")

    for needle, reason in FORBIDDEN.items():
        if needle in text:
            problems.append(f"{path.name}: {reason} ({needle!r})")

    if normalized(text, stem) != canonical:
        problems.append(
            f"{path.name}: differs from jobs/{CANONICAL}.job in more than the config, "
            "main, ntasks and time"
        )
    return problems


def check_log_folder() -> list[str]:
    """
    Check that jobs/logs/ exists in a fresh clone.

    Slurm does not create the folder for --output, and a job whose log folder is
    missing fails without writing any log at all.

    :returns: A list of problems; empty when the folder is tracked.
    """
    keep = JOBS_DIR / "logs" / ".gitkeep"
    if not keep.is_file():
        return ["jobs/logs/.gitkeep is missing, so a fresh clone has no log folder"]
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(keep)], cwd=paths.ROOT, capture_output=True
    )
    if ignored.returncode == 0:
        return ["jobs/logs/.gitkeep is git-ignored, so a fresh clone has no log folder"]
    return []


def main() -> None:
    """Run the checks and report."""
    run = load_run_module()
    canonical_path = JOBS_DIR / f"{CANONICAL}.job"
    if not canonical_path.is_file():
        print("JOBS CHECK FAILED")
        print(f"  - the canonical job {canonical_path.name} is missing")
        raise SystemExit(1)
    canonical = normalized(canonical_path.read_text(encoding="utf-8"), CANONICAL)

    jobs = {path.stem: path for path in sorted(JOBS_DIR.glob("*.job"))}
    configs = trainable_configs()
    problems = [f"config/{stem}.py has no job" for stem in configs if stem not in jobs]
    problems += [f"jobs/{stem}.job has no matching config" for stem in jobs if stem not in configs]
    for path in jobs.values():
        problems.extend(check_job(path, run, canonical))
    problems.extend(check_log_folder())

    if problems:
        print("JOBS CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("JOBS CHECK PASSED")
    print(f"  {len(jobs)} jobs, one per trainable config, all valid bash")
    print("  each starts a clean run.py -r into output/<config>_<job id> with the matching main")
    print("  each sets egl, single-threaded BLAS, the repo's own conda env and jobs/logs/")
    print("  none resumes, replays, sets PYTHONPATH, or points outside the repo")
    print(f"  all match jobs/{CANONICAL}.job apart from config, main, ntasks and time")
    print("  jobs/logs/ is tracked, so Slurm has somewhere to write logs")


if __name__ == "__main__":
    main()
