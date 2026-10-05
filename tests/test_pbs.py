"""Check the CHPC Lengau PBS scripts in hpc/pbs/ before anything is submitted.

Lengau time is charged to a project allocation, and a job that is wrong in a way
qsub still accepts (wrong project, a log path the scheduler cannot write, a stray
reference outside the repository) only fails after it has queued and started. So every
PBS script is checked here the way jobs/ is checked for Slurm.

These are *templates*, not one-per-config jobs: train.pbs and resume.pbs take
their config/output through `qsub -v`, so unlike the Slurm check this does NOT
require a referenced config to exist. What it does require is the cluster
contract every Lengau job must honour: the right project and queue, a real
resource request, log paths on lustre, the shared container helper, and nothing
pointing outside the repository or setting PYTHONPATH by hand.

The shared helper hpc/pbs/_run_in_container.sh is where the training environment
(egl, single-threaded BLAS, NCPUS pass-through) actually lives, so that contract
is checked once, there, rather than in every job.

Run with `make check-pbs`, or `python tests/test_pbs.py`.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

PBS_DIR = paths.ROOT / "hpc" / "pbs"
HELPER = "_run_in_container.sh"
# The allocation these jobs are charged to, and where lustre is mounted. The
# shipped scripts carry the placeholder below; set PROJECT here to your project
# code when you put it in the #PBS -P lines of hpc/pbs/*.pbs, so the check then
# guards the real value.
PROJECT = "<PROJECT>"
LUSTRE = "/mnt/lustre"

# Anything that points outside the repository or builds its own import path
# instead of leaving that to run.py / src/paths.py. Unlike the Slurm jobs, a
# PBS script MAY use `run.py -c` (resume.pbs and the gate resume on purpose) and
# MAY mention /scratch (the conda interpreter baked into the image lives there),
# so those are not forbidden here.
FORBIDDEN = {
    "revolve2-folding": "refers to a directory outside this repository",
    "PYTHONPATH": "sets PYTHONPATH itself; run.py builds it",
}

# The cluster contract every Lengau PBS job must honour. Checked as substrings
# so a job may add its own comment or spacing, but cannot drop one of these.
REQUIRED = (
    "set -euo pipefail",
    f"#PBS -P {PROJECT}",
    "source hpc/pbs/_run_in_container.sh",
)

# The environment training needs, checked once in the shared helper the jobs
# source rather than in every job.
HELPER_REQUIRED = (
    "set -euo pipefail",
    "MUJOCO_GL=egl",
    "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
    "NCPUS=",
    "--bind /mnt/lustre",
)


def pbs_directive(text: str, flag: str) -> str | None:
    """
    The value of a `#PBS -flag value` directive.

    :param text: The script.
    :param flag: The single-letter flag, e.g. "q" or "N".
    :returns: The value, or None if the script does not set it.
    """
    match = re.search(rf"^#PBS -{flag}\s+(\S.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def bash_ok(path: Path) -> str | None:
    """
    Parse a shell script with `bash -n`.

    :param path: The script.
    :returns: The parse error, or None when it parses.
    """
    parsed = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    return None if parsed.returncode == 0 else parsed.stderr.strip()


def check_pbs(path: Path) -> list[str]:
    """
    Check one PBS job script.

    :param path: The .pbs file.
    :returns: A list of problems; empty when the job is fine.
    """
    problems = []
    text = path.read_text(encoding="utf-8")

    error = bash_ok(path)
    if error:
        problems.append(f"{path.name}: bash -n failed: {error}")

    for line in REQUIRED:
        if line not in text:
            problems.append(f"{path.name}: missing {line!r}")

    if pbs_directive(text, "N") is None:
        problems.append(f"{path.name}: #PBS -N (job name) is missing")

    project = pbs_directive(text, "P")
    if project != PROJECT:
        problems.append(f"{path.name}: #PBS -P is {project!r}, expected {PROJECT!r}")

    if pbs_directive(text, "q") is None:
        problems.append(f"{path.name}: #PBS -q (queue) is missing")

    # -l appears more than once (select and walltime), so scan all of them.
    resources = re.findall(r"^#PBS -l\s+(\S.*)$", text, re.MULTILINE)
    joined = " ".join(resources)
    if not re.search(r"select=\d+:ncpus=\d+", joined):
        problems.append(f"{path.name}: #PBS -l select=<n>:ncpus=<n> is missing or malformed")
    if not re.search(r"walltime=\d", joined):
        problems.append(f"{path.name}: #PBS -l walltime=<time> is missing")

    for flag in ("o", "e"):
        value = pbs_directive(text, flag)
        if value is None:
            problems.append(f"{path.name}: #PBS -{flag} (log path) is missing")
        elif not value.startswith(LUSTRE):
            problems.append(
                f"{path.name}: #PBS -{flag} is {value!r}, must be on {LUSTRE} "
                "(the scheduler cannot write elsewhere)"
            )

    for needle, reason in FORBIDDEN.items():
        if needle in text:
            problems.append(f"{path.name}: {reason} ({needle!r})")

    return problems


def check_helper() -> list[str]:
    """
    Check the shared container helper the jobs source.

    :returns: A list of problems; empty when the helper is fine.
    """
    path = PBS_DIR / HELPER
    if not path.is_file():
        return [f"hpc/pbs/{HELPER} is missing; every PBS job sources it"]

    problems = []
    error = bash_ok(path)
    if error:
        problems.append(f"{HELPER}: bash -n failed: {error}")

    text = path.read_text(encoding="utf-8")
    for line in HELPER_REQUIRED:
        if line not in text:
            problems.append(f"{HELPER}: missing {line!r}")

    for needle, reason in FORBIDDEN.items():
        if needle in text:
            problems.append(f"{HELPER}: {reason} ({needle!r})")

    return problems


def main() -> None:
    """Run the checks and report."""
    if not PBS_DIR.is_dir():
        print("PBS CHECK FAILED")
        print(f"  - {PBS_DIR} does not exist")
        raise SystemExit(1)

    jobs = sorted(PBS_DIR.glob("*.pbs"))
    if not jobs:
        print("PBS CHECK FAILED")
        print(f"  - no .pbs scripts in {PBS_DIR}")
        raise SystemExit(1)

    problems = []
    for path in jobs:
        problems.extend(check_pbs(path))
    problems.extend(check_helper())

    if problems:
        print("PBS CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        raise SystemExit(1)

    print("PBS CHECK PASSED")
    print(f"  {len(jobs)} PBS scripts, all valid bash, all on project {PROJECT}")
    print(f"  each names a queue, requests select/ncpus and a walltime, logs on {LUSTRE}")
    print(f"  each sources hpc/pbs/{HELPER}, which sets egl, single-threaded BLAS and NCPUS")
    print("  none refers outside the repository or sets PYTHONPATH itself")


if __name__ == "__main__":
    main()
