"""The single-robot setup is untouched (R1.2, R1.3, R1.5, R1.7, R26.5). A standing check for every phase gate.

1. ``make check`` in ``Oribots/`` passes: every single-robot sub-check prints its
   ``CHECK PASSED`` line and the command exits 0.
2. Compared with the baseline commit (``4ece000``, where multibody work began), no
   tracked file outside ``multibody/`` was added, modified, deleted or renamed, and
   no new file git does not ignore appeared outside it, with two exceptions only:
   ``Makefile`` may gain exactly one ``check-multibody`` target that runs
   ``make -C multibody check``, and ``README.md`` may gain exactly one line
   mentioning ``multibody/``.

Skipped by ``make -C multibody check-fast`` (it is slow).
"""

import os
import re
import subprocess
import sys

from _check import check_main, require

import contest_paths

ROOT = contest_paths.ROOT
BASELINE = "4ece000"
SUB_CHECKS = (
    "verify", "check-bodies", "check-brain", "check-genotype", "check-evaluator", "check-head",
    "check-configs", "check-ea", "check-cmaes", "check-run", "check-workflow", "check-jobs", "check-pbs",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=True).stdout


def added_and_removed(path: str) -> tuple[list[str], list[str]]:
    diff = git("diff", BASELINE, "--", path)
    added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:] for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")]
    return added, removed


def check_root_edits() -> list[str]:
    problems = []
    changed = [line.split("\t") for line in git("diff", "--name-status", BASELINE, "--", ".", ":(exclude)multibody").splitlines()]
    for entry in changed:
        status, path = entry[0], entry[-1]
        if path not in ("Makefile", "README.md") or not status.startswith("M"):
            problems.append(f"{status} {path}")
    untracked = [
        line[3:] for line in git("status", "--porcelain", "--untracked-files=all").splitlines()
        if line.startswith("??") and not line[3:].startswith("multibody/")
    ]
    problems.extend(f"untracked {path}" for path in untracked)

    added, removed = added_and_removed("Makefile")
    if added or removed:
        body = [line for line in added if line.strip()]
        target_ok = (
            not removed
            and any(re.match(r"^check-multibody\s*:", line) for line in body)
            and all(re.match(r"^(check-multibody\s*:.*|\t\$\(MAKE\) -C multibody check|\tmake -C multibody check|\.PHONY:.*check-multibody.*)$", line) for line in body)
        )
        if not target_ok:
            problems.append(f"Makefile changes are not exactly one check-multibody target: +{body} -{removed}")
    added, removed = added_and_removed("README.md")
    if added or removed:
        if removed or len(added) != 1 or "multibody/" not in added[0]:
            problems.append(f"README.md changes are not exactly one multibody/ line: +{added} -{removed}")
    return problems


def run() -> None:
    # A clean environment: the multibody harness exports RUN_PYTHON relative to
    # multibody/ (and make passes its own flags down); the single-robot suite gets
    # this test's interpreter as an absolute path instead.
    environment = {k: v for k, v in os.environ.items() if k not in ("RUN_PYTHON", "MAKEFLAGS", "MFLAGS", "MAKELEVEL", "CHECK_TIMEOUT", "MULTIBODY_SKIP")}
    interpreter = os.path.abspath(sys.executable)
    result = subprocess.run(["make", "check", f"RUN_PYTHON={interpreter}"], cwd=str(ROOT), capture_output=True, text=True, env=environment)
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"single-robot `make check` failed (exit {result.returncode}):\n{output[-3000:]}")
    passed = len(re.findall(r"CHECK PASSED", output))
    require(passed >= len(SUB_CHECKS), f"only {passed} 'CHECK PASSED' lines for {len(SUB_CHECKS)} sub-checks:\n{output[-3000:]}")
    problems = check_root_edits()
    require(not problems, "changes outside multibody/ since " + BASELINE + ":\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    check_main("ISOLATION", run)
