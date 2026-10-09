# Check suite for the multibody ball contest.
#
#   make -C multibody check        run every test script (the phase-gate suite)
#   make -C multibody check-fast   skip the slow single-robot isolation test
#   make -C multibody list         list the test scripts that would run
#
# Each test is a plain script in tests/ (NOT a package: tests/ has no
# __init__.py, like Oribots/tests/, so the name "tests" never shadows the
# single-robot one). A script prints "<NAME> CHECK PASSED" and exits 0 on
# success, or "<NAME> CHECK FAILED" with a reason and a non-zero exit on
# failure. New test scripts are picked up automatically by the glob below, so
# adding tests/test_<x>.py needs no edit here.
#
# The suite runs each script with its own 1800 s watchdog (D-12) and exits 0
# only when every script it ran exited 0.

# The interpreter every test runs under. Defaults to the repository's venv;
# override for the conda env or a worktree's venv, e.g.
#   make -C multibody check RUN_PYTHON=/path/to/.venv.nosync/bin/python
# The default is written relative to this Makefile's parent (the Oribots root),
# so `make -C multibody` from the repo root finds it.
RUN_PYTHON ?= ../.venv.nosync/bin/python

# Per-script wall-clock limit in seconds (D-12). A script still running after
# this is killed and counts as a failure.
CHECK_TIMEOUT ?= 1800

# The isolation test is slow (it runs the whole single-robot `make check`), so
# `check-fast` skips it for quick local iteration while `check` keeps it for the
# phase gates. The runner skips any script whose name is listed here when
# MULTIBODY_SKIP is set.
ISOLATION_TEST := test_isolation.py

.PHONY: check check-fast list

check:
	@MULTIBODY_SKIP="" \
	 RUN_PYTHON="$(RUN_PYTHON)" CHECK_TIMEOUT="$(CHECK_TIMEOUT)" \
	 bash run_checks.sh

check-fast:
	@MULTIBODY_SKIP="$(ISOLATION_TEST)" \
	 RUN_PYTHON="$(RUN_PYTHON)" CHECK_TIMEOUT="$(CHECK_TIMEOUT)" \
	 bash run_checks.sh

list:
	@ls tests/test_*.py 2>/dev/null || echo "no test scripts yet"
