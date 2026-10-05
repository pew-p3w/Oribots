# One command per workflow. Run `make help` for the list.
#
#   Laptop:  make setup            (creates .venv.nosync with Python 3.11)
#   HPC:     make setup-conda         (creates .conda in the repo; the jobs in
#                                      jobs/ activate exactly that env)
#
# Both install the pins in requirements.txt, held to the full resolved set in
# constraints.txt so transitive packages match too, then the six vendored
# Revolve2 packages in engine/ in editable mode.

# The interpreter the venv is built from. On macOS the venv MUST come from a
# framework Python (Homebrew's python@3.11), or mjpython cannot dlopen
# libpython and the interactive viewer will not start. A uv/standalone
# python3.11 (often first on PATH) is not a framework build, so prefer
# Homebrew's when it is installed; otherwise fall back to whatever python3.11
# is on PATH (fine for headless use, e.g. the HPC uses conda instead).
PYTHON      ?= $(shell test -x /opt/homebrew/opt/python@3.11/bin/python3.11 \
	&& echo /opt/homebrew/opt/python@3.11/bin/python3.11 || echo python3.11)
# The venv is named .venv.nosync: iCloud Drive skips any path ending in
# .nosync, which stops it duplicating and hiding files inside the environment
# when the checkout lives under a synced folder such as ~/Documents.
VENV        ?= .venv.nosync
CONDA_ENV   ?= .conda
# The interpreter used by every target below. Override for the conda env:
#   make verify RUN_PYTHON=.conda/bin/python
RUN_PYTHON  ?= $(VENV)/bin/python
# Visual `make test` uses mjpython, not plain python: on macOS an interactive
# MuJoCo window must own the main thread, which only mjpython provides. run.py's
# 'auto' viewer then selects the native viewer because it detects mjpython.
VIEW_PYTHON ?= $(VENV)/bin/mjpython

ENGINE_PACKAGES = \
	engine/simulation \
	engine/modular_robot \
	engine/modular_robot_simulation \
	engine/experimentation \
	engine/ci_group \
	engine/simulators/mujoco_simulator

.PHONY: help setup setup-conda check verify check-bodies check-brain check-genotype check-evaluator check-head check-configs check-ea check-cmaes check-run check-workflow check-jobs check-pbs train test export

help:
	@echo "make setup         create .venv.nosync (Python 3.11) and install everything"
	@echo "make setup-conda   same, into a conda env (HPC): CONDA_ENV=<path>"
	@echo "make check         run all checks below"
	@echo "make verify        check the environment imports Revolve2 from engine/"
	@echo "make check-bodies  check every robot body matches the reference structure"
	@echo "make check-brain   check the brain reproduces the reference control values"
	@echo "make check-genotype   check mutation and crossover match the reference"
	@echo "make check-evaluator  check fitness scoring matches the reference"
	@echo "make check-head       check head mode and the pentagon spider match the reference values"
	@echo "make check-configs    check every config loads and agrees with the others"
	@echo "make check-ea         check the EA still makes the same choices"
	@echo "make check-cmaes      check the CMA-ES loop still makes the same choices"
	@echo "make check-run        check run.py refuses to overwrite an existing run"
	@echo "make check-workflow   train, export and resume both algorithms for real (about 20s)"
	@echo "make check-jobs       check the Slurm jobs start clean runs of real configs"
	@echo "make check-pbs        check the CHPC Lengau PBS scripts honour the cluster contract"
	@echo "make train         CONFIG=config/<body>.py [MAIN=src/ea/main.py] OUTPUT=output/<run>"
	@echo "make test          SNAPSHOT=output/<run>/gen<N>.pkl   (watch a saved robot)"
	@echo "make export        OUTPUT=output/<run>                (gen*.pkl -> generations.csv)"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -c constraints.txt -r requirements.txt
	$(VENV)/bin/python -m pip install --no-deps $(addprefix -e ,$(ENGINE_PACKAGES))

setup-conda:
	conda create -y -p $(CONDA_ENV) python=3.11 pip
	conda run -p $(CONDA_ENV) --no-capture-output python -m pip install -c constraints.txt -r requirements.txt
	conda run -p $(CONDA_ENV) --no-capture-output python -m pip install --no-deps $(addprefix -e ,$(ENGINE_PACKAGES))

check: verify check-bodies check-brain check-genotype check-evaluator check-head check-configs check-ea check-cmaes check-run check-workflow check-jobs check-pbs

verify:
	$(RUN_PYTHON) tests/test_environment.py

check-bodies:
	$(RUN_PYTHON) tests/test_bodies.py

check-brain:
	$(RUN_PYTHON) tests/test_brain.py

check-genotype:
	$(RUN_PYTHON) tests/test_genotype.py

check-evaluator:
	$(RUN_PYTHON) tests/test_evaluator.py

check-head:
	$(RUN_PYTHON) tests/test_head.py

check-configs:
	$(RUN_PYTHON) tests/test_configs.py

check-ea:
	$(RUN_PYTHON) tests/test_ea.py

check-cmaes:
	$(RUN_PYTHON) tests/test_cmaes.py

check-run:
	$(RUN_PYTHON) tests/test_run.py

check-workflow:
	$(RUN_PYTHON) tests/test_workflow.py

check-jobs:
	$(RUN_PYTHON) tests/test_jobs.py

check-pbs:
	$(RUN_PYTHON) tests/test_pbs.py

MAIN ?= src/ea/main.py

train:
	@test -n "$(CONFIG)" -a -n "$(OUTPUT)" || { echo "usage: make train CONFIG=config/<body>.py OUTPUT=output/<run> [MAIN=src/cmaes/main.py]"; exit 1; }
	$(RUN_PYTHON) run.py -r $(CONFIG) $(MAIN) $(OUTPUT)

test:
	@test -n "$(SNAPSHOT)" || { echo "usage: make test SNAPSHOT=output/<run>/gen<N>.pkl"; exit 1; }
	$(VIEW_PYTHON) run.py -t $(SNAPSHOT)

export:
	@test -n "$(OUTPUT)" || { echo "usage: make export OUTPUT=output/<run>"; exit 1; }
	$(RUN_PYTHON) run.py -o $(OUTPUT)
