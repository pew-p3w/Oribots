#!/bin/bash
# Shared helper for the CHPC Lengau PBS jobs: run a command inside the Oribots
# Singularity container, with lustre bound and the training environment set.
#
# Lengau is CentOS 7.9 (glibc 2.17), which is too old for Oribots' pinned stack
# (mujoco 3.10 etc. need glibc >= 2.28). The stack therefore runs inside a
# Rocky-8 Singularity image (see hpc/build_sif.sh and docker/Dockerfile). The
# repository itself is a lustre checkout that is bind-mounted into the container,
# so a code sync never needs a new image.
#
# Sourced by the PBS job scripts, not run directly. The caller must have cd'd to
# the repository root first. Usage:
#
#     source hpc/pbs/_run_in_container.sh
#     run_in_container python run.py -r config/x.py src/cmaes/main.py output/x
#
# Environment knobs (override in the job or the submit environment):
#   ORIBOTS_SIF   path to the .sif image      (default: $HOME/lustre/oribots.sif)
#   SINGULARITY_MODULE  module to load        (default: chpc/singularity/3.5.3)

set -euo pipefail

: "${ORIBOTS_SIF:=$HOME/lustre/oribots.sif}"
: "${SINGULARITY_MODULE:=chpc/singularity/3.5.3}"
# The conda interpreter baked into the image (see docker/Dockerfile: make
# setup-conda builds it at this path). The checkout's own engine/ still wins on
# sys.path via src/paths.py, so this runs the checkout's code with the image's
# pinned dependencies.
: "${ORIBOTS_CONDA_PYTHON:=/scratch/oribots/Oribots/.conda/bin/python}"

run_in_container() {
    local repo_dir
    repo_dir="$(pwd)"

    if [ ! -f "$repo_dir/run.py" ] || [ ! -d "$repo_dir/engine" ]; then
        echo "Not in the repository root ($repo_dir has no run.py/engine)." >&2
        return 1
    fi
    if [ ! -f "$ORIBOTS_SIF" ]; then
        echo "Singularity image not found: $ORIBOTS_SIF" >&2
        echo "Build it off-cluster and copy it here (see hpc/build_sif.sh)." >&2
        return 1
    fi

    module purge 2>/dev/null || true
    module load "$SINGULARITY_MODULE"

    # Bind lustre so the checkout and its output/ are visible in the container.
    # The container runs the checkout's own code (src/paths.py puts this repo's
    # engine/ on sys.path), NOT any copy baked into the image.
    #
    # NCPUS is set by PBS and NUM_SIMULATORS reads it; egl + single-threaded BLAS
    # match the Slurm jobs. --cleanenv keeps the host environment out, and only
    # what training needs is passed through.
    # The Python environment (interpreter + pinned deps) lives IN the image; the
    # lustre checkout has no .conda. So invoke the image's conda interpreter,
    # against the bind-mounted checkout's code. A command starting with "python"
    # is rewritten to that interpreter; anything else runs as given.
    local -a command=("$@")
    if [ "${command[0]:-}" = "python" ]; then
        command[0]="$ORIBOTS_CONDA_PYTHON"
    fi

    # Variables go in as SINGULARITYENV_<NAME>, which --cleanenv still passes
    # through. (`singularity exec --env` only exists from Singularity 3.6; Lengau
    # has 3.5.3, which rejects it: "unknown flag: --env".)
    SINGULARITYENV_MUJOCO_GL=egl \
    SINGULARITYENV_OMP_NUM_THREADS=1 \
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1 \
    SINGULARITYENV_MKL_NUM_THREADS=1 \
    SINGULARITYENV_NUMEXPR_NUM_THREADS=1 \
    SINGULARITYENV_NCPUS="${NCPUS:-}" \
    SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 \
    singularity exec \
        --cleanenv \
        --bind /mnt/lustre \
        --pwd "$repo_dir" \
        "$ORIBOTS_SIF" \
        "${command[@]}"
}
