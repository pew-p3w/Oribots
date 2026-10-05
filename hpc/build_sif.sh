#!/bin/bash
# Build the Oribots environment image and package it for CHPC Lengau.
#
#     bash hpc/build_sif.sh
#
# Set LENGAU_USER to a cluster username to have it filled into the printed scp
# command; otherwise the command shows <user>.
#
# WHY THIS EXISTS
# Lengau is CentOS 7.9 (glibc 2.17), too old for Oribots' pinned stack (mujoco
# 3.10 etc. need glibc >= 2.28), and it has no internet, so nothing can be
# pip- or docker-pulled there. This script therefore builds the same Rocky 8
# image the pre-HPC gate uses (docker/Dockerfile) and saves it as a tarball,
# which is converted to a Singularity .sif on Lengau, where Singularity 3.5.3 is
# available as a module.
#
# WHAT THIS SCRIPT DOES (on any machine with Docker, off-cluster)
#   1. Builds docker/Dockerfile for linux/amd64 on the Rocky 8 base, from a
#      fresh clone of what is committed (never the working tree, so no macOS
#      venv or iCloud-evicted files leak in).
#   2. `docker save`s the image to hpc/dist/oribots-<commit>.tar.
#   3. Prints the copy and on-cluster conversion commands, with real values.
#      It never connects to the cluster; those steps are run by hand.
#
# WHAT TO RUN AFTERWARDS (the script prints these with real values)
#   scp hpc/dist/oribots-<commit>.tar <user>@lengau.chpc.ac.za:~/lustre/
#   # then on a Lengau login node:
#   module load chpc/singularity/3.5.3
#   cd ~/lustre
#   singularity build oribots.sif docker-archive://oribots-<commit>.tar
#
# The image carries the Python environment (interpreter + pinned deps) in a
# conda env at IMAGE:/scratch/oribots/Oribots/.conda. The CODE is run from a
# bind-mounted lustre checkout at a SEPARATE path: src/paths.py prepends the
# checkout's engine/ roots to sys.path, so the checkout's own code loads. A code
# sync therefore needs no new image; rebuild only when the pinned dependencies
# change.
#
# ENGINE PATH: docker/Dockerfile deletes the engine's editable-install
# revolve2_*.pth files from the image's conda env; src/paths.py already puts the
# engine roots on sys.path, so nothing needs them. A bind-mounted checkout is
# therefore the ONLY engine on the path, and tests/test_environment.py (strict:
# every `revolve2` namespace entry must live under the running checkout's
# engine/) passes inside the container too, with the checkout bind-mounted at
# /mnt/lustre/users/<user>/Oribots and run by the image's
# /scratch/oribots/Oribots/.conda/bin/python.

set -euo pipefail

# Rocky 8, not 9: Lengau's CentOS 7 host kernel is 3.10, and a Rocky 8 userland
# is the safer match. Override if a test on Lengau shows otherwise.
BASE_IMAGE="${BASE_IMAGE:-rockylinux:8}"
# Only used to fill in the printed scp command.
LENGAU_USER="${LENGAU_USER:-<user>}"

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found; run this on a machine with Docker." >&2
    exit 1
fi
if ! docker info >/dev/null 2>&1; then
    echo "The Docker daemon is not running; start Docker Desktop first." >&2
    exit 1
fi

# Build from a fresh clone of HEAD, exactly as docker/run_gate.sh does, so the
# image is reproducible from what is committed and never sees the working tree.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git clone -q "$root" "$work/Oribots"
commit="$(git -C "$work/Oribots" rev-parse --short HEAD)"
if [ -n "$(git -C "$root" status --porcelain)" ]; then
    echo "Note: uncommitted changes are NOT in this image; it is built from commit $commit."
fi

tag="oribots-env:$commit"
echo "Building $tag from $BASE_IMAGE (linux/amd64)..."
docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f "$work/Oribots/docker/Dockerfile" -t "$tag" "$work/Oribots"

dist="$root/hpc/dist"
mkdir -p "$dist"
tarball="$dist/oribots-$commit.tar"
echo "Saving image to $tarball..."
docker save -o "$tarball" "$tag"

size="$(du -h "$tarball" | cut -f1)"
cat <<INSTRUCTIONS

Built and saved: $tarball  ($size)

Next, copy it to Lengau and convert it to a .sif there:

  scp "$tarball" ${LENGAU_USER}@lengau.chpc.ac.za:~/lustre/

  # then on the Lengau login node:
  module load chpc/singularity/3.5.3
  cd ~/lustre
  singularity build oribots.sif docker-archive://oribots-$commit.tar

If the unprivileged build or mksquashfs is disallowed on the login node, build
the .sif off-cluster instead (Apptainer in a Linux VM/container) and scp the
.sif over -- but first confirm a modern .sif opens in Singularity 3.5.3 (2019):

  singularity inspect oribots.sif
  singularity exec oribots.sif /opt/miniconda/bin/python --version

The PBS jobs expect the .sif at \$HOME/lustre/oribots.sif (override with
ORIBOTS_SIF). Prove the environment through the container before any real run.
The Python env lives IN the image (the lustre checkout has no .conda), so use
the image's interpreter against the bind-mounted checkout:

  cd ~/lustre/Oribots            # your lustre checkout of this repo
  module load chpc/singularity/3.5.3
  singularity exec --cleanenv --bind /mnt/lustre --pwd "\$PWD" \\
      ~/lustre/oribots.sif \\
      /scratch/oribots/Oribots/.conda/bin/python -c "import mujoco; print(mujoco.__version__)"

That confirms the pinned stack imports under Lengau's kernel. The strict
environment check also passes inside the container (see ENGINE PATH in the
header of hpc/build_sif.sh):

  singularity exec --cleanenv --bind /mnt/lustre --pwd "\$PWD" \\
      ~/lustre/oribots.sif \\
      /scratch/oribots/Oribots/.conda/bin/python tests/test_environment.py
INSTRUCTIONS
