#!/bin/bash
# Run the pre-HPC gate in Docker, on exactly what is committed.
#
#     docker/run_gate.sh
#
# Clones the repository's HEAD into a temporary folder (so uncommitted edits,
# the macOS venv and iCloud never reach the image), builds docker/Dockerfile
# from that clone for linux/amd64, and runs docker/gate.sh in it. Needs Docker
# Desktop running; on Apple silicon, enable its Rosetta setting for amd64
# emulation, since MuJoCo under QEMU is very slow.
#
# Override the base to mirror the cluster, for example:
#     BASE_IMAGE=rockylinux:9 docker/run_gate.sh

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

git clone -q "$root" "$work/Oribots"
commit="$(git -C "$work/Oribots" rev-parse --short HEAD)"
if [ -n "$(git -C "$root" status --porcelain)" ]; then
    echo "Note: uncommitted changes are NOT part of this gate; it tests commit $commit."
fi

build_args=()
[ -n "${BASE_IMAGE:-}" ] && build_args+=(--build-arg "BASE_IMAGE=$BASE_IMAGE")
[ -n "${MINICONDA_VERSION:-}" ] && build_args+=(--build-arg "MINICONDA_VERSION=$MINICONDA_VERSION")

docker build --platform linux/amd64 "${build_args[@]+"${build_args[@]}"}" \
    -f "$work/Oribots/docker/Dockerfile" -t "oribots-gate:$commit" "$work/Oribots"
docker run --rm --platform linux/amd64 "oribots-gate:$commit"
