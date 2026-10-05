# A Linux machine shaped like the HPC, for the pre-HPC gate (docker/gate.sh).
#
# Build and run it with docker/run_gate.sh, which builds from a fresh clone of
# what is committed, so the image never sees a macOS venv or uncommitted work.
#
# The base is a RHEL 8 family image because the locked mujoco, scipy, pillow
# and dm-tree wheels need glibc 2.28 or newer on x86_64; an image with a newer
# glibc than the cluster's would pass here and still fail there. Set
# BASE_IMAGE and MINICONDA_VERSION to match the target cluster's OS and conda
# version.

ARG BASE_IMAGE=rockylinux:8
FROM --platform=linux/amd64 ${BASE_IMAGE}

# A C compiler is needed at setup time: noise has no Linux wheel. Mesa provides
# EGL on the CPU, standing in for the cluster's MUJOCO_GL=egl. git lets the
# job check confirm jobs/logs/ is tracked. --allowerasing lets Rocky 9 swap its
# curl-minimal for curl.
RUN dnf install -y --allowerasing --setopt=install_weak_deps=False \
        gcc gcc-c++ make git which curl \
        mesa-libEGL mesa-libGL mesa-dri-drivers libglvnd-egl libglvnd-opengl \
    && dnf clean all

# Stands in for `module load python/miniconda3-py3.12`: conda on PATH, no env
# active.
ARG MINICONDA_VERSION=py312_24.7.1-0
RUN curl -fsSL -o /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/miniconda/bin:$PATH

WORKDIR /scratch/oribots/Oribots
COPY . .
RUN make setup-conda
# Oribots never relies on the engine's editable-install .pth files: src/paths.py
# puts the engine roots on sys.path itself. Removing them means a checkout
# bind-mounted at another path (Lengau, hpc/pbs/) is the ONLY engine on the
# path, instead of this image's copy trailing behind it, so the strict namespace
# check in tests/test_environment.py holds in the container too.
RUN rm -f .conda/lib/python3.11/site-packages/revolve2_*.pth \
    && ! ls .conda/lib/python3.11/site-packages/revolve2_*.pth 2>/dev/null

CMD ["bash", "docker/gate.sh"]
