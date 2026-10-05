#!/bin/bash
# The pre-HPC gate: prove the cluster workflow on Linux before any sbatch.
#
# Runs inside the image from docker/Dockerfile (start it with
# docker/run_gate.sh). It covers what macOS cannot: the Linux `fork` start
# method, MUJOCO_GL=egl, the conda env built by `make setup-conda`, and a real
# job script from jobs/ run with real conda activation. The visual modes (-t,
# --random-test) need a display and are not run here.

set -euo pipefail
cd "$(dirname "$0")/.."

# On the cluster the checkout persists between gates (hpc/gate.job), so clear
# the gate's own runs first; the overwrite guard would refuse them otherwise.
rm -rf output/_gate_4241 output/_gate_4242

PY=.conda/bin/python
export SLURM_NTASKS=4
export MUJOCO_GL=egl

step() { printf '\n=== %s\n' "$*"; }

step "Platform"
uname -m
# sed reads all of ldd's output; `head -1` could close the pipe early and
# pipefail would then stop the gate on ldd's SIGPIPE (exit 141).
ldd --version | sed -n 1p
grep -E '^PRETTY_NAME=' /etc/os-release
conda --version
"$PY" --version
echo "commit: $(git rev-parse --short HEAD)"

step "Multiprocessing uses fork, and the tiny config runs 4 simulators"
"$PY" - <<'EOF'
import importlib.util
import multiprocessing

method = multiprocessing.get_start_method()
assert method == "fork", f"start method is {method}, expected fork"

import sys
sys.path.insert(0, "src")
import paths
paths.install("config")
spec = importlib.util.spec_from_file_location("config", "config/_tiny.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
assert config.NUM_SIMULATORS == 4, f"NUM_SIMULATORS is {config.NUM_SIMULATORS}"
print(f"start method {method}, NUM_SIMULATORS {config.NUM_SIMULATORS}")
EOF

step "MuJoCo renders offscreen through EGL (Mesa on the CPU, not the cluster's driver)"
# Fatal everywhere: the jobs set MUJOCO_GL=egl, and with that set even
# `import mujoco` fails when EGL cannot load, so no training could run.
"$PY" - <<'EOF'
import mujoco

model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/>'
    '<geom type="sphere" size="0.2" rgba="1 0 0 1"/></worldbody></mujoco>'
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, 64, 64) as renderer:
    renderer.update_scene(data)
    image = renderer.render()
assert image.max() > 0, "EGL rendered a black frame"
print(f"rendered {image.shape}, max pixel {image.max()}")
EOF

step "make check"
make check RUN_PYTHON="$PY"

step "Real job scripts, with only the config swapped for the tiny fixtures"
# `module` is a shell function on the cluster; stub it. conda is real.
stub="$(mktemp -d)"
printf '#!/bin/bash\necho "[stub module] $*"\n' > "$stub/module"
chmod +x "$stub/module"
run_job() {
    local job="$1" config="$2" job_id="$3"
    sed -e "s#^CONFIG=config/[a-z_]*\.py#CONFIG=$config#" \
        -e "s#^OUTPUT=\"output/[a-z_]*_#OUTPUT=\"output/_gate_#" \
        "jobs/$job" > "$stub/$job"
    SLURM_SUBMIT_DIR="$PWD" SLURM_JOB_ID="$job_id" PATH="$stub:$PATH" \
        bash "$stub/$job"
    for file in gen1.pkl gen2.pkl; do
        test -s "output/_gate_$job_id/$file" || { echo "missing output/_gate_$job_id/$file"; exit 1; }
    done
}
run_job spider.job config/_tiny.py 4241
run_job simple_cmaes.job config/_tiny_cmaes.py 4242

step "-o and -c on the job outputs"
for run in output/_gate_4241 output/_gate_4242; do
    "$PY" run.py -o "$run"
    test -s "$run/generations.csv"
    "$PY" run.py -c "$run/gen2.pkl"
done

step "A second submit into the same folder is refused"
if SLURM_SUBMIT_DIR="$PWD" SLURM_JOB_ID=4241 PATH="$stub:$PATH" bash "$stub/spider.job" > "$stub/refused.log" 2>&1; then
    echo "the overwrite guard did not refuse"; exit 1
fi
grep -q "cannot be recovered" "$stub/refused.log"
echo "refused as expected"

printf '\nGATE PASSED at commit %s\n' "$(git rev-parse --short HEAD)"
