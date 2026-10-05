"""Oribots command-line entrypoint.

One script drives the whole workflow. Five modes, exactly one per invocation:

    python run.py -r CONFIG MAIN OUTPUT      train
    python run.py -t GEN_PKL [--viewer ...]  watch a saved controller
    python run.py -o OUTPUT_FOLDER           export gen*.pkl to one CSV
    python run.py -c GEN_PKL                 resume a run from a snapshot
    python run.py --random-test CONFIG       preview a body with a random brain

Training (`-r`, `-c`) runs in a child process so the simulator's own process
group can be cleaned up on interruption without taking this launcher down with
it. The child is a fresh interpreter, so its import path and bytecode policy are
passed through the environment by `paths.child_environment()`; do not rebuild
that here.

The in-process modes (`-t`, `--random-test`, and reading snapshots for `-o`/`-c`)
put the repository's directories on `sys.path` with `paths.install()`. Neither
depends on the `.pth` files `pip install -e` writes, which macOS/iCloud can
silently disable.

Snapshots created on a cluster store absolute `/scratch/...` paths, and
snapshots written by earlier versions of the code reference the body catalogue
under its former module name inside the engine namespace
(`revolve2.ci_group.modular_robots_v2`). `-t`/`-c` remap those paths to the
local checkout and alias that module name to `bodies`, so such snapshots still
load. `-c` dispatches on the snapshot's explicit `algorithm` field rather than
guessing from which keys are present.
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/ holds the path helper (src/paths.py) but is not importable until it is on
# sys.path, so add it first and then import paths.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import paths  # noqa: E402

# Avoid stale/offloaded .pyc in this launcher; the child gets the same via
# paths.child_environment(). Set before importing anything heavy.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
from typing import Any  # noqa: E402

CONFIG_PATH_ENV = "REVOLVE2_RUN_CONFIG_PATH"
MAIN_PATH_ENV = "REVOLVE2_RUN_MAIN_PATH"
OUTPUT_DIR_ENV = "REVOLVE2_RUN_OUTPUT_DIR"

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EA_MAIN = REPO_ROOT / "src" / "ea" / "main.py"

# What each algorithm needs its config to define. A config written for one
# algorithm lacks the other's search settings, so pairing them fails deep inside
# the training process with an AttributeError; checking here says so up front.
ALGORITHM_CONFIG_NAMES = {
    "ea": ("POPULATION_SIZE", "TOURNAMENT_SIZE", "MUTATE_STD", "MUTATION_PROBABILITY"),
    "cmaes": ("CMA_INITIAL_STD", "CMA_INITIAL_MEAN", "CMA_BOUNDS", "CMA_POPULATION_SIZE"),
}
ALGORITHM_NAMES = {"ea": "the evolutionary algorithm", "cmaes": "CMA-ES"}


def main() -> None:
    """Parse arguments and run the one selected workflow mode."""
    args = _parse_args()
    selected = [
        args.run is not None,
        args.test is not None,
        args.output_csv is not None,
        args.continue_run is not None,
        args.random_test is not None,
    ]
    if sum(bool(flag) for flag in selected) != 1:
        raise SystemExit("Choose exactly one of -r, -t, -o, -c, or --random-test.")

    if args.run is not None:
        config_path = _existing_file(args.run[0])
        main_path = _existing_file(args.run[1])
        output_dir = Path(args.run[2]).resolve()
        _require_config_fits_main(config_path, main_path)
        _require_fresh_output_dir(output_dir)
        _run_training(config_path=config_path, main_path=main_path, output_dir=output_dir)
    elif args.random_test is not None:
        _run_random_test(
            config_path=_existing_file(args.random_test),
            viewer_type=args.viewer,
            simulation_time=(
                args.sim_seconds
                if args.sim_seconds is not None
                else DEFAULT_RANDOM_TEST_SIM_SECONDS
            ),
        )
    elif args.test is not None:
        _run_test(
            snapshot_path=_existing_file(args.test),
            viewer_type=args.viewer,
            simulation_time=args.sim_seconds,  # None -> use the config's SIMULATION_TIME
        )
    elif args.output_csv is not None:
        _export_csv(_existing_dir(args.output_csv))
    else:
        _continue_training(_existing_file(args.continue_run))


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :returns: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train, test, or export Oribots runs.")
    parser.add_argument(
        "-r", "--run", nargs=3, metavar=("CONFIG", "MAIN", "OUTPUT"),
        help="Train with CONFIG file, MAIN file (e.g. src/ea/main.py), and OUTPUT folder.",
    )
    parser.add_argument(
        "-t", "--test", metavar="GEN_PKL",
        help="Visually test one generation snapshot, e.g. output/run/gen12.pkl.",
    )
    parser.add_argument(
        "-o", "--output-csv", metavar="OUTPUT_FOLDER",
        help="Export all gen*.pkl snapshots in a folder to generations.csv.",
    )
    parser.add_argument(
        "-c", "--continue-run", metavar="GEN_PKL",
        help="Continue training from a resumable generation snapshot.",
    )
    parser.add_argument(
        "--random-test", metavar="CONFIG",
        help="Preview a config's body with one random controller.",
    )
    parser.add_argument(
        "--viewer", choices=("auto", "native", "custom"), default="auto",
        help=(
            "Viewer for visual modes. Default 'auto' pairs the viewer to the "
            "launcher: 'native' under mjpython, 'custom' under plain python. On "
            "macOS an interactive window must own the main thread, which only "
            "mjpython provides, so use `mjpython run.py -t ...` to watch a robot."
        ),
    )
    parser.add_argument(
        "--sim-seconds",
        "--random-test-time",  # alias of --sim-seconds, kept for compatibility
        dest="sim_seconds",
        type=float,
        default=None,
        help=(
            "SIMULATED seconds to run a visual mode for. The viewer renders far "
            "faster than real time (~34x on an M-series Mac), so simulated "
            "seconds are not wall-clock seconds. Defaults to 1000 for "
            "--random-test (~30 s of watchable window); for -t it defaults to the "
            "config's SIMULATION_TIME and this flag overrides it when given."
        ),
    )
    return parser.parse_args()


# Simulated seconds for --random-test when --sim-seconds is not given.
# The custom viewer renders ~34x faster than real time, so ~1000 simulated
# seconds is roughly 30 wall-clock seconds of watchable window.
DEFAULT_RANDOM_TEST_SIM_SECONDS = 1000.0


# --------------------------------------------------------------------------- #
# Training (child process)
# --------------------------------------------------------------------------- #


def _require_config_fits_main(config_path: Path, main_path: Path) -> None:
    """
    Refuse a config and training main written for different algorithms.

    Each main declares its `ALGORITHM`, and each algorithm needs its own search
    settings in the config. Both files are read as text, so nothing is imported
    or started. A main that declares no algorithm is not checked.

    :param config_path: The config file.
    :param main_path: The training main.
    :raises SystemExit: If the config lacks settings the main needs.
    """
    import ast

    def top_level_assignments(path: Path) -> dict[str, Any]:
        """Module-level `NAME = <literal>` assignments in a file."""
        found: dict[str, Any] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    found[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    found[node.targets[0].id] = None
        return found

    algorithm = top_level_assignments(main_path).get("ALGORITHM")
    if algorithm not in ALGORITHM_CONFIG_NAMES:
        return

    defined = set(top_level_assignments(config_path))
    missing = [name for name in ALGORITHM_CONFIG_NAMES[algorithm] if name not in defined]
    if not missing:
        return

    other = next(a for a in ALGORITHM_CONFIG_NAMES if a != algorithm)
    looks_like_other = all(name in defined for name in ALGORITHM_CONFIG_NAMES[other])
    hint = (
        f"{config_path.name} looks like a config for {ALGORITHM_NAMES[other]}. Pair it with "
        f"src/{other}/main.py, or use a config written for {ALGORITHM_NAMES[algorithm]}."
        if looks_like_other
        else f"Use a config written for {ALGORITHM_NAMES[algorithm]}."
    )
    raise SystemExit(
        f"{main_path.parent.name}/{main_path.name} runs {ALGORITHM_NAMES[algorithm]}, but "
        f"{config_path.name} does not define {', '.join(missing)}.\n{hint}"
    )


def _require_fresh_output_dir(output_dir: Path) -> None:
    """
    Refuse to start a new run in a folder that already holds one.

    A run writes `gen1.pkl`, `gen2.pkl` and so on, so starting a second run in
    the same folder overwrites the first one's snapshots one by one. Those
    snapshots are the only record of what a run produced and the only way to
    replay a controller afterwards; a cluster run that produced them can take
    days. The run CSV would survive under a fresh timestamped name, which makes
    the loss easy to miss.

    Deleting is left to the person, deliberately: this refuses and explains
    rather than offering a flag that would make the accident a keystroke away.
    Resuming (`-c`) is meant to write into its own folder and does not come
    through here.

    :param output_dir: The folder a new run was asked to write to.
    :raises SystemExit: If the folder already contains generation snapshots or
        per-individual recording files.
    """
    def order(path: Path) -> tuple[int, str]:
        """Sort by generation number, tolerating names like gen100_extend200.pkl."""
        leading = path.stem.removeprefix("gen").split("_")[0]
        return (int(leading) if leading.isdigit() else -1, path.name)

    snapshots = sorted(output_dir.glob("gen*.pkl"), key=order)
    if not snapshots:
        # A run killed before its first snapshot has still recorded individuals;
        # a new run would truncate weights.csv and mix two runs' rows.
        recordings = sorted(output_dir.glob("individual_performance_*.csv")) + sorted(
            output_dir.glob("weights.csv")
        )
        if recordings:
            raise SystemExit(
                f"{output_dir} already holds a run's per-individual recording "
                f"({', '.join(path.name for path in recordings)}).\n"
                f"Starting a new run here would overwrite it.\n"
                f"Use a new output folder, or move this one aside first."
            )
        return
    latest = snapshots[-1]
    raise SystemExit(
        f"{output_dir} already holds {len(snapshots)} generation snapshots "
        f"({snapshots[0].name} to {latest.name}).\n"
        f"Starting a new run here would overwrite them, and they cannot be recovered.\n"
        f"Use a new output folder, or move this one aside first.\n"
        f"To carry on with that run instead (snapshots with an 'algorithm' field):\n"
        f"    python run.py -c {latest}"
    )


def _run_training(
    config_path: Path,
    main_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
) -> None:
    """
    Launch a training run in a child process.

    :param config_path: Config file path.
    :param main_path: Training main.py path.
    :param output_dir: Output folder for run artifacts.
    :param checkpoint_path: Optional checkpoint/snapshot to resume from.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    env = paths.child_environment(config_dir=config_path.parent)
    env[CONFIG_PATH_ENV] = str(config_path)
    env[MAIN_PATH_ENV] = str(main_path)
    env[OUTPUT_DIR_ENV] = str(output_dir)
    command = [sys.executable, str(main_path)]
    if checkpoint_path is not None:
        command.append(str(checkpoint_path))
    _run_child_process(command, env)


def _run_child_process(command: list[str], env: dict[str, str]) -> None:
    """
    Run the training process and clean up its process group on interruption.

    :param command: Command to execute.
    :param env: Environment for the child process.
    :raises subprocess.CalledProcessError: If the child exits with an error.
    """
    process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env, start_new_session=True)
    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_child(signum: int, _frame: Any) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("Stopping training process group...", file=sys.stderr, flush=True)
        _stop_process_group(process)
        raise SystemExit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, stop_child)
    signal.signal(signal.SIGTERM, stop_child)
    try:
        return_code = process.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if process.poll() is None:
            _stop_process_group(process)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    """
    Stop a child process group, escalating to SIGKILL if it does not exit.

    :param process: Child process.
    """
    if process.poll() is not None:
        return
    descendants = _descendant_pids(process.pid)
    _signal_pids(descendants, signal.SIGTERM)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if _wait_for_exit(process, timeout=3):
        return
    stubborn = sorted(set(descendants + _descendant_pids(process.pid)))
    _signal_pids(stubborn, signal.SIGKILL)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _wait_for_exit(process, timeout=3)


def _wait_for_exit(process: subprocess.Popen[Any], timeout: float) -> bool:
    """
    Wait briefly for a process to exit.

    :param process: Process to wait for.
    :param timeout: Maximum seconds to wait.
    :returns: True if the process exited.
    """
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return True


def _signal_pids(pids: list[int], signum: int) -> None:
    """
    Send a signal to specific process IDs, ignoring ones that are already gone.

    :param pids: Process IDs.
    :param signum: Signal to send.
    """
    for pid in pids:
        try:
            os.kill(pid, signum)
        except (ProcessLookupError, PermissionError):
            continue


def _descendant_pids(root_pid: int) -> list[int]:
    """
    Find descendant process IDs of a process using ps.

    :param root_pid: Root process ID.
    :returns: Descendant process IDs.
    """
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    children_by_parent: dict[int, list[int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        pid, parent_pid = int(parts[0]), int(parts[1])
        children_by_parent.setdefault(parent_pid, []).append(pid)
    descendants: list[int] = []
    stack = list(children_by_parent.get(root_pid, []))
    while stack:
        pid = stack.pop()
        descendants.append(pid)
        stack.extend(children_by_parent.get(pid, []))
    return descendants


# --------------------------------------------------------------------------- #
# Random-test (in process)
# --------------------------------------------------------------------------- #


def _run_random_test(config_path: Path, viewer_type: str, simulation_time: float) -> None:
    """
    Preview a config's body with one random controller.

    :param config_path: Config file path.
    :param viewer_type: Requested MuJoCo viewer.
    :param simulation_time: Maximum simulation seconds.
    """
    import numpy as np

    viewer_type = _resolve_viewer(viewer_type)
    config = _load_config_in_process(config_path)

    from ball_aware_brain import steering_parameter_count
    from revolve2.experimentation.rng import make_rng_time_seed
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )

    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
        active_hinges
    )
    num_parameters = cpg_network_structure.num_connections + steering_parameter_count(
        output_mapping=output_mapping,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
    )
    rng = make_rng_time_seed()
    weights = rng.random(size=num_parameters) * 2.0 - 1.0

    print(f"Random test config: {config_path}", flush=True)
    print(f"Body: {config.TEST_FILE}   active hinges: {len(active_hinges)}   "
          f"parameters: {len(weights)}", flush=True)

    _simulate_and_report(
        config=config,
        weights=weights,
        cpg_network_structure=cpg_network_structure,
        output_mapping=output_mapping,
        viewer_type=viewer_type,
        simulation_time=simulation_time,
        label="random controller",
    )


def _simulate_and_report(
    config: Any,
    weights: Any,
    cpg_network_structure: Any,
    output_mapping: list[Any],
    viewer_type: str,
    simulation_time: float,
    label: str,
) -> None:
    """
    Build one robot, simulate it toward a ball, and log distance and fitness.

    This is the single visual simulate-and-report path shared by `--random-test`
    (random weights) and `-t` (snapshot weights). The only difference between
    the callers is where `weights` comes from, so that is the only parameter
    that varies.

    :param config: The loaded config module.
    :param weights: Controller parameter vector to run.
    :param cpg_network_structure: CPG network structure for the body.
    :param output_mapping: CPG-output-to-hinge mapping for the body.
    :param viewer_type: Resolved MuJoCo viewer.
    :param simulation_time: Maximum simulation seconds.
    :param label: Human-readable description of what is being run.
    """
    import logging
    import math

    from ball_aware_brain import BallAwareCpgBrain
    from revolve2.ci_group.interactive_objects import Ball
    from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters
    from revolve2.experimentation.logging import setup_logging
    from revolve2.experimentation.rng import make_rng_time_seed
    from revolve2.modular_robot import ModularRobot
    from revolve2.modular_robot_simulation import (
        ModularRobotScene,
        StopOnRobotObjectDistance,
        simulate_scenes,
    )

    setup_logging()
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco  # noqa: F401  (import validates the runtime before simulating)

    from revolve2.simulators.mujoco_simulator import LocalSimulator

    rng = make_rng_time_seed()
    ball = Ball(
        radius=config.BALL_RADIUS,
        mass=config.BALL_MASS,
        pose=config.make_random_ball_pose(rng),
    )
    logging.info(f"Testing {label} ({len(weights)} parameters).")
    logging.info(f"Ball at x={ball.pose.position.x:.4f}, y={ball.pose.position.y:.4f}.")

    brain = BallAwareCpgBrain.from_params(
        params=weights,
        cpg_network_structure=cpg_network_structure,
        initial_state_uniform=math.sqrt(2) * 0.5,
        output_mapping=output_mapping,
        ball=ball,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
        steering_output_scale=config.FEEDBACK_OUTPUT_SCALE,
        distance_scale=config.FEEDBACK_DISTANCE_SCALE,
    )
    robot = ModularRobot(body=config.BODY, brain=brain)
    scene = ModularRobotScene(terrain=config.make_terrain())
    scene.add_robot(robot)
    scene.add_interactive_object(ball)
    scene.add_stop_condition(
        StopOnRobotObjectDistance(robot=robot, obj=ball, distance=config.BALL_REACHED_DISTANCE)
    )

    logging.info(f"Using {viewer_type} MuJoCo viewer.")
    simulator = LocalSimulator(viewer_type=viewer_type)
    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = simulation_time

    logging.info(f"Simulating for up to {simulation_time} seconds.")
    scene_states = simulate_scenes(
        simulator=simulator, batch_parameters=batch_parameters, scenes=scene
    )
    logging.info("Simulation finished.")

    _report_trial(config, scene_states, robot, ball, batch_parameters, simulation_time)


def _report_trial(
    config: Any,
    scene_states: list[Any],
    robot: Any,
    ball: Any,
    batch_parameters: Any,
    simulation_time: float,
) -> None:
    """
    Log initial/closest distance, improvement, progress, and weighted fitness.

    :param config: The loaded config module.
    :param scene_states: Sampled states from the simulation.
    :param robot: The simulated robot.
    :param ball: The target ball.
    :param batch_parameters: Batch parameters used (for sampling frequency).
    :param simulation_time: Maximum simulation seconds.
    """
    import logging
    import math

    from evaluator import _trial_fitness

    def distance(state: Any) -> float:
        robot_pos = state.get_modular_robot_simulation_state(robot).get_pose().position
        ball_pos = state._simulation_state.get_multi_body_system_pose(ball).position
        return math.sqrt((robot_pos.x - ball_pos.x) ** 2 + (robot_pos.y - ball_pos.y) ** 2)

    initial_dist = distance(scene_states[0])
    min_dist = min(distance(state) for state in scene_states)
    logging.info(f"Initial robot-to-ball distance: {initial_dist:.4f} m")
    logging.info(f"Closest robot-to-ball distance: {min_dist:.4f} m")
    logging.info(f"Distance improvement:           {initial_dist - min_dist:+.4f} m")

    progress = 0.0 if initial_dist <= 0.0 else (initial_dist - min_dist) / initial_dist
    logging.info(f"Signed distance progress:       {progress:+.4f}")

    fitness = _trial_fitness(
        scene_states, robot, ball, batch_parameters.sampling_frequency, simulation_time
    )
    logging.info(f"Weighted trial fitness:         {fitness:+.4f}")
    if min_dist <= config.BALL_REACHED_DISTANCE:
        logging.info(f"Reached ball threshold:          {config.BALL_REACHED_DISTANCE:.4f} m")


# --------------------------------------------------------------------------- #
# Test a snapshot (in process)
# --------------------------------------------------------------------------- #


def _run_test(
    snapshot_path: Path, viewer_type: str, simulation_time: float | None = None
) -> None:
    """
    Visually test one saved generation snapshot.

    Loads the snapshot, installs the config it was trained with (remapping
    cluster paths and the former body-catalogue module name to the local
    layout), then
    runs the same simulate-and-report path as `--random-test`, only with the
    snapshot's weights instead of random ones.

    :param snapshot_path: Snapshot pkl path.
    :param viewer_type: Requested MuJoCo viewer.
    :param simulation_time: Simulated seconds to run. When None, the config's
        SIMULATION_TIME is used; pass a value (via --sim-seconds) to watch longer.
    """
    import numpy as np

    viewer_type = _resolve_viewer(viewer_type)
    print(f"Loading generation snapshot: {snapshot_path}", flush=True)
    snapshot = _load_snapshot(snapshot_path)
    config = _install_snapshot_config(snapshot)

    from ball_aware_brain import steering_parameter_count
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )

    weights = np.asarray(snapshot.get("parameters", snapshot.get("best_parameters")))
    active_hinges = config.BODY.find_modules_of_type(ActiveHinge)
    cpg_network_structure, output_mapping = active_hinges_to_cpg_network_structure_neighbor(
        active_hinges
    )
    expected = cpg_network_structure.num_connections + steering_parameter_count(
        output_mapping=output_mapping,
        num_steering_inputs=config.FEEDBACK_NUM_INPUTS,
    )
    if len(weights) != expected:
        raise SystemExit(
            f"Snapshot has {len(weights)} parameters, but this body/config expects "
            f"{expected}. This usually means the snapshot is being tested with the "
            f"wrong config or body."
        )

    generation = snapshot.get("generation", "?")
    fitness = snapshot.get("fitness")
    print(f"Testing generation {generation}"
          + (f" (fitness {float(fitness):.4f})" if fitness is not None else ""), flush=True)

    _simulate_and_report(
        config=config,
        weights=weights,
        cpg_network_structure=cpg_network_structure,
        output_mapping=output_mapping,
        viewer_type=viewer_type,
        simulation_time=(
            simulation_time if simulation_time is not None else config.SIMULATION_TIME
        ),
        label=f"snapshot generation {generation}",
    )


# --------------------------------------------------------------------------- #
# Export snapshots to CSV
# --------------------------------------------------------------------------- #

CSV_FIELDNAMES = [
    "num_of_generation",
    "best_fitness",
    "worst_fitness",
    "best_parent_fitness",
    "best_offspring_fitness",
    "best_ever_fitness",
    "best_robot_weights",
    "worst_robot_weights",
]


def _export_csv(output_dir: Path) -> None:
    """
    Export every gen*.pkl snapshot in a folder to one generations.csv.

    :param output_dir: Folder containing gen*.pkl files.
    :raises SystemExit: If the folder has no snapshots.
    """
    import csv

    snapshots = sorted(output_dir.glob("gen*.pkl"), key=_generation_number)
    if not snapshots:
        raise SystemExit(f"No gen*.pkl files found in {output_dir}")

    csv_path = output_dir / "generations.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for snapshot_path in snapshots:
            snapshot = _load_snapshot(snapshot_path)
            row = snapshot.get("csv_row")
            if row is not None:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})
            else:
                writer.writerow(_derive_csv_row(snapshot))
    print(f"Wrote {csv_path}")


def _derive_csv_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Build a CSV row from a snapshot that has no stored `csv_row`.

    :param snapshot: Loaded snapshot.
    :returns: A row keyed by the canonical CSV fields.
    """
    import json

    def number(value: Any) -> str:
        return "" if value is None else f"{float(value):.17g}"

    def weights(value: Any) -> str:
        return json.dumps([float(v) for v in value]) if value is not None else ""

    return {
        "num_of_generation": int(snapshot["generation"]),
        "best_fitness": number(snapshot.get("best_fitness")),
        "worst_fitness": number(snapshot.get("worst_fitness")),
        "best_parent_fitness": number(snapshot.get("best_parent_fitness")),
        "best_offspring_fitness": number(snapshot.get("best_offspring_fitness")),
        "best_ever_fitness": number(snapshot.get("best_ever_fitness")),
        "best_robot_weights": weights(snapshot.get("best_parameters")),
        "worst_robot_weights": weights(snapshot.get("worst_parameters")),
    }


# --------------------------------------------------------------------------- #
# Continue a run
# --------------------------------------------------------------------------- #


def _continue_training(snapshot_path: Path) -> None:
    """
    Resume training from a resumable generation snapshot.

    Dispatches to the algorithm the snapshot names in its `algorithm` field,
    rather than inferring it from which keys are present.

    :param snapshot_path: Generation snapshot path.
    :raises SystemExit: If the snapshot is not resumable or names no known algorithm.
    """
    import pickle
    import tempfile

    snapshot = _load_snapshot(snapshot_path)
    algorithm = snapshot.get("algorithm")
    if algorithm not in ("ea", "cmaes"):
        raise SystemExit(
            f"Snapshot does not name a known algorithm to resume (algorithm={algorithm!r}). "
            f"Only snapshots that carry an explicit 'algorithm' field can be resumed."
        )
    _validate_resumable_snapshot(snapshot, snapshot_path, algorithm)

    output_dir = Path(
        snapshot.get("artifacts", {}).get("output_dir", snapshot_path.parent)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = _resolve_snapshot_path(Path(snapshot["code_paths"]["main_path"]))

    resume_checkpoint = output_dir / "resume_checkpoint.pkl"
    with open(resume_checkpoint, "wb") as handle:
        pickle.dump(snapshot, handle)
    print(
        f"Continuing {algorithm} from generation "
        f"{int(snapshot['completed_generations'])}; checkpoint: {resume_checkpoint}",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="oribots_resume_config_") as temp_dir:
        config_path = Path(temp_dir) / "config.py"
        config_path.write_text(snapshot["config_source"], encoding="utf-8")
        _run_training(
            config_path=config_path,
            main_path=main_path,
            output_dir=output_dir,
            checkpoint_path=resume_checkpoint,
        )


def _validate_resumable_snapshot(
    snapshot: dict[str, Any], snapshot_path: Path, algorithm: str
) -> None:
    """
    Check a snapshot carries the state needed to resume its algorithm.

    :param snapshot: Loaded snapshot.
    :param snapshot_path: Snapshot path (for the error message).
    :param algorithm: The algorithm the snapshot names.
    :raises SystemExit: If required keys are missing.
    """
    common = ["completed_generations", "training_ball_poses", "config_source", "code_paths"]
    per_algorithm = {
        "ea": [
            "population_parameters", "population_fitnesses", "rng_state",
            "reproducer_rng_state", "best_ever_parameters", "best_ever_fitness", "num_params",
        ],
        "cmaes": ["cma_state", "best_ever_parameters", "best_ever_fitness", "num_params"],
    }
    required = common + per_algorithm[algorithm]
    missing = [key for key in required if key not in snapshot]
    if missing:
        raise SystemExit(
            f"{snapshot_path} is testable but not resumable as {algorithm}: "
            f"missing {', '.join(missing)}."
        )


# --------------------------------------------------------------------------- #
# Snapshot loading and path remapping
# --------------------------------------------------------------------------- #


def _load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """
    Load a generation snapshot.

    :param snapshot_path: Snapshot path.
    :returns: The loaded snapshot.
    """
    import pickle

    # Snapshots from earlier versions reference the body catalogue under its
    # former module name inside the engine namespace. Alias that name to
    # `bodies` so unpickling and the config's imports resolve.
    _install_legacy_body_shim()
    with open(snapshot_path, "rb") as handle:
        return pickle.load(handle)


def _install_snapshot_config(snapshot: dict[str, Any]) -> Any:
    """
    Install the config a snapshot was trained with as the module `config`.

    Uses the snapshot's embedded `config_source` (always present and authoritative),
    executed under the local repository paths, so a snapshot whose stored config
    path points at a `/scratch/...` location that does not exist here still loads.

    :param snapshot: Loaded snapshot.
    :returns: The installed config module.
    """
    import types

    # The config directory the snapshot named, remapped locally when possible;
    # falls back to this repo's config/ so `from bodies import ...` resolves.
    config_dir = _local_config_dir(snapshot)
    paths.install(config_dir)
    _install_legacy_body_shim()
    for cached in ("config", "bodies", "evaluator", "genotype", "ball_aware_brain"):
        sys.modules.pop(cached, None)

    source = snapshot["config_source"]
    config_module = types.ModuleType("config")
    config_module.__file__ = str(config_dir / "config.py")
    sys.modules["config"] = config_module
    exec(compile(source, config_module.__file__, "exec"), config_module.__dict__)
    return config_module


def _local_config_dir(snapshot: dict[str, Any]) -> Path:
    """
    Find the local config directory for a snapshot.

    :param snapshot: Loaded snapshot.
    :returns: A local directory to treat as the config's home.
    """
    stored = snapshot.get("code_paths", {}).get("config_path")
    if stored:
        remapped = _resolve_snapshot_path(Path(stored))
        if remapped.exists():
            return remapped.parent
    return REPO_ROOT / "config"


def _install_legacy_body_shim() -> None:
    """
    Alias the former engine-namespace body module to the `bodies` module.

    Snapshots from earlier versions, and their stored configs, import
    `revolve2.ci_group.modular_robots_v2`, which does not hold this project's
    catalogue. Point that name at `bodies`.
    """
    if "revolve2.ci_group.modular_robots_v2" in sys.modules:
        return
    paths.install()
    try:
        import bodies
    except ImportError:
        return
    sys.modules["revolve2.ci_group.modular_robots_v2"] = bodies


def _resolve_snapshot_path(stored: Path) -> Path:
    """
    Resolve a path stored inside a snapshot onto this machine.

    Cluster snapshots hold absolute `/scratch/...` paths. When a stored path
    does not exist locally, remap it by its first recognized top-level directory
    (`engine`, `src`, `brains`, `config`) into this checkout.

    :param stored: Path saved in the snapshot.
    :returns: A local path when one can be found, else the original.
    """
    if stored.exists():
        return stored.resolve()
    top_level = {"engine", "src", "brains", "config"}
    parts = stored.parts
    for index, part in enumerate(parts):
        if part in top_level:
            candidate = REPO_ROOT.joinpath(*parts[index:])
            if candidate.exists():
                return candidate.resolve()
    return stored


def _generation_number(snapshot_path: Path) -> int:
    """
    Extract the integer generation number from a gen<N>.pkl filename.

    :param snapshot_path: Snapshot path.
    :returns: The generation number.
    """
    return int(snapshot_path.stem.removeprefix("gen"))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _running_under_mjpython() -> bool:
    """
    Whether this process is running under `mjpython`.

    `mjpython` (shipped with the `mujoco` package) sets `MJPYTHON_BIN` in the
    environment; plain `python` does not. `sys.executable` cannot tell them
    apart because both resolve to the same interpreter binary. `mjpython` is the
    only launcher on macOS that gives the viewer the main thread it needs.

    :returns: True if launched via mjpython.
    """
    return "MJPYTHON_BIN" in os.environ


def _resolve_viewer(viewer_type: str) -> str:
    """
    Pair the viewer to the launcher, refusing the combinations that do not work.

    The verified behaviour on macOS is: `mjpython` + `native` displays a window;
    plain `python` + `custom` runs but never composites (nothing appears);
    `mjpython` + `custom` crashes (`NSWindow` off the main thread). The native
    viewer needs the main thread, which only `mjpython` provides.

    `auto` (the default) picks the viewer that matches the launcher. An explicit
    request that cannot work is refused with the command that will, rather than
    silently producing an invisible or crashing window.

    :param viewer_type: Requested viewer: 'auto', 'native', or 'custom'.
    :returns: The viewer to actually use ('native' or 'custom').
    :raises SystemExit: If the requested viewer cannot work under this launcher.
    """
    under_mjpython = _running_under_mjpython()

    if viewer_type == "auto":
        resolved = "native" if under_mjpython else "custom"
        if not under_mjpython and sys.platform == "darwin":
            print(
                "Note: running under plain python, so using the 'custom' viewer. "
                "On macOS that window does not always appear; to watch the robot, "
                "relaunch with mjpython:\n"
                "    .venv.nosync/bin/mjpython run.py -t <snapshot>",
                flush=True,
            )
        return resolved

    if viewer_type == "custom" and under_mjpython:
        raise SystemExit(
            "The 'custom' viewer crashes under mjpython (NSWindow must be created "
            "on the main thread). Drop --viewer so 'auto' selects the native "
            "viewer, which is the one that works under mjpython."
        )

    if viewer_type == "native" and not under_mjpython and sys.platform == "darwin":
        raise SystemExit(
            "The 'native' viewer needs the main thread, which only mjpython gives "
            "it on macOS; under plain python it will not display. Relaunch with:\n"
            "    .venv.nosync/bin/mjpython run.py -t <snapshot>"
        )

    return viewer_type


def _load_config_in_process(config_path: Path) -> Any:
    """
    Load a config file as the module named `config` for an in-process mode.

    Puts the repository directories (including this config's own folder) on
    `sys.path` first, so the config's `from bodies import ...` resolves.

    :param config_path: Config file path.
    :returns: The loaded config module.
    """
    import importlib.util

    paths.install(config_path.parent)
    for cached in ("config", "bodies", "evaluator", "genotype", "ball_aware_brain"):
        sys.modules.pop(cached, None)
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    return module


def _existing_file(path_text: str) -> Path:
    """
    Resolve a path that must exist as a file.

    :param path_text: Path text.
    :returns: The resolved path.
    :raises SystemExit: If the path does not exist.
    """
    path = Path(path_text).resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return path


def _existing_dir(path_text: str) -> Path:
    """
    Resolve a path that must exist as a directory.

    :param path_text: Path text.
    :returns: The resolved directory path.
    :raises SystemExit: If the path is not a directory.
    """
    path = Path(path_text).resolve()
    if not path.is_dir():
        raise SystemExit(f"Directory not found: {path}")
    return path


if __name__ == "__main__":
    main()
