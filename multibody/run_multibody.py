"""Train, resume, watch and replay multibody ball contests (the Entry_Point, R20).

Run from ``Oribots/``::

    python multibody/run_multibody.py -r multibody/configs/<config>.py output/<run>
    python multibody/run_multibody.py -c output/<run>/gen<N>.pkl
    mjpython multibody/run_multibody.py -t output/<run>/gen<N>.pkl [--layout J] [--sim-seconds S]
    python multibody/run_multibody.py -t output/<run>/gen<N>.pkl --headless
    python multibody/run_multibody.py --replay output/<run>/gen<N>.pkl GEN CAND LAYOUT [--headless]
    python multibody/run_multibody.py -t output/<run>/gen<N>.pkl --video match.mp4 [--video-speed 20]

``-r`` starts a new run in a fresh output folder; ``-c`` continues a run from a
snapshot (in that snapshot's folder); ``-t`` plays the best candidate of a
snapshot's generation on one of the run's layouts; ``--replay`` re-simulates one
recorded match and checks every recorded per-robot value against it. Playback
runs ``PLAYBACK_SIMULATION_TIME`` seconds unless ``--sim-seconds`` says
otherwise. The native viewer needs ``mjpython`` on macOS; ``--headless`` needs no
display at all.

Training runs in a child process in its own process group: an interrupt or a
SIGTERM stops it and all its workers (the single-robot ``run.py`` mechanism).
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contest_paths  # noqa: E402

contest_paths.install()
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import arena  # noqa: E402
import config_loader  # noqa: E402
import match  # noqa: E402
import match_recording  # noqa: E402
import match_worker  # noqa: E402
from layer_cmaes import CONFIG_PATH_ENV, OUTPUT_DIR_ENV, SnapshotRefused, load_snapshot  # noqa: E402

LAYER_CMAES = contest_paths.MULTIBODY / "layer_cmaes.py"
EXIT_REFUSED = 2


def _single_robot_runner() -> Any:
    """The single-robot ``run.py``, loaded by path for its process-group launcher (not edited)."""
    spec = importlib.util.spec_from_file_location("_oribots_run", contest_paths.ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse(message: str) -> None:
    print(f"Refused: {message}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_REFUSED)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def start_training(config_path: Path, output_dir: Path) -> None:
    """
    ``-r``: validate, refuse a folder that already holds a run, then train (R20.1, R19.5).
    """
    try:
        config_loader.load_contest(config_path)
    except config_loader.ConfigRefused as error:
        _refuse(str(error))
    existing = match_recording.existing_run_files(output_dir)
    if existing:
        _refuse(f"{output_dir} already holds a run ({', '.join(existing)}); choose a new output folder")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = contest_paths.child_environment()
    environment[CONFIG_PATH_ENV] = str(config_path.resolve())
    environment[OUTPUT_DIR_ENV] = str(output_dir.resolve())
    _launch([sys.executable, str(LAYER_CMAES)], environment)


def continue_training(snapshot_path: Path) -> None:
    """
    ``-c``: continue a run from its snapshot, in the snapshot's folder (R18.1, R18.5, R18.6).
    """
    try:
        snapshot = load_snapshot(snapshot_path)
    except SnapshotRefused as error:
        _refuse(str(error))
    total = int(snapshot["settings_values"]["num_generations"])
    done = int(snapshot["completed_generations"])
    if done >= total:
        print(f"The run is already complete ({snapshot_path.name} is generation {done} of {total}).", flush=True)
        raise SystemExit(0)
    _launch([sys.executable, str(LAYER_CMAES), str(snapshot_path.resolve())], contest_paths.child_environment())


def _launch(command: list[str], environment: dict[str, str]) -> None:
    runner = _single_robot_runner()
    try:
        runner._run_child_process(command, environment)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


# --------------------------------------------------------------------------- #
# Playback and replay
# --------------------------------------------------------------------------- #


def _resolve_viewer(headless: bool) -> Any:
    """The viewer to use, or None for headless; refuses the native viewer without mjpython (R20.9)."""
    if headless:
        return None
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        _refuse(
            "the native viewer needs mjpython on macOS. Relaunch the same playback with:\n"
            f"    .venv.nosync/bin/mjpython {' '.join(sys.argv)}\n"
            "or add --headless to play it without a window."
        )
    from revolve2.simulators.mujoco_simulator.viewers import ViewerType

    return ViewerType.NATIVE


def _play(
    settings: config_loader.ContestSettings,
    layout_index: int,
    layouts: list[arena.Layout],
    candidate: np.ndarray,
    reference: np.ndarray,
    seconds: float,
    viewer: Any,
    video: tuple[Path, float] | None = None,
) -> tuple[list[str], match_worker.MatchOutcome]:
    """Simulate one match in this process (optionally into a video) and score it."""
    config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    structure, mapping = config_loader.body_wiring(settings.body)
    controllers = match.slot_controllers_for(settings, candidate, layout_index, reference)
    task = match.make_match_task(
        settings, layouts[layout_index], layout_index, controllers, structure, mapping,
        scene_id=0, simulation_time=seconds,
    )
    if video is not None:
        import match_video

        states = match_video.record_match(task, settings.terrain_size, video[0], speed=video[1])
        print(f"Wrote {video[0]} ({match_video.frame_count(video[0])} frames, x{video[1]:g}).", flush=True)
    else:
        states = match_worker.simulate_match_states(task, headless=viewer is None, viewer_type=viewer)
    print(f"Simulated {len(states) - 1} samples after t = 0 ({seconds:g} s at {task['sampling_frequency']:g} Hz).", flush=True)
    return task["roles"], match_worker.score_states(task, states)


def _print_metrics(roles: list[str], outcome: match_worker.MatchOutcome) -> None:
    """One line per slot with exact values (R20.7)."""
    if outcome.failure is not None:
        print(f"Match failed: {outcome.failure}", flush=True)
        return
    for slot, (role, metrics) in enumerate(zip(roles, outcome.metrics)):
        print(
            f"slot {slot} {role}: d0={metrics.d0!r} min_distance={metrics.min_distance!r} "
            f"possession_share={metrics.possession_share!r} first_reach_bonus={metrics.first_reach_bonus!r} "
            f"most_possession_bonus={metrics.most_possession_bonus!r} score={metrics.score!r}",
            flush=True,
        )


def _playback_seconds(settings: config_loader.ContestSettings, override: float | None) -> float:
    seconds = settings.playback_simulation_time if override is None else float(override)
    if not (math.isfinite(seconds) and seconds > 0):
        _refuse(f"--sim-seconds {override}: the playback length must be greater than 0 s")
    return seconds


def _load_run(snapshot_path: Path) -> tuple[dict[str, Any], config_loader.ContestSettings, list[arena.Layout], np.ndarray]:
    try:
        snapshot = load_snapshot(snapshot_path)
    except SnapshotRefused as error:
        _refuse(str(error))
    settings = config_loader.settings_from_snapshot(snapshot)
    layouts = arena.layouts_from_plain(snapshot["layouts"])
    reference = settings.reference_layer if settings.reference_layer is not None else match.zero_output_layer(settings)
    return snapshot, settings, layouts, reference


def play_best(snapshot_path: Path, layout_index: int | None, seconds: float | None, headless: bool, video: tuple[Path, float] | None = None) -> None:
    """``-t``: play the snapshot generation's best candidate (R20.2, R20.4-20.7, R20.10)."""
    snapshot, settings, layouts, reference = _load_run(snapshot_path)
    layout_index = 0 if layout_index is None else layout_index
    if not 0 <= layout_index < len(layouts):
        _refuse(f"--layout {layout_index}: must be 0 to {len(layouts) - 1}")
    length = _playback_seconds(settings, seconds)
    viewer = _resolve_viewer(headless or video is not None)
    candidate = np.asarray(snapshot["best_layer_params"], dtype=np.float64)
    print(
        f"Playing generation {snapshot['generation']}'s best candidate (score {snapshot['best_score']!r}) "
        f"on layout {layout_index} for {length:g} s, opponents: {settings.opponents}.",
        flush=True,
    )
    roles, outcome = _play(settings, layout_index, layouts, candidate, reference, length, viewer, video)
    _print_metrics(roles, outcome)


def replay(snapshot_path: Path, generation: int, candidate_index: int, layout_index: int, seconds: float | None, headless: bool, video: tuple[Path, float] | None = None) -> None:
    """``--replay``: re-simulate one recorded match and compare it with its record (R20.3, R20.8)."""
    snapshot, settings, layouts, reference = _load_run(snapshot_path)
    if not 0 <= layout_index < len(layouts):
        _refuse(f"LAYOUT {layout_index}: must be 0 to {len(layouts) - 1}")
    folder = snapshot_path.parent
    names = snapshot["record_paths"]
    candidates_file = folder / names["candidates"]
    robots_file = folder / names["per_robot"]
    if not (candidates_file.is_file() and robots_file.is_file()):
        _refuse(f"the run's record files are missing from {folder}")
    row = next(
        (r for r in match_recording.read_rows(candidates_file)
         if r["generation"] == str(generation) and r["candidate_index"] == str(candidate_index)),
        None,
    )
    if row is None:
        _refuse(f"no recorded candidate {candidate_index} in generation {generation} ({candidates_file.name})")
    recorded = [
        r for r in match_recording.read_rows(robots_file)
        if r["generation"] == str(generation) and r["candidate_index"] == str(candidate_index)
        and r["layout_index"] == str(layout_index)
    ]
    recorded.sort(key=lambda r: int(r["slot"]))
    length = _playback_seconds(settings, seconds)
    viewer = _resolve_viewer(headless or video is not None)
    candidate = np.asarray(eval_parameters(row["layer_params"]), dtype=np.float64)
    roles, outcome = _play(settings, layout_index, layouts, candidate, reference, length, viewer, video)
    _print_metrics(roles, outcome)
    if length != settings.simulation_time:
        print(
            f"Comparison skipped: the playback length ({length:g} s) differs from the recorded "
            f"SIMULATION_TIME ({settings.simulation_time:g} s).",
            flush=True,
        )
        return
    mismatches = compare_with_record(recorded, outcome, settings.failed_match_score)
    for slot, verdicts in enumerate(mismatches):
        print(f"slot {slot}: " + ", ".join(f"{name} {'equal' if ok else 'DIFFERS'}" for name, ok in verdicts), flush=True)
    all_equal = all(ok for verdicts in mismatches for _, ok in verdicts)
    print("Replay matches the record bit for bit." if all_equal else "Replay DIFFERS from the record.", flush=True)
    if not all_equal:
        raise SystemExit(1)


def eval_parameters(text: str) -> list[float]:
    """Parse a recorded parameter list (JSON written by the single-robot serializer)."""
    import json

    return [float(value) for value in json.loads(text)]


def compare_with_record(recorded: list[dict[str, str]], outcome: match_worker.MatchOutcome, failed_score: float) -> list[list[tuple[str, bool]]]:
    """
    Compare a replayed match with its recorded per-robot rows, value by value.

    :returns: Per slot, (column, equal?) pairs.
    """
    verdicts = []
    for slot, row in enumerate(recorded):
        pairs = []
        failed_now = outcome.failure is not None
        pairs.append(("failed", (row["failed"] == "True") == failed_now))
        for column in match_recording.METRIC_COLUMNS:
            stored = float(row[column])
            if failed_now:
                value = failed_score if column == "score" else float("nan")
            else:
                value = float(getattr(outcome.metrics[slot], column))
            same = (math.isnan(stored) and math.isnan(value)) or stored == value
            pairs.append((column, same))
        verdicts.append(pairs)
    return verdicts


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, resume, watch and replay multibody contests.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-r", nargs=2, metavar=("CONFIG", "OUTPUT"), help="start a new run")
    group.add_argument("-c", metavar="SNAPSHOT", help="continue a run from its snapshot")
    group.add_argument("-t", metavar="SNAPSHOT", help="play a snapshot's best candidate")
    group.add_argument("--replay", nargs=4, metavar=("SNAPSHOT", "GEN", "CAND", "LAYOUT"), help="re-simulate a recorded match")
    parser.add_argument("--layout", type=int, default=None, help="layout index for -t (default 0)")
    parser.add_argument("--sim-seconds", type=float, default=None, help="playback length override")
    parser.add_argument("--headless", action="store_true", help="play without a window")
    parser.add_argument("--viewer", choices=["native"], default="native", help="viewer for -t/--replay")
    parser.add_argument("--video", type=Path, default=None, help="record -t/--replay to this .mp4 instead of opening a window")
    parser.add_argument("--video-speed", type=float, default=20.0, help="simulated seconds per video second (default 20)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Dispatch the command line."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    video = None
    if args.video is not None:
        if not (math.isfinite(args.video_speed) and args.video_speed > 0):
            _refuse(f"--video-speed {args.video_speed}: must be greater than 0")
        if args.video.suffix.lower() != ".mp4":
            _refuse(f"--video {args.video}: the file must end in .mp4")
        video = (args.video, args.video_speed)
    if args.r:
        start_training(Path(args.r[0]), Path(args.r[1]))
    elif args.c:
        continue_training(Path(args.c))
    elif args.t:
        play_best(Path(args.t), args.layout, args.sim_seconds, args.headless, video)
    else:
        snapshot, generation, candidate, layout = args.replay
        try:
            numbers = int(generation), int(candidate), int(layout)
        except ValueError:
            _refuse(f"--replay GEN CAND LAYOUT must be integers, got {generation} {candidate} {layout}")
        replay(Path(snapshot), *numbers, args.sim_seconds, args.headless, video)


if __name__ == "__main__":
    main()
