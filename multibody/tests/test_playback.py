"""Headless playback and replay (R20.2-20.10, phase 6 code side).

From a real one-generation ``_tiny`` run:

- ``-t --headless`` plays the generation's best candidate on layout 0 and on
  ``--layout 1``, for the stored PLAYBACK_SIMULATION_TIME, printing one line per slot;
- ``--replay`` at the recorded length reproduces every recorded per-robot value
  bit for bit, and at a different length runs the full length and says the
  comparison was skipped;
- a playback longer than 600 s runs its full length (R20.5);
- invalid arguments, and the native viewer without mjpython on macOS, are
  refused before simulating, the latter with the mjpython command (R20.9).

Watching the native viewer itself is a manual step (phase 6 gate).
"""

import os
import sys

from _check import check_main, require, temp_workspace

import _runs


def play(workspace, name, args, env=None):
    log = workspace / f"{name}.log"
    process = _runs.subprocess.Popen(
        [sys.executable, str(_runs.RUNNER), *args], cwd=str(_runs.ROOT),
        stdout=open(log, "w"), stderr=_runs.subprocess.STDOUT, env=env,
    )
    status = process.wait(timeout=1500)
    return status, log.read_text()


def run() -> None:
    with temp_workspace() as workspace:
        config = _runs.tiny_config(workspace, NUM_GENERATIONS=1, PLAYBACK_SIMULATION_TIME=40)
        output = workspace / "run"
        require(_runs.run(["-r", str(config), str(output)], workspace / "train.log") == 0, "training failed")
        snapshot = str(output / "gen1.pkl")
        rows = _runs.record_rows(output)["candidates_"]
        candidate = rows[1].split(",")[1]

        status, text = play(workspace, "t0", ["-t", snapshot, "--headless"])
        require(status == 0 and "slot 0" in text and "slot 1" in text, f"-t failed: {text[-1500:]}")
        require("Learner" in text and "Opponent" in text, "-t did not show the roles")
        require("Simulated 200 samples" in text, f"-t did not play the 40 s PLAYBACK_SIMULATION_TIME: {text[-800:]}")
        status, text = play(workspace, "t1", ["-t", snapshot, "--layout", "1", "--headless"])
        require(status == 0 and "on layout 1" in text, "-t --layout 1 failed")

        status, text = play(workspace, "replay", ["--replay", snapshot, "1", candidate, "1", "--sim-seconds", "30", "--headless"])
        require(status == 0 and "matches the record bit for bit" in text, f"replay at the recorded length: {text[-1500:]}")
        status, text = play(workspace, "replay_long", ["--replay", snapshot, "1", candidate, "0", "--headless"])
        require(status == 0 and "Comparison skipped" in text and "40 s" in text and "30 s" in text,
                f"replay at a different length: {text[-1500:]}")

        status, text = play(workspace, "long", ["-t", snapshot, "--sim-seconds", "610", "--headless"])
        require(status == 0 and "Simulated 3050 samples" in text, f"a 610 s playback did not run its full length: {text[-800:]}")

        # A video records the same match: identical per-robot results, the right
        # number of frames for the speed, and frames that are not blank.
        video = workspace / "match.mp4"
        status, recorded = play(workspace, "video", ["-t", snapshot, "--sim-seconds", "10", "--video", str(video), "--video-speed", "10"])
        status_h, headless = play(workspace, "video_ref", ["-t", snapshot, "--sim-seconds", "10", "--headless"])
        slots = lambda text: [line for line in text.splitlines() if line.startswith("slot ")]
        require(status == 0 and status_h == 0 and slots(recorded) and slots(recorded) == slots(headless),
                f"the video's results differ from a headless playback:\n{recorded[-800:]}\n{headless[-800:]}")
        import match_video

        require(match_video.frame_count(video) == 10 * 30 // 10 + 1, f"{match_video.frame_count(video)} frames, expected 31")
        require(match_video.mean_brightness(video, 15) > 10, "video frames are blank")
        status, text = play(workspace, "bad_video", ["-t", snapshot, "--video", str(workspace / "x.avi")])
        require(status != 0 and ".mp4" in text, "a non-.mp4 video path was not refused")

        refusals = [
            (["-t", snapshot, "--layout", "2", "--headless"], "--layout 2"),
            (["-t", snapshot, "--sim-seconds", "0", "--headless"], "greater than 0"),
            (["--replay", snapshot, "7", "0", "0", "--headless"], "no recorded candidate"),
            (["--replay", snapshot, "1", "0", "9", "--headless"], "LAYOUT 9"),
        ]
        for args, words in refusals:
            status, text = play(workspace, "refuse", args)
            require(status != 0 and words in text and "Simulated" not in text, f"{args}: expected a refusal with '{words}': {text[-600:]}")
        # The viewer selection itself (the window cannot be opened in the suite):
        # under mjpython it must resolve to the native viewer without error.
        import run_multibody
        from revolve2.simulators.mujoco_simulator.viewers import ViewerType

        os.environ["MJPYTHON_BIN"] = "mjpython"
        try:
            require(run_multibody._resolve_viewer(False) is ViewerType.NATIVE, "viewer selection under mjpython")
            require(run_multibody._resolve_viewer(True) is None, "headless selection")
        finally:
            del os.environ["MJPYTHON_BIN"]
        if sys.platform == "darwin":
            env = {k: v for k, v in os.environ.items() if k != "MJPYTHON_BIN"}
            status, text = play(workspace, "viewer", ["-t", snapshot], env=env)
            require(status != 0 and "mjpython" in text and "-t" in text and "Simulated" not in text,
                    f"native viewer without mjpython was not refused with the mjpython command: {text[-600:]}")


if __name__ == "__main__":
    check_main("PLAYBACK", run)
