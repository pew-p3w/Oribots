"""Self-contained snapshots (Property 13, R17).

From a real one-generation ``_tiny`` run:

- R17.4: loading a snapshot and writing it again gives equal contents, and the
  CMA-ES state draws bit-identical next candidates;
- R17.7: it loads in a fresh Python process that has the pinned packages but no
  multibody module on its path;
- R17.3, R17.8: a run folder copied elsewhere, with the original config and
  weights files gone, still plays back (headless) and resumes in its new folder,
  touching nothing at the old paths;
- R17.9, R18.5: a non-snapshot file, a single-robot snapshot and an incomplete
  multibody snapshot are refused with a non-zero exit and no new files.
"""

import os
import pickle
import shutil
import subprocess
import sys

from _check import check_main, require, temp_workspace

import _runs
import layer_cmaes


def run() -> None:
    with temp_workspace() as workspace:
        weights = workspace / "weights_gen1.pkl"
        shutil.copy(_runs.FIXTURES / "fake_single_robot_gen1.pkl", weights)
        config = _runs.tiny_config(workspace, NUM_GENERATIONS=2, TRAINED_WEIGHTS_FILE=str(weights))
        original = workspace / "original"
        status = _runs.run(["-r", str(config), str(original)], workspace / "train.log")
        require(status == 0, f"training exited with {status}")
        snapshot_path = original / "gen1.pkl"
        snapshot = _runs.load(snapshot_path)

        # Round trip.
        copy_dir = workspace / "roundtrip"
        copy_dir.mkdir()
        layer_cmaes.write_snapshot(copy_dir, 1, _runs.load(snapshot_path))
        differ = _runs.snapshot_differences(snapshot, _runs.load(copy_dir / "gen1.pkl"), ignore=())
        require(not differ, f"round trip changed {differ}")

        # Loads without any multibody module importable.
        environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        code = (
            "import sys, pickle; sys.path.insert(0, %r); import paths; paths.install(); "
            "d = pickle.load(open(%r, 'rb')); "
            "assert not any(m in sys.modules for m in ('config_loader', 'layer_cmaes', 'arena', 'match')); "
            "print(d['algorithm'])"
        ) % (str(_runs.ROOT / "src"), str(snapshot_path))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/", env=environment)
        require(result.returncode == 0 and result.stdout.strip() == "multibody_cmaes",
                f"snapshot did not load without multibody modules: {result.stderr[-1500:]}")

        # Move the run: copy the folder, delete the config and the weights it named.
        moved = workspace / "moved"
        shutil.copytree(original, moved)
        (moved / "gen2.pkl").unlink()
        before_original = sorted(p.name for p in original.iterdir())
        mtimes = {p.name: p.stat().st_mtime_ns for p in original.iterdir()}
        config.unlink()
        weights.unlink()
        status = _runs.run(["-t", str(moved / "gen1.pkl"), "--headless"], workspace / "play.log")
        require(status == 0, f"headless playback of the moved run exited with {status}: {(workspace / 'play.log').read_text()[-1500:]}")
        status = _runs.run(["-c", str(moved / "gen1.pkl")], workspace / "resume.log")
        require(status == 0, f"resume of the moved run exited with {status}")
        require((moved / "gen2.pkl").exists(), "the moved run did not continue in its new folder")
        require(sorted(p.name for p in original.iterdir()) == before_original
                and all(p.stat().st_mtime_ns == mtimes[p.name] for p in original.iterdir()),
                "the resumed copy touched the original folder")
        require(not _runs.snapshot_differences(_runs.load(original / "gen2.pkl"), _runs.load(moved / "gen2.pkl")),
                "the moved run's gen2 differs from the original's")

        # Refusals.
        junk = workspace / "junk"
        junk.mkdir()
        (junk / "gen1.pkl").write_bytes(b"not a pickle")
        with open(junk / "gen2.pkl", "wb") as handle:
            pickle.dump({"algorithm": "ea", "best_parameters": [0.0]}, handle)
        partial = {k: v for k, v in snapshot.items() if k != "cma_state"}
        with open(junk / "gen3.pkl", "wb") as handle:
            pickle.dump(partial, handle)
        listing = sorted(p.name for p in junk.iterdir())
        for name, words in (("gen1.pkl", ["cannot be loaded"]), ("gen2.pkl", ["'ea'"]), ("gen3.pkl", ["missing", "cma_state"])):
            for flag in ("-c", "-t"):
                log = workspace / f"refuse_{name}_{flag}.log"
                args = [flag, str(junk / name)] + (["--headless"] if flag == "-t" else [])
                status = _runs.run(args, log)
                text = log.read_text()
                require(status != 0, f"{flag} {name} was not refused")
                require(all(w in text for w in words) and name in text, f"{flag} {name}: refusal text {text[-500:]}")
        require(sorted(p.name for p in junk.iterdir()) == listing, "a refused snapshot's folder changed")


if __name__ == "__main__":
    check_main("SNAPSHOT", run)
