"""Check every config loads standalone and means what it should.

Configs are flat duplicates with nothing shared, which keeps each one readable
on its own but makes copy-paste mistakes easy: a config that still carries
another body's name, or one whose constants quietly drifted. These checks
cover both, plus the sampling rules the training ball placement relies on.

Run with `make check-configs`, or `python tests/test_configs.py`.
"""

import importlib.util
import math
import shutil
import sys
from pathlib import Path

# Python reuses cached bytecode when a source file's size and whole-second
# mtime are unchanged. Editing a config and rerunning within the same second,
# without changing its length, therefore reads the previous values - which is
# exactly what a small edit like 0.3 -> 0.0 looks like. run.py sets this same
# flag for the same reason.
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402


ROOT = paths.ROOT
CONFIG_DIR = ROOT / "config"

# Names every config must define for the runner, evaluator and brain to work.
REQUIRED_NAMES = [
    "BODY", "TEST_FILE", "TERRAIN_SIZE", "BALL_RADIUS", "BALL_MASS",
    "BALL_REACHED_DISTANCE", "BALL_SPAWN_MARGIN",
    "MIN_TRAINING_BALL_DISTANCE_FRACTION", "NUM_TRAINING_BALL_POSES",
    "FEEDBACK_NUM_INPUTS", "FEEDBACK_OUTPUT_SCALE", "FEEDBACK_DISTANCE_SCALE",
    "NO_PROGRESS_PENALTY", "NO_PROGRESS_EPSILON", "FITNESS_PROGRESS_WEIGHT",
    "FITNESS_REACHED_BONUS_WEIGHT", "FITNESS_TIME_TO_REACH_WEIGHT",
    "FITNESS_ALIGNMENT_WEIGHT", "NUM_GENERATIONS", "SIMULATION_TIME",
    "NUM_SIMULATORS", "HEADLESS",
]
REQUIRED_FUNCTIONS = [
    "parameter_filename", "make_terrain", "make_random_ball_pose", "make_training_ball_pose",
]
EA_ONLY_NAMES = ["POPULATION_SIZE", "TOURNAMENT_SIZE", "MUTATE_STD", "MUTATION_PROBABILITY"]
CMAES_ONLY_NAMES = ["CMA_INITIAL_STD", "CMA_INITIAL_MEAN", "CMA_BOUNDS", "CMA_POPULATION_SIZE"]

# Settings that must be identical in every config. A difference here is drift,
# not intent, and would make two runs incomparable for a reason nobody recorded.
SHARED_VALUES = {
    "BALL_RADIUS": 0.3,
    "BALL_MASS": 0.1,
    "BALL_SPAWN_MARGIN": 0.4,
    "MIN_TRAINING_BALL_DISTANCE_FRACTION": 0.3,
    "FEEDBACK_NUM_INPUTS": 4,
    "FEEDBACK_OUTPUT_SCALE": 0.5,
    "NO_PROGRESS_PENALTY": 0.01,
    "NO_PROGRESS_EPSILON": 1e-6,
    "FITNESS_PROGRESS_WEIGHT": 1.0,
    "FITNESS_REACHED_BONUS_WEIGHT": 0.20,
    "FITNESS_TIME_TO_REACH_WEIGHT": 0.10,
    "FITNESS_ALIGNMENT_WEIGHT": 0.05,
    "HEADLESS": True,
}

# The only configs allowed to differ from the settings every real body shares.
FIXTURES = {"_tiny", "_tiny_cmaes"}


def load_config(path: Path):
    """
    Load one config the way the runner does: as a module named `config`.

    :param path: The config file.
    :returns: The loaded module.
    """
    for cached in ("config", "bodies"):
        sys.modules.pop(cached, None)
    spec = importlib.util.spec_from_file_location("config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)
    return module


def check_config(path: Path, module) -> list[str]:
    """
    Check one loaded config.

    :param path: The config file.
    :param module: The loaded config module.
    :returns: A list of problems; empty when the config is sound.
    """
    import numpy as np

    stem = path.stem
    problems = []

    for name in REQUIRED_NAMES:
        if not hasattr(module, name):
            problems.append(f"{stem}: missing {name}")
    for name in REQUIRED_FUNCTIONS:
        if not callable(getattr(module, name, None)):
            problems.append(f"{stem}: missing function {name}()")
    if problems:
        return problems

    is_cmaes = stem.endswith("_cmaes")
    for name in (CMAES_ONLY_NAMES if is_cmaes else EA_ONLY_NAMES):
        if not hasattr(module, name):
            problems.append(f"{stem}: missing {name}")
    for name in (EA_ONLY_NAMES if is_cmaes else CMAES_ONLY_NAMES):
        if hasattr(module, name):
            problems.append(f"{stem}: defines {name}, which belongs to the other algorithm")

    # A config that still names another experiment would write its results over
    # that experiment's files.
    if module.TEST_FILE != stem:
        problems.append(f"{stem}: TEST_FILE is {module.TEST_FILE!r}, expected {stem!r}")
    if module.parameter_filename() != f"parameters_{stem}.npy":
        problems.append(f"{stem}: parameter_filename() is {module.parameter_filename()!r}")

    # Config filenames become importable top-level modules, because the config
    # directory goes on sys.path.
    if stem in sys.stdlib_module_names:
        problems.append(f"{stem}: filename shadows the standard library module {stem!r}")

    if stem not in FIXTURES:
        for name, expected in SHARED_VALUES.items():
            actual = getattr(module, name)
            if actual != expected:
                problems.append(f"{stem}: {name} is {actual!r}, expected the shared {expected!r}")

    # FEEDBACK_DISTANCE_SCALE must track the arena, not be a fixed guess: the
    # furthest a ball can spawn from the robot's start at the origin.
    expected_scale = math.sqrt(
        (module.TERRAIN_SIZE.x / 2.0 - max(module.BALL_SPAWN_MARGIN, module.BALL_RADIUS)) ** 2
        + (module.TERRAIN_SIZE.y / 2.0 - max(module.BALL_SPAWN_MARGIN, module.BALL_RADIUS)) ** 2
    )
    if abs(module.FEEDBACK_DISTANCE_SCALE - expected_scale) > 1e-9:
        problems.append(
            f"{stem}: FEEDBACK_DISTANCE_SCALE is {module.FEEDBACK_DISTANCE_SCALE}, "
            f"not the {expected_scale} the terrain implies"
        )

    # A laptop must not be asked to start dozens of simulator processes.
    if not 1 <= module.NUM_SIMULATORS <= 32:
        problems.append(f"{stem}: NUM_SIMULATORS is {module.NUM_SIMULATORS}")

    # Ball placement: training poses must clear the minimum distance, and every
    # pose must sit inside the arena.
    rng = np.random.default_rng(3)
    margin = max(module.BALL_SPAWN_MARGIN, module.BALL_RADIUS)
    half_x = module.TERRAIN_SIZE.x / 2.0 - margin
    half_y = module.TERRAIN_SIZE.y / 2.0 - margin
    min_distance = module.MIN_TRAINING_BALL_DISTANCE_FRACTION * min(
        module.TERRAIN_SIZE.x, module.TERRAIN_SIZE.y
    )
    for _ in range(300):
        pose = module.make_training_ball_pose(rng)
        distance = math.hypot(pose.position.x, pose.position.y)
        if distance < min_distance - 1e-9:
            problems.append(
                f"{stem}: make_training_ball_pose returned a ball {distance:.3f} m away, "
                f"nearer than the {min_distance:.3f} m minimum"
            )
            break
        if abs(pose.position.x) > half_x + 1e-9 or abs(pose.position.y) > half_y + 1e-9:
            problems.append(f"{stem}: make_training_ball_pose returned a ball outside the arena")
            break
    for _ in range(300):
        pose = module.make_random_ball_pose(rng)
        if abs(pose.position.x) > half_x + 1e-9 or abs(pose.position.y) > half_y + 1e-9:
            problems.append(f"{stem}: make_random_ball_pose returned a ball outside the arena")
            break
        if abs(pose.position.z - module.BALL_RADIUS) > 1e-9:
            problems.append(f"{stem}: make_random_ball_pose placed the ball off the ground")
            break

    return problems


def main() -> None:
    """Load every config and report problems."""
    paths.install()
    sys.path.insert(0, str(CONFIG_DIR))

    # Drop any bytecode left by an earlier run, which could otherwise hide a
    # change to a config made within the same second (see the note above).
    cache = CONFIG_DIR / "__pycache__"
    if cache.is_dir():
        shutil.rmtree(cache)

    config_paths = sorted(p for p in CONFIG_DIR.glob("*.py") if p.stem != "bodies")
    if not config_paths:
        raise SystemExit("no configs found")

    problems = []
    loaded = {}
    for path in config_paths:
        try:
            module = load_config(path)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            problems.append(f"{path.stem}: failed to load: {error}")
            continue
        loaded[path.stem] = {
            name: getattr(module, name)
            for name in REQUIRED_NAMES + EA_ONLY_NAMES + CMAES_ONLY_NAMES
            if hasattr(module, name)
        }
        problems.extend(check_config(path, module))


    if problems:
        print("CONFIG CHECK FAILED")
        for problem in problems[:30]:
            print(f"  - {problem}")
        if len(problems) > 30:
            print(f"  ... and {len(problems) - 30} more")
        raise SystemExit(1)

    ea = sorted(s for s in loaded if not s.endswith("_cmaes") and s not in FIXTURES)
    cmaes = sorted(s for s in loaded if s.endswith("_cmaes"))
    print("CONFIG CHECK PASSED")
    print(f"  {len(ea)} EA configs, {len(cmaes)} CMA-ES configs, {len(FIXTURES)} test fixture")
    print(f"  CMA-ES: {', '.join(cmaes)}")
    print("  each loads standalone, names its own outputs, and shares the common settings")
    print("  ball placement respects the minimum training distance and the arena bounds")


if __name__ == "__main__":
    main()
