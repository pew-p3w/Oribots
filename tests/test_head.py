"""Check head mode and the pentagon spider against their reference values.

Replays the frozen real simulation states in `head_reference.json` through
robot_frame, the brain, the evaluator and the engine stop condition, in head
mode and in core mode, and compares every value with the reference values
computed from the same states. Also checks the pentagon spider's structure and
built core geometry, the 8-ball training layout, and that the two spider_8ball
configs describe the same experiment. See `tests/head_fingerprint.py`.

Run with `make check-head`, or `python tests/test_head.py`.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import head_fingerprint  # noqa: E402

ROOT = paths.ROOT
CONFIGS = ("spider_8ball", "spider_8ball_cmaes")
ALGORITHM_NAMES = {
    "TEST_FILE",
    "POPULATION_SIZE",
    "TOURNAMENT_SIZE",
    "MUTATE_STD",
    "MUTATION_PROBABILITY",
    "CMA_INITIAL_STD",
    "CMA_INITIAL_MEAN",
    "CMA_BOUNDS",
    "CMA_POPULATION_SIZE",
}


def load_config(stem: str):
    """Load a config file as its own module (not as `config`)."""
    spec = importlib.util.spec_from_file_location(f"_head_check_{stem}", ROOT / "config" / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_modules() -> dict:
    """
    Import the modules the way the runner does.

    :returns: The modules and functions `head_fingerprint.fingerprint` needs.
    """
    paths.install()
    sys.path.insert(0, str(ROOT / "config"))
    reference = head_fingerprint.load_reference()
    # The evaluator imports `config` when it is imported, so one must exist.
    sys.modules["config"] = head_fingerprint.make_config(
        reference["inputs"]["head_offset"], True, reference["inputs"]["simulation_time"]
    )
    import ball_aware_brain
    import bodies
    import evaluator
    import robot_frame
    from revolve2.modular_robot_simulation import _modular_robot_scene

    for module in (ball_aware_brain, bodies, evaluator, robot_frame, _modular_robot_scene):
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise SystemExit(f"{module.__name__} was imported from outside this repository: {module.__file__}")

    def set_config(module) -> None:
        sys.modules["config"] = module
        evaluator.config = module

    def spider_pentagon():
        body = bodies.spider_pentagon()
        return body, body.core_pentagon.front_face.middle

    return {
        "robot_frame": robot_frame,
        "brain": ball_aware_brain,
        "evaluator": evaluator,
        "scene": _modular_robot_scene,
        "set_config": set_config,
        "spider_pentagon": spider_pentagon,
        "make_training_ball_poses": load_config("spider_8ball").make_training_ball_poses,
    }


def check_invariants(reference: dict, data: dict) -> list[str]:
    """
    Check what makes the reference comparison meaningful, and the configs.

    :returns: A list of problems; empty when all hold.
    """
    import math

    problems = []
    sequences = data["sequences"]
    # The scenario must actually exercise reaching the ball in head mode, and
    # must not sit on the edge of the reach threshold.
    if sequences["evolved_reaches"]["head"]["first_stop_index"] is None:
        problems.append("evolved_reaches never reaches the ball in head mode: the reach path is untested")
    if sequences["random_genome"]["head"]["num_stops"] != 0:
        problems.append("random_genome reaches the ball: it was meant to be the non-reaching case")
    for name, sequence in sequences.items():
        for mode in ("head", "legacy"):
            if sequence[mode]["min_threshold_margin"] < 1e-6:
                problems.append(f"{name} {mode}: a sample sits within 1e-6 m of the reach threshold")
    head = data["pentagon"]
    if len(head["core_geometries"]) < 5:
        problems.append("the pentagon core is not built from five slabs (is the engine hook missing?)")
    # Both configs: head mode on, the reference head, the same experiment.
    configs = {stem: load_config(stem) for stem in CONFIGS}
    for stem, module in configs.items():
        if not module.USE_HEAD_COORDINATES:
            problems.append(f"{stem}: USE_HEAD_COORDINATES is off")
        if [round(v, 12) for v in module.HEAD_OFFSET] != head["head_offset"]:
            problems.append(f"{stem}: HEAD_OFFSET {module.HEAD_OFFSET} differs from the reference")
    shared = [
        {
            name: value
            for name, value in vars(module).items()
            if name.isupper() and name not in ALGORITHM_NAMES and name not in {"BODY", "HEAD", "NUM_SIMULATORS"}
        }
        for module in configs.values()
    ]
    differing = sorted(
        name for name in set(shared[0]) | set(shared[1]) if repr(shared[0].get(name)) != repr(shared[1].get(name))
    )
    if differing:
        problems.append(f"spider_8ball and spider_8ball_cmaes differ in {differing}")
    # The layout: two balls per quadrant, 9.5 m to 10.5 m away.
    quadrants = [0, 0, 0, 0]
    for x, y, _ in data["ball_layout"]:
        quadrants[int((math.atan2(y, x) % (2 * math.pi)) // (math.pi / 2))] += 1
        if not 9.5 - 1e-9 <= math.hypot(x, y) <= 10.5 + 1e-9:
            problems.append(f"training ball at ({x}, {y}) is not 9.5-10.5 m away")
    if quadrants != [2, 2, 2, 2]:
        problems.append(f"training balls per quadrant are {quadrants}, expected two each")
    return problems


def main() -> None:
    """Run the comparison and the invariants, and report."""
    modules = load_modules()
    reference = head_fingerprint.load_reference()
    data = head_fingerprint.fingerprint(modules, reference["inputs"])
    problems = head_fingerprint.compare(reference["fingerprint"], data)
    problems += check_invariants(reference, data)
    if problems:
        print("HEAD CHECK FAILED")
        for problem in problems[:40]:
            print(f"  {problem}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        raise SystemExit(1)
    sequences = data["sequences"]
    print("HEAD CHECK PASSED")
    print(
        f"  {head_fingerprint.value_count(data)} values identical to the reference "
        f"(digest {head_fingerprint.digest(data)})"
    )
    print(
        "  frozen real states: "
        + ", ".join(
            f"{name} ({sequence['num_samples']} samples, head-mode stop at "
            f"{sequence['head']['first_stop_index']})"
            for name, sequence in sequences.items()
        )
    )
    print(
        f"  pentagon spider: {data['pentagon']['num_active_hinges']} hinges, head at "
        f"{data['pentagon']['head_offset']}, core of {len(data['pentagon']['core_geometries'])} geometries"
    )
    print("  spider_8ball and spider_8ball_cmaes describe the same experiment; 8 balls, 2 per quadrant")


if __name__ == "__main__":
    main()
