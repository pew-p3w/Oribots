"""Check the evaluator against its reference values.

Two things are checked. First, the fitness calculation must produce exactly the
reference values for a set of scripted trials. Second, the evaluator must return
raw signed fitness and never clip to `[0, 1]`: the clip belongs to the EA, and
CMA-ES needs the unclipped values.

Run with `make check-evaluator`, or `python tests/test_evaluator.py`.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import evaluator_fingerprint  # noqa: E402
from compare import differences  # noqa: E402

ROOT = paths.ROOT


def load_evaluator_module():
    """
    Import the evaluator the way a training run does.

    :returns: The evaluator module.
    :raises SystemExit: If it cannot be imported as a bare module.
    """
    sys.modules["config"] = evaluator_fingerprint.make_stub_config()
    sys.path.insert(0, str(ROOT / "brains"))
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import evaluator
    except ImportError as error:
        raise SystemExit(f"could not import `evaluator` as a top-level module: {error}")
    return evaluator


def check_no_clipping(evaluator_module) -> list[str]:
    """
    Check that averaging across ball poses returns raw signed fitness.

    The simulation is not run: `_evaluate_scenes` (one ball's batch) is replaced
    with scripted per-pose scores, and the one-batch-per-ball path is forced, so
    only the averaging and the absence of clipping are exercised.

    :param evaluator_module: The evaluator module.
    :returns: A list of problems; empty when the behaviour is correct.
    """
    problems: list[str] = []

    def stub_pose(index: int):
        """A ball pose with the coordinates the evaluator logs."""
        return evaluator_fingerprint._Pose(evaluator_fingerprint._Position(float(index), 0.0, 0.3))

    cases = [
        # per-pose scores, expected raw mean, the mean clamped to [0, 1]
        ([-2.0, -1.0], -1.5, 0.0),
        ([-0.4, 0.2], -0.1, 0.0),
        ([0.5, 1.5], 1.0, 1.0),
        ([3.0, 3.0], 3.0, 1.0),
        ([0.25, 0.75], 0.5, 0.5),
    ]

    saved = os.environ.get("REVOLVE2_SCORE_IN_WORKERS")
    os.environ["REVOLVE2_SCORE_IN_WORKERS"] = "0"
    try:
        results = []
        for per_pose_scores, _, _ in cases:
            evaluator = object.__new__(evaluator_module.Evaluator)
            remaining = list(per_pose_scores)

            def fake_scenes(population, ball_pose, with_movement, _remaining=remaining):
                """Return the next scripted per-pose score for a one-member population."""
                return [_remaining.pop(0)], [None]

            evaluator._evaluate_scenes = fake_scenes
            results.append(
                evaluator_module.Evaluator.evaluate_on_ball_poses(
                    evaluator, [object()], [stub_pose(i) for i in range(len(per_pose_scores))]
                )
            )
    finally:
        if saved is None:
            os.environ.pop("REVOLVE2_SCORE_IN_WORKERS", None)
        else:
            os.environ["REVOLVE2_SCORE_IN_WORKERS"] = saved

    for (per_pose_scores, expected_raw, old_clipped), result in zip(cases, results):
        actual = result[0]
        if abs(actual - expected_raw) > 1e-12:
            problems.append(
                f"mean of {per_pose_scores} was {actual}, expected the raw {expected_raw}"
            )
        # Clipping the raw result must give the clamped value, which is what the
        # EA records (and selects on, with SELECT_ON_RAW_FITNESS = False).
        if abs(min(max(actual, 0.0), 1.0) - old_clipped) > 1e-12:
            problems.append(
                f"clipping the raw mean of {per_pose_scores} gave "
                f"{min(max(actual, 0.0), 1.0)}, expected {old_clipped}"
            )

    # No ball poses is a programming error, not a silent zero.
    evaluator = object.__new__(evaluator_module.Evaluator)
    try:
        evaluator_module.Evaluator.evaluate_on_ball_poses(evaluator, [object()], [])
    except ValueError:
        pass
    else:
        problems.append("evaluate_on_ball_poses accepted an empty list of ball poses")

    # The evaluator must not depend on the genotype representation.
    source = (ROOT / "src" / "evaluator.py").read_text()
    if "from genotype import" in source or "import genotype" in source:
        problems.append("evaluator imports genotype; it should take plain parameter vectors")

    return problems


def main() -> None:
    """Compare the evaluator with the reference and report differences."""
    paths.install()
    evaluator_module = load_evaluator_module()

    reference = evaluator_fingerprint.load_reference()
    current = evaluator_fingerprint.fingerprint(evaluator_module)
    problems = differences(reference, current)
    problems.extend(check_no_clipping(evaluator_module))

    if problems:
        print("EVALUATOR CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        raise SystemExit(1)

    trials = current["trials"]
    knocked = trials["reaches_then_ball_knocked_away"]
    print("EVALUATOR CHECK PASSED")
    print(f"  {len(trials)} scripted trials score identically to the reference")
    print("  fitness is returned raw and signed; clipping it gives the expected clamped values")
    print(
        "  closest-approach scoring holds: a robot that reached the ball and knocked it away "
        f"scores {knocked['progress_from_closest']:+.2f} progress, "
        f"not {knocked['progress_from_final']:+.2f} as final-state scoring gave"
    )


if __name__ == "__main__":
    main()
