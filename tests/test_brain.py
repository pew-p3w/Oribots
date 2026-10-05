"""Check the brain against its reference values.

`brains/ball_aware_brain.py` is the single controller shared by both
algorithms. This test drives it through a fixed scripted scenario and compares
every steering input, CPG state and hinge target against
`brain_reference.json`. It also covers the invariants that scenario does not reach.

Run with `make check-brain`, or `python tests/test_brain.py`.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import brain_fingerprint  # noqa: E402

ROOT = paths.ROOT


def load_modules():
    """
    Import the brain and the body catalogue the way the runner will.

    Both are imported as bare top-level modules, from `brains/` and `config/`.

    :returns: The brain module and the bodies module.
    :raises SystemExit: If either cannot be imported as a bare module.
    """
    sys.path.insert(0, str(ROOT / "config"))
    sys.path.insert(0, str(ROOT / "brains"))
    try:
        import ball_aware_brain
        import bodies
    except ImportError as error:
        raise SystemExit(f"could not import brain/bodies as top-level modules: {error}")
    return ball_aware_brain, bodies


def check_invariants(brain_module) -> list[str]:
    """
    Check brain properties the scripted scenario does not pin down.

    :param brain_module: The brain module.
    :returns: A list of problems; empty when all hold.
    """
    import numpy as np

    problems: list[str] = []

    # steering_parameter_count is one weight per hinge per input.
    for hinges, inputs in ((1, 4), (6, 4), (8, 3), (14, 4)):
        mapping = [(index, object()) for index in range(hinges)]
        expected = hinges * inputs
        actual = brain_module.steering_parameter_count(output_mapping=mapping, num_steering_inputs=inputs)
        if actual != expected:
            problems.append(f"steering_parameter_count({hinges} hinges, {inputs} inputs) = {actual}, expected {expected}")

    # _wrap_angle must map any angle into [-pi, pi].
    for angle in np.linspace(-8 * math.pi, 8 * math.pi, 401):
        wrapped = brain_module._wrap_angle(float(angle))
        if not -math.pi - 1e-9 <= wrapped <= math.pi + 1e-9:
            problems.append(f"_wrap_angle({angle}) = {wrapped}, outside [-pi, pi]")
            break
        if abs((math.cos(wrapped) - math.cos(angle))) > 1e-9 or abs(math.sin(wrapped) - math.sin(angle)) > 1e-9:
            problems.append(f"_wrap_angle({angle}) = {wrapped} is not the same angle")
            break

    # The distance input is normalized and clipped to [0, 1]; the first two
    # inputs are a sine/cosine pair; the fourth is a constant bias.
    from pyrr import Quaternion, Vector3

    robot, ball = object(), object()
    for distance in (0.0, 1.0, brain_fingerprint.DISTANCE_SCALE, brain_fingerprint.DISTANCE_SCALE * 5):
        for bearing in (0.0, 1.0, -2.5, 3.0):
            state = brain_fingerprint._SimulationState(
                {
                    id(robot): brain_fingerprint._Pose(Vector3([0.0, 0.0, 0.0]), Quaternion.from_z_rotation(bearing)),
                    id(ball): brain_fingerprint._Pose(Vector3([distance, 0.0, 0.3]), Quaternion()),
                }
            )
            sensor_state = brain_fingerprint._SensorState(state, brain_fingerprint._Mapping(robot))
            inputs = brain_module._ball_relative_inputs(
                sensor_state=sensor_state,
                ball=ball,
                distance_scale=brain_fingerprint.DISTANCE_SCALE,
            )
            if len(inputs) != 4:
                problems.append(f"_ball_relative_inputs returned {len(inputs)} values, expected 4")
                break
            if not 0.0 <= inputs[2] <= 1.0:
                problems.append(f"distance input {inputs[2]} not clipped to [0, 1] at distance {distance}")
            if abs(inputs[0] ** 2 + inputs[1] ** 2 - 1.0) > 1e-9:
                problems.append(f"first two inputs are not a unit sin/cos pair at bearing {bearing}")
            if inputs[3] != 1.0:
                problems.append(f"bias input is {inputs[3]}, expected 1.0")

    return problems


def main() -> None:
    """Compare the brain with the reference and report differences."""
    paths.install()
    brain_module, bodies_module = load_modules()

    reference = brain_fingerprint.load_reference()
    current = brain_fingerprint.fingerprint(brain_module, bodies_module)
    problems = brain_fingerprint.compare(reference, current)
    problems.extend(check_invariants(brain_module))

    if problems:
        print("BRAIN CHECK FAILED")
        for problem in problems[:25]:
            print(f"  - {problem}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        raise SystemExit(1)

    print("BRAIN CHECK PASSED")
    print(f"  {brain_fingerprint.value_count(current)} recorded values identical to the reference")
    print(f"  digest {brain_fingerprint.digest(current)}")
    for name in brain_fingerprint.BODIES:
        trace = current["bodies"][name]
        print(
            f"  {name:<16} {trace['num_active_hinges']:>2} hinges, {trace['num_parameters']:>3} parameters, "
            f"{brain_fingerprint.NUM_CONTROL_STEPS} control steps traced"
        )
    print("  invariants: parameter count, angle wrapping, input shape/clipping/bias all hold")


if __name__ == "__main__":
    main()
