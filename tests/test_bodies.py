"""Check the body catalogue against its reference structures.

This test proves that no body in `config/bodies.py` has changed shape: every
body is rebuilt and compared, module by module, against `body_reference.json`.

Run with `make check-bodies`, or `python tests/test_bodies.py`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402

import body_fingerprint  # noqa: E402

CONFIG_DIR = paths.CONFIG


def load_bodies_module():
    """
    Import the catalogue the same way a config file does.

    Configs run with `config/` on `sys.path` and use `from bodies import ...`,
    so the test must import it as a bare top-level module. Importing it as
    `config.bodies` would pass here and still fail at runtime.

    :returns: The imported `bodies` module.
    :raises SystemExit: If the module cannot be imported as a bare module.
    """
    sys.path.insert(0, str(CONFIG_DIR))
    try:
        from bodies import all_bodies, get  # noqa: F401 - the import form is the assertion
    except ImportError as error:
        raise SystemExit(f"could not import `bodies` as a top-level module: {error}")
    return sys.modules["bodies"]


def main() -> None:
    """Compare the catalogue with the reference and report differences."""
    paths.install()
    bodies = load_bodies_module()

    reference = body_fingerprint.load_reference()
    current = body_fingerprint.fingerprint_catalogue(bodies)
    problems = body_fingerprint.compare(reference, current)

    # Every body must be reachable by name, and the parameter counts the brain
    # will derive from them must match the reference exactly.
    for name in sorted(current["bodies"]):
        short_name = name.removesuffix("_v2")
        try:
            bodies.get(short_name)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            problems.append(f"get({short_name!r}) failed: {error}")

    if problems:
        print("BODY CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    total_hinges = sum(fp["counts"]["active_hinges"] for fp in current["bodies"].values())
    print("BODY CHECK PASSED")
    print(f"  {len(current['bodies'])} bodies, structurally identical to the reference")
    print(f"  {total_hinges} active hinges and {len(current['module_geometry'])} distinct module geometries compared")
    for name in ("spider_v2", "gecko_v2"):
        counts = current["bodies"][name]["counts"]
        print(f"  {name:<12} {counts['active_hinges']} hinges -> {counts['controller_parameters']} controller parameters")


if __name__ == "__main__":
    main()
