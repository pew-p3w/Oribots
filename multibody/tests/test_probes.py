"""Phase 1 gate: the six engine probes pass (R21, R23.7).

Probe 2 is judged on the robot construction the Match_Builder uses: one shared
body, falling back to per-robot copies only if sharing fails (R21.7). A
mutation (every robot sent robot 0's hinge targets) must make probe 2 fail, so
the probe is known to detect mixed-up commands.
"""

from _check import check_main, require

import arena_probes
import config_loader


def run() -> None:
    results = arena_probes.run_all()
    problems = []
    for name, failures in results.items():
        if name == "hinge_commands_shared":
            continue
        problems.extend(f"{name}: {failure}" for failure in failures)
    require(not problems, "probe failures:\n  " + "\n  ".join(problems))
    require(
        not results["hinge_commands_shared"],
        "robots sharing one body did not obey their own hinge commands; the Match_Builder "
        "default (shared body) must change to per_robot_body_copy=True",
    )

    settings = config_loader.load_contest(arena_probes.TINY_CONFIG)
    mutated = arena_probes.probe_hinge_commands(settings, _send_robot0_targets=True)
    require(bool(mutated), "the hinge probe did not notice robots receiving another robot's commands")


if __name__ == "__main__":
    check_main("PROBES", run)
