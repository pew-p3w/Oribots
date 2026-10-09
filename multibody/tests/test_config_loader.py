"""The Config_Loader: accepts the fixture config, refuses bad ones with clear messages (R2-R4, R7, R14, R16).

Every refusal is checked for the setting it must name. The trained-weights
paths use the task 1.5 fixtures (a seeded vector, and a fake single-robot
snapshot carrying a real stored config), including the cross-check against
that stored config (R4.9). Also: installing the base config as the process's
``config`` module (R4.5) and rebuilding the settings from snapshot contents
alone (R17.3).
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from _check import check_main, require, temp_workspace

import config_loader
from config_loader import ConfigRefused

MULTIBODY = config_loader.contest_paths.MULTIBODY
FIXTURES = MULTIBODY / "tests" / "fixtures"
TINY = MULTIBODY / "configs" / "_tiny.py"


def tiny_source(**overrides) -> str:
    """The _tiny config with settings replaced (``None`` deletes one) and absolute fixture paths."""
    lines = []
    for line in TINY.read_text(encoding="utf-8").splitlines():
        name = line.split("=")[0].strip() if "=" in line and not line.startswith((" ", "#")) else None
        if name in overrides:
            continue
        lines.append(line)
    text = "\n".join(lines).replace('"../tests/fixtures/', f'"{FIXTURES}/')
    for name, value in overrides.items():
        if value is not None:
            text += f"\n{name} = {value!r}"
    return text + "\n"


def refused(workspace: Path, name: str, must_mention: list[str], **overrides) -> None:
    path = workspace / f"{name}.py"
    path.write_text(tiny_source(**overrides), encoding="utf-8")
    try:
        config_loader.load_contest(path)
    except ConfigRefused as error:
        text = str(error)
        missing = [word for word in must_mention if word not in text]
        require(not missing, f"{name}: refusal does not mention {missing}: {text}")
        return
    raise AssertionError(f"{name}: the config was accepted")


def run() -> None:
    settings = config_loader.load_contest(TINY)
    require(settings.k == 2 and settings.cma_population_size == 4 and settings.num_layouts == 2, "tiny settings")
    require(settings.simulation_time == 30.0 and 1 <= settings.num_generations <= 3, "tiny length")
    require(settings.trained_weights.shape == (44,), "44 trained weights")
    require(settings.base_config_name == "spider_8ball" and settings.opponents == "reference", "base / mode")
    require(settings.failed_match_score == -(2.0 + 1.0) - 1.0, f"D-13 default {settings.failed_match_score}")
    require(settings.possession_distance == settings.ball_reached_distance, "POSSESSION_DISTANCE default")
    require(settings.wall_height == 4 * settings.ball_radius and settings.wall_thickness == 0.2, "wall defaults")
    require(settings.feedback_distance_scale == settings.feedback_distance_scale and abs(settings.feedback_distance_scale - 20.6475) < 1e-3,
            f"FEEDBACK_DISTANCE_SCALE kept from the base config: {settings.feedback_distance_scale}")

    with temp_workspace() as workspace:
        bad_weights = workspace / "short.npy"
        np.save(bad_weights, np.zeros(43))
        nan_weights = workspace / "nan.npy"
        values = np.load(FIXTURES / "trained_weights_seed.npy")
        values[7] = np.nan
        np.save(nan_weights, values)
        refused(workspace, "bad_k", ["K", "{2, 3, 4, 5}"], K=6)
        refused(workspace, "bad_l", ["HIDDEN_SIZE", "6 to 10"], HIDDEN_SIZE=11)
        refused(workspace, "bad_opponents", ["OPPONENTS", "'reference', 'self', 'population'"], OPPONENTS="Reference")
        refused(workspace, "population", ["population", "not yet available"], OPPONENTS="population")
        refused(workspace, "no_head", ["ant", "Head_Coordinates are required", "USE_HEAD_COORDINATES"], BASE_CONFIG="ant")
        refused(workspace, "missing_base", ["no_such_base", "does not exist"], BASE_CONFIG="no_such_base")
        refused(workspace, "short_weights", ["44", "43"], TRAINED_WEIGHTS_FILE=str(bad_weights), TRAINED_WEIGHTS_FIELD=None)
        refused(workspace, "nan_weights", ["index 7"], TRAINED_WEIGHTS_FILE=str(nan_weights), TRAINED_WEIGHTS_FIELD=None)
        refused(workspace, "no_file", ["does not exist"], TRAINED_WEIGHTS_FILE=str(workspace / "gone.npy"))
        refused(workspace, "bad_field", ["nope", "best_ever_parameters", "best_parameters"], TRAINED_WEIGHTS_FIELD="nope")
        refused(workspace, "ring_through_wall", ["START_RADIUS", "TERRAIN_SIZE"], START_RADIUS=2.3)
        refused(workspace, "gaps_too_big", ["MIN_ARC_GAP", "circumference"], MIN_ARC_GAP=5.0)
        refused(workspace, "missing_settings", ["START_RADIUS", "NUM_GENERATIONS"], START_RADIUS=None, NUM_GENERATIONS=None)
        refused(workspace, "bad_bounds", ["CMA_BOUNDS"], CMA_BOUNDS=(0.0, 3.0))
        refused(workspace, "w1_outside_bounds", ["W1_INIT_STD", "CMA_BOUNDS"], W1_INIT_STD=5.0, CMA_BOUNDS=(-1.0, 1.0))
        refused(workspace, "bad_population", ["CMA_POPULATION_SIZE"], CMA_POPULATION_SIZE=1)
        refused(workspace, "bad_vref", ["V_REF"], V_REF=0.0)
        refused(workspace, "bad_most", ["MOST_POSSESSION_BONUS"], MOST_POSSESSION_BONUS=-1.0)
        refused(workspace, "bad_failed_fraction", ["MAX_FAILED_MATCH_FRACTION"], MAX_FAILED_MATCH_FRACTION=1.5)
        refused(workspace, "low_wall", ["WALL_HEIGHT", "ball diameter"], WALL_HEIGHT=0.5)

        # R4.9: a trained run whose stored config differs is refused, naming the setting.
        with open(FIXTURES / "fake_single_robot_gen1.pkl", "rb") as handle:
            snapshot = pickle.load(handle)
        snapshot["config_source"] = snapshot["config_source"].replace("FEEDBACK_OUTPUT_SCALE = 0.5", "FEEDBACK_OUTPUT_SCALE = 0.6")
        changed = workspace / "changed_gen1.pkl"
        with open(changed, "wb") as handle:
            pickle.dump(snapshot, handle)
        refused(workspace, "cross_check", ["FEEDBACK_OUTPUT_SCALE", "0.6", "0.5"], TRAINED_WEIGHTS_FILE=str(changed))
        snapshot["config_source"] = snapshot["config_source"].replace("FEEDBACK_OUTPUT_SCALE = 0.6", "FEEDBACK_OUTPUT_SCALE = 0.5").replace(
            "HEAD = BODY.core_pentagon.front_face.middle", "HEAD = BODY.core_pentagon.back_face.middle")
        with open(changed, "wb") as handle:
            pickle.dump(snapshot, handle)
        refused(workspace, "cross_check_head", ["HEAD", "back_face"], TRAINED_WEIGHTS_FILE=str(changed))

        # A .npy source loads with length and finiteness checks only (R4.10).
        npy = workspace / "npy.py"
        npy.write_text(tiny_source(TRAINED_WEIGHTS_FILE=str(FIXTURES / "trained_weights_seed.npy"), TRAINED_WEIGHTS_FIELD=None))
        require(config_loader.load_contest(npy).trained_weights.shape == (44,), ".npy source")

    # R4.5: the base config becomes the module named config, replacing another one.
    import types

    sys.modules["config"] = types.ModuleType("config")
    module = config_loader.install_base_config_as_config(settings.base_config_source, settings.base_config_name)
    require(sys.modules["config"] is module, "config not installed")
    require(config_loader.install_base_config_as_config(settings.base_config_source) is module, "not idempotent")
    import robot_frame

    require(tuple(robot_frame.reference_offset()) == settings.head_offset, "robot_frame does not see the head offset")

    # R17.3: settings rebuilt from snapshot contents alone match.
    fake = {
        "multibody_config_source": settings.multibody_config_source,
        "base_config_source": settings.base_config_source,
        "trained_weights": settings.trained_weights,
        "reference_layer": None,
        "settings_values": config_loader.settings_values(settings),
    }
    rebuilt = config_loader.settings_from_snapshot(fake)
    require(config_loader.settings_values(rebuilt) == config_loader.settings_values(settings), "rebuilt settings differ")


if __name__ == "__main__":
    check_main("CONFIG_LOADER", run)
