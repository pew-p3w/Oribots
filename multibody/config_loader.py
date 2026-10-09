"""Load and validate a multibody contest config (the Config_Loader).

A multibody config is a plain Python file in ``multibody/configs/``. It names
one single-robot *base config* in ``Oribots/config/`` (body, head, ball and
feedback settings, taken unchanged) and one *trained-weights source* (a
single-robot ``gen<N>.pkl`` field or a ``.npy`` file), plus the contest settings.
Everything is validated before any match runs or any file is written; a problem
raises :class:`ConfigRefused` with a message naming the setting.

Two ways in:

- :func:`load_contest` reads the files (a new run, the pilot);
- :func:`settings_from_snapshot` rebuilds the same settings from a multibody
  snapshot alone (resume, playback), never re-reading the original files (R17.3).

The base config is executed under a private module name, never imported as
``config``. :func:`install_base_config_as_config` then registers that same module
as ``config`` in a process that builds or scores scenes, because the single-robot
``robot_frame`` helpers read the process-global ``config`` (R4.5).
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import pickle
import sys
import types
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()

import arena  # noqa: E402

# The algorithm name every multibody snapshot carries (R17.2). It is neither
# "ea" nor "cmaes", so the single-robot run.py refuses these snapshots (R17.6).
MULTIBODY_ALGORITHM = "multibody_cmaes"

OPPONENT_MODES = ("reference", "self", "population")
ALLOWED_K = (2, 3, 4, 5)
HIDDEN_SIZE_RANGE = (6, 10)

# Settings the Base_Config must define (R2.1); taken from it unchanged.
BASE_SETTINGS = (
    "BODY",
    "HEAD",
    "HEAD_OFFSET",
    "USE_HEAD_COORDINATES",
    "BALL_RADIUS",
    "BALL_MASS",
    "BALL_REACHED_DISTANCE",
    "FEEDBACK_NUM_INPUTS",
    "FEEDBACK_OUTPUT_SCALE",
    "FEEDBACK_DISTANCE_SCALE",
)

# Settings with no default: a config that leaves one out is refused (R2.11).
REQUIRED_SETTINGS = (
    "BASE_CONFIG",
    "TRAINED_WEIGHTS_FILE",
    "K",
    "TERRAIN_SIZE",
    "START_RADIUS",
    "MIN_ARC_GAP",
    "NUM_LAYOUTS",
    "LAYOUT_SEED",
    "LAYER_OUTPUT_SCALE",
    "W1_INIT_STD",
    "W1_INIT_SEED",
    "CMA_INITIAL_STD",
    "CMA_BOUNDS",
    "NUM_GENERATIONS",
)

# Settings the trained weights' stored single-robot config must agree on with the
# Base_Config (R4.9).
CROSS_CHECKED_SETTINGS = (
    "USE_HEAD_COORDINATES",
    "HEAD_OFFSET",
    "FEEDBACK_DISTANCE_SCALE",
    "FEEDBACK_NUM_INPUTS",
    "FEEDBACK_OUTPUT_SCALE",
    "BALL_RADIUS",
)

_BASE_MODULE_PREFIX = "_multibody_base_config_"
_CONTEST_MODULE_NAME = "_multibody_contest_config"


class ConfigRefused(Exception):
    """A multibody config, base config or trained-weights source was refused."""


@dataclass(frozen=True, eq=False)
class ContestSettings:
    """Everything one contest run needs, validated (frozen interface, design.md)."""

    # identity
    base_config_name: str
    base_config_source: str
    multibody_config_source: str
    trained_weights: np.ndarray
    trained_weights_source: str
    # body / brain (taken from Base_Config, R2.1)
    body: Any
    head_offset: tuple[float, float, float]
    feedback_num_inputs: int
    feedback_output_scale: float
    feedback_distance_scale: float
    ball_radius: float
    ball_mass: float
    ball_reached_distance: float
    num_active_hinges: int
    cpg_plus_steering_count: int
    # contest
    k: int
    hidden_size: int
    opponents: str
    reference_layer: np.ndarray | None
    layer_output_scale: float
    # arena / layouts
    terrain_size: tuple[float, float]
    wall_height: float
    wall_thickness: float
    start_radius: float
    min_arc_gap: float
    num_layouts: int
    layout_seed: int
    spawn_height_tolerance: float
    footprint_radius: float
    standing_height: float
    # match / fitness
    simulation_time: float
    playback_simulation_time: float
    possession_distance: float
    v_ref: float
    time_bonus_cap: float
    possession_weight: float
    possession_penalty: float
    first_reach_bonus: float
    most_possession_bonus: float
    failed_match_score: float
    max_failed_match_fraction: float
    # optimizer
    cma_population_size: int
    cma_initial_std: float
    cma_bounds: tuple[float, float]
    num_generations: int
    w1_init_std: float
    w1_init_seed: int
    worker_count: int

    @property
    def layer_param_count(self) -> int:
        """4L + nL, the length of one Layer_Parameters vector (R5.3)."""
        return 4 * self.hidden_size + self.num_active_hinges * self.hidden_size


# Names of the plain (non-array, non-object) settings, stored in every snapshot.
SCALAR_FIELDS = tuple(
    f.name
    for f in fields(ContestSettings)
    if f.name
    not in {
        "base_config_source",
        "multibody_config_source",
        "trained_weights",
        "body",
        "reference_layer",
    }
)


def settings_values(settings: ContestSettings) -> dict[str, Any]:
    """
    The plain settings of a run as a dict of builtins (for a snapshot, R17.2/R17.7).

    :param settings: The settings.
    :returns: Field name to value (tuples become lists).
    """
    values = {}
    for name in SCALAR_FIELDS:
        value = getattr(settings, name)
        values[name] = list(value) if isinstance(value, tuple) else value
    return values


# --------------------------------------------------------------------------- #
# Body wiring shared by every component
# --------------------------------------------------------------------------- #


def body_wiring(body: Any) -> tuple[Any, list[Any]]:
    """
    The CPG network structure and hinge output mapping of a body, exactly as the
    single-robot evaluator builds them, so trained weights line up row by row.

    :param body: The robot body.
    :returns: (cpg_network_structure, output_mapping).
    """
    from revolve2.modular_robot.body.base import ActiveHinge
    from revolve2.modular_robot.brain.cpg import (
        active_hinges_to_cpg_network_structure_neighbor,
    )

    return active_hinges_to_cpg_network_structure_neighbor(
        body.find_modules_of_type(ActiveHinge)
    )


# --------------------------------------------------------------------------- #
# Base config: private load, and installing it as ``config`` (R4.5)
# --------------------------------------------------------------------------- #

_BASE_MODULES: dict[str, types.ModuleType] = {}


def _source_key(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def load_base_module(name: str, source: str, origin: str | None = None) -> types.ModuleType:
    """
    Execute a base config's source under a private module name, once per process.

    :param name: The base config's name (for messages).
    :param source: Its source text.
    :param origin: The file it came from, if any (becomes ``__file__``).
    :returns: The module.
    :raises ConfigRefused: If the source fails to execute.
    """
    key = _source_key(source)
    module = _BASE_MODULES.get(key)
    if module is not None:
        return module
    module_name = _BASE_MODULE_PREFIX + key
    module = types.ModuleType(module_name)
    module.__file__ = origin or f"<stored base config {name}>"
    sys.modules[module_name] = module
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    except Exception as error:  # noqa: BLE001 - any failure refuses the config
        del sys.modules[module_name]
        raise ConfigRefused(
            f"Base_Config '{name}' failed to load: {type(error).__name__}: {error}"
        ) from error
    _BASE_MODULES[key] = module
    return module


def install_base_config_as_config(base_config_source: str, base_config_name: str = "base") -> types.ModuleType:
    """
    Make the module named ``config`` hold the Base_Config, in this process (R4.5).

    Idempotent: the same source is executed once per process and the same module
    object is registered, replacing any other module named ``config``.

    :param base_config_source: The Base_Config's source text.
    :param base_config_name: Its name (for messages).
    :returns: The installed module.
    """
    module = load_base_module(base_config_name, base_config_source)
    if sys.modules.get("config") is not module:
        sys.modules["config"] = module
    return module


# --------------------------------------------------------------------------- #
# Small validators
# --------------------------------------------------------------------------- #


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _positive(name: str, value: Any, problems: list[str]) -> None:
    if not (_finite(value) and float(value) > 0):
        problems.append(f"{name} = {value!r}: must be a finite number greater than 0")


def _int_at_least(name: str, value: Any, minimum: int, problems: list[str]) -> None:
    if not (_is_int(value) and int(value) >= minimum):
        problems.append(f"{name} = {value!r}: must be an integer of at least {minimum}")


def _finite_number(name: str, value: Any, problems: list[str]) -> None:
    if not _finite(value):
        problems.append(f"{name} = {value!r}: must be a finite number")


def default_worker_count() -> int:
    """
    D-7: SLURM_NTASKS, else NCPUS, else min(CPU count, 8), as the single-robot configs.

    :returns: The worker count.
    """
    return int(
        os.environ.get("SLURM_NTASKS") or os.environ.get("NCPUS") or min(os.cpu_count() or 1, 8)
    )


# --------------------------------------------------------------------------- #
# Trained weights (R4)
# --------------------------------------------------------------------------- #


def _load_pickle(path: Path) -> Any:
    try:
        import cma  # noqa: F401 - a CMA-ES snapshot's optimizer state needs it
    except ImportError:  # pragma: no cover - cma is pinned in requirements.txt
        pass
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_trained_weights(file: Path, field: str | None) -> tuple[np.ndarray, str, str | None]:
    """
    Load the frozen controller's parameter vector (R4.1, R4.6, R4.7).

    :param file: A single-robot ``gen<N>.pkl`` snapshot or a ``.npy`` file.
    :param field: The snapshot field to read (required for a snapshot, unused for ``.npy``).
    :returns: (vector, description of the source, the snapshot's stored config source or None).
    :raises ConfigRefused: If the file or field is missing or unreadable, or a value is not finite.
    """
    if not file.is_file():
        raise ConfigRefused(f"Trained_Weights file does not exist: {file}")
    stored_config_source = None
    if file.suffix == ".npy":
        try:
            weights = np.load(file, allow_pickle=False)
        except Exception as error:  # noqa: BLE001
            raise ConfigRefused(f"Trained_Weights file {file} cannot be read: {error}") from error
        source = str(file)
    else:
        if not field:
            raise ConfigRefused(
                f"Trained_Weights file {file} is a snapshot, but no TRAINED_WEIGHTS_FIELD "
                "names which parameter field to use"
            )
        try:
            snapshot = _load_pickle(file)
        except Exception as error:  # noqa: BLE001
            raise ConfigRefused(f"Trained_Weights file {file} cannot be read: {error}") from error
        if not isinstance(snapshot, dict):
            raise ConfigRefused(f"Trained_Weights file {file} is not a single-robot snapshot")
        if field not in snapshot:
            options = sorted(name for name in snapshot if str(name).endswith("parameters"))
            raise ConfigRefused(
                f"Trained_Weights file {file} has no field '{field}'. "
                f"Fields ending in 'parameters': {options}"
            )
        weights = snapshot[field]
        stored = snapshot.get("config_source")
        stored_config_source = stored if isinstance(stored, str) else None
        source = f"{file} [{field}]"
    weights = np.asarray(weights)
    if weights.dtype.kind not in "fiu":
        raise ConfigRefused(f"Trained_Weights from {source} are not numeric (dtype {weights.dtype})")
    weights = weights.astype(np.float64)
    if weights.ndim == 1:
        bad = np.flatnonzero(~np.isfinite(weights))
        if bad.size:
            raise ConfigRefused(
                f"Trained_Weights from {source} hold a non-finite value at index {int(bad[0])}"
            )
    return weights, source, stored_config_source


def _evaluate_assignments(source: str, names: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Evaluate a config source statement by statement, skipping statements that fail.

    A stored single-robot config may import modules that only existed in the
    repository it ran in; those statements (and anything depending on them) are
    skipped rather than failing the whole check.

    :param source: Config source text.
    :param names: Top-level names of interest.
    :returns: (values that could be evaluated, assignment expression text per name).
    """
    tree = ast.parse(source)
    namespace: dict[str, Any] = {"__name__": "_multibody_cross_check"}
    expressions = _assignment_expressions(source)
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        try:
            exec(compile(ast.Module(body=[statement], type_ignores=[]), "<stored config>", "exec"), namespace)  # noqa: S102
        except Exception:  # noqa: BLE001 - see docstring
            continue
    return {name: namespace[name] for name in names if name in namespace}, expressions


def _call_name(expression: str) -> str | None:
    """The function name of a call expression (``a.b.f()`` -> ``f``), else None."""
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
    return None


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if _is_number(a) and _is_number(b):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    try:
        aa, bb = tuple(float(v) for v in a), tuple(float(v) for v in b)
    except (TypeError, ValueError):
        return a == b
    return len(aa) == len(bb) and all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(aa, bb))


def cross_check_trained_config(
    stored_source: str, base_name: str, base_source: str, base_module: types.ModuleType
) -> list[str]:
    """
    Compare the trained run's stored config with the Base_Config (R4.9).

    Values are compared where the stored source can be evaluated here. Where it
    cannot (for example ``HEAD_OFFSET``, computed from a body module that only
    existed in the repository the run trained in), the assignment expressions of
    the setting and of what it depends on (``HEAD``) are compared instead, and the
    body is compared by the name of the function that builds it.

    :returns: One message per mismatch; empty when the two describe the same controller.
    """
    stored_values, stored_expressions = _evaluate_assignments(stored_source, CROSS_CHECKED_SETTINGS)
    base_expressions = _assignment_expressions(base_source)
    problems = []
    for name in CROSS_CHECKED_SETTINGS:
        base_value = getattr(base_module, name, None)
        if name in stored_values:
            if not _same_value(stored_values[name], base_value):
                problems.append(
                    f"{name}: trained run used {stored_values[name]!r}, Base_Config '{base_name}' "
                    f"has {base_value!r}"
                )
            continue
        stored_expression = stored_expressions.get(name)
        if stored_expression is None:
            problems.append(f"{name}: the trained run's stored config does not set it")
            continue
        dependencies = [name] + (["HEAD"] if name == "HEAD_OFFSET" else [])
        for dependency in dependencies:
            if stored_expressions.get(dependency) != base_expressions.get(dependency):
                problems.append(
                    f"{dependency}: trained run has '{stored_expressions.get(dependency)}', "
                    f"Base_Config '{base_name}' has '{base_expressions.get(dependency)}' "
                    "(compared as expressions: the stored value cannot be evaluated here)"
                )
    stored_body = _call_name(stored_expressions.get("BODY", ""))
    base_body = _call_name(base_expressions.get("BODY", ""))
    if stored_body is not None and base_body is not None and stored_body != base_body:
        problems.append(
            f"BODY: trained run built '{stored_body}', Base_Config '{base_name}' builds '{base_body}'"
        )
    return problems


def _assignment_expressions(source: str) -> dict[str, str]:
    expressions = {}
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    expressions[target.id] = ast.unparse(statement.value)
    return expressions


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _exec_contest_source(source: str, origin: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": _CONTEST_MODULE_NAME, "__file__": origin}
    try:
        exec(compile(source, origin, "exec"), namespace)  # noqa: S102
    except Exception as error:  # noqa: BLE001
        raise ConfigRefused(f"Multibody_Config {origin} failed to load: {type(error).__name__}: {error}") from error
    return namespace


def load_contest(config_path: str | Path) -> ContestSettings:
    """
    Load and validate a multibody config from its files (a new run).

    :param config_path: The multibody config file.
    :returns: The validated settings.
    :raises ConfigRefused: On any problem, before any match or file is written.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise ConfigRefused(f"Multibody_Config does not exist: {path}")
    source = path.read_text(encoding="utf-8")
    contest = _exec_contest_source(source, str(path))
    _require_settings(contest, path)
    base_name = contest["BASE_CONFIG"]
    base_path = contest_paths.CONFIG / f"{base_name}.py"
    if not (isinstance(base_name, str) and base_path.is_file()):
        raise ConfigRefused(f"Base_Config '{base_name}' does not exist in {contest_paths.CONFIG}")
    base_source = base_path.read_text(encoding="utf-8")
    base_module = load_base_module(base_name, base_source, str(base_path))

    weights_file = Path(str(contest["TRAINED_WEIGHTS_FILE"]))
    if not weights_file.is_absolute():
        weights_file = (path.parent / weights_file).resolve()
    weights, weights_source, stored_config = load_trained_weights(
        weights_file, contest.get("TRAINED_WEIGHTS_FIELD")
    )
    reference_layer = _load_reference_layer(contest, path)
    settings = _build_settings(
        contest=contest,
        contest_source=source,
        base_name=base_name,
        base_source=base_source,
        base_module=base_module,
        trained_weights=weights,
        trained_weights_source=weights_source,
        reference_layer=reference_layer,
    )
    if stored_config is not None:
        problems = cross_check_trained_config(stored_config, base_name, base_source, base_module)
        if problems:
            raise ConfigRefused(
                f"Trained_Weights {weights_source} were trained with a config that does not "
                f"match Base_Config '{base_name}':\n  " + "\n  ".join(problems)
            )
    return settings


def settings_from_snapshot(snapshot: dict[str, Any]) -> ContestSettings:
    """
    Rebuild a run's settings from a multibody snapshot alone (R17.3).

    The stored sources, trained weights and reference layer are used; the
    original config, base config and weights files are never read.

    :param snapshot: A loaded multibody snapshot.
    :returns: The settings.
    :raises ConfigRefused: If the stored settings no longer validate with this code.
    """
    contest = _exec_contest_source(snapshot["multibody_config_source"], "<stored multibody config>")
    base_name = snapshot["settings_values"]["base_config_name"]
    base_source = snapshot["base_config_source"]
    base_module = load_base_module(base_name, base_source)
    reference = snapshot.get("reference_layer")
    return _build_settings(
        contest=contest,
        contest_source=snapshot["multibody_config_source"],
        base_name=base_name,
        base_source=base_source,
        base_module=base_module,
        trained_weights=np.asarray(snapshot["trained_weights"], dtype=np.float64),
        trained_weights_source=snapshot["settings_values"]["trained_weights_source"],
        reference_layer=None if reference is None else np.asarray(reference, dtype=np.float64),
        skip_requirements=("TRAINED_WEIGHTS_FILE",),
    )


def _require_settings(contest: dict[str, Any], origin: Any, skip: tuple[str, ...] = ()) -> None:
    missing = [name for name in REQUIRED_SETTINGS if name not in skip and name not in contest]
    if missing:
        raise ConfigRefused(
            f"Multibody_Config {origin} does not set: {', '.join(missing)} (these have no default)"
        )


def _load_reference_layer(contest: dict[str, Any], path: Path) -> np.ndarray | None:
    """
    The Reference_Layer the Opponents use, if the config names one (R14.4, R14.10).

    ``REFERENCE_LAYER = {"snapshot": <multibody gen<N>.pkl>, "which": "best" | "best_ever"}``.

    :returns: The layer, or None for the Zero_Output_Layer.
    """
    spec = contest.get("REFERENCE_LAYER")
    if spec is None:
        return None
    if not isinstance(spec, dict) or "snapshot" not in spec:
        raise ConfigRefused(
            "REFERENCE_LAYER must be None or {'snapshot': <multibody gen<N>.pkl>, 'which': 'best' | 'best_ever'}"
        )
    snapshot_path = Path(str(spec["snapshot"]))
    if not snapshot_path.is_absolute():
        snapshot_path = (path.parent / snapshot_path).resolve()
    try:
        snapshot = _load_pickle(snapshot_path)
    except Exception as error:  # noqa: BLE001
        raise ConfigRefused(f"REFERENCE_LAYER snapshot {snapshot_path} cannot be read: {error}") from error
    if not isinstance(snapshot, dict) or snapshot.get("algorithm") != MULTIBODY_ALGORITHM:
        found = snapshot.get("algorithm") if isinstance(snapshot, dict) else type(snapshot).__name__
        raise ConfigRefused(
            f"REFERENCE_LAYER snapshot {snapshot_path} is not a multibody snapshot "
            f"(algorithm {found!r}, expected {MULTIBODY_ALGORITHM!r})"
        )
    which = spec.get("which", "best_ever")
    key = {"best": "best_layer_params", "best_ever": "best_ever_layer_params"}.get(which)
    if key is None:
        raise ConfigRefused(f"REFERENCE_LAYER 'which' = {which!r}: must be 'best' or 'best_ever'")
    return np.asarray(snapshot[key], dtype=np.float64)


def _build_settings(
    *,
    contest: dict[str, Any],
    contest_source: str,
    base_name: str,
    base_source: str,
    base_module: types.ModuleType,
    trained_weights: np.ndarray,
    trained_weights_source: str,
    reference_layer: np.ndarray | None,
    skip_requirements: tuple[str, ...] = (),
) -> ContestSettings:
    """Validate everything and assemble the settings (shared by both ways in)."""
    _require_settings(contest, contest.get("__file__"), skip_requirements)

    # Head coordinates first (R3.1), then the other Base_Config contents (R2.1, R2.12).
    _require_head_coordinates(base_name, base_module)
    missing_base = [name for name in BASE_SETTINGS if not hasattr(base_module, name)]
    if missing_base:
        raise ConfigRefused(f"Base_Config '{base_name}' does not define: {', '.join(missing_base)}")
    if base_module.FEEDBACK_NUM_INPUTS != 4:
        raise ConfigRefused(
            f"Base_Config '{base_name}' sets FEEDBACK_NUM_INPUTS = {base_module.FEEDBACK_NUM_INPUTS!r}; "
            "the contest brain needs exactly 4"
        )
    head_offset = tuple(float(v) for v in base_module.HEAD_OFFSET)
    body = base_module.BODY
    cpg_structure, output_mapping = body_wiring(body)
    num_hinges = len(output_mapping)
    expected_weights = cpg_structure.num_connections + num_hinges * 4

    # Trained weights (R4.2).
    if trained_weights.ndim != 1 or trained_weights.size != expected_weights:
        raise ConfigRefused(
            f"Trained_Weights from {trained_weights_source}: expected a 1-D vector of "
            f"{expected_weights} values (CPG + steering for Base_Config '{base_name}'), "
            f"loaded {trained_weights.size} values with shape {tuple(trained_weights.shape)}"
        )

    problems: list[str] = []
    get = contest.get
    # Contest settings (R2.3-2.7).
    k = get("K")
    if not (_is_int(k) and int(k) in ALLOWED_K):
        problems.append(f"K = {k!r}: must be an integer in {{2, 3, 4, 5}}")
    hidden = get("HIDDEN_SIZE", 8)
    if not (_is_int(hidden) and HIDDEN_SIZE_RANGE[0] <= int(hidden) <= HIDDEN_SIZE_RANGE[1]):
        problems.append(f"HIDDEN_SIZE (L) = {hidden!r}: must be an integer from 6 to 10 inclusive")
    opponents = get("OPPONENTS", "reference")
    if not (isinstance(opponents, str) and opponents in OPPONENT_MODES):
        problems.append(f"OPPONENTS = {opponents!r}: must be one of 'reference', 'self', 'population'")
    if problems:
        raise ConfigRefused("Multibody_Config refused:\n  " + "\n  ".join(problems))
    if opponents == "population":
        raise ConfigRefused(
            "OPPONENTS = 'population': the population opponent mode is planned but not yet available"
        )

    ball_radius = float(base_module.BALL_RADIUS)
    terrain = get("TERRAIN_SIZE")
    try:
        terrain_size = (float(terrain[0]), float(terrain[1]))
    except (TypeError, ValueError, IndexError):
        terrain_size = None
        problems.append(f"TERRAIN_SIZE = {terrain!r}: must be two numbers (x, y)")
    if terrain_size is not None:
        for side in terrain_size:
            if not (math.isfinite(side) and side > 0):
                problems.append(f"TERRAIN_SIZE = {terrain!r}: both sides must be finite numbers greater than 0")
                break
    wall_height = get("WALL_HEIGHT", 4.0 * ball_radius)
    wall_thickness = get("WALL_THICKNESS", 0.2)
    _positive("WALL_HEIGHT", wall_height, problems)
    _positive("WALL_THICKNESS", wall_thickness, problems)
    for name in ("START_RADIUS", "LAYER_OUTPUT_SCALE", "CMA_INITIAL_STD", "W1_INIT_STD"):
        _positive(name, get(name), problems)
    _finite_number("MIN_ARC_GAP", get("MIN_ARC_GAP"), problems)
    if _finite(get("MIN_ARC_GAP")) and float(get("MIN_ARC_GAP")) < 0:
        problems.append(f"MIN_ARC_GAP = {get('MIN_ARC_GAP')!r}: must not be negative")
    _int_at_least("NUM_LAYOUTS", get("NUM_LAYOUTS"), 1, problems)
    _int_at_least("LAYOUT_SEED", get("LAYOUT_SEED"), 0, problems)
    _int_at_least("W1_INIT_SEED", get("W1_INIT_SEED"), 0, problems)
    _int_at_least("NUM_GENERATIONS", get("NUM_GENERATIONS"), 1, problems)
    population = get("CMA_POPULATION_SIZE", 24)
    _int_at_least("CMA_POPULATION_SIZE", population, 2, problems)
    workers = get("NUM_WORKERS", None)
    if workers is None:
        workers = default_worker_count()
    _int_at_least("NUM_WORKERS", workers, 1, problems)
    spawn_tolerance = get("SPAWN_HEIGHT_TOLERANCE", 0.01)
    _positive("SPAWN_HEIGHT_TOLERANCE", spawn_tolerance, problems)
    simulation_time = get("SIMULATION_TIME", 700)
    _positive("SIMULATION_TIME", simulation_time, problems)
    playback_time = get("PLAYBACK_SIMULATION_TIME", simulation_time)
    _positive("PLAYBACK_SIMULATION_TIME", playback_time, problems)
    possession_distance = get("POSSESSION_DISTANCE", float(base_module.BALL_REACHED_DISTANCE))
    _positive("POSSESSION_DISTANCE", possession_distance, problems)
    v_ref = get("V_REF", 0.01)
    _positive("V_REF", v_ref, problems)
    time_bonus_cap = get("TIME_BONUS_CAP", 2.0)
    _finite_number("TIME_BONUS_CAP", time_bonus_cap, problems)
    possession_weight = get("POSSESSION_WEIGHT", 3.0)
    possession_penalty = get("POSSESSION_PENALTY", 1.0)
    first_reach_bonus = get("FIRST_REACH_BONUS", 0.5)
    for name, value in (
        ("POSSESSION_WEIGHT", possession_weight),
        ("POSSESSION_PENALTY", possession_penalty),
        ("FIRST_REACH_BONUS", first_reach_bonus),
    ):
        _finite_number(name, value, problems)
    most_bonus = get("MOST_POSSESSION_BONUS", 2.0)
    if not (_finite(most_bonus) and float(most_bonus) >= 0):
        problems.append(f"MOST_POSSESSION_BONUS = {most_bonus!r}: must be a finite number of at least 0")
    failed_score = get("FAILED_MATCH_SCORE", None)
    if failed_score is None and _finite(time_bonus_cap) and _finite(possession_penalty):
        failed_score = -(float(time_bonus_cap) + float(possession_penalty)) - 1.0
    _finite_number("FAILED_MATCH_SCORE", failed_score, problems)
    max_failed = get("MAX_FAILED_MATCH_FRACTION", 0.5)
    if not (_finite(max_failed) and 0.0 <= float(max_failed) <= 1.0):
        problems.append(f"MAX_FAILED_MATCH_FRACTION = {max_failed!r}: must be a number in [0, 1]")
    bounds = get("CMA_BOUNDS")
    try:
        lower, upper = float(bounds[0]), float(bounds[1])
        if len(bounds) != 2 or not (math.isfinite(lower) and math.isfinite(upper) and lower < 0 < upper):
            raise ValueError
        cma_bounds = (lower, upper)
    except (TypeError, ValueError, IndexError):
        cma_bounds = None
        problems.append(f"CMA_BOUNDS = {bounds!r}: must be two finite numbers with lower < 0 < upper")
    if problems:
        raise ConfigRefused("Multibody_Config refused:\n  " + "\n  ".join(problems))

    # Walls must hold the ball and the robots (R6.5).
    footprint = arena.footprint_radius(body)
    height = arena.standing_height(body)
    if float(wall_height) <= 2.0 * ball_radius or float(wall_height) <= height:
        raise ConfigRefused(
            f"WALL_HEIGHT = {wall_height} m must exceed both the ball diameter "
            f"({2.0 * ball_radius} m) and the body's standing height ({height:.4f} m)"
        )

    # Layout feasibility before sampling (R7.6, R7.10).
    layout_problems = arena.feasibility_problems(
        k=int(k),
        terrain_size=terrain_size,
        start_radius=float(get("START_RADIUS")),
        min_arc_gap=float(get("MIN_ARC_GAP")),
        footprint=footprint,
        ball_radius=ball_radius,
        head_offset=head_offset,
        possession_distance=float(possession_distance),
    )
    if layout_problems:
        raise ConfigRefused("Start layouts are infeasible:\n  " + "\n  ".join(layout_problems))

    layer_count = 4 * int(hidden) + num_hinges * int(hidden)
    if reference_layer is not None:
        if reference_layer.ndim != 1 or reference_layer.size != layer_count:
            raise ConfigRefused(
                f"REFERENCE_LAYER: expected {layer_count} values (4L + nL with L={hidden}, n={num_hinges}), "
                f"got {reference_layer.size}"
            )

    # The initial W1 must sit inside the CMA-ES bounds (R16.7).
    w1 = initial_w1(int(hidden), float(get("W1_INIT_STD")), int(get("W1_INIT_SEED")))
    if np.any(w1 < cma_bounds[0]) or np.any(w1 > cma_bounds[1]):
        raise ConfigRefused(
            f"The initial W1 drawn with W1_INIT_STD = {get('W1_INIT_STD')} and W1_INIT_SEED = "
            f"{get('W1_INIT_SEED')} leaves the CMA_BOUNDS {list(cma_bounds)} "
            f"(range {float(w1.min()):.4f} .. {float(w1.max()):.4f})"
        )

    return ContestSettings(
        base_config_name=base_name,
        base_config_source=base_source,
        multibody_config_source=contest_source,
        trained_weights=trained_weights,
        trained_weights_source=trained_weights_source,
        body=body,
        head_offset=head_offset,
        feedback_num_inputs=int(base_module.FEEDBACK_NUM_INPUTS),
        feedback_output_scale=float(base_module.FEEDBACK_OUTPUT_SCALE),
        feedback_distance_scale=float(base_module.FEEDBACK_DISTANCE_SCALE),
        ball_radius=ball_radius,
        ball_mass=float(base_module.BALL_MASS),
        ball_reached_distance=float(base_module.BALL_REACHED_DISTANCE),
        num_active_hinges=num_hinges,
        cpg_plus_steering_count=expected_weights,
        k=int(k),
        hidden_size=int(hidden),
        opponents=opponents,
        reference_layer=reference_layer,
        layer_output_scale=float(get("LAYER_OUTPUT_SCALE")),
        terrain_size=terrain_size,
        wall_height=float(wall_height),
        wall_thickness=float(wall_thickness),
        start_radius=float(get("START_RADIUS")),
        min_arc_gap=float(get("MIN_ARC_GAP")),
        num_layouts=int(get("NUM_LAYOUTS")),
        layout_seed=int(get("LAYOUT_SEED")),
        spawn_height_tolerance=float(spawn_tolerance),
        footprint_radius=float(footprint),
        standing_height=float(height),
        simulation_time=float(simulation_time),
        playback_simulation_time=float(playback_time),
        possession_distance=float(possession_distance),
        v_ref=float(v_ref),
        time_bonus_cap=float(time_bonus_cap),
        possession_weight=float(possession_weight),
        possession_penalty=float(possession_penalty),
        first_reach_bonus=float(first_reach_bonus),
        most_possession_bonus=float(most_bonus),
        failed_match_score=float(failed_score),
        max_failed_match_fraction=float(max_failed),
        cma_population_size=int(population),
        cma_initial_std=float(get("CMA_INITIAL_STD")),
        cma_bounds=cma_bounds,
        num_generations=int(get("NUM_GENERATIONS")),
        w1_init_std=float(get("W1_INIT_STD")),
        w1_init_seed=int(get("W1_INIT_SEED")),
        worker_count=int(workers),
    )


def _require_head_coordinates(base_name: str, base_module: types.ModuleType) -> None:
    """
    R3.1: the Base_Config must measure from a head on the robot's +x axis.

    :raises ConfigRefused: Naming the Base_Config and the missing or invalid setting.
    """
    use = getattr(base_module, "USE_HEAD_COORDINATES", None)
    head = getattr(base_module, "HEAD_OFFSET", None)
    problem = None
    if use is not True:
        problem = f"USE_HEAD_COORDINATES is {use!r}, must be True" if hasattr(base_module, "USE_HEAD_COORDINATES") else "USE_HEAD_COORDINATES is not set"
    elif head is None:
        problem = "HEAD_OFFSET is not set"
    else:
        try:
            values = tuple(float(v) for v in head)
        except (TypeError, ValueError):
            values = ()
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            problem = f"HEAD_OFFSET = {head!r} is not three finite numbers"
        elif not (values[0] > 0 and abs(values[1]) <= 1e-3):
            problem = f"HEAD_OFFSET = {head!r} is not on the robot's +x heading axis (x > 0, |y| <= 0.001)"
    if problem is not None:
        raise ConfigRefused(
            f"Base_Config '{base_name}': Head_Coordinates are required (a head the robot measures "
            f"the ball from); {problem}"
        )


def initial_w1(hidden_size: int, std: float, seed: int) -> np.ndarray:
    """
    The seeded initial W1 values of the CMA-ES mean (R16.2), as a flat L*4 vector.

    :param hidden_size: L.
    :param std: W1_INIT_STD.
    :param seed: W1_INIT_SEED.
    :returns: L*4 values, bit-identical for the same arguments.
    """
    return np.random.default_rng(int(seed)).normal(0.0, float(std), size=4 * int(hidden_size))
