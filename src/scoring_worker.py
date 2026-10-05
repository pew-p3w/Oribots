"""Entry point that runs inside a worker process: simulate one scene and score it.

Kept separate from ``evaluator.py`` on purpose. ``evaluator`` needs the run's
config at import time, and a freshly started worker process (macOS and Windows
start workers from scratch instead of copying the parent, as Linux does) does
not have it yet. This module has no such requirement: it loads the config first
and only then imports the evaluator.
"""

import importlib.util
import sys
from typing import Any


def _ensure_config(config_path: str) -> None:
    """
    Make sure the module named ``config`` is the run's config.

    A no-op when the worker was copied from a parent that already loaded it.

    :param config_path: Path of the run's config file.
    """
    module = sys.modules.get("config")
    if module is not None and hasattr(module, "BODY"):
        return
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = module
    spec.loader.exec_module(module)


def simulate_and_score(task: dict[str, Any]) -> tuple[float, dict[str, float] | None]:
    """
    Simulate one scene and score it, without sending the states back.

    :param task: Everything needed for one trial (see Evaluator).
    :returns: The trial fitness and, if requested, its movement record.
    """
    _ensure_config(task["config_path"])
    import evaluator

    return evaluator.simulate_and_score_scene(task)
