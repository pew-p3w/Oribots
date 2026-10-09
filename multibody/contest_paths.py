"""Make the single-robot modules and the multibody modules importable.

The multibody modules are plain top-level modules (``arena``, ``match``, ...)
that live in ``Oribots/multibody/``, next to the single-robot roots that
``src/paths.py`` manages (the six vendored engine packages, ``src/``,
``brains/`` and ``config/``). Every multibody entry point (the runner, the
training child, the workers, the tests) calls :func:`install` first, so all of
them import the same files the same way.

The multibody directory is never a package import (``multibody.arena``): the
training child and its workers get it through ``PYTHONPATH`` (see
:func:`child_environment`), exactly as the single-robot child gets ``src/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MULTIBODY = Path(__file__).resolve().parent
ROOT = MULTIBODY.parent
SRC = ROOT / "src"
CONFIG = ROOT / "config"


def _single_robot_paths():
    """
    Import the single-robot path authority (``src/paths.py``).

    :returns: The ``paths`` module.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import paths

    return paths


def install() -> None:
    """Put the engine, ``src/``, ``brains/``, ``config/`` and ``multibody/`` on ``sys.path``."""
    _single_robot_paths().install(CONFIG)
    if str(MULTIBODY) not in sys.path:
        sys.path.insert(0, str(MULTIBODY))


def child_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build the environment for a training child process.

    Same as the single-robot ``paths.child_environment`` (engine, ``src/``,
    ``brains/``, the base config directory ``config/``, no bytecode), with the
    multibody directory put first on ``PYTHONPATH``.

    :param base: Environment to extend. Defaults to the current one.
    :returns: The environment to hand to the child process.
    """
    environment = _single_robot_paths().child_environment(config_dir=CONFIG, base=base)
    environment["PYTHONPATH"] = os.pathsep.join([str(MULTIBODY), environment["PYTHONPATH"]])
    return environment
