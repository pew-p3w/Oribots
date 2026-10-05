"""Where the project's modules live, and how to make them importable.

Everything the project imports comes from inside this repository, and this
module is the single place that says where from. The runner, both algorithms
and the tests all call it, so the import path is defined once.

Two facts make this necessary:

`revolve2` is a namespace package split across the six vendored directories
under `engine/`. It is importable only once all six are on `sys.path`, and then
`revolve2.__path__` spans all six.

The `.pth` files that `pip install -e` writes cannot be relied on here. macOS
silently ignores a `.pth` file carrying the "hidden" flag, which iCloud Drive
sets on files under ~/Documents, and that quietly breaks every `revolve2`
import in an environment that installed perfectly. Putting the roots on the
path directly sidesteps the whole question.

Bootstrapping: this module lives in `src/`, which is itself one of the
directories it adds, so a file cannot `import paths` until `src/` is on the
path. Callers need two lines first:

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import paths

That prelude is deliberate; it does not mean that `src/` or `config/` should be
turned into a package: the runner registers the chosen config file as a module
named `config`, so a `config` package would be shadowed and
`from config.bodies import ...` would fail. Bodies, brain, evaluator and
genotype are all imported as bare top-level modules instead.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
CONFIG = ROOT / "config"
BRAINS = ROOT / "brains"
SRC = ROOT / "src"

NUM_ENGINE_PACKAGES = 6


def engine_roots() -> list[Path]:
    """
    Find the vendored Revolve2 package roots that make up the `revolve2` namespace.

    :returns: The six directories that each contain a `revolve2` package.
    :raises SystemExit: If the expected six are not all present.
    """
    roots = sorted(
        {
            found.parent
            for pattern in ("*/revolve2", "simulators/*/revolve2")
            for found in ENGINE.glob(pattern)
            if found.is_dir()
        }
    )
    if len(roots) != NUM_ENGINE_PACKAGES:
        raise SystemExit(
            f"expected {NUM_ENGINE_PACKAGES} Revolve2 packages under {ENGINE}, "
            f"found {len(roots)}: {[root.name for root in roots]}"
        )
    return roots


def all_roots(config_dir: Path | str | None = None) -> list[Path]:
    """
    Every directory that has to be importable for a run.

    :param config_dir: The directory holding the config in use. Defaults to
        this repository's `config/`; a snapshot being replayed may name another.
    :returns: The engine roots followed by `src/`, `brains/` and the config directory.
    """
    chosen_config = Path(config_dir) if config_dir is not None else CONFIG
    roots = engine_roots() + [SRC, BRAINS, chosen_config]
    if chosen_config.resolve() != CONFIG.resolve():
        roots.append(CONFIG)
    return roots


def install(config_dir: Path | str | None = None) -> list[Path]:
    """
    Put everything the project needs at the front of `sys.path`.

    Safe to call more than once: a root already present is left where it is.

    :param config_dir: The directory holding the config in use.
    :returns: The roots that were added.
    """
    added = []
    for root in reversed(all_roots(config_dir)):
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)
            added.append(root)
    return added


def child_environment(
    config_dir: Path | str | None = None,
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build the environment for a training child process.

    The child is a fresh interpreter, so it inherits neither `sys.path` nor
    `sys.dont_write_bytecode` from the parent. Both are passed through the
    environment instead. Bytecode writing is disabled because Python treats
    cached bytecode as current when a source file's size and whole-second
    mtime are unchanged, so editing a config and relaunching within the same
    second can otherwise run the previous settings.

    :param config_dir: The directory holding the config in use.
    :param base: Environment to extend. Defaults to the current one.
    :returns: The environment to hand to the child process.
    """
    environment = dict(os.environ if base is None else base)
    entries = [str(root) for root in all_roots(config_dir)]
    existing = environment.get("PYTHONPATH", "")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment
