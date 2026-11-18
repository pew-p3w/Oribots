"""The physics settings of a run: integrator, timestep, and the batch parameters.

Option 1 of PERFORMANCE_OPTIONS.md. The integrator and timestep used to be fixed
(RK4 at 1 ms). A config may now choose them::

    SIM_INTEGRATOR = "implicitfast"   # 'RK4' (default), 'Euler', 'implicit', 'implicitfast'
    SIM_TIMESTEP = 0.002              # seconds (default 0.001)

A config that sets neither gets exactly the old physics, so every existing
config, snapshot and reference value is unchanged. A snapshot stores its
config's source, so `run.py -t` replays a controller with the physics it was
trained with.

Everything that builds simulation parameters (the evaluator, its worker
processes, `run.py -t`) asks this module, so training, scoring and replay can
never use different physics.

`tests/test_physics.py` checks the validation and that the settings reach the
compiled MuJoCo model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

DEFAULT_INTEGRATOR = "RK4"
DEFAULT_TIMESTEP = 0.001
INTEGRATORS = ("RK4", "Euler", "implicit", "implicitfast")

# How far a step ratio may be from a whole number and still count as whole.
_WHOLE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PhysicsSettings:
    """The physics a run uses."""

    integrator: str
    timestep: float

    @property
    def is_default(self) -> bool:
        """Whether this is the original physics (RK4 at 1 ms)."""
        return self.integrator == DEFAULT_INTEGRATOR and self.timestep == DEFAULT_TIMESTEP


def _config(config: Any | None) -> Any:
    """
    Get the config to read from.

    :param config: An explicit config, or None for the run's `config` module.
    :returns: The config.
    :raises RuntimeError: If there is no config loaded.
    """
    if config is not None:
        return config
    module = sys.modules.get("config")
    if module is None:
        raise RuntimeError("No config is loaded (the module named 'config' is missing).")
    return module


def steps_per(interval: float, timestep: float) -> int:
    """
    How many physics steps make up an interval, which must be a whole number.

    :param interval: Interval in seconds (a control or sampling period).
    :param timestep: Physics timestep in seconds.
    :returns: The whole number of steps.
    :raises ValueError: If the interval is not a whole number of steps.
    """
    ratio = interval / timestep
    whole = round(ratio)
    if whole < 1 or abs(ratio - whole) > _WHOLE_TOLERANCE:
        raise ValueError(
            f"SIM_TIMESTEP {timestep:g} s does not divide {interval:g} s into whole steps "
            f"({ratio:g}). Pick a timestep that does, for example 0.001, 0.002, 0.0025 or 0.005."
        )
    return int(whole)


def settings(config: Any | None = None) -> PhysicsSettings:
    """
    Read and validate a config's physics settings.

    :param config: The config (default: the run's `config` module).
    :returns: The settings.
    :raises ValueError: If the integrator is unknown or the timestep is unusable.
    """
    from revolve2.ci_group.simulation_parameters import (
        STANDARD_CONTROL_FREQUENCY,
        STANDARD_SAMPLING_FREQUENCY,
    )

    cfg = _config(config)
    integrator = getattr(cfg, "SIM_INTEGRATOR", DEFAULT_INTEGRATOR)
    timestep = getattr(cfg, "SIM_TIMESTEP", DEFAULT_TIMESTEP)
    if integrator not in INTEGRATORS:
        raise ValueError(f"SIM_INTEGRATOR {integrator!r} is not one of {', '.join(INTEGRATORS)}.")
    if isinstance(timestep, bool) or not isinstance(timestep, (int, float)) or not timestep > 0:
        raise ValueError(f"SIM_TIMESTEP must be a positive number of seconds, got {timestep!r}.")
    timestep = float(timestep)
    # Control and sampling must land on whole steps, or they would drift off
    # their intended rate as the timestep changes.
    steps_per(1.0 / STANDARD_CONTROL_FREQUENCY, timestep)
    steps_per(1.0 / STANDARD_SAMPLING_FREQUENCY, timestep)
    return PhysicsSettings(integrator=integrator, timestep=timestep)


def batch_parameters(config: Any | None = None, simulation_time: float | None = None) -> Any:
    """
    Build the simulation batch parameters for a run.

    :param config: The config (default: the run's `config` module).
    :param simulation_time: Seconds to simulate (default: the config's SIMULATION_TIME).
    :returns: BatchParameters carrying the run's integrator and timestep.
    """
    from revolve2.ci_group.simulation_parameters import make_standard_batch_parameters

    cfg = _config(config)
    chosen = settings(cfg)
    parameters = make_standard_batch_parameters(simulation_timestep=chosen.timestep)
    parameters.integrator = chosen.integrator
    parameters.simulation_time = cfg.SIMULATION_TIME if simulation_time is None else simulation_time
    return parameters
