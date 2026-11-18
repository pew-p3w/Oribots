"""Run the fast trial loop in C++ (src/native/trial_loop.cpp). Results are bit-identical.

``fast_trial`` removed the per-tick Python objects; what is left of the Python
loop is the brain step and the step bookkeeping, about a fifth of a trial at
implicitfast 5 ms. This moves that loop into C++. Model building and scoring
stay in Python (``fast_trial._Trial`` and ``fast_trial._score``), so only the
loop itself changes.

The C++ loop gives exactly the Python loop's numbers because every routine
whose rounding is not fixed by IEEE 754 is the very one Python uses, passed in
as a function pointer: NumPy's OpenBLAS ``ddot`` and ``dgemv`` (``np.dot``,
``np.matmul``), NumPy's float64 ``tanh`` loop, and the C library behind
Python's ``math`` (``sin``, ``cos``, ``atan2``, ``pow``). ``tests/test_native_trial.py``
checks the two loops agree bit for bit.

Build the library once with ``python src/native/build_native.py`` (needs a C++
compiler: g++ on Linux, MinGW-w64 g++ on Windows). Without it, or if anything
here cannot be found, trials run in the Python loop as before.

On by default when the library is built; ``NATIVE_TRIAL = False`` in a config,
or ``REVOLVE2_NATIVE_TRIAL=0``, turns it off.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

NATIVE_DIR = Path(__file__).resolve().parent / "native"
LIBRARY = NATIVE_DIR / ("trial_loop.dll" if sys.platform == "win32" else "trial_loop.so")
ABI_VERSION = 1

_P = ctypes.c_void_p
_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int32)


class _Funcs(ctypes.Structure):
    _fields_ = [
        ("mj_step", _P), ("mj_forward", _P),
        ("sin", _P), ("cos", _P), ("atan2", _P), ("pow", _P),
        ("ddot", _P), ("dgemv", _P),
        ("tanh_loop", _P), ("tanh_data", _P),
    ]


class _Spec(ctypes.Structure):
    _fields_ = [
        ("control_step", ctypes.c_double),
        ("sample_step", ctypes.c_double),
        ("end", ctypes.c_double),
        ("robot_id", ctypes.c_int32),
        ("ball_id", ctypes.c_int32),
        ("n_state", ctypes.c_int32),
        ("state", _D),
        ("weights", _D),
        ("work", _D),
        ("n_out", ctypes.c_int32),
        ("state_index", _I),
        ("hinge_range", _D),
        ("steering", _D),
        ("n_in", ctypes.c_int32),
        ("pos_index", _I),
        ("vel_index", _I),
        ("steering_scale", ctypes.c_double),
        ("distance_scale", ctypes.c_double),
        ("brain_has_offset", ctypes.c_int32),
        ("brain_offset", ctypes.c_double * 3),
        ("n_stops", ctypes.c_int32),
        ("stop_robot", _I),
        ("stop_object", _I),
        ("stop_distance", _D),
        ("stop_has_offset", _I),
        ("stop_offset", _D),
        ("max_samples", ctypes.c_int32),
        ("robot_pos", _D),
        ("robot_quat", _D),
        ("ball_pos", _D),
    ]


class _UFunc(ctypes.Structure):
    """The start of NumPy's PyUFuncObject (numpy/core/include/numpy/ufuncobject.h)."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t), ("ob_type", ctypes.c_void_p),
        ("nin", ctypes.c_int), ("nout", ctypes.c_int), ("nargs", ctypes.c_int), ("identity", ctypes.c_int),
        ("functions", ctypes.POINTER(ctypes.c_void_p)), ("data", ctypes.POINTER(ctypes.c_void_p)),
        ("ntypes", ctypes.c_int), ("reserved1", ctypes.c_int), ("name", ctypes.c_char_p),
        ("types", ctypes.POINTER(ctypes.c_char)),
    ]


_loaded: tuple[Any, _Funcs] | None = None
_failed: str | None = None


def _address(library: Any, name: str) -> int:
    return ctypes.cast(getattr(library, name), ctypes.c_void_p).value


def _mujoco_library() -> Any:
    import mujoco

    folder = os.path.dirname(mujoco.__file__)
    if sys.platform == "win32":
        return ctypes.CDLL(os.path.join(folder, "mujoco.dll"))
    candidates = sorted(glob.glob(os.path.join(folder, "libmujoco.so*"))) + sorted(
        glob.glob(os.path.join(folder, "libmujoco*.dylib"))
    )
    return ctypes.CDLL(candidates[0])


def _math_library() -> Any:
    """The C library Python's math module calls for sin, cos, atan2 and pow."""
    if sys.platform == "win32":
        return ctypes.CDLL("ucrtbase.dll")
    if sys.platform == "darwin":
        return ctypes.CDLL("libSystem.dylib")
    return ctypes.CDLL("libm.so.6")


def _numpy_blas() -> tuple[int, int]:
    """NumPy's own OpenBLAS cblas_ddot / cblas_dgemv (ILP64, as NumPy calls them)."""
    folder = os.path.join(os.path.dirname(np.__file__), "..", "numpy.libs")
    for path in sorted(glob.glob(os.path.join(folder, "*openblas*"))):
        library = ctypes.CDLL(path)
        for suffix in ("64_", ""):
            if hasattr(library, f"cblas_ddot{suffix}") and hasattr(library, f"cblas_dgemv{suffix}"):
                if suffix != "64_":
                    raise RuntimeError("NumPy's OpenBLAS is not ILP64; the native loop assumes it is.")
                return _address(library, f"cblas_ddot{suffix}"), _address(library, f"cblas_dgemv{suffix}")
    raise RuntimeError("NumPy's bundled OpenBLAS was not found.")


def _numpy_tanh_loop() -> tuple[int, int]:
    """NumPy's float64 -> float64 tanh inner loop, and its data pointer."""
    ufunc = _UFunc.from_address(id(np.tanh))
    if ufunc.name != b"tanh" or ufunc.nargs != 2:
        raise RuntimeError("Unexpected PyUFuncObject layout.")
    double = np.dtype(np.float64).num
    for i in range(ufunc.ntypes):
        if ord(ufunc.types[2 * i]) == double and ord(ufunc.types[2 * i + 1]) == double:
            return ufunc.functions[i], ufunc.data[i] or 0
    raise RuntimeError("NumPy's float64 tanh loop was not found.")


def _load() -> tuple[Any, _Funcs] | None:
    """Load the library and collect the routines, once per process."""
    global _loaded, _failed
    if _loaded is not None or _failed is not None:
        return _loaded
    try:
        if not LIBRARY.exists():
            raise FileNotFoundError(f"{LIBRARY.name} is not built (python src/native/build_native.py)")
        library = ctypes.CDLL(str(LIBRARY))
        library.trial_loop_abi_version.restype = ctypes.c_int32
        if library.trial_loop_abi_version() != ABI_VERSION:
            raise RuntimeError(f"{LIBRARY.name} is out of date; rebuild it.")
        import mujoco

        library.trial_loop_mujoco_version.restype = ctypes.c_int32
        built_for, running = library.trial_loop_mujoco_version(), mujoco.mj_version()
        if built_for != running:
            raise RuntimeError(
                f"{LIBRARY.name} was built for MuJoCo {built_for}, but {running} is installed; rebuild it."
            )
        library.run_trial.restype = ctypes.c_int32
        library.run_trial.argtypes = [ctypes.POINTER(_Funcs), ctypes.POINTER(_Spec), ctypes.c_void_p, ctypes.c_void_p]
        mujoco_library = _mujoco_library()
        libm = _math_library()
        ddot, dgemv = _numpy_blas()
        tanh_loop, tanh_data = _numpy_tanh_loop()
        funcs = _Funcs(
            mj_step=_address(mujoco_library, "mj_step"),
            mj_forward=_address(mujoco_library, "mj_forward"),
            sin=_address(libm, "sin"), cos=_address(libm, "cos"),
            atan2=_address(libm, "atan2"), pow=_address(libm, "pow"),
            ddot=ddot, dgemv=dgemv, tanh_loop=tanh_loop, tanh_data=tanh_data,
        )
        # Keep the libraries loaded for as long as the function pointers live.
        _loaded = ((library, mujoco_library, libm), funcs)
    except Exception as error:  # noqa: BLE001 - any failure means: use the Python loop
        _failed = f"{type(error).__name__}: {error}"
        logging.info(f"Native trial loop unavailable, using the Python loop ({_failed}).")
    return _loaded


def enabled() -> bool:
    """Whether the native loop is switched on (it is unless turned off)."""
    import config

    return os.environ.get("REVOLVE2_NATIVE_TRIAL", "1") != "0" and bool(
        getattr(config, "NATIVE_TRIAL", True)
    )


def available() -> bool:
    """Whether the library is built and everything it needs was found."""
    return _load() is not None


def _array(values: Any, dtype: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=dtype))


def run(trial: Any, task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    ``fast_trial._run`` in C++.

    :param trial: A ``fast_trial._Trial``.
    :param task: The worker task.
    :returns: The same arrays as ``fast_trial._run``, or None when this trial
        is not one the native loop handles (the caller then uses Python).
    :raises RuntimeError: If MuJoCo's clock leaves the predicted schedule.
    """
    loaded = _load()
    if loaded is None:
        return None
    _, funcs = loaded

    weights = trial.weights
    steering = trial.steering
    # The loop calls BLAS on these exactly as NumPy does for C-ordered float64
    # arrays; anything else goes through Python.
    if not (
        isinstance(weights, np.ndarray) and weights.dtype == np.float64 and weights.flags.c_contiguous
        and weights.ndim == 2 and weights.shape[0] == weights.shape[1] == len(trial.state)
        and isinstance(steering, np.ndarray) and steering.dtype == np.float64
        and steering.flags.c_contiguous and steering.ndim == 2 and steering.shape[1] == 4
        and isinstance(trial.state, np.ndarray) and trial.state.dtype == np.float64
    ):
        return None

    zero = (0.0, 0.0, 0.0)
    n_state = len(trial.state)
    state = trial.state.copy()
    work = np.empty(5 * n_state)
    state_index = _array([o[0] for o in trial.outputs], np.int32)
    hinge_range = _array([o[1] for o in trial.outputs], np.float64)
    pos_index = _array(trial.position_indices, np.int32)
    vel_index = _array(trial.velocity_indices, np.int32)
    stops = trial.stops or [(0, 0, 0.0, zero)]
    stop_robot = _array([s[0] for s in stops], np.int32)
    stop_object = _array([s[1] for s in stops], np.int32)
    stop_distance = _array([s[2] for s in stops], np.float64)
    stop_has_offset = _array([int(tuple(s[3]) != zero) for s in stops], np.int32)
    stop_offset = _array([list(s[3]) for s in stops], np.float64)

    end = float("inf") if task["simulation_time"] is None else float(task["simulation_time"])
    if not math.isfinite(end):
        return None
    max_samples = int(end / task["sample_step"]) + 16
    robot_pos = np.empty((max_samples, 3))
    robot_quat = np.empty((max_samples, 4))
    ball_pos = np.empty((max_samples, 3))

    def p(a: np.ndarray, kind: Any = _D) -> Any:
        return a.ctypes.data_as(kind)

    spec = _Spec(
        control_step=task["control_step"],
        sample_step=task["sample_step"],
        end=end,
        robot_id=trial.robot_id,
        ball_id=trial.ball_id,
        n_state=n_state,
        state=p(state), weights=p(weights), work=p(work),
        n_out=len(trial.outputs),
        state_index=p(state_index, _I), hinge_range=p(hinge_range), steering=p(steering),
        n_in=steering.shape[1],
        pos_index=p(pos_index, _I), vel_index=p(vel_index, _I),
        steering_scale=trial.steering_scale,
        distance_scale=trial.distance_scale,
        brain_has_offset=int(tuple(trial.brain_offset) != zero),
        brain_offset=(ctypes.c_double * 3)(*[float(v) for v in trial.brain_offset]),
        n_stops=len(trial.stops),
        stop_robot=p(stop_robot, _I), stop_object=p(stop_object, _I),
        stop_distance=p(stop_distance), stop_has_offset=p(stop_has_offset, _I),
        stop_offset=p(stop_offset),
        max_samples=max_samples,
        robot_pos=p(robot_pos), robot_quat=p(robot_quat), ball_pos=p(ball_pos),
    )
    library = loaded[0][0]
    count = library.run_trial(ctypes.byref(funcs), ctypes.byref(spec), trial.model._address, trial.data._address)
    if count == -1:
        raise RuntimeError(
            "MuJoCo's time left the predicted step schedule in the native trial loop. "
            "Set REVOLVE2_NATIVE_TRIAL=0 and report this."
        )
    if count < 0:
        raise RuntimeError(f"Native trial loop failed (code {count}).")
    trial.state = state
    return robot_pos[:count].copy(), robot_quat[:count].copy(), ball_pos[:count].copy()
