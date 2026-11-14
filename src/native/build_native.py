"""Build the C++ trial loop (trial_loop.cpp) into a library next to this file.

    python src/native/build_native.py            (Windows: .venv\\Scripts\\python ...)

Needs a C++ compiler:

- Linux / the cluster: the system g++ (module load gcc, if needed).
- Windows: MinGW-w64 g++, for example WinLibs (a zip; no installer, no admin).
  Put its mingw64\\bin on PATH, or set CXX to the full path of g++.exe.
- macOS: clang++ from the Xcode command line tools.

The only other input is MuJoCo's headers, which the mujoco Python package
ships. The library calls MuJoCo through function pointers, so nothing is
linked against MuJoCo itself.

-ffp-contract=off and no -ffast-math are required: they stop the compiler
fusing a*b+c into one rounding, which would change the results.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "trial_loop.cpp"
OUTPUT = HERE / ("trial_loop.dll" if sys.platform == "win32" else "trial_loop.so")


def find_compiler() -> str:
    """g++ (or clang++ on macOS), from CXX, PATH, or a WinLibs folder."""
    candidates = [os.environ.get("CXX")]
    candidates += [shutil.which(name) for name in ("g++", "clang++", "c++")]
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "winlibs" / "mingw64" / "bin" / "g++.exe"
        candidates.append(str(local))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit(
        "No C++ compiler found. Install g++ (Windows: WinLibs MinGW-w64), put it on PATH "
        "or set CXX, and run this again. Training works without it, in the Python loop."
    )


def mujoco_include() -> Path:
    import mujoco

    include = Path(mujoco.__file__).parent / "include" / "mujoco"
    if not (include / "mujoco" / "mujoco.h").exists():
        raise SystemExit(f"MuJoCo headers not found under {include}.")
    return include


def main() -> None:
    compiler = find_compiler()
    command = [
        compiler, "-O2", "-std=c++17", "-shared",
        "-ffp-contract=off", "-fno-fast-math",
        f"-I{mujoco_include()}",
        str(SOURCE), "-o", str(OUTPUT),
    ]
    if sys.platform == "win32":
        # No MinGW runtime DLLs needed next to the library.
        command += ["-static-libgcc", "-static-libstdc++", "-static"]
    else:
        command += ["-fPIC"]
    print(" ".join(command))
    env = dict(os.environ)
    # MinGW's g++ finds its own helper programs through PATH.
    env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
    subprocess.run(command, check=True, env=env)
    print(f"Built {OUTPUT}")


if __name__ == "__main__":
    main()
