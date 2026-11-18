# Differences from upstream Revolve2 (v1.2.0)

`engine/` is a vendored copy of the six Revolve2 packages this project uses:
`simulation`, `modular_robot`, `modular_robot_simulation`, `experimentation`,
`ci_group` and `simulators/mujoco_simulator`. Everything is unchanged from
upstream except the items below.

`modular_robot_physical` (robot hardware only) is not vendored.

## Source changes

- `simulators/mujoco_simulator/revolve2/simulators/mujoco_simulator/_simulate_scene.py`:
  `import cv2` moved from module level into the video-recording branch. OpenCV is
  only needed when recording video, and importing it eagerly caused NumPy/OpenCV
  binary-compatibility failures on some machines.
- `modular_robot_simulation/revolve2/modular_robot_simulation/_build_multi_body_systems/_builders/_core_builder.py`:
  a core may supply its own shape by defining `make_geometries(pose, texture)`;
  without it the core is one box, exactly as upstream. The pentagon spider's
  five-sided core (`config/bodies.py`) needs this. **If a copy of the engine lacks
  this hook, the pentagon core silently becomes one big box.**
- `modular_robot_simulation/revolve2/modular_robot_simulation/_modular_robot_scene.py`:
  - `physical_orientation(pose)` returns a robot's true orientation. The MuJoCo
    simulator reports a body's orientation by reading MuJoCo's (w, x, y, z)
    quaternion as pyrr's (x, y, z, w), so the reported orientation is a
    permutation of the physical one: a robot turning about the vertical axis is
    reported as turning about x, so "the robot's +x axis" computed from it is always
    the world's +x, whichever way the robot faces (and a robot physically spawns
    facing world -x). The simulator itself is left as it is, so its reported
    orientations keep their upstream meaning; only head mode
    (`src/robot_frame.py`) uses the true orientation.
  - `StopOnRobotObjectDistance` gained `reference_offset` (default `(0, 0, 0)`, the
    core, as upstream), so "ball reached" can be measured from a head.
- Configurable integrator (OrigbotsRewrite, option 1 of `PERFORMANCE_OPTIONS.md`).
  Upstream hard-codes RK4. Every default is still `"RK4"`, so a caller that does
  not pass an integrator gets exactly the upstream physics:
  - `simulation/.../simulator/_batch_parameters.py`: `BatchParameters` gained
    `integrator: str = "RK4"`.
  - `simulators/.../_scene_to_model.py`: `scene_to_model(..., integrator="RK4")`
    sets `option.integrator` from it instead of the literal `"RK4"`.
  - `simulators/.../_simulate_scene.py`: `simulate_scene(..., integrator="RK4")`
    passes it to `scene_to_model`.
  - `simulators/.../_local_simulator.py`: passes `batch.parameters.integrator`
    on to `simulate_scene`.
  The project chooses it through `src/physics.py` (`SIM_INTEGRATOR`,
  `SIM_TIMESTEP` in a config).
- `ci_group/revolve2/ci_group/terrains.py`: `from noise import pnoise2` moved into
  the rugged-heightmap function, the only user. `noise` has no Windows wheel and
  needs a C compiler; every shipped config uses the flat terrain, so Windows
  installs skip it (`requirements.txt` marker).

## Manifest changes (`pyproject.toml`)

- All six: `readme` now points at the top-level `README.md` of this repository.
- `ci_group`:
  - Removed the Cython build step (`[tool.poetry.build]` script, the `include`
    of the compiled module, and `Cython`/`numpy`/`setuptools` from
    `[build-system] requires`). It compiled the morphological-novelty module,
    which this project does not use. The module's source is still present.
  - Removed the unused dependencies `multineat`, `sqlalchemy`, `Cython`,
    `setuptools`, `opencv-python` and `opencv-contrib-python`.
    Kept `noise`, `numpy` and the `revolve2-modular-robot-simulation` link.

The packages are installed with `pip install --no-deps -e`; third-party
dependencies and their pinned versions are listed in the top-level
`requirements.txt`, not in these manifests.
