"""Record a match to an MP4 at a fixed speed, without a window.

The native viewer shows simulated time only as fast as the computer can simulate
it, so a contest of several robots crawls on screen. A video plays at a fixed
speed instead (``speed`` simulated seconds per video second).

The match is stepped with exactly the engine's loop (``_simulate_scene.py``:
control at the control step, a state at t = 0, one each sample step, one at the
end), so the states, and the scores computed from them, equal those of a
headless playback. Frames come from MuJoCo's offscreen renderer (MUJOCO_GL
picks the backend: cgl on macOS, egl on a cluster) and are written with OpenCV.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import contest_paths

contest_paths.install()

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_SPEED = 20.0
DEFAULT_FPS = 30
DEFAULT_SIZE = (1280, 720)


def _camera(terrain_size: tuple[float, float]) -> Any:
    """A fixed camera looking down at the whole arena at an angle."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.0)
    camera.distance = 0.85 * math.hypot(*terrain_size)
    camera.elevation = -60.0
    camera.azimuth = 90.0
    return camera


def record_match(
    task: dict[str, Any],
    terrain_size: tuple[float, float],
    path: Path,
    speed: float = DEFAULT_SPEED,
    fps: int = DEFAULT_FPS,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> list[Any]:
    """
    Simulate a match task, writing a video, and return its states.

    :param task: A match task (``match.make_match_task``).
    :param terrain_size: The arena's floor size, for the camera.
    :param path: The .mp4 to write.
    :param speed: Simulated seconds per video second.
    :param fps: Video frames per second.
    :param size: Video (width, height).
    :returns: The scene simulation states (t = 0 first), as a headless playback returns them.
    :raises ValueError: If the speed, fps or size are not positive.
    """
    from revolve2.modular_robot_simulation._scene_simulation_state import SceneSimulationState
    from revolve2.simulators.mujoco_simulator._control_interface_impl import ControlInterfaceImpl
    from revolve2.simulators.mujoco_simulator._scene_to_model import scene_to_model
    from revolve2.simulators.mujoco_simulator._simulation_state_impl import SimulationStateImpl

    if not (speed > 0 and fps > 0 and size[0] > 0 and size[1] > 0):
        raise ValueError("video speed, fps and size must be positive")
    scene = task["scene"]
    model, mapping = scene_to_model(scene, task["simulation_timestep"], cast_shadows=task["cast_shadows"], fast_sim=task["fast_sim"])
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, size[1])
    data = mujoco.MjData(model)
    control = ControlInterfaceImpl(data=data, abstraction_to_mujoco_mapping=mapping)
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    camera = _camera(terrain_size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    frame_step = speed / fps

    def write_frame() -> None:
        renderer.update_scene(data, camera)
        frame = renderer.render()
        label = f"t = {data.time:6.1f} s   x{speed:g}"
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    control_step, sample_step, end = task["control_step"], task["sample_step"], task["simulation_time"]
    states = []
    try:
        mujoco.mj_forward(model, data)
        states.append(SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping, camera_views={}))
        write_frame()
        last_control = last_sample = 0.0
        next_frame = frame_step
        while (time := data.time) < end:
            if time >= last_control + control_step:
                last_control = math.floor(time / control_step) * control_step
                state = SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping, camera_views={})
                scene.handler.handle(state, control, control_step)
            if time >= last_sample + sample_step:
                last_sample = int(time / sample_step) * sample_step
                states.append(SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping, camera_views={}))
            mujoco.mj_step(model, data)
            if data.time >= next_frame:
                write_frame()
                next_frame += frame_step
        states.append(SimulationStateImpl(data=data, abstraction_to_mujoco_mapping=mapping, camera_views={}))
    finally:
        writer.release()
        renderer.close()
    return [SceneSimulationState(state, task["mapping"]) for state in states]


def frame_count(path: Path) -> int:
    """
    The number of frames in a video file.

    :returns: The frame count.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def mean_brightness(path: Path, index: int = 0) -> float:
    """
    The mean pixel value of one frame (to check frames are not blank).

    :returns: The mean over all pixels and channels.
    """
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        return float(np.mean(frame)) if ok else 0.0
    finally:
        capture.release()
