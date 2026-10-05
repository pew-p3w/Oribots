"""Different viewer implementations for mujoco."""

from ._viewer_type import ViewerType

__all__ = [
    "CustomMujocoViewer",
    "CustomMujocoViewerMode",
    "NativeMujocoViewer",
    "ViewerType",
]


def __getattr__(name: str):
    """
    Lazily import viewer implementations.

    Viewer modules can initialize GUI-related dependencies, so importing them only
    when requested keeps simulator startup from blocking before a scene is run.
    """
    if name in {"CustomMujocoViewer", "CustomMujocoViewerMode"}:
        from ._custom_mujoco_viewer import CustomMujocoViewer, CustomMujocoViewerMode

        return {
            "CustomMujocoViewer": CustomMujocoViewer,
            "CustomMujocoViewerMode": CustomMujocoViewerMode,
        }[name]
    if name == "NativeMujocoViewer":
        from ._native_mujoco_viewer import NativeMujocoViewer

        return NativeMujocoViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
