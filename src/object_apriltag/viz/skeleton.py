"""Object skeleton model for visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from object_apriltag.calibration import DEFAULT_OBJECT_MODEL_PATH
from object_apriltag.layout import MarkerLayout, layout_point_to_camera

DEFAULT_AXIS_LIMITS = (-0.5, 0.5, -0.5, 0.5, 0.0, 2.0)
MODEL_FRAME_NAME = "marker_model"


@dataclass(frozen=True)
class ObjectModel:
    """Parsed object skeleton model for visualization.

    Attributes:
        units: Length units declared in the object model JSON.
        keypoint_names: Ordered keypoint names.
        keypoints: Keypoint positions in marker_model coordinates.
        skeleton_edges: Undirected skeleton edges as name pairs.
        object_points: Stacked keypoint positions as ``(N, 3)`` float32 array.
    """

    units: str
    keypoint_names: tuple[str, ...]
    keypoints: dict[str, np.ndarray]
    skeleton_edges: tuple[tuple[str, str], ...]
    object_points: np.ndarray


def _as_point3(value: Any, field_name: str) -> np.ndarray:
    """Parse a JSON keypoint value into a 3D point.

    Args:
        value: JSON value as ``[x, y]`` or ``[x, y, z]``.
        field_name: Field name for error messages.

    Returns:
        3D point as float64 array.

    Raises:
        ValueError: If the value is not 2D or 3D coordinates.
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape == (2,):
        return np.array([array[0], array[1], 0.0], dtype=np.float64)
    if array.shape == (3,):
        return array
    raise ValueError(f"{field_name} must be [x, y] or [x, y, z] coordinates.")


def object_model_from_data(data: dict[str, Any]) -> ObjectModel:
    """Parse an object model JSON dict into an ``ObjectModel``.

    Args:
        data: Parsed object model JSON.

    Returns:
        Visualization object model.

    Raises:
        ValueError: If required fields are missing or malformed.
    """
    units = str(data.get("units", "meters"))
    coordinate_frame = data.get("coordinate_frame", MODEL_FRAME_NAME)
    if coordinate_frame != MODEL_FRAME_NAME:
        raise ValueError(
            f"object model coordinate_frame must be {MODEL_FRAME_NAME!r}, got {coordinate_frame!r}."
        )
    keypoints_raw = data.get("keypoints")
    if not isinstance(keypoints_raw, dict) or not keypoints_raw:
        raise ValueError("Object model must contain a non-empty 'keypoints' object.")

    keypoint_names = tuple(str(name) for name in keypoints_raw)
    keypoints = {
        name: _as_point3(keypoints_raw[name], f"keypoints.{name}")
        for name in keypoint_names
    }

    skeleton_raw = data.get("skeleton")
    if not isinstance(skeleton_raw, list) or not skeleton_raw:
        raise ValueError("Object model must contain a non-empty 'skeleton' list.")

    skeleton_edges: list[tuple[str, str]] = []
    for index, edge in enumerate(skeleton_raw):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"skeleton[{index}] must be [start, end] keypoint names.")
        start_name, end_name = str(edge[0]), str(edge[1])
        for name in (start_name, end_name):
            if name not in keypoints:
                raise ValueError(f"skeleton[{index}] references unknown keypoint {name!r}.")
        skeleton_edges.append((start_name, end_name))

    object_points = np.asarray([keypoints[name] for name in keypoint_names], dtype=np.float32)
    return ObjectModel(
        units=units,
        keypoint_names=keypoint_names,
        keypoints=keypoints,
        skeleton_edges=tuple(skeleton_edges),
        object_points=object_points,
    )


def load_object_model(path: str | Path) -> ObjectModel:
    """Load an object model JSON file for visualization.

    Args:
        path: Path to the object model JSON file.

    Returns:
        Parsed visualization object model.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Object model file not found: {path}")
    return object_model_from_data(json.loads(path.read_text(encoding="utf-8")))


def object_world_points_from_pose(
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    model: ObjectModel,
    marker_model: MarkerLayout,
) -> dict[str, list[float]]:
    """Transform object-model keypoints into camera-frame world points.

    Args:
        object_rotation: Fused object rotation in camera frame.
        object_origin: Fused object origin in camera frame.
        model: Object skeleton model.
        marker_model: Marker layout defining the object frame.

    Returns:
        Keypoint positions in camera frame keyed by name.
    """
    return {
        name: layout_point_to_camera(
            model.keypoints[name],
            object_rotation,
            object_origin,
            marker_model,
        ).tolist()
        for name in model.keypoint_names
    }
