"""Paddle skeleton model for visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from paddle_apriltag.calibration import DEFAULT_PADDLE_MODEL_PATH
DEFAULT_AXIS_LIMITS = (-0.5, 0.5, -0.5, 0.5, 0.0, 2.0)


@dataclass(frozen=True)
class PaddleModel:
    units: str
    keypoint_names: tuple[str, ...]
    keypoints: dict[str, np.ndarray]
    skeleton_edges: tuple[tuple[str, str], ...]
    object_points: np.ndarray


def _as_point3(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape == (2,):
        return np.array([array[0], array[1], 0.0], dtype=np.float64)
    if array.shape == (3,):
        return array
    raise ValueError(f"{field_name} must be [x, y] or [x, y, z] coordinates.")


def load_paddle_model(path: str | Path) -> PaddleModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Paddle model file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    units = str(data.get("units", "meters"))
    keypoints_raw = data.get("keypoints")
    if not isinstance(keypoints_raw, dict) or not keypoints_raw:
        raise ValueError("Paddle model must contain a non-empty 'keypoints' object.")

    keypoint_names = tuple(str(name) for name in keypoints_raw)
    keypoints = {
        name: _as_point3(keypoints_raw[name], f"keypoints.{name}")
        for name in keypoint_names
    }

    skeleton_raw = data.get("skeleton")
    if not isinstance(skeleton_raw, list) or not skeleton_raw:
        raise ValueError("Paddle model must contain a non-empty 'skeleton' list.")

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
    return PaddleModel(
        units=units,
        keypoint_names=keypoint_names,
        keypoints=keypoints,
        skeleton_edges=tuple(skeleton_edges),
        object_points=object_points,
    )


def paddle_world_points_from_pose(
    paddle_rotation: np.ndarray,
    paddle_origin: np.ndarray,
    model: PaddleModel,
) -> dict[str, list[float]]:
    world_points = (paddle_rotation @ model.object_points.T + paddle_origin.reshape(3, 1)).T
    return {
        name: point.tolist()
        for name, point in zip(model.keypoint_names, world_points, strict=True)
    }
