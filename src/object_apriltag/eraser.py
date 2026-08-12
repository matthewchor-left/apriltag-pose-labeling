"""Load eraser plane geometry for live tag masking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_apriltag.layout import CORNER_NAMES, MarkerModel, layout_point_to_camera, object_reference_origin

ERASER_ORIGIN_REFERENCE_MARKER_CENTER = "reference_marker_center"


def _project_camera_point(
    point_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    point = np.asarray(point_camera, dtype=np.float64).reshape(1, 1, 3)
    projected, _ = cv2.projectPoints(
        point,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(2)


@dataclass(frozen=True)
class EraserPlane:
    plane_id: str | None
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left


@dataclass(frozen=True)
class EraserModel:
    units: str
    origin: str
    planes: tuple[EraserPlane, ...]


def _as_offset3(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must be [dx, dy, dz] offsets.")
    return array


def plane_from_dict(payload: dict[str, Any], index: int) -> EraserPlane:
    missing = [name for name in CORNER_NAMES if name not in payload]
    if missing:
        raise ValueError(
            f"Eraser plane {index} must include all four corners "
            f"{list(CORNER_NAMES)}; missing {missing}."
        )

    raw_plane_id = payload.get("plane_id", payload.get("id"))
    plane_id = str(raw_plane_id) if raw_plane_id is not None else None

    prefix = f"planes[{index}]"
    return EraserPlane(
        plane_id=plane_id,
        top_left=_as_offset3(payload["top_left"], f"{prefix}.top_left"),
        top_right=_as_offset3(payload["top_right"], f"{prefix}.top_right"),
        bottom_right=_as_offset3(payload["bottom_right"], f"{prefix}.bottom_right"),
        bottom_left=_as_offset3(payload["bottom_left"], f"{prefix}.bottom_left"),
    )


def load_eraser_model(path: str | Path) -> EraserModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eraser model file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    units = str(data.get("units", "meters"))
    origin = str(data.get("origin", ERASER_ORIGIN_REFERENCE_MARKER_CENTER))
    if origin != ERASER_ORIGIN_REFERENCE_MARKER_CENTER:
        raise ValueError(
            f"Unsupported eraser origin {origin!r}; "
            f"expected {ERASER_ORIGIN_REFERENCE_MARKER_CENTER!r}."
        )

    planes_raw = data.get("planes")
    if not isinstance(planes_raw, list) or not planes_raw:
        raise ValueError("Eraser model must contain a non-empty 'planes' array.")

    planes = tuple(plane_from_dict(payload, index) for index, payload in enumerate(planes_raw))
    return EraserModel(units=units, origin=origin, planes=planes)


def clip_polygon_to_rect(polygon: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Clip a polygon to the image rectangle [0, width] x [0, height]."""
    if len(polygon) < 3:
        return None

    def _clip(points: np.ndarray, inside, intersect) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        output: list[np.ndarray] = []
        previous = points[-1]
        for current in points:
            current_inside = inside(current)
            previous_inside = inside(previous)
            if current_inside:
                if previous_inside:
                    output.append(current)
                else:
                    output.append(intersect(previous, current))
                    output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
        if not output:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(output, dtype=np.float64)

    def _intersect_x(edge_x: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta_x = b[0] - a[0]
        t = 0.0 if delta_x == 0.0 else (edge_x - a[0]) / delta_x
        return np.array([edge_x, a[1] + t * (b[1] - a[1])], dtype=np.float64)

    def _intersect_y(edge_y: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta_y = b[1] - a[1]
        t = 0.0 if delta_y == 0.0 else (edge_y - a[1]) / delta_y
        return np.array([a[0] + t * (b[0] - a[0]), edge_y], dtype=np.float64)

    points = polygon.astype(np.float64)
    for edge_value, inside, intersect in (
        (0.0, lambda p: p[0] >= 0.0, lambda a, b: _intersect_x(0.0, a, b)),
        (float(width), lambda p: p[0] <= float(width), lambda a, b: _intersect_x(float(width), a, b)),
        (0.0, lambda p: p[1] >= 0.0, lambda a, b: _intersect_y(0.0, a, b)),
        (float(height), lambda p: p[1] <= float(height), lambda a, b: _intersect_y(float(height), a, b)),
    ):
        del edge_value
        points = _clip(points, inside, intersect)
        if len(points) < 3:
            return None
    return points


def eraser_offset_to_model_point(offset: np.ndarray, marker_model: MarkerModel) -> np.ndarray:
    return object_reference_origin(marker_model) + offset


def project_eraser_plane(
    plane_corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    marker_model: MarkerModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    image_points: list[np.ndarray] = []
    for offset in plane_corners:
        model_point = eraser_offset_to_model_point(offset, marker_model)
        camera_point = layout_point_to_camera(model_point, object_rotation, object_origin, marker_model)
        if camera_point[2] <= 0.0:
            continue
        projected = _project_camera_point(camera_point, camera_matrix, dist_coeffs)
        if not (np.isfinite(projected[0]) and np.isfinite(projected[1])):
            continue
        image_points.append(projected)

    if len(image_points) < 3:
        return None

    polygon = np.asarray(image_points, dtype=np.float64)
    return clip_polygon_to_rect(polygon, image_width, image_height)


def project_eraser_planes(
    eraser_model: EraserModel,
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    marker_model: MarkerModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for plane in eraser_model.planes:
        polygon = project_eraser_plane(
            plane.corners(),
            object_rotation,
            object_origin,
            marker_model,
            camera_matrix,
            dist_coeffs,
            image_width=image_width,
            image_height=image_height,
        )
        if polygon is not None:
            polygons.append(polygon)
    return polygons
