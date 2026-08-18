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
    """Project a single 3D camera-frame point to distorted image coordinates.

    Args:
        point_camera: 3D point in the camera frame.
        camera_matrix: ``3x3`` intrinsic matrix.
        dist_coeffs: Distortion coefficients passed to ``cv2.projectPoints``.

    Returns:
        ``(2,)`` image coordinates in pixels.
    """
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
    """Rectangular eraser mask plane defined by corner offsets from the reference marker.

    Attributes:
        plane_id: Optional identifier for the plane.
        top_left: ``[dx, dy, dz]`` offset from the reference marker center.
        top_right: ``[dx, dy, dz]`` offset from the reference marker center.
        bottom_right: ``[dx, dy, dz]`` offset from the reference marker center.
        bottom_left: ``[dx, dy, dz]`` offset from the reference marker center.
    """

    plane_id: str | None
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the four corner offsets in top-left, clockwise order."""
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left


@dataclass(frozen=True)
class EraserModel:
    """Collection of eraser mask planes loaded from JSON.

    Attributes:
        units: Length unit string (typically ``meters``).
        origin: Coordinate origin name (must be ``reference_marker_center``).
        planes: Eraser planes to project and mask in the image.
    """

    units: str
    origin: str
    planes: tuple[EraserPlane, ...]


def _as_offset3(value: Any, field_name: str) -> np.ndarray:
    """Parse a JSON value into a length-3 offset vector.

    Args:
        value: Raw JSON array value.
        field_name: Dotted field path used in error messages.

    Returns:
        ``(3,)`` float64 offset vector.

    Raises:
        ValueError: If ``value`` is not a length-3 numeric array.
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{field_name} must be [dx, dy, dz] offsets.")
    return array


def plane_from_dict(payload: dict[str, Any], index: int) -> EraserPlane:
    """Parse one eraser plane object from JSON.

    Args:
        payload: Plane object containing four named corners.
        index: Zero-based plane index used in validation errors.

    Returns:
        Parsed ``EraserPlane``.

    Raises:
        ValueError: If required corners or offset shapes are missing or invalid.
    """
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
    """Load and validate an eraser model JSON file.

    Args:
        path: Path to an eraser model JSON file.

    Returns:
        Parsed eraser geometry.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required fields are missing or use unsupported values.
    """
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
    """Clip a polygon to the image rectangle ``[0, width] x [0, height]``.

    Args:
        polygon: ``(N, 2)`` polygon vertices in image coordinates.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Clipped polygon with at least three vertices, or ``None`` when clipping
        removes too many points.
    """
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
    """Convert an eraser offset from the reference marker center to model coordinates.

    Args:
        offset: ``(3,)`` offset from the reference marker center.
        marker_model: Marker layout model supplying the reference origin.

    Returns:
        3D point in the object model frame.
    """
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
    """Project one eraser plane into a clipped image-space polygon.

    Args:
        plane_corners: Four model-frame corner offsets in clockwise order.
        object_rotation: ``3x3`` object rotation in the camera frame.
        object_origin: Object origin in the camera frame.
        marker_model: Marker layout model for reference-frame conversion.
        camera_matrix: ``3x3`` intrinsic matrix.
        dist_coeffs: Distortion coefficients passed to ``cv2.projectPoints``.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Clipped ``(N, 2)`` polygon in image coordinates, or ``None`` when fewer than
        three corners project in front of the camera.
    """
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
    """Project all eraser planes that are visible into image-space polygons.

    Args:
        eraser_model: Loaded eraser geometry.
        object_rotation: ``3x3`` object rotation in the camera frame.
        object_origin: Object origin in the camera frame.
        marker_model: Marker layout model for reference-frame conversion.
        camera_matrix: ``3x3`` intrinsic matrix.
        dist_coeffs: Distortion coefficients passed to ``cv2.projectPoints``.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        List of clipped polygons; planes that fail projection are omitted.
    """
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


def erase_with_mask(
    frame: np.ndarray,
    plate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Replace masked pixels in ``frame`` with values from ``plate``.

    Args:
        frame: Input BGR image.
        plate: Background plate with the same shape as ``frame``.
        mask: Single-channel mask; pixels with value > 0 are replaced.

    Returns:
        Copy of ``frame`` with masked pixels taken from ``plate``.

    Raises:
        ValueError: ``plate`` shape does not match ``frame``.
    """
    if plate.shape != frame.shape:
        raise ValueError("Background plate must match the frame shape.")
    output = frame.copy()
    output[mask > 0] = plate[mask > 0]
    return output


def build_eraser_mask(
    frame_shape: tuple[int, int],
    polygons: list[np.ndarray],
) -> np.ndarray:
    """Rasterize projected eraser polygons into a binary mask."""
    mask = np.zeros(frame_shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 255)
    return mask


def erase_with_planes(
    frame: np.ndarray,
    plate: np.ndarray,
    polygons: list[np.ndarray],
) -> np.ndarray:
    """Erase projected eraser regions by compositing ``plate`` through a polygon mask."""
    if not polygons:
        return frame.copy()
    mask = build_eraser_mask(frame.shape[:2], polygons)
    return erase_with_mask(frame, plate, mask)
