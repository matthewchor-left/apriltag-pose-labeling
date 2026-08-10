"""Load marker sticker layout on the paddle and derive marker-to-paddle transforms.

Layout coordinate frame (matches OpenCV camera axes when marker 0 faces the camera):
  +X: right in the image
  +Y: down in the image
  +Z: into the scene (away from the camera)

Marker 0 (front rubber) lies on the z = 0 plane. The back marker and edge markers
use positive Z because they are farther into the scene.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")

MARKER_LAYOUT_COLORS: dict[int, str] = {
    0: "#e41a1c",
    1: "#377eb8",
    2: "#4daf4a",
    3: "#984ea3",
    4: "#ff7f00",
}
MARKER_LAYOUT_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (60, 26, 228),
    1: (184, 126, 55),
    2: (74, 175, 77),
    3: (163, 78, 152),
    4: (0, 127, 255),
}
CORNER_LABELS = {
    "top_left": "tl",
    "top_right": "tr",
    "bottom_right": "br",
    "bottom_left": "bl",
}


from paddle_apriltag.calibration import DEFAULT_MARKER_LAYOUT_PATH


@dataclass(frozen=True)
class MarkerFootprint:
    marker_id: int
    top_left: np.ndarray
    top_right: np.ndarray
    bottom_right: np.ndarray
    bottom_left: np.ndarray
    orientation: np.ndarray

    def corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left

    def corners_by_name(self) -> dict[str, np.ndarray]:
        return {
            "top_left": self.top_left,
            "top_right": self.top_right,
            "bottom_right": self.bottom_right,
            "bottom_left": self.bottom_left,
        }


@dataclass(frozen=True)
class MarkerToPaddle:
    offset: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class MarkerLayout:
    reference_marker_id: int
    units: str
    marker_size_m: float
    footprints: dict[int, MarkerFootprint]
    transforms: dict[int, MarkerToPaddle]

    @property
    def marker_ids(self) -> set[int]:
        return set(self.footprints)


def _as_point3(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape == (2,):
        return np.array([array[0], array[1], 0.0], dtype=np.float64)
    if array.shape == (3,):
        return array
    raise ValueError(f"{field_name} must be [x, y] or [x, y, z] coordinates.")


def footprint_edge_lengths(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_right: np.ndarray,
    bottom_left: np.ndarray,
) -> tuple[float, float, float, float]:
    return (
        float(np.linalg.norm(top_right - top_left)),
        float(np.linalg.norm(bottom_right - top_right)),
        float(np.linalg.norm(bottom_left - bottom_right)),
        float(np.linalg.norm(top_left - bottom_left)),
    )


def rectangle_center(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_right: np.ndarray,
    bottom_left: np.ndarray,
) -> np.ndarray:
    return (top_left + top_right + bottom_right + bottom_left) / 4.0


def marker_origin_on_paddle(bottom_left: np.ndarray, bottom_right: np.ndarray) -> np.ndarray:
    return (bottom_left + bottom_right) / 2.0


def footprint_orientation(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> np.ndarray:
    x_axis = bottom_right - bottom_left
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 0.0:
        raise ValueError("Degenerate footprint: bottom edge has zero length.")
    x_axis /= x_norm

    y_axis = top_left - bottom_left
    y_norm = np.linalg.norm(y_axis)
    if y_norm <= 0.0:
        raise ValueError("Degenerate footprint: left edge has zero length.")
    y_axis /= y_norm

    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm <= 0.0:
        raise ValueError("Degenerate footprint: sticker axes are not independent.")
    z_axis /= z_norm

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    orientation = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(orientation) < 0.0:
        raise ValueError("Footprint orientation is improper (det < 0).")
    return orientation


def footprint_from_dict(marker_id: int, payload: dict[str, Any]) -> MarkerFootprint:
    missing = [name for name in CORNER_NAMES if name not in payload]
    if missing:
        raise ValueError(
            f"Marker {marker_id} must include all four corners "
            f"{list(CORNER_NAMES)}; missing {missing}."
        )

    top_left = _as_point3(payload["top_left"], f"markers.{marker_id}.top_left")
    top_right = _as_point3(payload["top_right"], f"markers.{marker_id}.top_right")
    bottom_right = _as_point3(payload["bottom_right"], f"markers.{marker_id}.bottom_right")
    bottom_left = _as_point3(payload["bottom_left"], f"markers.{marker_id}.bottom_left")
    orientation = footprint_orientation(top_left, top_right, bottom_left, bottom_right)
    return MarkerFootprint(
        marker_id=marker_id,
        top_left=top_left,
        top_right=top_right,
        bottom_right=bottom_right,
        bottom_left=bottom_left,
        orientation=orientation,
    )


def derive_marker_to_paddle_transform(
    footprint: MarkerFootprint,
    reference_orientation: np.ndarray,
    paddle_origin: np.ndarray,
) -> MarkerToPaddle:
    marker_origin = marker_origin_on_paddle(footprint.bottom_left, footprint.bottom_right)
    delta_layout = paddle_origin - marker_origin
    rotation = footprint.orientation.T @ reference_orientation
    offset = footprint.orientation.T @ delta_layout
    if np.linalg.det(rotation) < 0.0:
        raise ValueError(f"Marker {footprint.marker_id}: improper marker-to-paddle rotation.")
    return MarkerToPaddle(offset=offset, rotation=rotation)


def derive_marker_to_paddle_transforms(
    footprints: dict[int, MarkerFootprint],
    reference_marker_id: int,
) -> dict[int, MarkerToPaddle]:
    if reference_marker_id not in footprints:
        raise KeyError(f"reference_marker_id {reference_marker_id} is not present in markers.")

    reference = footprints[reference_marker_id]
    reference_orientation = reference.orientation
    paddle_origin = rectangle_center(*reference.corners())
    return {
        marker_id: derive_marker_to_paddle_transform(
            footprint,
            reference_orientation,
            paddle_origin,
        )
        for marker_id, footprint in footprints.items()
    }


def load_marker_layout(path: str | Path) -> MarkerLayout:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Marker layout file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    reference_marker_id = int(data.get("reference_marker_id", 0))
    units = str(data.get("units", "meters"))
    if "marker_size_m" not in data:
        raise ValueError("Marker layout must include 'marker_size_m'.")
    marker_size_m = float(data["marker_size_m"])
    if marker_size_m <= 0.0:
        raise ValueError(f"marker_size_m must be positive, got {marker_size_m}.")

    markers_raw = data.get("markers")
    if not isinstance(markers_raw, dict) or not markers_raw:
        raise ValueError("Marker layout must contain a non-empty 'markers' object.")

    footprints = {
        int(marker_id): footprint_from_dict(int(marker_id), payload)
        for marker_id, payload in markers_raw.items()
    }
    validate_all_footprint_sizes(footprints, marker_size_m)
    transforms = derive_marker_to_paddle_transforms(footprints, reference_marker_id)
    return MarkerLayout(
        reference_marker_id=reference_marker_id,
        units=units,
        marker_size_m=marker_size_m,
        footprints=footprints,
        transforms=transforms,
    )


def validate_footprint_size(
    footprint: MarkerFootprint,
    marker_size_m: float,
    tolerance: float = 1e-4,
) -> None:
    edge_labels = ("top", "right", "bottom", "left")
    for label, value in zip(edge_labels, footprint_edge_lengths(*footprint.corners()), strict=True):
        if abs(value - marker_size_m) > tolerance:
            corners = footprint.corners_by_name()
            raise ValueError(
                f"Marker {footprint.marker_id} {label} edge {value:.6f} m does not match "
                f"marker_size_m {marker_size_m:.6f} m. "
                f"corners={{{', '.join(f'{name}={corners[name].tolist()}' for name in CORNER_NAMES)}}}"
            )


def validate_all_footprint_sizes(
    footprints: dict[int, MarkerFootprint],
    marker_size_m: float,
    tolerance: float = 1e-4,
) -> None:
    for footprint in footprints.values():
        validate_footprint_size(footprint, marker_size_m, tolerance=tolerance)


def layout_axis_limits(
    layout: MarkerLayout,
    padding_m: float = 0.02,
) -> tuple[float, float, float, float, float, float]:
    points = []
    for footprint in layout.footprints.values():
        points.extend(footprint.corners())
    stacked = np.stack(points, axis=0)
    mins = stacked.min(axis=0) - padding_m
    maxs = stacked.max(axis=0) + padding_m
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


def paddle_reference_footprint(layout: MarkerLayout) -> MarkerFootprint:
    return layout.footprints[layout.reference_marker_id]


def paddle_reference_origin(layout: MarkerLayout) -> np.ndarray:
    return rectangle_center(*paddle_reference_footprint(layout).corners())


def paddle_reference_orientation(layout: MarkerLayout) -> np.ndarray:
    return paddle_reference_footprint(layout).orientation


def layout_point_to_paddle_frame(point_layout: np.ndarray, layout: MarkerLayout) -> np.ndarray:
    origin = paddle_reference_origin(layout)
    orientation = paddle_reference_orientation(layout)
    return orientation.T @ (point_layout - origin)


def layout_point_to_camera(
    point_layout: np.ndarray,
    paddle_rotation: np.ndarray,
    paddle_origin: np.ndarray,
    layout: MarkerLayout,
) -> np.ndarray:
    point_paddle = layout_point_to_paddle_frame(point_layout, layout)
    return paddle_rotation @ point_paddle + paddle_origin


def marker_color(marker_id: int) -> str:
    return MARKER_LAYOUT_COLORS.get(marker_id, "#888888")


def marker_color_bgr(marker_id: int) -> tuple[int, int, int]:
    return MARKER_LAYOUT_COLORS_BGR.get(marker_id, (136, 136, 136))
