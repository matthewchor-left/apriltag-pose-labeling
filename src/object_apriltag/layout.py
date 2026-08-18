"""Load marker sticker layout on the object and derive marker-to-object transforms.

Layout coordinate frame (marker model / object model / ``coordinate_frame: marker_model``):
  +X: right in the image when the reference marker faces the camera
  +Y: down in the image
  +Z: into the scene (away from the camera)

``marker_model.json``, ``object_model.json``, and eraser geometry all store 3D points
in this frame. ``ObjectPose`` maps reference-marker-centered coordinates into the
camera frame without an extra axis flip.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")

REFERENCE_MARKER_COLOR = "#e41a1c"
REFERENCE_MARKER_COLOR_BGR = (60, 26, 228)
DEFAULT_MARKER_COLOR = "#ffff00"
DEFAULT_MARKER_COLOR_BGR = (0, 255, 255)
CORNER_LABELS = {
    "top_left": "tl",
    "top_right": "tr",
    "bottom_right": "br",
    "bottom_left": "bl",
}


from object_apriltag.calibration import DEFAULT_MARKER_MODEL_PATH


@dataclass(frozen=True)
class MarkerFootprint:
    """Physical corner geometry and orientation for one marker sticker.

    Attributes:
        marker_id: AprilTag ID for this footprint.
        top_left: Top-left corner in layout coordinates, meters.
        top_right: Top-right corner in layout coordinates, meters.
        bottom_right: Bottom-right corner in layout coordinates, meters.
        bottom_left: Bottom-left corner in layout coordinates, meters.
        orientation: ``(3, 3)`` rotation matrix whose columns are sticker axes.
    """

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
class MarkerToObject:
    """Rigid transform from a marker frame to the object reference frame.

    Attributes:
        offset: Translation from marker origin to object origin in marker frame.
        rotation: ``(3, 3)`` rotation from marker frame to object frame.
    """

    offset: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class MarkerLayout:
    """Complete marker sticker layout and derived marker-to-object transforms.

    Attributes:
        reference_marker_id: Marker ID defining the object reference frame.
        units: Length unit label stored in the model JSON.
        marker_size_m: Default physical edge length for markers in meters.
        marker_sizes_m: Per-marker edge lengths keyed by marker ID.
        footprints: Footprint geometry keyed by marker ID.
        transforms: Marker-to-object transforms keyed by marker ID.
        anchor_marker_ids: Optional subset of markers used as layout anchors.
    """

    reference_marker_id: int
    units: str
    marker_size_m: float
    marker_sizes_m: dict[int, float]
    footprints: dict[int, MarkerFootprint]
    transforms: dict[int, MarkerToObject]
    anchor_marker_ids: tuple[int, ...] | None = None

    @property
    def marker_ids(self) -> set[int]:
        return set(self.footprints)

    def marker_size_for(self, marker_id: int) -> float:
        """Return the physical edge length for a marker ID.

        Args:
            marker_id: Marker ID to look up.

        Returns:
            Marker edge length in meters.
        """
        return self.marker_sizes_m[marker_id]


def resolve_anchor_marker_ids_for_layout(
    anchor_marker_ids: Sequence[int] | None,
    marker_ids: Iterable[int],
) -> tuple[int, ...]:
    """Resolve anchor marker IDs against markers present in a layout.

    Args:
        anchor_marker_ids: Optional explicit anchor list; when ``None``, all
            present markers are used.
        marker_ids: Marker IDs available in the current layout.

    Returns:
        Tuple of anchor marker IDs present in ``marker_ids``, falling back to all
        present IDs when the explicit list is empty.
    """
    present = set(marker_ids)
    if anchor_marker_ids is None:
        return tuple(sorted(present))
    resolved = tuple(marker_id for marker_id in anchor_marker_ids if marker_id in present)
    return resolved or tuple(sorted(present))


def _as_point3(value: Any, field_name: str) -> np.ndarray:
    """Parse a JSON coordinate value into a 3D layout point.

    Args:
        value: ``[x, y]`` or ``[x, y, z]`` coordinate sequence.
        field_name: Field name used in validation error messages.

    Returns:
        ``(3,)`` float64 point; missing Z defaults to ``0.0``.

    Raises:
        ValueError: If ``value`` is not a 2- or 3-element coordinate.
    """
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
    """Measure the four edge lengths of a marker footprint quadrilateral.

    Args:
        top_left: Top-left corner in layout coordinates.
        top_right: Top-right corner in layout coordinates.
        bottom_right: Bottom-right corner in layout coordinates.
        bottom_left: Bottom-left corner in layout coordinates.

    Returns:
        Tuple ``(top, right, bottom, left)`` edge lengths in meters.
    """
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
    """Return the center of a quadrilateral defined by four corners.

    Args:
        top_left: Top-left corner in layout coordinates.
        top_right: Top-right corner in layout coordinates.
        bottom_right: Bottom-right corner in layout coordinates.
        bottom_left: Bottom-left corner in layout coordinates.

    Returns:
        ``(3,)`` centroid of the four corners.
    """
    return (top_left + top_right + bottom_right + bottom_left) / 4.0


def footprint_corner_with_padding(
    footprint: MarkerFootprint,
    corner_name: str,
    padding_m: float,
) -> np.ndarray:
    """Offset a footprint corner inward by a uniform padding distance.

    Args:
        footprint: Marker footprint geometry.
        corner_name: Corner name; one of ``CORNER_NAMES``.
        padding_m: Inward offset distance in meters.

    Returns:
        Padded corner position in layout coordinates.

    Raises:
        ValueError: If ``corner_name`` is invalid, ``padding_m`` is negative or
            non-finite, or the footprint geometry is degenerate.
    """
    if corner_name not in CORNER_NAMES:
        raise ValueError(
            f"corner must be one of {list(CORNER_NAMES)}, got {corner_name!r}."
        )
    if not np.isfinite(padding_m):
        raise ValueError(f"padding_m must be finite, got {padding_m}.")
    if padding_m < 0.0:
        raise ValueError(f"padding_m must be >= 0, got {padding_m}.")

    corners_by_name = footprint.corners_by_name()
    corner = corners_by_name[corner_name]
    if padding_m == 0.0:
        return corner.copy()

    corner_index = CORNER_NAMES.index(corner_name)
    prev_corner = corners_by_name[CORNER_NAMES[(corner_index - 1) % 4]]
    next_corner = corners_by_name[CORNER_NAMES[(corner_index + 1) % 4]]
    z = footprint.orientation[:, 2]
    inward_hint = rectangle_center(*footprint.corners()) - corner

    normals: list[np.ndarray] = []
    for edge in (corner - prev_corner, next_corner - corner):
        edge_norm = np.linalg.norm(edge)
        if edge_norm <= 0.0:
            raise ValueError(
                f"Degenerate footprint edge at {corner_name!r} for marker {footprint.marker_id}."
            )
        normal = np.cross(z, edge)
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 0.0:
            raise ValueError(
                f"Degenerate in-plane normal at {corner_name!r} for marker {footprint.marker_id}."
            )
        normal /= normal_norm
        if np.dot(normal, inward_hint) > 0.0:
            normal = -normal
        normals.append(normal)

    n1, n2 = normals
    denominator = 1.0 + float(np.dot(n1, n2))
    if abs(denominator) <= 1e-12:
        raise ValueError(
            f"Cannot apply padding at {corner_name!r} for marker {footprint.marker_id}: "
            "incident edge normals are opposed."
        )
    delta = padding_m * (n1 + n2) / denominator
    return corner + delta


def marker_origin_on_object(bottom_left: np.ndarray, bottom_right: np.ndarray) -> np.ndarray:
    """Return the marker origin at the midpoint of the bottom edge.

    Args:
        bottom_left: Bottom-left corner in layout coordinates.
        bottom_right: Bottom-right corner in layout coordinates.

    Returns:
        ``(3,)`` marker origin on the object surface.
    """
    return (bottom_left + bottom_right) / 2.0


def footprint_orientation(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
) -> np.ndarray:
    """Build a right-handed orientation matrix from footprint corner geometry.

    X axis follows the bottom edge; Y axis follows the left edge; Z axis is the
    outward sticker normal.

    Args:
        top_left: Top-left corner in layout coordinates.
        top_right: Top-right corner in layout coordinates.
        bottom_left: Bottom-left corner in layout coordinates.
        bottom_right: Bottom-right corner in layout coordinates.

    Returns:
        ``(3, 3)`` orientation matrix with orthonormal columns.

    Raises:
        ValueError: If any edge is degenerate or the resulting basis is improper.
    """
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
    """Parse a marker footprint from a JSON object payload.

    Args:
        marker_id: Marker ID for error messages and the returned footprint.
        payload: Dictionary containing all four corner coordinates.

    Returns:
        Parsed ``MarkerFootprint``.

    Raises:
        ValueError: If required corners are missing or geometry is degenerate.
    """
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


def derive_marker_to_object_transform(
    footprint: MarkerFootprint,
    reference_orientation: np.ndarray,
    object_origin: np.ndarray,
) -> MarkerToObject:
    """Derive the rigid transform from one marker frame to the object frame.

    Args:
        footprint: Marker footprint geometry.
        reference_orientation: ``(3, 3)`` orientation of the reference marker.
        object_origin: Object reference origin in layout coordinates.

    Returns:
        ``MarkerToObject`` transform for the marker.

    Raises:
        ValueError: If the derived rotation is improper (det < 0).
    """
    marker_origin = marker_origin_on_object(footprint.bottom_left, footprint.bottom_right)
    delta_layout = object_origin - marker_origin
    rotation = footprint.orientation.T @ reference_orientation
    offset = footprint.orientation.T @ delta_layout
    if np.linalg.det(rotation) < 0.0:
        raise ValueError(f"Marker {footprint.marker_id}: improper marker-to-object rotation.")
    return MarkerToObject(offset=offset, rotation=rotation)


def derive_marker_to_object_transforms(
    footprints: dict[int, MarkerFootprint],
    reference_marker_id: int,
) -> dict[int, MarkerToObject]:
    """Derive marker-to-object transforms for all footprints in a layout.

    Args:
        footprints: Footprint geometry keyed by marker ID.
        reference_marker_id: Marker ID defining the object reference frame.

    Returns:
        Marker-to-object transforms keyed by marker ID.

    Raises:
        KeyError: If ``reference_marker_id`` is not present in ``footprints``.
    """
    if reference_marker_id not in footprints:
        raise KeyError(f"reference_marker_id {reference_marker_id} is not present in markers.")

    reference = footprints[reference_marker_id]
    reference_orientation = reference.orientation
    object_origin = rectangle_center(*reference.corners())
    return {
        marker_id: derive_marker_to_object_transform(
            footprint,
            reference_orientation,
            object_origin,
        )
        for marker_id, footprint in footprints.items()
    }


def _point_to_json(point: np.ndarray) -> list[float]:
    """Serialize a layout point to a JSON coordinate list.

    Args:
        point: ``(2,)`` or ``(3,)`` coordinate array.

    Returns:
        ``[x, y]`` or ``[x, y, z]`` list of floats.
    """
    array = np.asarray(point, dtype=np.float64).reshape(-1)
    if array.shape == (2,):
        return [float(array[0]), float(array[1])]
    return [float(array[0]), float(array[1]), float(array[2])]


def footprint_to_dict(footprint: MarkerFootprint) -> dict[str, list[float]]:
    """Serialize a marker footprint to a JSON-compatible dictionary.

    Args:
        footprint: Marker footprint to serialize.

    Returns:
        Dictionary with the four corner coordinate lists.
    """
    return {
        "top_left": _point_to_json(footprint.top_left),
        "top_right": _point_to_json(footprint.top_right),
        "bottom_right": _point_to_json(footprint.bottom_right),
        "bottom_left": _point_to_json(footprint.bottom_left),
    }


def marker_layout_to_dict(layout: MarkerLayout) -> dict[str, Any]:
    """Serialize a marker layout to a JSON-compatible dictionary.

    Args:
        layout: Marker layout to serialize.

    Returns:
        Dictionary suitable for writing to ``marker_model.json``.
    """
    markers: dict[str, Any] = {}
    for marker_id, footprint in sorted(layout.footprints.items()):
        payload = footprint_to_dict(footprint)
        size_m = layout.marker_sizes_m[marker_id]
        if size_m != layout.marker_size_m:
            payload["size_m"] = size_m
        markers[str(marker_id)] = payload
    payload: dict[str, Any] = {
        "reference_marker_id": layout.reference_marker_id,
        "units": layout.units,
        "marker_size_m": layout.marker_size_m,
        "markers": markers,
    }
    if layout.anchor_marker_ids is not None:
        payload["anchor_marker_ids"] = list(layout.anchor_marker_ids)
    return payload


def resolve_marker_sizes(
    marker_ids: set[int] | frozenset[int],
    default_size_m: float,
    overrides: dict[int, float] | None = None,
) -> dict[int, float]:
    """Resolve per-marker physical edge lengths from defaults and overrides.

    Args:
        marker_ids: Marker IDs present in the layout.
        default_size_m: Default edge length in meters.
        overrides: Optional per-marker size overrides.

    Returns:
        Marker edge lengths keyed by marker ID.

    Raises:
        ValueError: If default or override sizes are invalid or reference unknown
            marker IDs.
    """
    if default_size_m <= 0.0 or not np.isfinite(default_size_m):
        raise ValueError(f"marker_size_m must be positive and finite, got {default_size_m}.")
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(marker_ids))
    if unknown:
        raise ValueError(f"marker size overrides are not subset of markers; extra {unknown}.")
    for marker_id, size in overrides.items():
        if size <= 0.0 or not np.isfinite(size):
            raise ValueError(
                f"markers.{marker_id}.size_m must be positive and finite, got {size}."
            )
    return {
        marker_id: float(overrides.get(marker_id, default_size_m))
        for marker_id in marker_ids
    }


def build_marker_layout(
    reference_marker_id: int,
    marker_size_m: float,
    footprints: dict[int, MarkerFootprint],
    units: str = "meters",
    marker_sizes_m: dict[int, float] | None = None,
    anchor_marker_ids: Sequence[int] | None = None,
) -> MarkerLayout:
    """Construct a validated marker layout from footprints and metadata.

    Args:
        reference_marker_id: Marker ID defining the object reference frame.
        marker_size_m: Default physical edge length in meters.
        footprints: Footprint geometry keyed by marker ID.
        units: Length unit label stored in the model.
        marker_sizes_m: Optional explicit per-marker sizes; must cover all
            footprints when provided.
        anchor_marker_ids: Optional subset of markers used as layout anchors.

    Returns:
        Fully derived ``MarkerLayout``.

    Raises:
        ValueError: If marker sizes are inconsistent or footprint validation
            fails.
    """
    footprint_ids = set(footprints)
    if marker_sizes_m is None:
        resolved_sizes = resolve_marker_sizes(footprint_ids, marker_size_m)
    else:
        size_ids = set(marker_sizes_m)
        if size_ids != footprint_ids:
            missing = sorted(footprint_ids - size_ids)
            extra = sorted(size_ids - footprint_ids)
            if missing:
                raise ValueError(f"marker_sizes_m missing footprint marker IDs: {missing}.")
            raise ValueError(f"marker_sizes_m contains unexpected marker IDs: {extra}.")
        resolved_sizes = dict(marker_sizes_m)
    validate_all_footprint_sizes(footprints, resolved_sizes)
    transforms = derive_marker_to_object_transforms(footprints, reference_marker_id)
    resolved_anchors = resolve_anchor_marker_ids_for_layout(anchor_marker_ids, footprints)
    return MarkerLayout(
        reference_marker_id=reference_marker_id,
        units=units,
        marker_size_m=marker_size_m,
        marker_sizes_m=resolved_sizes,
        footprints=footprints,
        transforms=transforms,
        anchor_marker_ids=resolved_anchors,
    )


def save_marker_model(path: str | Path, layout: MarkerLayout) -> None:
    """Atomically write a validated marker model JSON file.

    Args:
        path: Destination path for ``marker_model.json``.
        layout: Marker layout to serialize.

    Raises:
        OSError: If the temporary file cannot be written or replaced.
    """
    path = Path(path)
    payload = marker_layout_to_dict(layout)
    text = json.dumps(payload, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_marker_model(path: str | Path) -> MarkerLayout:
    """Load and validate a marker model JSON file.

    Args:
        path: Path to ``marker_model.json``.

    Returns:
        Parsed and validated ``MarkerLayout``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required fields are missing or geometry is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Marker model file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    reference_marker_id = int(data.get("reference_marker_id", 0))
    units = str(data.get("units", "meters"))
    if "marker_size_m" not in data:
        raise ValueError("Marker model must include 'marker_size_m'.")
    marker_size_m = float(data["marker_size_m"])
    if marker_size_m <= 0.0 or not np.isfinite(marker_size_m):
        raise ValueError(
            f"marker_size_m must be positive and finite, got {marker_size_m}."
        )

    markers_raw = data.get("markers")
    if not isinstance(markers_raw, dict) or not markers_raw:
        raise ValueError("Marker model must contain a non-empty 'markers' object.")

    size_overrides: dict[int, float] = {}
    footprints: dict[int, MarkerFootprint] = {}
    for marker_id_text, payload in markers_raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"markers.{marker_id_text} must be an object.")
        marker_id = int(marker_id_text)
        footprint_payload = {key: value for key, value in payload.items() if key != "size_m"}
        if "size_m" in payload:
            size_m = float(payload["size_m"])
            if size_m <= 0.0 or not np.isfinite(size_m):
                raise ValueError(
                    f"markers.{marker_id}.size_m must be positive and finite, got {size_m}."
                )
            size_overrides[marker_id] = size_m
        footprints[marker_id] = footprint_from_dict(marker_id, footprint_payload)
    marker_sizes_m = resolve_marker_sizes(set(footprints), marker_size_m, size_overrides)
    validate_all_footprint_sizes(footprints, marker_sizes_m)
    transforms = derive_marker_to_object_transforms(footprints, reference_marker_id)
    anchor_marker_ids = _parse_anchor_marker_ids_field(
        data.get("anchor_marker_ids"),
        footprints,
        reference_marker_id,
    )
    return MarkerLayout(
        reference_marker_id=reference_marker_id,
        units=units,
        marker_size_m=marker_size_m,
        marker_sizes_m=marker_sizes_m,
        footprints=footprints,
        transforms=transforms,
        anchor_marker_ids=anchor_marker_ids,
    )


def _parse_anchor_marker_ids_field(
    raw: Any,
    footprints: dict[int, MarkerFootprint],
    reference_marker_id: int,
) -> tuple[int, ...] | None:
    """Parse and validate the optional ``anchor_marker_ids`` JSON field.

    Args:
        raw: Raw JSON value for ``anchor_marker_ids``.
        footprints: Footprint geometry keyed by marker ID.
        reference_marker_id: Marker ID defining the object reference frame.

    Returns:
        Parsed anchor marker ID tuple, or ``None`` when the field is absent.

    Raises:
        ValueError: If the field is present but malformed or inconsistent.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("anchor_marker_ids must be a non-empty list of marker IDs when present.")
    try:
        anchor_marker_ids = tuple(int(marker_id) for marker_id in raw)
    except (TypeError, ValueError) as error:
        raise ValueError("anchor_marker_ids must contain integer marker IDs.") from error
    if len(set(anchor_marker_ids)) != len(anchor_marker_ids):
        duplicates = sorted({marker_id for marker_id in anchor_marker_ids if anchor_marker_ids.count(marker_id) > 1})
        raise ValueError(f"anchor_marker_ids contains duplicates: {duplicates}.")
    unknown = sorted(set(anchor_marker_ids) - set(footprints))
    if unknown:
        raise ValueError(f"anchor_marker_ids are not subset of markers; extra {unknown}.")
    if reference_marker_id not in anchor_marker_ids:
        raise ValueError(
            f"reference_marker_id {reference_marker_id} must appear in anchor_marker_ids."
        )
    return anchor_marker_ids


def validate_footprint_size(
    footprint: MarkerFootprint,
    marker_size_m: float,
    tolerance: float = 1e-4,
) -> None:
    """Validate that footprint edge lengths match the expected marker size.

    Args:
        footprint: Marker footprint to validate.
        marker_size_m: Expected physical edge length in meters.
        tolerance: Maximum allowed absolute edge-length error in meters.

    Raises:
        ValueError: If any edge length differs from ``marker_size_m`` beyond
            ``tolerance``.
    """
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
    marker_sizes_m: float | dict[int, float],
    tolerance: float = 1e-4,
) -> None:
    """Validate edge lengths for all footprints against expected marker sizes.

    Args:
        footprints: Footprint geometry keyed by marker ID.
        marker_sizes_m: Uniform expected size or per-marker size mapping.
        tolerance: Maximum allowed absolute edge-length error in meters.

    Raises:
        ValueError: If a footprint is missing from ``marker_sizes_m`` or any
            edge length is out of tolerance.
    """
    if isinstance(marker_sizes_m, (int, float)):
        expected_by_id = {footprint.marker_id: float(marker_sizes_m) for footprint in footprints.values()}
    else:
        expected_by_id = marker_sizes_m
    for footprint in footprints.values():
        marker_id = footprint.marker_id
        if marker_id not in expected_by_id:
            raise ValueError(f"marker_sizes_m missing footprint marker ID {marker_id}.")
        validate_footprint_size(footprint, expected_by_id[marker_id], tolerance=tolerance)


def layout_axis_limits(
    layout: MarkerLayout,
    padding_m: float = 0.02,
) -> tuple[float, float, float, float, float, float]:
    """Return axis-aligned bounds enclosing all layout footprint corners.

    Args:
        layout: Marker layout to bound.
        padding_m: Extra margin added on every axis in meters.

    Returns:
        Tuple ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
    """
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


def object_reference_footprint(layout: MarkerLayout) -> MarkerFootprint:
    """Return the footprint of the layout reference marker.

    Args:
        layout: Marker layout defining the reference marker ID.

    Returns:
        Reference marker ``MarkerFootprint``.
    """
    return layout.footprints[layout.reference_marker_id]


def object_reference_origin(layout: MarkerLayout) -> np.ndarray:
    """Return the object reference origin in layout coordinates.

    Args:
        layout: Marker layout defining the reference marker footprint.

    Returns:
        ``(3,)`` center of the reference marker footprint.
    """
    return rectangle_center(*object_reference_footprint(layout).corners())


def object_reference_orientation(layout: MarkerLayout) -> np.ndarray:
    """Return the object reference orientation in layout coordinates.

    Args:
        layout: Marker layout defining the reference marker footprint.

    Returns:
        ``(3, 3)`` orientation matrix of the reference marker.
    """
    return object_reference_footprint(layout).orientation


def layout_point_to_object_frame(point_layout: np.ndarray, layout: MarkerLayout) -> np.ndarray:
    """Transform a layout point into reference-marker-centered coordinates.

    Args:
        point_layout: ``(3,)`` point in marker model layout coordinates.
        layout: Marker layout defining the reference frame.

    Returns:
        ``(3,)`` point expressed in the reference marker sticker frame (origin at
        reference center, axes from the reference footprint orientation).
    """
    origin = object_reference_origin(layout)
    orientation = object_reference_orientation(layout)
    return orientation.T @ (np.asarray(point_layout, dtype=np.float64).reshape(3) - origin)


def layout_point_to_camera(
    point_layout: np.ndarray,
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    layout: MarkerLayout,
) -> np.ndarray:
    """Transform a layout point into the OpenCV camera frame.

    Args:
        point_layout: ``(3,)`` point in layout coordinates.
        object_rotation: ``(3, 3)`` object rotation in the camera frame.
        object_origin: Object origin in the camera frame, meters.
        layout: Marker layout defining the reference frame.

    Returns:
        ``(3,)`` point in the camera frame.
    """
    point_object = layout_point_to_object_frame(point_layout, layout)
    return object_rotation @ point_object + object_origin


def camera_point_to_layout_point(
    point_camera: np.ndarray,
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    layout: MarkerLayout,
) -> np.ndarray:
    """Transform a camera-frame point back into layout coordinates.

    Args:
        point_camera: ``(3,)`` point in the camera frame.
        object_rotation: ``(3, 3)`` object rotation in the camera frame.
        object_origin: Object origin in the camera frame, meters.
        layout: Marker layout defining the reference frame.

    Returns:
        ``(3,)`` point in layout coordinates.
    """
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    point_object = object_rotation.T @ (point - object_origin)
    orientation = object_reference_orientation(layout)
    origin = object_reference_origin(layout)
    return origin + orientation @ point_object


def marker_color(marker_id: int, reference_marker_id: int) -> str:
    return REFERENCE_MARKER_COLOR if marker_id == reference_marker_id else DEFAULT_MARKER_COLOR


def marker_color_bgr(marker_id: int, reference_marker_id: int) -> tuple[int, int, int]:
    return (
        REFERENCE_MARKER_COLOR_BGR
        if marker_id == reference_marker_id
        else DEFAULT_MARKER_COLOR_BGR
    )


MarkerModel = MarkerLayout
