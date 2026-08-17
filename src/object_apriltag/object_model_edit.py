"""Object model load/save and keypoint source helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from object_apriltag.layout import (
    CORNER_NAMES,
    MarkerLayout,
    footprint_corner_with_padding,
)
from object_apriltag.viz.skeleton import MODEL_FRAME_NAME, ObjectModel, object_model_from_data


def load_object_model_document(path: str | Path) -> tuple[ObjectModel, dict[str, Any]]:
    """Load an object model JSON file into runtime and raw document forms.

    Args:
        path: Path to an ``object_model.json`` file.

    Returns:
        Tuple of parsed ``ObjectModel`` and the original JSON document dict.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    return object_model_from_data(document), document


def _parse_source_marker_id(value: Any, field_name: str) -> int:
    """Parse a marker ID from JSON into a non-boolean integer.

    Args:
        value: Raw JSON value for a marker ID field.
        field_name: Dotted field path used in error messages.

    Returns:
        Parsed integer marker ID.

    Raises:
        ValueError: If ``value`` is not a valid integer marker ID.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer marker id, got {value!r}.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return int(text, 10)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be an integer marker id, got {value!r}."
            ) from exc
    raise ValueError(f"{field_name} must be an integer marker id, got {value!r}.")


def _parse_padding_mm(value: Any, field_name: str) -> float:
    """Parse a non-negative padding distance from millimeters to meters.

    Args:
        value: Raw JSON value for a padding field.
        field_name: Dotted field path used in error messages.

    Returns:
        Padding distance in meters.

    Raises:
        ValueError: If ``value`` is not a finite non-negative number.
    """
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number in millimeters, got {value!r}.")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number in millimeters, got {value!r}.")
    padding_mm = float(value)
    if not np.isfinite(padding_mm):
        raise ValueError(f"{field_name} must be finite, got {padding_mm}.")
    if padding_mm < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {padding_mm}.")
    return padding_mm / 1000.0


def parse_keypoint_sources(document: dict[str, Any]) -> dict[str, tuple[int, str, float]]:
    """Parse ``keypoint_sources`` entries from an object model document.

    Args:
        document: Parsed object-model JSON root object.

    Returns:
        Map from keypoint name to ``(marker_id, corner_name, padding_m)`` tuples.

    Raises:
        ValueError: If ``keypoint_sources`` is missing, empty, or contains invalid entries.
    """
    raw = document.get("keypoint_sources")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "Object model must contain a non-empty 'keypoint_sources' object when "
            "updating from marker calibration."
        )

    sources: dict[str, tuple[int, str, float]] = {}
    for keypoint_name, payload in raw.items():
        name = str(keypoint_name)
        if not name:
            raise ValueError("keypoint_sources keys must be non-empty keypoint names.")
        if not isinstance(payload, dict):
            raise ValueError(
                f"keypoint_sources.{name} must be an object with marker_id and corner."
            )
        if "marker_id" not in payload:
            raise ValueError(f"keypoint_sources.{name} is missing marker_id.")
        if "corner" not in payload:
            raise ValueError(f"keypoint_sources.{name} is missing corner.")
        marker_id = _parse_source_marker_id(
            payload["marker_id"],
            f"keypoint_sources.{name}.marker_id",
        )
        corner = str(payload["corner"])
        if corner not in CORNER_NAMES:
            raise ValueError(
                f"keypoint_sources.{name}.corner must be one of {list(CORNER_NAMES)}, "
                f"got {corner!r}."
            )
        padding_m = 0.0
        if "padding_mm" in payload:
            padding_m = _parse_padding_mm(
                payload["padding_mm"],
                f"keypoint_sources.{name}.padding_mm",
            )
        sources[name] = (marker_id, corner, padding_m)
    return sources


def apply_keypoint_sources_from_layout(
    model: ObjectModel,
    document: dict[str, Any],
    layout: MarkerLayout,
) -> ObjectModel:
    """Update keypoint positions from solved marker layout and source metadata.

    Args:
        model: Current object model to update.
        document: Parsed object-model JSON containing ``keypoint_sources``.
        layout: Solved marker layout with footprint geometry.

    Returns:
        Object model with keypoints moved to layout-derived corner positions.

    Raises:
        ValueError: If coordinate frame, keypoint names, or marker IDs are inconsistent.
    """
    coordinate_frame = document.get("coordinate_frame", MODEL_FRAME_NAME)
    if coordinate_frame != MODEL_FRAME_NAME:
        raise ValueError(
            f"object model coordinate_frame must be {MODEL_FRAME_NAME!r}, got {coordinate_frame!r}."
        )

    sources = parse_keypoint_sources(document)
    names = ordered_keypoint_names(document, model)
    updated = model
    for keypoint_name, (marker_id, corner, padding_m) in sources.items():
        if keypoint_name not in model.keypoints:
            raise ValueError(
                f"keypoint_sources.{keypoint_name} references unknown keypoint {keypoint_name!r}."
            )
        if marker_id not in layout.footprints:
            raise ValueError(
                f"keypoint_sources.{keypoint_name} references marker {marker_id}, "
                f"which is not present in the solved marker layout."
            )
        footprint = layout.footprints[marker_id]
        corner_point = footprint_corner_with_padding(footprint, corner, padding_m)
        updated = object_model_with_keypoint(
            updated,
            keypoint_name,
            corner_point,
            keypoint_names=names,
        )
    return updated


def missing_source_marker_ids(
    layout: MarkerLayout,
    sources: Mapping[str, tuple[int, str, float]],
) -> tuple[int, ...]:
    """Return sorted source marker IDs absent from a solved marker layout.

    Args:
        layout: Solved marker layout footprint map.
        sources: Parsed keypoint source metadata keyed by keypoint name.

    Returns:
        Sorted marker IDs referenced by ``sources`` but missing from ``layout``.
    """
    referenced = {marker_id for marker_id, _, _ in sources.values()}
    missing = sorted(referenced - set(layout.footprints.keys()))
    return tuple(missing)


def _keypoint_source_to_json(
    marker_id: int,
    corner: str,
    padding_m: float,
) -> dict[str, Any]:
    """Serialize one keypoint source entry for object-model JSON."""
    payload: dict[str, Any] = {"marker_id": marker_id, "corner": corner}
    if padding_m > 0.0:
        payload["padding_mm"] = float(padding_m * 1000.0)
    return payload


def build_object_model_document_from_layout(
    layout: MarkerLayout,
    sources: Mapping[str, tuple[int, str, float]],
    skeleton: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Build a complete object-model JSON document from layout-derived keypoints.

    Args:
        layout: Solved marker layout with footprint geometry.
        sources: Keypoint source metadata keyed by keypoint name.
        skeleton: Undirected skeleton edges as keypoint name pairs.

    Returns:
        Object-model JSON payload with generated ``keypoints``.

    Raises:
        ValueError: When skeleton coverage, source markers, or geometry are inconsistent.
    """
    skeleton_names = _skeleton_keypoint_names(skeleton)
    missing_sources = sorted(set(skeleton_names) - set(sources))
    if missing_sources:
        raise ValueError(
            f"skeleton keypoints are missing keypoint_sources: {missing_sources}."
        )

    missing_markers = missing_source_marker_ids(layout, sources)
    if missing_markers:
        raise ValueError(
            "keypoint_sources reference markers missing from the solved layout: "
            f"{list(missing_markers)}."
        )

    keypoints: dict[str, list[float]] = {}
    keypoint_sources_json: dict[str, dict[str, Any]] = {}
    for name, (marker_id, corner, padding_m) in sources.items():
        footprint = layout.footprints[marker_id]
        point = footprint_corner_with_padding(footprint, corner, padding_m)
        keypoints[name] = _layout_point_to_json(point)
        keypoint_sources_json[name] = _keypoint_source_to_json(marker_id, corner, padding_m)

    document = {
        "units": "meters",
        "coordinate_frame": MODEL_FRAME_NAME,
        "keypoint_sources": keypoint_sources_json,
        "keypoints": keypoints,
        "skeleton": [[start_name, end_name] for start_name, end_name in skeleton],
    }
    object_model_from_data(document)
    return document


def _skeleton_keypoint_names(skeleton: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Collect unique skeleton endpoint names in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for start_name, end_name in skeleton:
        for name in (start_name, end_name):
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def save_object_model_document(path: str | Path, document: dict[str, Any]) -> None:
    """Atomically write a validated object-model JSON document.

    Args:
        path: Destination ``object_model.json`` path.
        document: Parsed object-model JSON root object to serialize.

    Raises:
        ValueError: If the document fails object-model validation.
        OSError: If the temporary file cannot be written or replaced.
    """
    object_model_from_data(document)
    path = Path(path)
    text = json.dumps(document, indent=2) + "\n"
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


def ordered_keypoint_names(document: dict[str, Any], model: ObjectModel) -> tuple[str, ...]:
    """Return keypoint names in document order, appending any model-only names.

    Args:
        document: Parsed object-model JSON that may define a ``keypoints`` object.
        model: Object model supplying the authoritative keypoint set.

    Returns:
        Keypoint names in stable display/save order.
    """
    raw = document.get("keypoints")
    ordered: list[str] = []
    if isinstance(raw, dict):
        ordered.extend(str(name) for name in raw if str(name) in model.keypoints)
    for name in model.keypoint_names:
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def object_model_with_keypoint(
    model: ObjectModel,
    keypoint_id: str,
    point_layout: np.ndarray,
    *,
    keypoint_names: tuple[str, ...] | None = None,
) -> ObjectModel:
    """Return a copy of ``model`` with one keypoint position replaced.

    Args:
        model: Base object model.
        keypoint_id: Keypoint name to update.
        point_layout: New 3D position in the object layout frame.
        keypoint_names: Optional explicit keypoint ordering; defaults to ``model`` order.

    Returns:
        Updated object model with refreshed ``object_points`` array.
    """
    keypoints = dict(model.keypoints)
    keypoints[keypoint_id] = np.asarray(point_layout, dtype=np.float64).reshape(3)
    if keypoint_names is None:
        names = model.keypoint_names if keypoint_id in model.keypoint_names else model.keypoint_names + (keypoint_id,)
    else:
        names = keypoint_names
    object_points = np.asarray([keypoints[name] for name in names], dtype=np.float32)
    return ObjectModel(
        units=model.units,
        keypoint_names=names,
        keypoints=keypoints,
        skeleton_edges=model.skeleton_edges,
        object_points=object_points,
    )


def _layout_point_to_json(point: np.ndarray) -> list[float]:
    """Serialize a 3D layout point as a JSON-friendly float triple."""
    array = np.asarray(point, dtype=np.float64).reshape(3)
    return [float(array[0]), float(array[1]), float(array[2])]


def save_object_model_keypoints(
    path: str | Path,
    model: ObjectModel,
    document: dict[str, Any],
) -> None:
    """Atomically write updated keypoint coordinates back to an object model JSON file.

    Args:
        path: Destination ``object_model.json`` path.
        model: Object model with updated keypoint positions.
        document: Original JSON document to preserve non-keypoint fields.

    Notes:
        Uses a temporary file and ``os.replace`` so partial writes do not corrupt
        the on-disk model.
    """
    path = Path(path)
    payload = dict(document)
    names = ordered_keypoint_names(document, model)
    payload["keypoints"] = {
        name: _layout_point_to_json(model.keypoints[name])
        for name in names
    }
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
