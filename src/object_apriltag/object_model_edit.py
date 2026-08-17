"""Object model load/save and keypoint source helpers."""

from __future__ import annotations

import json
import os
import tempfile
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
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    return object_model_from_data(document), document


def _parse_source_marker_id(value: Any, field_name: str) -> int:
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


def ordered_keypoint_names(document: dict[str, Any], model: ObjectModel) -> tuple[str, ...]:
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
    array = np.asarray(point, dtype=np.float64).reshape(3)
    return [float(array[0]), float(array[1]), float(array[2])]


def save_object_model_keypoints(
    path: str | Path,
    model: ObjectModel,
    document: dict[str, Any],
) -> None:
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
