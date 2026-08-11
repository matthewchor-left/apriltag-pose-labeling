"""Load eraser plane geometry for live tag masking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from paddle_apriltag.layout import CORNER_NAMES

DEFAULT_ERASER_MODEL_PATH = Path("calibration") / "eraser_model.json"
ERASER_ORIGIN_REFERENCE_MARKER_CENTER = "reference_marker_center"


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

    plane_id = payload.get("id")
    if plane_id is not None:
        plane_id = str(plane_id)

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
