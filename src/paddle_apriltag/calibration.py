"""Load camera calibration JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CALIBRATION_DIR = Path("calibration")
DEFAULT_CALIBRATION_PATH = DEFAULT_CALIBRATION_DIR / "camera_calibration.json"
DEFAULT_MARKER_LAYOUT_PATH = DEFAULT_CALIBRATION_DIR / "marker_layout.json"
DEFAULT_PADDLE_MODEL_PATH = DEFAULT_CALIBRATION_DIR / "paddle_model.json"


def save_calibration_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_calibration_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None, str | None]:
    data = load_calibration_dict(path)
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)

    image_width = None
    image_height = None
    if "image_width" in data and "image_height" in data:
        image_width = int(data["image_width"])
        image_height = int(data["image_height"])
    elif "image_size" in data:
        image_width = int(data["image_size"][0])
        image_height = int(data["image_size"][1])

    source = str(data["calibration_source"]) if "calibration_source" in data else None
    return camera_matrix, dist_coeffs, image_width, image_height, source
