"""Load camera intrinsics and resolve config profile paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CAMERA_PROFILE = "nexplaygroundcam"
DEFAULT_OBJECT_PROFILE = "object_01"


def config_dir() -> Path:
    """Return the repo ``config/`` directory."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config"
        if (candidate / "Model").is_dir() and (candidate / "Camera").is_dir():
            return candidate
    return Path("config")


CONFIG_DIR = config_dir()
CAMERA_DIR = CONFIG_DIR / "Camera"
MODEL_DIR = CONFIG_DIR / "Model"


def camera_profile_dir(profile: str) -> Path:
    return CAMERA_DIR / profile


def object_profile_dir(profile: str) -> Path:
    return MODEL_DIR / profile


def camera_intrinsics_path(profile: str = DEFAULT_CAMERA_PROFILE) -> Path:
    return camera_profile_dir(profile) / "intrinsics.json"


def camera_uvcc_path(profile: str = DEFAULT_CAMERA_PROFILE) -> Path:
    return camera_profile_dir(profile) / "uvcc.json"


def camera_device_path(profile: str = DEFAULT_CAMERA_PROFILE) -> Path:
    return camera_profile_dir(profile) / "device.json"


def marker_model_path(profile: str = DEFAULT_OBJECT_PROFILE) -> Path:
    return object_profile_dir(profile) / "marker_model.json"


def eraser_model_path(profile: str = DEFAULT_OBJECT_PROFILE) -> Path:
    return object_profile_dir(profile) / "eraser_model.json"


def object_model_path(profile: str = DEFAULT_OBJECT_PROFILE) -> Path:
    return object_profile_dir(profile) / "object_model.json"


DEFAULT_INTRINSICS_PATH = camera_intrinsics_path()
DEFAULT_CALIBRATION_PATH = DEFAULT_INTRINSICS_PATH
DEFAULT_MARKER_MODEL_PATH = marker_model_path()
DEFAULT_ERASER_MODEL_PATH = eraser_model_path()
DEFAULT_OBJECT_MODEL_PATH = object_model_path()


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


def require_calibration_image_size(
    image_width: int | None,
    image_height: int | None,
    path: str | Path,
) -> tuple[int, int]:
    if image_width is None or image_height is None:
        raise ValueError(
            f"Calibration file must contain image_width and image_height (or image_size): {path}"
        )
    return image_width, image_height
