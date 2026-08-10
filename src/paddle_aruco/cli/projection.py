"""Convert an OpenCV camera calibration into OpenGL projection data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from paddle_aruco.calibration import DEFAULT_CALIBRATION_PATH, load_intrinsics


def camera_projection(
    camera_matrix: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
) -> np.ndarray:
    """Return an OpenGL projection matrix for OpenGL camera coordinates.

    OpenGL camera coordinates use +X right, +Y up, and -Z forward. Convert an
    OpenCV camera-space point with ``diag(1, -1, -1, 1)`` before this matrix.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not 0 < near < far:
        raise ValueError("Require 0 < near < far.")

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    return np.array(
        [
            [2 * fx / width, 0, 1 - 2 * cx / width, 0],
            [0, 2 * fy / height, 2 * cy / height - 1, 0],
            [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0],
        ],
        dtype=np.float64,
    )


def calibration_graphics_data(
    camera_matrix: np.ndarray,
    width: int,
    height: int,
    near: float,
    far: float,
    sensor_width_mm: float | None = None,
    sensor_height_mm: float | None = None,
) -> dict[str, object]:
    """Derive graphics-relevant values from OpenCV intrinsics."""
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    data: dict[str, object] = {
        "image_size_px": [width, height],
        "focal_length_px": {"x": fx, "y": fy},
        "principal_point_px": {"x": cx, "y": cy},
        "field_of_view_degrees": {
            "horizontal": math.degrees(2 * math.atan(width / (2 * fx))),
            "vertical": math.degrees(2 * math.atan(height / (2 * fy))),
        },
        "near_far": {"near": near, "far": far},
        "opencv_to_opengl_camera": np.diag([1.0, -1.0, -1.0, 1.0]).tolist(),
        "opengl_projection_matrix": camera_projection(
            camera_matrix, width, height, near, far
        ).tolist(),
    }
    if sensor_width_mm is not None or sensor_height_mm is not None:
        if sensor_width_mm is None or sensor_height_mm is None:
            raise ValueError("Provide both sensor dimensions, or neither.")
        if sensor_width_mm <= 0 or sensor_height_mm <= 0:
            raise ValueError("Sensor dimensions must be positive.")
        pixel_pitch_x = sensor_width_mm / width
        pixel_pitch_y = sensor_height_mm / height
        data["pixel_density_px_per_mm"] = {
            "x": width / sensor_width_mm,
            "y": height / sensor_height_mm,
        }
        data["pixel_pitch_mm"] = {"x": pixel_pitch_x, "y": pixel_pitch_y}
        data["focal_length_mm"] = {"x": fx * pixel_pitch_x, "y": fy * pixel_pitch_y}
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an OpenCV calibration file to OpenGL projection data."
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--near", type=float, default=0.01, help="Near plane in metres.")
    parser.add_argument("--far", type=float, default=100.0, help="Far plane in metres.")
    parser.add_argument("--sensor-width-mm", type=float)
    parser.add_argument("--sensor-height-mm", type=float)
    args = parser.parse_args()

    camera_matrix, _, width, height, _ = load_intrinsics(args.calibration)
    if width is None or height is None:
        raise ValueError("Calibration file must contain image dimensions.")
    print(
        json.dumps(
            calibration_graphics_data(
                camera_matrix,
                width,
                height,
                args.near,
                args.far,
                args.sensor_width_mm,
                args.sensor_height_mm,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
