"""Camera projection helpers for visualization."""

from __future__ import annotations

import cv2
import numpy as np


def project_camera_point(
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


def _to_xy(point: np.ndarray) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


# Paddle frame unit vectors: +X left->right, +Y handle->tip, +Z out of rubber.
PADDLE_UNIT_AXES = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def paddle_axis_image_points(
    paddle_rotation: np.ndarray,
    paddle_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Return origin and paddle +X/+Y/+Z axis endpoints in the image."""
    origin_xy = project_camera_point(paddle_origin, camera_matrix, dist_coeffs)
    ends = [
        project_camera_point(
            paddle_origin + paddle_rotation @ (axis * axis_length_m),
            camera_matrix,
            dist_coeffs,
        )
        for axis in PADDLE_UNIT_AXES
    ]
    return _to_xy(origin_xy), _to_xy(ends[0]), _to_xy(ends[1]), _to_xy(ends[2])


def marker_axis_image_points(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float | None = None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    from paddle_aruco.pose import estimate_marker_pose

    if axis_length_m is None:
        axis_length_m = marker_size_m * 0.35

    rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    axis_points = np.array([[0.0, axis_length_m, 0.0], [axis_length_m, 0.0, 0.0]], dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(2, 2)

    origin = project_camera_point(tvec.reshape(3), camera_matrix, dist_coeffs)
    origin_xy = (int(round(origin[0])), int(round(origin[1])))
    y_end = (int(round(projected[0, 0])), int(round(projected[0, 1])))
    x_end = (int(round(projected[1, 0])), int(round(projected[1, 1])))
    return origin_xy, y_end, x_end


def paddle_origin_image_coords(
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    layout,
) -> tuple[int, int]:
    from paddle_aruco.pose import paddle_pose_from_marker

    paddle_rotation, paddle_origin = paddle_pose_from_marker(
        corners, marker_id, marker_size_m, camera_matrix, dist_coeffs, layout
    )
    del paddle_rotation
    point = project_camera_point(paddle_origin, camera_matrix, dist_coeffs)
    return int(round(point[0])), int(round(point[1]))
