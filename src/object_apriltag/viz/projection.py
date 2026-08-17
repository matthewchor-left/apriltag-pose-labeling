"""Camera projection helpers for visualization."""

from __future__ import annotations

import cv2
import numpy as np

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def opencv_image_point(point_xy: np.ndarray | tuple[float, float]) -> tuple[int, int] | None:
    """Convert a projected point to an OpenCV-safe integer pixel coordinate.

    Args:
        point_xy: Image coordinates as ``(x, y)``.

    Returns:
        Rounded integer pixel coordinate, or ``None`` when unusable.
    """
    xy = np.asarray(point_xy, dtype=np.float64).reshape(2)
    if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
        return None
    x = int(round(xy[0]))
    y = int(round(xy[1]))
    if x < INT32_MIN or x > INT32_MAX or y < INT32_MIN or y > INT32_MAX:
        return None
    return x, y


def _require_opencv_image_point(point_xy: np.ndarray | tuple[float, float]) -> tuple[int, int]:
    """Require an OpenCV-safe integer pixel coordinate.

    Args:
        point_xy: Image coordinates as ``(x, y)``.

    Returns:
        Rounded integer pixel coordinate.

    Raises:
        ValueError: If the point is non-finite or outside int32 range.
    """
    point = opencv_image_point(point_xy)
    if point is None:
        raise ValueError("projected image point is not OpenCV-safe")
    return point


def project_camera_point(
    point_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project one camera-frame 3D point to image coordinates.

    Args:
        point_camera: 3D point in the camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        Image coordinates as ``(2,)`` float array.
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


def object_axis_image_points(
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Project fused object X/Y/Z axis tips and origin to image coordinates.

    Args:
        object_rotation: Object rotation in camera frame.
        object_origin: Object origin in camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        axis_length_m: Axis length in meters.

    Returns:
        Tuple of origin, X-tip, Y-tip, and Z-tip integer pixel coordinates.

    Raises:
        ValueError: If any projected point is not OpenCV-safe.
    """
    axis_points = np.array(
        [
            object_origin + object_rotation @ np.array([axis_length_m, 0.0, 0.0]),
            object_origin + object_rotation @ np.array([0.0, axis_length_m, 0.0]),
            object_origin + object_rotation @ np.array([0.0, 0.0, axis_length_m]),
        ],
        dtype=np.float64,
    )
    origin_xy = project_camera_point(object_origin, camera_matrix, dist_coeffs)
    x_xy = project_camera_point(axis_points[0], camera_matrix, dist_coeffs)
    y_xy = project_camera_point(axis_points[1], camera_matrix, dist_coeffs)
    z_xy = project_camera_point(axis_points[2], camera_matrix, dist_coeffs)
    return (
        _require_opencv_image_point(origin_xy),
        _require_opencv_image_point(x_xy),
        _require_opencv_image_point(y_xy),
        _require_opencv_image_point(z_xy),
    )


def marker_axis_image_points(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float | None = None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Project marker-local X/Y axes from detected corners to image coordinates.

    Args:
        corners: Detected marker corners.
        marker_size_m: Physical marker edge length in meters.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        axis_length_m: Axis length in meters; defaults to ``0.35 * marker_size_m``.

    Returns:
        Tuple of origin, Y-tip, and X-tip integer pixel coordinates.

    Raises:
        ValueError: If any projected point is not OpenCV-safe.
    """
    from object_apriltag.pose import estimate_marker_pose

    if axis_length_m is None:
        axis_length_m = marker_size_m * 0.35

    rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    axis_points = np.array([[0.0, axis_length_m, 0.0], [axis_length_m, 0.0, 0.0]], dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(2, 2)

    origin = project_camera_point(tvec.reshape(3), camera_matrix, dist_coeffs)
    origin_xy = _require_opencv_image_point(origin)
    y_end = _require_opencv_image_point(projected[0])
    x_end = _require_opencv_image_point(projected[1])
    return origin_xy, y_end, x_end


def object_origin_image_coords(
    corners: np.ndarray,
    marker_id: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    layout,
) -> tuple[int, int]:
    """Project the fused object origin inferred from one marker to image coordinates.

    Args:
        corners: Detected marker corners.
        marker_id: Marker ID used to infer object pose.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        layout: Marker layout.

    Returns:
        Object origin as integer pixel coordinates.

    Raises:
        ValueError: If the projected origin is not OpenCV-safe.
    """
    from object_apriltag.pose import object_pose_from_marker

    object_rotation, object_origin = object_pose_from_marker(
        corners, marker_id, layout, camera_matrix, dist_coeffs
    )
    del object_rotation
    point = project_camera_point(object_origin, camera_matrix, dist_coeffs)
    return _require_opencv_image_point(point)
