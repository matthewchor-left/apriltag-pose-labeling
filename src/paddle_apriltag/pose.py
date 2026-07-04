"""Marker pose estimation and multi-marker fusion."""

from __future__ import annotations

import cv2
import numpy as np

from paddle_apriltag.layout import MarkerLayout

Detection = tuple[np.ndarray, int]


def marker_corner_object_points(marker_size_m: float) -> np.ndarray:
    """3D corners in marker frame: origin at bottom-center, +Y toward tag top."""
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, 0.0, 0.0],
            [half, 0.0, 0.0],
            [half, marker_size_m, 0.0],
            [-half, marker_size_m, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_pose(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = marker_corner_object_points(marker_size_m)
    image_points = corners.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed to estimate marker pose.")
    return rvec, tvec


def marker_reprojection_error(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    object_points = marker_corner_object_points(marker_size_m)
    image_points = corners.reshape(4, 2).astype(np.float32)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)))


def mean_reprojection_error(
    detections: list[Detection],
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float | None:
    errors: list[float] = []
    for corners, _ in detections:
        try:
            errors.append(marker_reprojection_error(corners, marker_size_m, camera_matrix, dist_coeffs))
        except RuntimeError:
            continue
    return float(np.mean(errors)) if errors else None


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s],
            dtype=np.float64,
        )
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return np.array(
            [(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
             (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s],
            dtype=np.float64,
        )
    if rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return np.array(
            [(rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s,
             0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s],
            dtype=np.float64,
        )
    s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return np.array(
        [(rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s,
         (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s],
        dtype=np.float64,
    )


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def fuse_rotations(rotations: list[np.ndarray]) -> np.ndarray | None:
    if not rotations:
        return None
    if len(rotations) == 1:
        return rotations[0].astype(np.float64)

    quaternions = [_rotation_matrix_to_quaternion(rotation) for rotation in rotations]
    reference = quaternions[0]
    aligned = [q if np.dot(q, reference) >= 0.0 else -q for q in quaternions]
    mean_quaternion = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean_quaternion)
    if norm <= 0.0:
        return None
    return _quaternion_to_rotation_matrix(mean_quaternion / norm)


def paddle_pose_from_marker_pose(
    rvec: np.ndarray,
    tvec: np.ndarray,
    marker_id: int,
    layout: MarkerLayout,
) -> tuple[np.ndarray, np.ndarray]:
    marker_rotation, _ = cv2.Rodrigues(rvec)
    marker_rotation = marker_rotation.astype(np.float64)
    transform = layout.transforms[marker_id]
    paddle_rotation = marker_rotation @ transform.rotation
    if np.linalg.det(paddle_rotation) < 0.0:
        raise RuntimeError(f"Paddle rotation for marker {marker_id} is improper.")
    paddle_origin = marker_rotation @ transform.offset + tvec.reshape(3)
    return paddle_rotation, paddle_origin.astype(np.float64)


def paddle_pose_from_marker(
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    layout: MarkerLayout,
) -> tuple[np.ndarray, np.ndarray]:
    rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    return paddle_pose_from_marker_pose(rvec, tvec, marker_id, layout)


def estimate_fused_pose(
    detections: list[Detection],
    layout: MarkerLayout,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    origins: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for corners, marker_id in detections:
        try:
            paddle_rotation, paddle_origin = paddle_pose_from_marker(
                corners, marker_id, marker_size_m, camera_matrix, dist_coeffs, layout
            )
            origins.append(paddle_origin)
            rotations.append(paddle_rotation)
        except (RuntimeError, KeyError):
            continue

    if not origins:
        return None, None
    fused_origin = np.mean(np.stack(origins, axis=0), axis=0)
    fused_rotation = fuse_rotations(rotations)
    return fused_origin, fused_rotation
