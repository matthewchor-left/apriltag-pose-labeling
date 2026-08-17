"""Marker pose estimation and multi-marker fusion."""

from __future__ import annotations

import cv2
import numpy as np

from object_apriltag.layout import (
    MarkerLayout,
    OBJECT_AXIS_FLIP,
    layout_point_to_camera,
    layout_point_to_object_frame,
    object_reference_origin,
)

Detection = tuple[np.ndarray, int]
PoseTuple = tuple[np.ndarray, np.ndarray]

GLOBAL_PNP_REPROJECTION_ERROR_PX = 3.0
GLOBAL_PNP_ITERATIONS = 100
GLOBAL_PNP_CONFIDENCE = 0.99

# object_model.json uses +X left→right and +Z out of the rubber surface.


def marker_corner_object_points(marker_size_m: float) -> np.ndarray:
    """Return 3D marker corner points in the marker frame.

    Origin is at bottom-center; +X is marker-right; +Y points toward the tag top.
    Row order matches OpenCV/AprilTag detection order: top-left, top-right,
    bottom-right, bottom-left.

    Args:
        marker_size_m: Physical edge length of the square marker in meters.

    Returns:
        ``(4, 3)`` float32 array of corner coordinates.
    """
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, marker_size_m, 0.0],
            [half, marker_size_m, 0.0],
            [half, 0.0, 0.0],
            [-half, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_pose(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate marker pose from detected image corners via IPPE.

    Args:
        corners: Detected marker corners, shape ``(1, 4, 2)`` or ``(4, 2)``.
        marker_size_m: Physical marker edge length in meters.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        Tuple ``(rvec, tvec)`` giving marker pose in the OpenCV camera frame.

    Raises:
        RuntimeError: If ``cv2.solvePnP`` fails.
    """
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
    """Compute mean pixel reprojection error for a single marker detection.

    Args:
        corners: Detected marker corners, shape ``(1, 4, 2)`` or ``(4, 2)``.
        marker_size_m: Physical marker edge length in meters.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        Mean Euclidean distance in pixels between projected and detected corners.

    Raises:
        RuntimeError: If marker pose estimation fails.
    """
    rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    object_points = marker_corner_object_points(marker_size_m)
    image_points = corners.reshape(4, 2).astype(np.float32)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    return float(np.mean(np.linalg.norm(projected.reshape(4, 2) - image_points, axis=1)))


def mean_reprojection_error(
    detections: list[Detection],
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float | None:
    """Average per-marker reprojection error across a detection set.

    Markers that fail pose estimation or are absent from the layout are skipped.

    Args:
        detections: List of ``(corners, marker_id)`` detections.
        layout: Marker layout with per-marker physical sizes.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        Mean pixel error over successful markers, or ``None`` when no marker
        could be evaluated.
    """
    errors: list[float] = []
    for corners, marker_id in detections:
        try:
            marker_size_m = layout.marker_size_for(marker_id)
            errors.append(marker_reprojection_error(corners, marker_size_m, camera_matrix, dist_coeffs))
        except (RuntimeError, KeyError):
            continue
    return float(np.mean(errors)) if errors else None


def layout_reprojection_errors(
    detections: list[Detection],
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float] | None:
    """Measure layout-consistency reprojection error for detected markers.

    Projects layout footprint corners through the supplied object pose and
    compares them to detected image corners.

    Args:
        detections: List of ``(corners, marker_id)`` detections.
        object_rotation: ``(3, 3)`` object rotation in the camera frame.
        object_origin: Object origin in the camera frame, meters.
        layout: Marker layout with footprint geometry.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        Tuple ``(mean_error_px, max_error_px)``, or ``None`` when no valid
        corner correspondences exist.
    """
    errors: list[float] = []
    zero_rvec = np.zeros(3, dtype=np.float64)
    zero_tvec = np.zeros(3, dtype=np.float64)
    for corners, marker_id in detections:
        if marker_id not in layout.footprints:
            continue
        footprint = layout.footprints[marker_id]
        detected = corners.reshape(4, 2).astype(np.float64)
        for index, point_layout in enumerate(footprint.corners()):
            camera_point = layout_point_to_camera(
                point_layout, object_rotation, object_origin, layout
            )
            projected, _ = cv2.projectPoints(
                camera_point.reshape(1, 1, 3).astype(np.float64),
                zero_rvec,
                zero_tvec,
                camera_matrix,
                dist_coeffs,
            )
            projected_xy = projected.reshape(2)
            if not np.all(np.isfinite(projected_xy)):
                continue
            errors.append(float(np.linalg.norm(projected_xy - detected[index])))
    if not errors:
        return None
    return float(np.mean(errors)), float(np.max(errors))


def reference_marker_camera_position(
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    layout: MarkerLayout,
) -> np.ndarray:
    """Return the reference marker center in the OpenCV camera frame.

    Args:
        object_rotation: ``(3, 3)`` object rotation in the camera frame.
        object_origin: Object origin in the camera frame, meters.
        layout: Marker layout defining the reference marker footprint.

    Returns:
        ``(3,)`` camera-frame position of the reference marker center in meters.
    """
    return layout_point_to_camera(
        object_reference_origin(layout),
        object_rotation,
        object_origin,
        layout,
    )


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion.

    Args:
        rotation: ``(3, 3)`` rotation matrix.

    Returns:
        ``(4,)`` quaternion in ``(w, x, y, z)`` order.
    """
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
    """Convert a unit quaternion to a rotation matrix.

    Args:
        quaternion: ``(4,)`` quaternion in ``(w, x, y, z)`` order.

    Returns:
        ``(3, 3)`` rotation matrix.
    """
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
    """Fuse multiple rotation matrices by quaternion averaging.

    Args:
        rotations: List of ``(3, 3)`` rotation matrices.

    Returns:
        Fused ``(3, 3)`` rotation matrix, or ``None`` when input is empty or
        averaging degenerates.
    """
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


def object_pose_from_marker_pose(
    rvec: np.ndarray,
    tvec: np.ndarray,
    marker_id: int,
    layout: MarkerLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a single-marker camera pose into object-frame pose.

    Args:
        rvec: Marker rotation as a Rodrigues vector.
        tvec: Marker translation in the camera frame, meters.
        marker_id: ID of the observed marker in the layout.
        layout: Marker layout with per-marker transforms.

    Returns:
        Tuple ``(object_rotation, object_origin)`` in the OpenCV camera frame.

    Raises:
        RuntimeError: If the derived object rotation is improper (det < 0).
    """
    marker_rotation, _ = cv2.Rodrigues(rvec)
    marker_rotation = marker_rotation.astype(np.float64)
    transform = layout.transforms[marker_id]
    object_rotation = marker_rotation @ transform.rotation @ OBJECT_AXIS_FLIP
    if np.linalg.det(object_rotation) < 0.0:
        raise RuntimeError(f"Object rotation for marker {marker_id} is improper.")
    object_origin = marker_rotation @ transform.offset + tvec.reshape(3)
    return object_rotation, object_origin.astype(np.float64)


def object_pose_from_marker(
    corners: np.ndarray,
    marker_id: int,
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate object pose from one detected marker.

    Args:
        corners: Detected marker corners, shape ``(1, 4, 2)`` or ``(4, 2)``.
        marker_id: ID of the observed marker in the layout.
        layout: Marker layout with per-marker sizes and transforms.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        Tuple ``(object_rotation, object_origin)`` in the OpenCV camera frame.

    Raises:
        RuntimeError: If marker pose estimation fails or object rotation is improper.
    """
    marker_size_m = layout.marker_size_for(marker_id)
    rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    return object_pose_from_marker_pose(rvec, tvec, marker_id, layout)


def _global_pose_correspondences(
    detections: list[Detection],
    layout: MarkerLayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D-2D correspondences for global layout pose estimation.

    Args:
        detections: List of ``(corners, marker_id)`` detections.
        layout: Marker layout with footprint geometry.

    Returns:
        Tuple ``(object_points, image_points, marker_ids)`` where
        ``object_points`` is ``(N, 3)``, ``image_points`` is ``(N, 2)``, and
        ``marker_ids`` records the source marker for each correspondence.
    """
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    marker_ids: list[int] = []
    for corners, marker_id in detections:
        footprint = layout.footprints.get(marker_id)
        if footprint is None:
            continue
        detected = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        if not np.all(np.isfinite(detected)):
            continue
        for point_layout, point_image in zip(
            footprint.corners(), detected, strict=True
        ):
            object_points.append(layout_point_to_object_frame(point_layout, layout))
            image_points.append(point_image)
            marker_ids.append(marker_id)
    return (
        np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        np.asarray(marker_ids, dtype=np.int32),
    )


def _has_multiple_ippe_solutions(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> bool:
    """Return whether IPPE yields multiple pose hypotheses for the correspondences.

    Args:
        object_points: ``(N, 3)`` 3D points in the object frame.
        image_points: ``(N, 2)`` corresponding image points.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        ``True`` when ``cv2.solvePnPGeneric`` reports more than one IPPE solution.
    """
    if object_points.shape[0] < 4:
        return False
    try:
        ok, rvecs, _, _ = cv2.solvePnPGeneric(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return False
    if not ok or rvecs is None:
        return False
    return len(rvecs) > 1


def estimate_global_layout_pose(
    detections: list[Detection],
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> PoseTuple | None:
    """Estimate layout-wide object pose from multi-marker RANSAC and LM refinement.

    Requires at least two distinct markers with inlier correspondences after
    RANSAC. Returns ``None`` when ambiguity, insufficient markers, or invalid
    geometry is detected.

    Args:
        detections: List of ``(corners, marker_id)`` detections.
        layout: Marker layout with footprint geometry.
        camera_matrix: ``(3, 3)`` camera intrinsics matrix.
        dist_coeffs: Distortion coefficients for the camera model.

    Returns:
        Tuple ``(object_origin, object_rotation)`` in the OpenCV camera frame,
        or ``None`` when pose cannot be estimated reliably.
    """
    object_points, image_points, marker_ids = _global_pose_correspondences(
        detections, layout
    )

    if len(set(marker_ids.tolist())) < 2:
        return None
    if _has_multiple_ippe_solutions(
        object_points, image_points, camera_matrix, dist_coeffs
    ):
        return None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=GLOBAL_PNP_ITERATIONS,
            reprojectionError=GLOBAL_PNP_REPROJECTION_ERROR_PX,
            confidence=GLOBAL_PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None
    if not ok or inliers is None:
        return None

    inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
    if len(set(marker_ids[inlier_indices].tolist())) < 2:
        return None
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[inlier_indices],
            image_points[inlier_indices],
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )
    except cv2.error:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    origin = np.asarray(tvec, dtype=np.float64).reshape(3)
    camera_points = (
        np.asarray(rotation, dtype=np.float64)
        @ object_points[inlier_indices].T
    ).T + origin
    if (
        not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(origin))
        or np.any(camera_points[:, 2] <= 0.0)
    ):
        return None
    return origin, np.asarray(rotation, dtype=np.float64)
