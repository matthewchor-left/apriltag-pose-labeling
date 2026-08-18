"""Marker pose estimation and multi-marker fusion."""

from __future__ import annotations

import cv2
import numpy as np

from object_apriltag.layout import (
    MarkerLayout,
    layout_point_to_camera,
    layout_point_to_object_frame,
    object_reference_origin,
)

Detection = tuple[np.ndarray, int]
PoseTuple = tuple[np.ndarray, np.ndarray]

GLOBAL_PNP_REPROJECTION_ERROR_PX = 3.0
GLOBAL_PNP_ITERATIONS = 100
GLOBAL_PNP_CONFIDENCE = 0.99
# Observed-image gate before global PnP (see estimate_global_layout_pose).
GLOBAL_MARKER_MIN_MEAN_EDGE_PX = 25.0
# Max mean corner reprojection error as a fraction of observed mean marker edge
# length. 0.05 ~= subpixel corner noise on a tag resolved to ~20+ px/edge.
GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR = 0.05
# Minimum angle (degrees) between marker plane normals across every valid IPPE
# branch combination for a pair to count as confidently nonparallel.
GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG = 20.0

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
    object_rotation = marker_rotation @ transform.rotation
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


def _detected_corners_valid(corners: np.ndarray) -> bool:
    """Return whether detected marker corners are finite and non-degenerate."""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        return False
    twice_area = 0.0
    for index in range(4):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % 4]
        twice_area += x1 * y2 - x2 * y1
    return abs(twice_area) > 1e-6


def _mean_marker_edge_length_px(corners: np.ndarray) -> float:
    """Return the mean side length of a quadrilateral marker in image pixels."""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return float(np.mean(edges))


def _mean_reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3).astype(np.float32),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected_xy = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(image_points.reshape(-1, 2) - projected_xy, axis=1)))


def _marker_corners_in_front_of_camera(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> bool:
    """Return whether all marker corners lie in front of the camera (Z > 0).

    Runtime IPPE gating uses this cheirality check on the four transformed
    corners rather than calibration's marker +Z normal heuristic. That matches
    the trust boundary here: reject physically impossible poses while keeping
    plane-normal comparisons unsigned (``abs(dot)``) across IPPE branches.
    """
    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (np.asarray(rotation, dtype=np.float64) @ object_points.T).T
    camera_points += np.asarray(tvec, dtype=np.float64).reshape(1, 3)
    return bool(
        np.all(np.isfinite(camera_points))
        and np.all(camera_points[:, 2] > 0.0)
    )


def _marker_plane_normal_camera(rotation: np.ndarray) -> np.ndarray | None:
    normal = np.asarray(rotation, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0 or not np.isfinite(norm):
        return None
    return normal / norm


def _relative_reprojection_error(
    mean_error_px: float,
    mean_edge_px: float,
) -> float | None:
    if mean_edge_px <= 0.0 or not np.isfinite(mean_edge_px):
        return None
    if not np.isfinite(mean_error_px):
        return None
    return mean_error_px / mean_edge_px


def _ippe_marker_candidates(
    corners: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Solve IPPE once and return valid branch hypotheses for the observed gate.

    Each candidate must have finite pose, all four marker corners in front of
    the camera, and relative reprojection error within
    ``GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR``. Returns an empty list when
    corners are non-finite, degenerate, too small, or no branch passes.

    Returns:
        List of ``(rvec, rotation, plane_normal, relative_reproj_error)`` tuples.
    """
    if not _detected_corners_valid(corners):
        return []

    mean_edge_px = _mean_marker_edge_length_px(corners)
    if mean_edge_px < GLOBAL_MARKER_MIN_MEAN_EDGE_PX:
        return []

    object_points = marker_corner_object_points(marker_size_m)
    image_points = corners.reshape(4, 2).astype(np.float32)
    try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return []
    if not ok or rvecs is None or tvecs is None:
        return []

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        rotation = np.asarray(rotation, dtype=np.float64)
        if not np.all(np.isfinite(rotation)):
            continue
        if not _marker_corners_in_front_of_camera(object_points, rvec, tvec):
            continue
        plane_normal = _marker_plane_normal_camera(rotation)
        if plane_normal is None:
            continue
        mean_error_px = _mean_reprojection_error_px(
            object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        relative_error = _relative_reprojection_error(mean_error_px, mean_edge_px)
        if relative_error is None or relative_error > GLOBAL_MARKER_MAX_RELATIVE_REPROJ_ERROR:
            continue
        candidates.append((rvec, rotation, plane_normal, relative_error))
    return candidates


def _pair_minimum_normal_angle_deg(
    normals_a: list[np.ndarray],
    normals_b: list[np.ndarray],
) -> float | None:
    """Return the smallest unsigned normal angle across all branch combinations."""
    if not normals_a or not normals_b:
        return None
    min_angle_rad = float(np.pi)
    for normal_a in normals_a:
        for normal_b in normals_b:
            cosine = float(np.clip(abs(np.dot(normal_a, normal_b)), 0.0, 1.0))
            min_angle_rad = min(min_angle_rad, float(np.arccos(cosine)))
    return float(np.rad2deg(min_angle_rad))


def _observed_marker_plane_gate_passes(
    detections: list[Detection],
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> bool:
    """Return whether observed marker planes are confidently nonparallel.

    Builds per-marker IPPE candidate pools from image corners (one IPPE solve
    per marker ID), then requires at least one marker pair whose minimum unsigned
    plane-normal angle across every valid branch combination is at least
    ``GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG``.

    Duplicate detections for the same marker ID are ignored after the first
    in-list occurrence with a layout footprint. ``ObjectDetector`` normally emits
    unique IDs; this policy hardens the trust boundary for arbitrary callers.

    This gate tests **nonparallel observed planes**, not exact non-coplanarity.
    Parallel planes at different depths remain conservatively rejected because
    their image-observed normals stay aligned.
    """
    reliable: dict[int, tuple[float, list[np.ndarray], list[float]]] = {}
    processed_marker_ids: set[int] = set()
    detected_layout_marker_ids = sorted(
        {
            marker_id
            for _, marker_id in detections
            if marker_id in layout.footprints
        }
    )
    for corners, marker_id in detections:
        if marker_id not in layout.footprints:
            continue
        if marker_id in processed_marker_ids:
            continue
        processed_marker_ids.add(marker_id)
        try:
            marker_size_m = layout.marker_size_for(marker_id)
        except KeyError:
            continue
        try:
            candidates = _ippe_marker_candidates(
                corners, marker_size_m, camera_matrix, dist_coeffs
            )
        except (ValueError, TypeError):
            continue
        if not candidates:
            continue
        mean_edge_px = _mean_marker_edge_length_px(corners)
        normals = [candidate[2] for candidate in candidates]
        rel_errors = [candidate[3] for candidate in candidates]
        reliable[marker_id] = (mean_edge_px, normals, rel_errors)

    print(
        "[pose observability] "
        f"markers={detected_layout_marker_ids}"
    )
    for marker_id in sorted(reliable):
        mean_edge_px, _, rel_errors = reliable[marker_id]
        rel_text = ",".join(f"{error:.4f}" for error in rel_errors)
        print(
            "[pose observability] "
            f"m{marker_id}: edge={mean_edge_px:.1f}px rel_err=[{rel_text}]"
        )

    reliable_ids = sorted(reliable)
    confident_pair = False
    min_angle_threshold = GLOBAL_MARKER_PAIR_MIN_NORMAL_ANGLE_DEG
    for left_index in range(len(reliable_ids)):
        for right_index in range(left_index + 1, len(reliable_ids)):
            marker_a = reliable_ids[left_index]
            marker_b = reliable_ids[right_index]
            _, normals_a, _ = reliable[marker_a]
            _, normals_b, _ = reliable[marker_b]
            min_angle = _pair_minimum_normal_angle_deg(normals_a, normals_b)
            if min_angle is None:
                pair_pass = False
            else:
                pair_pass = min_angle >= min_angle_threshold
                if pair_pass:
                    confident_pair = True
            status = "pass" if pair_pass else "fail"
            angle_text = "n/a" if min_angle is None else f"{min_angle:.1f}"
            print(
                "[pose observability] "
                f"pair({marker_a},{marker_b}): min_angle={angle_text}° {status}"
            )
    return confident_pair


def estimate_global_layout_pose(
    detections: list[Detection],
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> PoseTuple | None:
    """Estimate layout-wide object pose from multi-marker RANSAC and LM refinement.

    Requires at least two distinct markers with inlier correspondences after
    RANSAC. Returns ``None`` when observed marker planes are not confidently
    nonparallel (see ``_observed_marker_plane_gate_passes``), when marker count
    is insufficient, or when RANSAC or refinement fails.

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
    if not _observed_marker_plane_gate_passes(
        detections, layout, camera_matrix, dist_coeffs
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
