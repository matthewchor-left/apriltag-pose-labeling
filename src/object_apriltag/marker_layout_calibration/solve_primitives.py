"""Shared marker-layout solve types and pure geometry primitives."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration.types import MarkerPair

OptimizationCheckpointStage = Literal[
    "graph_initialization",
    "initial_bundle_adjustment",
    "post_pruning_refit",
]


@dataclass
class BundleAdjustmentRunProfiler:
    """Per-run bundle-adjustment sector timings and projection counts.

    Timing accounting (do not double-count):
    - Per-run BA wall sectors sum as setup + least_squares + post.
    - ``residual_callback_total`` is measured inside ``least_squares`` only.
    - ``least_squares_overhead`` is the unobservable SciPy remainder (sparse
      Jacobian finite-difference probes, linear algebra, etc.).

    Count semantics:
    - ``projection_calls``: total per-corner projection evaluations (scalar-era
      metric; one per corner sent through the projection path each callback).
    - ``opencv_projectpoints_invocations``: ``cv2.projectPoints`` calls (one per
      residual callback when poses are finite).
    - ``batched_corner_count``: corners batched into each ``projectPoints`` call,
      summed across callbacks.
    - ``projection_loop`` (timing): pose gathering, batched transforms, OpenCV
      projection, and residual assembly inside each residual callback.
    """

    parameter_count: int = 0
    residual_count: int = 0
    residual_callback_invocations: int = 0
    projection_calls: int = 0
    opencv_projectpoints_invocations: int = 0
    batched_corner_count: int = 0
    setup_seconds: float = 0.0
    least_squares_seconds: float = 0.0
    post_seconds: float = 0.0
    residual_callback_total_seconds: float = 0.0
    residual_unpack_seconds: float = 0.0
    projection_loop_seconds: float = 0.0

    def timing_seconds(self) -> dict[str, float]:
        """Return sector timings for one bundle-adjustment run.

        Returns:
            Dict with setup, least_squares, post, residual callback sectors, and
            ``least_squares_overhead`` (SciPy remainder not attributed to callbacks).
        """
        residual_callback_other = max(
            0.0,
            self.residual_callback_total_seconds
            - self.residual_unpack_seconds
            - self.projection_loop_seconds,
        )
        least_squares_overhead = max(
            0.0,
            self.least_squares_seconds - self.residual_callback_total_seconds,
        )
        return {
            "setup": self.setup_seconds,
            "least_squares": self.least_squares_seconds,
            "post": self.post_seconds,
            "residual_callback_total": self.residual_callback_total_seconds,
            "residual_unpack": self.residual_unpack_seconds,
            "projection_loop": self.projection_loop_seconds,
            "residual_callback_other": residual_callback_other,
            "least_squares_overhead": least_squares_overhead,
        }

    def counts(self) -> dict[str, int]:
        """Return projection and residual callback counts for one BA run.

        Returns:
            Dict with parameter, residual, callback, projection, and batching counts.
        """
        return {
            "parameter_count": self.parameter_count,
            "residual_count": self.residual_count,
            "residual_callback_invocations": self.residual_callback_invocations,
            "projection_calls": self.projection_calls,
            "opencv_projectpoints_invocations": self.opencv_projectpoints_invocations,
            "batched_corner_count": self.batched_corner_count,
        }


@dataclass
class CalibrationSolveDiagnostics:
    """Optional benchmark collector for solve-stage timings and optimizer runs.

    Attributes:
        solve_stages_seconds: Accumulated wall time per named solve stage.
        optimizer_runs: Per-run optimizer metadata appended by ``record_optimizer_run``.
    """

    solve_stages_seconds: dict[str, float] = field(default_factory=dict)
    optimizer_runs: list[dict[str, Any]] = field(default_factory=list)
    stage_prefix: str = ""

    def scoped(self, prefix: str) -> CalibrationSolveDiagnostics:
        """Share this collector while prefixing stage names for a nested solve."""
        normalized = prefix if prefix.endswith(".") else f"{prefix}."
        return CalibrationSolveDiagnostics(
            solve_stages_seconds=self.solve_stages_seconds,
            optimizer_runs=self.optimizer_runs,
            stage_prefix=f"{self.stage_prefix}{normalized}",
        )

    def stage_name(self, stage: str) -> str:
        """Return a stage name qualified for this collector's scope."""
        return f"{self.stage_prefix}{stage}"


@contextmanager
def timed_solve_stage(
    diagnostics: CalibrationSolveDiagnostics | None,
    stage: str | None,
):
    """Accumulate wall time for a named solve stage when diagnostics are enabled.

    Args:
        diagnostics: Optional diagnostics collector; no timing when ``None``.
        stage: Stage name key used in ``solve_stages_seconds``.

    Yields:
        Control to the wrapped solve-stage body.
    """
    if diagnostics is None or stage is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        qualified_stage = diagnostics.stage_name(stage)
        diagnostics.solve_stages_seconds[qualified_stage] = (
            diagnostics.solve_stages_seconds.get(qualified_stage, 0.0) + elapsed
        )


def record_optimizer_run(
    diagnostics: CalibrationSolveDiagnostics | None,
    *,
    stage_name: str | None,
    active_frame_count: int,
    inlier_corner_count: int,
    result: Any | None,
    profiler: BundleAdjustmentRunProfiler | None = None,
) -> None:
    """Append one optimizer run record to solve diagnostics.

    Args:
        diagnostics: Optional diagnostics collector; no-op when ``None``.
        stage_name: Bundle-adjustment stage label for the run.
        active_frame_count: Frames with active layout poses in the run.
        inlier_corner_count: Inlier corners optimized in the run.
        result: SciPy least-squares result, or ``None`` after ``ValueError``.
        profiler: Optional per-run profiler supplying timing and count sectors.
    """
    if diagnostics is None or stage_name is None:
        return
    qualified_stage = diagnostics.stage_name(stage_name)
    entry: dict[str, Any] = {
        "stage": qualified_stage,
        "active_frame_count": active_frame_count,
        "inlier_corner_count": inlier_corner_count,
    }
    if result is not None:
        njev = getattr(result, "njev", None)
        entry["nfev"] = int(result.nfev)
        entry["njev"] = int(njev) if njev is not None else None
        entry["status"] = int(result.status)
        entry["cost"] = float(result.cost)
    else:
        entry["nfev"] = None
        entry["njev"] = None
        entry["status"] = None
        entry["cost"] = None
    if profiler is not None:
        entry["timing_seconds"] = profiler.timing_seconds()
        entry["counts"] = profiler.counts()
    diagnostics.optimizer_runs.append(entry)


@dataclass(frozen=True)
class MarkerCandidate:
    """One IPPE pose hypothesis for a marker in a single frame.

    Attributes:
        rvec: OpenCV rotation vector in the camera frame.
        tvec: OpenCV translation vector in the camera frame.
        rotation: Rotation matrix derived from ``rvec``.
        reprojection_rms_px: RMS reprojection error for this hypothesis in pixels.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    reprojection_rms_px: float


@dataclass(frozen=True)
class PairConsensus:
    """Robust low-to-high relative transform with per-frame inlier hypotheses.

    Attributes:
        marker_a: Low marker ID in the pair ordering.
        marker_b: High marker ID in the pair ordering.
        rotation_ba: Rotation mapping low marker frame to high marker frame.
        translation_ba: Translation mapping low marker frame to high marker frame.
        inlier_frames: Frame indices contributing inlier hypotheses.
        inlier_hypotheses: Per-frame ``(rotation_ba, translation_ba)`` inlier values.
    """

    marker_a: int
    marker_b: int
    rotation_ba: np.ndarray
    translation_ba: np.ndarray
    inlier_frames: tuple[int, ...]
    inlier_hypotheses: dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CornerObservation:
    """One corner image measurement linked to frame, marker, and corner index.

    Attributes:
        frame_index: Observation frame index.
        marker_id: Marker ID for the corner.
        corner_index: Corner index ``0`` through ``3`` on the marker.
        image_point: Observed pixel coordinates, shape ``(2,)``.
    """

    frame_index: int
    marker_id: int
    corner_index: int
    image_point: np.ndarray


@dataclass(frozen=True)
class BundleAdjustmentObservationLayout:
    """Immutable active-corner arrays for one bundle-adjustment run.

    Built once per ``run_bundle_adjustment`` call. Residual order matches the
    scalar loop: inlier observations in ``corner_observations`` list order.

    Attributes:
        object_points: Stacked object-frame corner coordinates for inliers.
        image_points: Stacked observed pixel coordinates for inliers.
        marker_ids: Marker ID per inlier observation.
        frame_indices: Frame index per inlier observation.
        unique_marker_ids: Sorted unique marker IDs among inliers.
        unique_frame_indices: Sorted unique frame indices among inliers.
        marker_pose_slots: Index into unique marker pose parameters per observation.
        frame_pose_slots: Index into unique frame pose parameters per observation.
        observation_count: Number of inlier corner observations.
    """

    object_points: np.ndarray
    image_points: np.ndarray
    marker_ids: np.ndarray
    frame_indices: np.ndarray
    unique_marker_ids: np.ndarray
    unique_frame_indices: np.ndarray
    marker_pose_slots: np.ndarray
    frame_pose_slots: np.ndarray
    observation_count: int


def _empty_bundle_adjustment_observation_layout() -> BundleAdjustmentObservationLayout:
    """Return an empty observation layout when no inlier corners remain.

    Returns:
        Layout with zero observations and empty index arrays.
    """
    empty_object = np.empty((0, 3), dtype=np.float64)
    empty_image = np.empty((0, 2), dtype=np.float64)
    empty_index = np.empty(0, dtype=np.int64)
    return BundleAdjustmentObservationLayout(
        empty_object,
        empty_image,
        empty_index,
        empty_index,
        empty_index,
        empty_index,
        empty_index,
        empty_index,
        0,
    )


def build_bundle_adjustment_observation_layout(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
    object_points_by_marker: dict[int, np.ndarray],
) -> BundleAdjustmentObservationLayout:
    """Pack inlier corners into contiguous arrays for batched BA residuals.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.
        object_points_by_marker: Object-frame corner coordinates per marker.

    Returns:
        Contiguous inlier arrays with unique marker and frame slot mappings for
        batched residual evaluation.
    """
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    marker_ids: list[int] = []
    frame_indices: list[int] = []
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        object_points.append(
            object_points_by_marker[observation.marker_id][observation.corner_index]
        )
        image_points.append(observation.image_point)
        marker_ids.append(observation.marker_id)
        frame_indices.append(observation.frame_index)
    if not object_points:
        return _empty_bundle_adjustment_observation_layout()
    marker_ids_arr = np.asarray(marker_ids, dtype=np.int64)
    frame_indices_arr = np.asarray(frame_indices, dtype=np.int64)
    unique_marker_ids, marker_pose_slots = np.unique(marker_ids_arr, return_inverse=True)
    unique_frame_indices, frame_pose_slots = np.unique(frame_indices_arr, return_inverse=True)
    return BundleAdjustmentObservationLayout(
        object_points=np.stack(object_points, axis=0),
        image_points=np.stack(image_points, axis=0),
        marker_ids=marker_ids_arr,
        frame_indices=frame_indices_arr,
        unique_marker_ids=unique_marker_ids,
        unique_frame_indices=unique_frame_indices,
        marker_pose_slots=marker_pose_slots.astype(np.int64, copy=False),
        frame_pose_slots=frame_pose_slots.astype(np.int64, copy=False),
        observation_count=len(object_points),
    )


def copy_marker_poses(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Deep-copy marker poses for checkpointing.

    Args:
        marker_poses: Object-frame marker poses keyed by marker ID.

    Returns:
        Independent copy of every rotation and translation array.
    """
    return {
        marker_id: (rotation.copy(), translation.copy())
        for marker_id, (rotation, translation) in marker_poses.items()
    }


def copy_frame_poses(
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Deep-copy per-frame layout poses for checkpointing.

    Args:
        frame_poses: Per-frame layout poses, with ``None`` for unassigned frames.

    Returns:
        Independent copy of each finite layout pose entry.
    """
    return [
        None if frame_pose is None else (frame_pose[0].copy(), frame_pose[1].copy())
        for frame_pose in frame_poses
    ]


def snapshot_pair_consensus(
    pair_consensus: dict[MarkerPair, PairConsensus],
) -> dict[MarkerPair, PairConsensus]:
    """Deep-copy pair consensus for optimization checkpoints.

    Args:
        pair_consensus: Gated pair consensus edges to snapshot.

    Returns:
        Independent copy of every edge rotation, translation, and inlier map.
    """
    return {
        pair: PairConsensus(
            marker_a=edge.marker_a,
            marker_b=edge.marker_b,
            rotation_ba=edge.rotation_ba.copy(),
            translation_ba=edge.translation_ba.copy(),
            inlier_frames=tuple(edge.inlier_frames),
            inlier_hypotheses={
                frame_index: (rotation.copy(), translation.copy())
                for frame_index, (rotation, translation) in edge.inlier_hypotheses.items()
            },
        )
        for pair, edge in pair_consensus.items()
    }


def poses_are_finite(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
) -> bool:
    """Check whether all marker and frame poses are finite.

    Args:
        marker_poses: Object-frame marker poses keyed by marker ID.
        frame_poses: Per-frame layout poses, with ``None`` for unassigned frames.

    Returns:
        False when any rotation or translation contains non-finite values.
    """
    for rotation, translation in marker_poses.values():
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            return False
    for frame_pose in frame_poses:
        if frame_pose is None:
            continue
        rotation, translation = frame_pose
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            return False
    return True


def mask_corner_observations_for_frames(
    corner_observations: Sequence[CornerObservation],
    allowed_frames: frozenset[int],
) -> np.ndarray:
    """Build a boolean mask for corners in allowed frames.

    Args:
        corner_observations: Full per-corner observation list.
        allowed_frames: Frame indices treated as inliers.

    Returns:
        Boolean array parallel to ``corner_observations``.
    """
    return np.array(
        [observation.frame_index in allowed_frames for observation in corner_observations],
        dtype=bool,
    )


def complete_markers_per_frame(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
) -> dict[int, set[int]]:
    """List markers with four inlier corners per frame.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.

    Returns:
        Mapping from frame index to marker IDs with all four corners inlier.
    """
    corner_counts: dict[tuple[int, int], int] = {}
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        key = (observation.frame_index, observation.marker_id)
        corner_counts[key] = corner_counts.get(key, 0) + 1
    complete: dict[int, set[int]] = {}
    for (frame_index, marker_id), count in corner_counts.items():
        if count == 4:
            complete.setdefault(frame_index, set()).add(marker_id)
    return complete


def drop_frames_without_covisibility(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
) -> np.ndarray:
    """Iteratively drop inlier corners from frames without co-visibility.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Initial inlier mask updated in place logically.

    Returns:
        Updated inlier mask where each retained frame has at least two complete
        markers.

    Notes:
        Repeats until stable: frames with fewer than two four-corner markers lose
        all inlier corners.
    """
    updated = inlier_mask.copy()
    while True:
        complete = complete_markers_per_frame(corner_observations, updated)
        valid_frames = {
            frame_index
            for frame_index, marker_ids in complete.items()
            if len(marker_ids) >= 2
        }
        changed = False
        for index, observation in enumerate(corner_observations):
            if updated[index] and observation.frame_index not in valid_frames:
                updated[index] = False
                changed = True
        if not changed:
            break
    return updated


def covisible_frames_from_inliers(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
) -> frozenset[int]:
    """Collect frame indices with at least two complete inlier markers.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.

    Returns:
        Frame indices where at least two markers have four inlier corners.
    """
    complete = complete_markers_per_frame(corner_observations, inlier_mask)
    return frozenset(
        frame_index
        for frame_index, marker_ids in complete.items()
        if len(marker_ids) >= 2
    )


def covisible_frame_count(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
) -> int:
    """Count frames with at least two complete inlier markers.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.

    Returns:
        Number of co-visible frames under the four-corner completeness rule.
    """
    return len(covisible_frames_from_inliers(corner_observations, inlier_mask))


def project_corner(
    corner_index: int,
    marker_id: int,
    marker_pose: tuple[np.ndarray, np.ndarray],
    frame_pose: tuple[np.ndarray, np.ndarray],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project one marker corner through object, layout, and camera models.

    Args:
        corner_index: Corner index on the marker.
        marker_id: Marker ID for object-point lookup.
        marker_pose: Object-frame marker ``(rotation, translation)``.
        frame_pose: Object-to-camera layout ``(rotation, translation)``.
        object_points_by_marker: Object-frame corner coordinates per marker.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.

    Returns:
        Projected pixel coordinates, shape ``(2,)``.

    Notes:
        Camera extrinsics are identity; layout pose maps object points to the camera.
    """
    marker_rotation, marker_translation = marker_pose
    frame_rotation, frame_translation = frame_pose
    object_points = object_points_by_marker[marker_id]
    point_layout = marker_rotation @ object_points[corner_index] + marker_translation
    point_camera = frame_rotation @ point_layout + frame_translation
    projected, _ = cv2.projectPoints(
        point_camera.reshape(1, 1, 3).astype(np.float32),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(2)


def corner_errors(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Compute per-corner reprojection errors for inlier observations.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.
        marker_poses: Object-frame marker poses keyed by marker ID.
        frame_poses: Per-frame layout poses.
        object_points_by_marker: Object-frame corner coordinates per marker.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.

    Returns:
        Per-corner Euclidean pixel error; ``inf`` for outliers or missing poses.
    """
    errors = np.zeros(len(corner_observations), dtype=np.float64)
    for index, (observation, keep) in enumerate(zip(corner_observations, inlier_mask, strict=True)):
        if not keep:
            errors[index] = np.inf
            continue
        frame_pose = frame_poses[observation.frame_index]
        marker_pose = marker_poses.get(observation.marker_id)
        if frame_pose is None or marker_pose is None:
            errors[index] = np.inf
            continue
        projected = project_corner(
            observation.corner_index,
            observation.marker_id,
            marker_pose,
            frame_pose,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
        )
        errors[index] = float(np.linalg.norm(projected - observation.image_point))
    return errors


def positive_depth_failure(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    object_points_by_marker: dict[int, np.ndarray],
    min_depth_m: float = 1e-4,
) -> str | None:
    """Fail when any inlier corner lies behind the camera.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.
        marker_poses: Object-frame marker poses keyed by marker ID.
        frame_poses: Per-frame layout poses.
        object_points_by_marker: Object-frame corner coordinates per marker.
        min_depth_m: Minimum positive camera-space Z, in meters.

    Returns:
        Failure message when depth is non-positive or poses are missing, else ``None``.
    """
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        frame_pose = frame_poses[observation.frame_index]
        marker_pose = marker_poses.get(observation.marker_id)
        if frame_pose is None or marker_pose is None:
            return "Bundle adjustment produced a frame or marker pose with missing state."
        marker_rotation, marker_translation = marker_pose
        frame_rotation, frame_translation = frame_pose
        object_points = object_points_by_marker[observation.marker_id]
        point_layout = marker_rotation @ object_points[observation.corner_index] + marker_translation
        point_camera = frame_rotation @ point_layout + frame_translation
        if not np.all(np.isfinite(point_camera)) or float(point_camera[2]) <= min_depth_m:
            return (
                f"Bundle adjustment produced non-positive depth for marker "
                f"{observation.marker_id} in frame {observation.frame_index}."
            )
    return None


def connected_marker_ids(
    pair_consensus: dict[MarkerPair, PairConsensus],
    reference_marker_id: int,
) -> set[int]:
    """Collect markers reachable from the reference along pair-consensus edges.

    Args:
        pair_consensus: Accepted pair consensus edges.
        reference_marker_id: Root marker for breadth-first expansion.

    Returns:
        Set of marker IDs connected to ``reference_marker_id``.
    """
    graph: dict[int, set[int]] = {}
    for marker_a, marker_b in pair_consensus:
        graph.setdefault(marker_a, set()).add(marker_b)
        graph.setdefault(marker_b, set()).add(marker_a)

    connected = {reference_marker_id}
    queue = [reference_marker_id]
    while queue:
        current = queue.pop()
        for neighbor in graph.get(current, set()):
            if neighbor not in connected:
                connected.add(neighbor)
                queue.append(neighbor)
    return connected


def missing_from_graph(
    pair_consensus: dict[MarkerPair, PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
) -> frozenset[int]:
    """List expected marker IDs not connected to the reference in the pair graph.

    Args:
        pair_consensus: Accepted pair consensus edges.
        expected_ids: Full set of marker IDs targeted by calibration.
        reference_marker_id: Root marker for connectivity analysis.

    Returns:
        Expected marker IDs absent from the connected component of the reference.
    """
    connected = connected_marker_ids(pair_consensus, reference_marker_id)
    return frozenset(set(expected_ids) - connected)


def rotation_geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Compute geodesic angle between two rotation matrices.

    Args:
        rotation_a: First rotation matrix, shape ``(3, 3)``.
        rotation_b: Second rotation matrix, shape ``(3, 3)``.

    Returns:
        Geodesic angle in degrees between ``rotation_a`` and ``rotation_b``.
    """
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a unit quaternion.

    Args:
        rotation: Rotation matrix, shape ``(3, 3)``.

    Returns:
        Unit quaternion ``(w, x, y, z)`` with stable branch selection.
    """
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    if rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return np.array(
        [
            (rotation[1, 0] - rotation[0, 1]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
            (rotation[1, 2] + rotation[2, 1]) / s,
            0.25 * s,
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion to a rotation matrix.

    Args:
        quaternion: Unit quaternion ``(w, x, y, z)``.

    Returns:
        Rotation matrix, shape ``(3, 3)``.
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


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    """Average rotations via quaternion mean with hemisphere alignment.

    Args:
        rotations: Rotation matrices to average.

    Returns:
        Mean rotation matrix; copies the sole input when only one rotation is given.

    Notes:
        Aligns quaternion signs to the first sample before Euclidean averaging in
        quaternion space.
    """
    if len(rotations) == 1:
        return rotations[0].copy()
    quaternions = [rotation_matrix_to_quaternion(rotation) for rotation in rotations]
    reference = quaternions[0]
    aligned = [quaternion if np.dot(quaternion, reference) >= 0.0 else -quaternion for quaternion in quaternions]
    mean = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 0.0:
        return rotations[0].copy()
    return quaternion_to_rotation_matrix(mean / norm)


def average_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Average poses with quaternion rotation mean and Euclidean translation mean.

    Args:
        poses: List of ``(rotation, translation)`` pose tuples.

    Returns:
        Tuple of averaged rotation and translation.
    """
    rotations = [rotation for rotation, _ in poses]
    translations = np.stack([translation for _, translation in poses], axis=0)
    return average_rotations(rotations), np.mean(translations, axis=0)
