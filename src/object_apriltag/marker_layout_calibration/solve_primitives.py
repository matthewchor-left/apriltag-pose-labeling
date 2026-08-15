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
    """Per-run BA sector timings and counts when solve diagnostics are enabled.

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
    """Optional benchmark collector for solve-stage timings and BA optimizer runs."""

    solve_stages_seconds: dict[str, float] = field(default_factory=dict)
    optimizer_runs: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def timed_solve_stage(
    diagnostics: CalibrationSolveDiagnostics | None,
    stage: str | None,
):
    if diagnostics is None or stage is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        diagnostics.solve_stages_seconds[stage] = (
            diagnostics.solve_stages_seconds.get(stage, 0.0) + elapsed
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
    """Append one optimizer run; ``result`` may be None after a SciPy ValueError."""
    if diagnostics is None or stage_name is None:
        return
    entry: dict[str, Any] = {
        "stage": stage_name,
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
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    reprojection_rms_px: float


@dataclass(frozen=True)
class PairConsensus:
    marker_a: int
    marker_b: int
    rotation_ba: np.ndarray
    translation_ba: np.ndarray
    inlier_frames: tuple[int, ...]
    inlier_hypotheses: dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CornerObservation:
    frame_index: int
    marker_id: int
    corner_index: int
    image_point: np.ndarray


@dataclass(frozen=True)
class BundleAdjustmentObservationLayout:
    """Immutable active-corner arrays for one bundle-adjustment run.

    Built once per ``run_bundle_adjustment`` call. Residual order matches the
    scalar loop: inlier observations in ``corner_observations`` list order.
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
    return {
        marker_id: (rotation.copy(), translation.copy())
        for marker_id, (rotation, translation) in marker_poses.items()
    }


def copy_frame_poses(
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    return [
        None if frame_pose is None else (frame_pose[0].copy(), frame_pose[1].copy())
        for frame_pose in frame_poses
    ]


def snapshot_pair_consensus(
    pair_consensus: dict[MarkerPair, PairConsensus],
) -> dict[MarkerPair, PairConsensus]:
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
    return np.array(
        [observation.frame_index in allowed_frames for observation in corner_observations],
        dtype=bool,
    )


def complete_markers_per_frame(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
) -> dict[int, set[int]]:
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
    connected = connected_marker_ids(pair_consensus, reference_marker_id)
    return frozenset(set(expected_ids) - connected)


def rotation_geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
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
    rotations = [rotation for rotation, _ in poses]
    translations = np.stack([translation for _, translation in poses], axis=0)
    return average_rotations(rotations), np.mean(translations, axis=0)
