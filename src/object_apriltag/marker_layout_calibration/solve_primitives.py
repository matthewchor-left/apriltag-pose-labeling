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
) -> None:
    if diagnostics is None or stage_name is None or result is None:
        return
    njev = getattr(result, "njev", None)
    diagnostics.optimizer_runs.append(
        {
            "stage": stage_name,
            "nfev": int(result.nfev),
            "njev": int(njev) if njev is not None else None,
            "status": int(result.status),
            "cost": float(result.cost),
            "active_frame_count": active_frame_count,
            "inlier_corner_count": inlier_corner_count,
        }
    )


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
