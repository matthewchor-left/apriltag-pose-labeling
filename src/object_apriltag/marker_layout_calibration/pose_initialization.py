"""Reference gauge, pose graph propagation, frame poses, and corner observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from object_apriltag.layout import footprint_orientation, marker_origin_on_object

from object_apriltag.marker_layout_calibration.discrete_graph import (
    make_dropped_pair_edge,
    maybe_restore_weak_connectivity,
)
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CornerObservation,
    MarkerPair,
    PairConsensus,
    average_poses,
    average_rotations,
)
from object_apriltag.marker_layout_calibration.solve_quality import pair_translation_gate
from object_apriltag.marker_layout_calibration.types import (
    CalibrationSettings,
    DroppedPairEdge,
    FrameObservation,
    RestoredPairEdge,
)

def restrict_pair_consensus_to_frames(
    pair_consensus: dict[MarkerPair, PairConsensus],
    allowed_frames: frozenset[int],
    expected_ids: list[int],
    reference_marker_id: int,
    settings: CalibrationSettings,
    *,
    marker_sizes_m: Mapping[int, float],
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[dict[MarkerPair, PairConsensus], str | None, tuple[DroppedPairEdge, ...]]:
    rotation_gate = settings.pair_rotation_rms_gate_deg
    updated: dict[MarkerPair, PairConsensus] = {}
    weak_pool: dict[MarkerPair, PairConsensus] = {}
    dropped: list[DroppedPairEdge] = []
    for pair, edge in pair_consensus.items():
        translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
        supported_frames = tuple(
            sorted(frame_index for frame_index in edge.inlier_frames if frame_index in allowed_frames)
        )
        if len(supported_frames) < settings.min_inliers_per_edge:
            if supported_frames:
                weak_pool[pair] = PairConsensus(
                    marker_a=edge.marker_a,
                    marker_b=edge.marker_b,
                    rotation_ba=edge.rotation_ba,
                    translation_ba=edge.translation_ba,
                    inlier_frames=supported_frames,
                    inlier_hypotheses={
                        frame_index: edge.inlier_hypotheses[frame_index]
                        for frame_index in supported_frames
                    },
                )
            dropped.append(
                make_dropped_pair_edge(
                    pair,
                    "assignment_support",
                    "insufficient_support",
                    observed_count=len(edge.inlier_frames),
                    supported_count=len(supported_frames),
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        updated[pair] = PairConsensus(
            marker_a=edge.marker_a,
            marker_b=edge.marker_b,
            rotation_ba=edge.rotation_ba,
            translation_ba=edge.translation_ba,
            inlier_frames=supported_frames,
            inlier_hypotheses={
                frame_index: edge.inlier_hypotheses[frame_index] for frame_index in supported_frames
            },
        )

    failure = maybe_restore_weak_connectivity(
        updated,
        weak_pool,
        dropped,
        expected_ids,
        reference_marker_id,
        "assignment_support",
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    if failure is not None:
        return updated, failure, tuple(dropped)
    return updated, None, tuple(dropped)


def markers_in_frame_indices(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    frame_indices: frozenset[int],
) -> set[int]:
    markers: set[int] = set()
    for frame_index, (_, markers_in_frame) in enumerate(observations):
        if frame_index in frame_indices:
            markers.update(markers_in_frame)
    return markers


def reference_gauge_pose(marker_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    half = marker_size_m / 2.0
    top_left = np.array([-half, -half, 0.0], dtype=np.float64)
    top_right = np.array([half, -half, 0.0], dtype=np.float64)
    bottom_right = np.array([half, half, 0.0], dtype=np.float64)
    bottom_left = np.array([-half, half, 0.0], dtype=np.float64)
    rotation = footprint_orientation(top_left, top_right, bottom_left, bottom_right)
    translation = marker_origin_on_object(bottom_left, bottom_right)
    return rotation, translation


def initialize_marker_poses(
    reference_marker_id: int,
    ref_rotation: np.ndarray,
    ref_translation: np.ndarray,
    expected_ids: list[int],
    pair_consensus: dict[MarkerPair, PairConsensus],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {
        reference_marker_id: (ref_rotation.copy(), ref_translation.copy())
    }
    graph: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for pair, edge in pair_consensus.items():
        marker_a, marker_b = pair
        graph.setdefault(marker_a, []).append((marker_b, edge.rotation_ba, edge.translation_ba))
        graph.setdefault(marker_b, []).append(
            (
                marker_a,
                edge.rotation_ba.T,
                -edge.rotation_ba.T @ edge.translation_ba,
            )
        )

    queue = [reference_marker_id]
    while queue:
        parent_id = queue.pop()
        parent_rotation, parent_translation = poses[parent_id]
        for child_id, rotation_cp, translation_cp in graph.get(parent_id, []):
            if child_id in poses or child_id not in expected_ids:
                continue
            child_rotation = parent_rotation @ rotation_cp
            child_translation = parent_rotation @ translation_cp + parent_translation
            poses[child_id] = (child_rotation, child_translation)
            queue.append(child_id)
    return poses


def initialize_frame_poses(
    assigned_candidates: dict[int, dict[int, MarkerCandidate]],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_count: int,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * frame_count
    for frame_index, assignment in assigned_candidates.items():
        estimates: list[tuple[np.ndarray, np.ndarray]] = []
        for marker_id, candidate in assignment.items():
            if marker_id not in marker_poses:
                continue
            marker_rotation, marker_translation = marker_poses[marker_id]
            layout_rotation = candidate.rotation @ marker_rotation.T
            layout_translation = candidate.tvec - layout_rotation @ marker_translation
            estimates.append((layout_rotation, layout_translation))
        if estimates:
            frame_poses[frame_index] = average_poses(estimates)
    return frame_poses


def build_corner_observations(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    expected_ids: list[int],
) -> list[CornerObservation]:
    expected_set = set(expected_ids)
    corner_observations: list[CornerObservation] = []
    for frame_index, (_, markers) in enumerate(observations):
        for marker_id, corners in markers.items():
            if marker_id not in expected_set:
                continue
            for corner_index in range(4):
                corner_observations.append(
                    CornerObservation(
                        frame_index=frame_index,
                        marker_id=marker_id,
                        corner_index=corner_index,
                        image_point=corners[corner_index],
                    )
                )
    return corner_observations


def synth_marker_corners(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    marker_ids: Sequence[int],
    layout_rotation: np.ndarray,
    layout_translation: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[int, np.ndarray]:
    markers: dict[int, np.ndarray] = {}
    for marker_id in marker_ids:
        marker_rotation, marker_translation = marker_poses[marker_id]
        image_corners = []
        for corner_index in range(4):
            layout_point = marker_rotation @ object_points[corner_index] + marker_translation
            camera_point = layout_rotation @ layout_point + layout_translation
            projected, _ = cv2.projectPoints(
                camera_point.reshape(1, 1, 3).astype(np.float32),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                camera_matrix,
                dist_coeffs,
            )
            image_corners.append(projected.reshape(2))
        markers[marker_id] = np.stack(image_corners, axis=0)
    return markers


def synth_pair_observations(
    num_frames: int,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    corrupt_frames: frozenset[int] = frozenset(),
    corrupt_offset: np.ndarray | None = None,
    varying_corrupt: bool = False,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    base_wrong_offset = (
        np.array([0.20, 0.0, -0.08], dtype=np.float64)
        if corrupt_offset is None
        else np.asarray(corrupt_offset, dtype=np.float64)
    )
    for frame_index in range(num_frames):
        layout_rotation, _ = cv2.Rodrigues(np.array([0.1, -0.15 + 0.002 * frame_index, 0.05]))
        layout_translation = np.array([0.02, -0.01, 0.6 + 0.002 * frame_index], dtype=np.float64)
        frame_poses = dict(marker_poses)
        if frame_index in corrupt_frames:
            offset = base_wrong_offset
            if varying_corrupt:
                offset = base_wrong_offset + np.array([0.01 * frame_index, 0.0, 0.0])
            frame_poses[1] = (marker_poses[1][0], marker_poses[0][1] + offset)
        markers = synth_marker_corners(
            frame_poses,
            (1, 0),
            layout_rotation,
            layout_translation,
            object_points,
            camera_matrix,
            dist_coeffs,
        )
        observations.append(FrameObservation(frame_id=frame_index, markers=markers))
    return observations
