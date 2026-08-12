"""Multi-view marker layout calibration from co-visible AprilTag corners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from object_apriltag.layout import (
    CORNER_NAMES,
    MarkerFootprint,
    MarkerLayout,
    build_marker_layout,
    footprint_from_dict,
    footprint_orientation,
    marker_origin_on_object,
)
from object_apriltag.pose import marker_corner_object_points

MarkerPair = tuple[int, int]


@dataclass(frozen=True)
class FrameObservation:
    """One camera sample with expected marker corners in OpenCV order."""

    frame_id: str | int
    markers: dict[int, np.ndarray]


@dataclass(frozen=True)
class CalibrationSettings:
    min_inliers_per_edge: int = 20
    reprojection_rms_gate_px: float = 2.0
    pair_translation_rms_gate_ratio: float = 0.10
    pair_rotation_rms_gate_deg: float = 5.0
    huber_delta_px: float = 1.0
    corner_outlier_px: float = 3.0
    max_ba_iterations: int = 50


@dataclass(frozen=True)
class EdgeDiagnostics:
    marker_a: int
    marker_b: int
    inlier_count: int
    translation_rms_m: float
    rotation_rms_deg: float


@dataclass(frozen=True)
class CalibrationQualityReport:
    reprojection_rms_px: float
    per_marker_reprojection_rms_px: dict[int, float]
    edges: tuple[EdgeDiagnostics, ...]
    pair_translation_rms_max_m: float
    pair_rotation_rms_max_deg: float
    frame_count: int
    observation_count: int
    inlier_corner_count: int
    input_frame_count: int
    rejected_frame_count: int
    accepted_frame_count: int
    connected_marker_ids: frozenset[int]
    missing_expected_ids: frozenset[int]
    unused_expected_ids: frozenset[int]


@dataclass(frozen=True)
class CalibrationResult:
    layout: MarkerLayout | None
    quality: CalibrationQualityReport | None
    failure_reason: str | None


@dataclass(frozen=True)
class _MarkerCandidate:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    reprojection_rms_px: float


@dataclass(frozen=True)
class _PairConsensus:
    marker_a: int
    marker_b: int
    rotation_ba: np.ndarray
    translation_ba: np.ndarray
    inlier_frames: tuple[int, ...]
    inlier_hypotheses: dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class _CornerObservation:
    frame_index: int
    marker_id: int
    corner_index: int
    image_point: np.ndarray


def _validate_settings(settings: CalibrationSettings) -> str | None:
    if settings.min_inliers_per_edge <= 0:
        return "CalibrationSettings.min_inliers_per_edge must be positive."
    if settings.max_ba_iterations <= 0:
        return "CalibrationSettings.max_ba_iterations must be positive."
    positive_float_fields = (
        ("reprojection_rms_gate_px", settings.reprojection_rms_gate_px),
        ("pair_translation_rms_gate_ratio", settings.pair_translation_rms_gate_ratio),
        ("pair_rotation_rms_gate_deg", settings.pair_rotation_rms_gate_deg),
        ("huber_delta_px", settings.huber_delta_px),
        ("corner_outlier_px", settings.corner_outlier_px),
    )
    for field_name, value in positive_float_fields:
        if not np.isfinite(value) or value <= 0.0:
            return f"CalibrationSettings.{field_name} must be finite and positive."
    return None


def _parse_expected_marker_ids(
    expected_marker_ids: Sequence[int],
    reference_marker_id: int,
) -> tuple[list[int] | None, str | None]:
    try:
        marker_ids = [int(marker_id) for marker_id in expected_marker_ids]
    except (TypeError, ValueError):
        return None, "expected_marker_ids must contain integer marker IDs."
    if not marker_ids:
        return None, "expected_marker_ids is empty."

    seen: set[int] = set()
    duplicates: set[int] = set()
    for marker_id in marker_ids:
        if marker_id in seen:
            duplicates.add(marker_id)
        seen.add(marker_id)
    if duplicates:
        return None, f"expected_marker_ids contains duplicates: {sorted(duplicates)}."

    expected_ids = sorted(marker_ids)
    if int(reference_marker_id) not in seen:
        return None, f"reference_marker_id {reference_marker_id} is not in expected_marker_ids."
    return expected_ids, None


def _validate_marker_size(marker_size_m: float) -> str | None:
    if not np.isfinite(marker_size_m) or marker_size_m <= 0.0:
        return "marker_size_m must be a finite positive number."
    return None


def _validate_camera_inputs(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    try:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, "camera_matrix must be a numeric 3x3 matrix."
    if matrix.shape != (3, 3):
        return None, None, f"camera_matrix must have shape (3, 3), got {matrix.shape}."
    if not np.all(np.isfinite(matrix)):
        return None, None, "camera_matrix must contain only finite values."

    try:
        distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    except (TypeError, ValueError):
        return None, None, "dist_coeffs must be numeric."
    if distortion.size == 0:
        return None, None, "dist_coeffs must not be empty."
    if not np.all(np.isfinite(distortion)):
        return None, None, "dist_coeffs must contain only finite values."
    return matrix, distortion, None


def _parse_marker_corners(
    corners: np.ndarray,
    frame_id: str | int,
    marker_id: int,
) -> tuple[np.ndarray | None, str | None]:
    try:
        array = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError):
        return None, (
            f"Malformed corners for frame {frame_id!r}, marker {marker_id}: "
            "corners must be numeric."
        )
    if array.size != 8:
        return None, (
            f"Malformed corners for frame {frame_id!r}, marker {marker_id}: "
            f"expected 4x2 corners, got {array.size} values."
        )
    array = array.reshape(4, 2)
    if array.shape != (4, 2) or not np.all(np.isfinite(array)):
        return None, (
            f"Malformed corners for frame {frame_id!r}, marker {marker_id}: "
            "corners must be finite 4x2 points."
        )
    return array, None


def _validate_observations(
    observations: Sequence[FrameObservation],
    expected_ids: list[int],
) -> str | None:
    expected_set = set(expected_ids)
    seen_frame_ids: set[str | int] = set()
    for observation in observations:
        if observation.frame_id in seen_frame_ids:
            return f"Duplicate FrameObservation.frame_id: {observation.frame_id!r}."
        seen_frame_ids.add(observation.frame_id)
        for marker_id, corners in observation.markers.items():
            marker_id = int(marker_id)
            if marker_id not in expected_set:
                continue
            _, failure = _parse_marker_corners(corners, observation.frame_id, marker_id)
            if failure is not None:
                return failure
    return None


def calibrate_marker_layout(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    expected_marker_ids: Sequence[int],
    reference_marker_id: int,
    marker_size_m: float,
    settings: CalibrationSettings | None = None,
) -> CalibrationResult:
    """Estimate a connected marker layout or refuse with a structured reason."""
    settings = settings or CalibrationSettings()
    settings_failure = _validate_settings(settings)
    if settings_failure is not None:
        return CalibrationResult(None, None, settings_failure)

    expected_ids, expected_failure = _parse_expected_marker_ids(
        expected_marker_ids,
        reference_marker_id,
    )
    if expected_failure is not None:
        return CalibrationResult(None, None, expected_failure)

    marker_size_failure = _validate_marker_size(marker_size_m)
    if marker_size_failure is not None:
        return CalibrationResult(None, None, marker_size_failure)

    camera_matrix, dist_coeffs, camera_failure = _validate_camera_inputs(
        camera_matrix,
        dist_coeffs,
    )
    if camera_failure is not None:
        return CalibrationResult(None, None, camera_failure)

    observations_failure = _validate_observations(observations, expected_ids)
    if observations_failure is not None:
        return CalibrationResult(None, None, observations_failure)

    object_points = marker_corner_object_points(marker_size_m).astype(np.float64)

    normalized_observations = _normalize_observations(observations, expected_ids)
    if not normalized_observations:
        missing = frozenset(expected_ids)
        return CalibrationResult(
            None,
            _empty_quality(missing, frozenset(), input_frame_count=len(observations)),
            f"No usable observations for expected marker IDs; missing {sorted(missing)}.",
        )

    observed_ids = {
        marker_id
        for _, markers in normalized_observations
        for marker_id in markers
    }
    never_observed = sorted(set(expected_ids) - observed_ids)
    if never_observed:
        return CalibrationResult(
            None,
            _empty_quality(
                frozenset(never_observed),
                frozenset(observed_ids),
                input_frame_count=len(normalized_observations),
            ),
            f"Expected marker IDs never observed: {never_observed}.",
        )

    frame_candidates = _estimate_frame_candidates(
        normalized_observations,
        object_points,
        camera_matrix,
        dist_coeffs,
    )
    if not frame_candidates:
        return CalibrationResult(
            None,
            _empty_quality(
                frozenset(expected_ids),
                frozenset(),
                input_frame_count=len(normalized_observations),
            ),
            "No valid IPPE marker poses found in any frame.",
        )

    pair_hypotheses = _collect_pair_hypotheses(frame_candidates, expected_ids)
    pair_consensus, pair_failure = _estimate_pair_consensus(
        pair_hypotheses,
        expected_ids,
        reference_marker_id,
        marker_size_m,
        settings,
    )
    if pair_failure is not None:
        missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing,
                input_frame_count=len(normalized_observations),
                rejected_frame_count=0,
                accepted_frame_count=0,
                observation_count=0,
            ),
            pair_failure,
        )

    assigned_candidates, rejected_frames = _assign_ippe_candidates(
        frame_candidates,
        pair_consensus,
        settings,
        marker_size_m,
    )
    input_frame_count = len(normalized_observations)
    rejected_frame_count = len(rejected_frames)
    accepted_frames = frozenset(assigned_candidates)
    accepted_frame_count = len(accepted_frames)
    if accepted_frame_count == 0:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                _missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=0,
                observation_count=0,
            ),
            "No frames with assignable IPPE candidates remain after rejecting inconsistent samples.",
        )

    pair_consensus, assignment_support_failure = _restrict_pair_consensus_to_frames(
        pair_consensus,
        accepted_frames,
        expected_ids,
        reference_marker_id,
        settings,
    )
    if assignment_support_failure is not None:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                _missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=0,
            ),
            assignment_support_failure,
        )

    markers_in_accepted_frames = _markers_in_frame_indices(normalized_observations, accepted_frames)
    missing_after_rejection = sorted(set(expected_ids) - markers_in_accepted_frames)
    if missing_after_rejection:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                frozenset(missing_after_rejection),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=0,
            ),
            f"Expected marker IDs have no accepted-frame observations after rejection: {missing_after_rejection}.",
        )

    ref_rotation, ref_translation = _reference_gauge_pose(marker_size_m)
    marker_poses = _initialize_marker_poses(
        reference_marker_id,
        ref_rotation,
        ref_translation,
        expected_ids,
        pair_consensus,
    )
    frame_poses = _initialize_frame_poses(
        assigned_candidates,
        marker_poses,
        len(normalized_observations),
    )

    corner_observations = _build_corner_observations(normalized_observations, expected_ids)
    inlier_mask = _mask_corner_observations_for_frames(corner_observations, accepted_frames)
    non_reference_ids = [marker_id for marker_id in expected_ids if marker_id != reference_marker_id]

    marker_poses, frame_poses, inlier_mask, ba_failure = _run_bundle_adjustment(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_ids,
        object_points,
        camera_matrix,
        dist_coeffs,
        settings,
    )
    if ba_failure is not None:
        return CalibrationResult(None, None, ba_failure)

    marker_poses, frame_poses, inlier_mask, pair_consensus, prune_failure = _prune_and_refit(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_ids,
        expected_ids,
        pair_consensus,
        accepted_frames,
        object_points,
        camera_matrix,
        dist_coeffs,
        settings,
    )
    if prune_failure is not None:
        missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing,
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=_covisible_frame_count(corner_observations, inlier_mask),
                observation_count=int(np.count_nonzero(inlier_mask)),
            ),
            prune_failure,
        )

    connected_ids = _connected_marker_ids(pair_consensus, reference_marker_id)
    missing_ids = frozenset(set(expected_ids) - connected_ids)
    final_accepted_frame_count = _covisible_frame_count(corner_observations, inlier_mask)
    quality = _build_quality_report(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        pair_consensus,
        expected_ids,
        reference_marker_id,
        missing_ids,
        input_frame_count,
        rejected_frame_count,
        final_accepted_frame_count,
        object_points,
        camera_matrix,
        dist_coeffs,
    )

    gate_failure = _check_quality_gates(quality, settings, marker_size_m, expected_ids)
    if gate_failure is not None:
        return CalibrationResult(None, quality, gate_failure)
    if missing_ids:
        return CalibrationResult(
            None,
            quality,
            f"Expected marker IDs are not connected to reference: {sorted(missing_ids)}.",
        )

    footprints = _footprints_from_poses(marker_poses, marker_size_m)
    if set(footprints) != set(expected_ids):
        absent = sorted(set(expected_ids) - set(footprints))
        return CalibrationResult(
            None,
            quality,
            f"Calibration did not produce all expected marker footprints; missing {absent}.",
        )

    layout = build_marker_layout(
        reference_marker_id=reference_marker_id,
        marker_size_m=marker_size_m,
        footprints=footprints,
    )
    return CalibrationResult(layout, quality, None)


def _normalize_observations(
    observations: Sequence[FrameObservation],
    expected_ids: list[int],
) -> list[tuple[str | int, dict[int, np.ndarray]]]:
    expected_set = set(expected_ids)
    normalized: list[tuple[str | int, dict[int, np.ndarray]]] = []
    for observation in observations:
        markers: dict[int, np.ndarray] = {}
        for marker_id, corners in observation.markers.items():
            marker_id = int(marker_id)
            if marker_id not in expected_set:
                continue
            array, failure = _parse_marker_corners(corners, observation.frame_id, marker_id)
            if failure is not None or array is None:
                continue
            markers[marker_id] = array
        if len(markers) >= 2:
            normalized.append((observation.frame_id, markers))
    return normalized


def _estimate_frame_candidates(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[int, dict[int, list[_MarkerCandidate]]]]:
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]] = []
    for frame_index, (_, markers) in enumerate(observations):
        candidates: dict[int, list[_MarkerCandidate]] = {}
        for marker_id, image_points in markers.items():
            marker_candidates = _ippe_candidates(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
            )
            if marker_candidates:
                candidates[marker_id] = marker_candidates
        if len(candidates) >= 2:
            frame_candidates.append((frame_index, candidates))
    return frame_candidates


def _ippe_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[_MarkerCandidate]:
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or rvecs is None or tvecs is None:
        return []

    candidates: list[_MarkerCandidate] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rotation, _ = cv2.Rodrigues(rvec)
        if not _is_marker_facing_camera(rotation):
            continue
        rms = _reprojection_rms(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        candidates.append(
            _MarkerCandidate(
                rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
                tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
                rotation=rotation.astype(np.float64),
                reprojection_rms_px=rms,
            )
        )
    return candidates


def _is_marker_facing_camera(rotation: np.ndarray) -> bool:
    normal = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(normal[2]) < 0.0


def _reprojection_rms(
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
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(image_points.reshape(-1, 2) - projected, axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def _relative_marker_transform(
    parent: _MarkerCandidate,
    child: _MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points in child marker frame to parent marker frame."""
    rotation = parent.rotation.T @ child.rotation
    translation = parent.rotation.T @ (child.tvec - parent.tvec)
    return rotation, translation


def _transform_high_in_low(
    low: _MarkerCandidate,
    high: _MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points in the high marker frame to the low marker frame."""
    return _relative_marker_transform(low, high)


def _pair_translation_gate(settings: CalibrationSettings, marker_size_m: float) -> float:
    return settings.pair_translation_rms_gate_ratio * marker_size_m


def _collect_pair_hypotheses(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    expected_ids: list[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    expected_set = set(expected_ids)
    for frame_index, candidates in frame_candidates:
        marker_ids = sorted(marker_id for marker_id in candidates if marker_id in expected_set)
        for index_a, marker_low in enumerate(marker_ids):
            for marker_high in marker_ids[index_a + 1 :]:
                pair = (marker_low, marker_high)
                for candidate_low in candidates[marker_low]:
                    for candidate_high in candidates[marker_high]:
                        rotation_ba, translation_ba = _transform_high_in_low(
                            candidate_low,
                            candidate_high,
                        )
                        hypotheses.setdefault(pair, []).append(
                            (rotation_ba, translation_ba, frame_index)
                        )
    return hypotheses


def _estimate_pair_consensus(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    expected_ids: list[int],
    reference_marker_id: int,
    marker_size_m: float,
    settings: CalibrationSettings,
) -> tuple[dict[MarkerPair, _PairConsensus], str | None]:
    translation_gate = _pair_translation_gate(settings, marker_size_m)
    rotation_gate = settings.pair_rotation_rms_gate_deg
    consensus: dict[MarkerPair, _PairConsensus] = {}

    for pair, hypotheses in pair_hypotheses.items():
        unique_frames = {frame_index for _, _, frame_index in hypotheses}
        if len(unique_frames) < settings.min_inliers_per_edge:
            continue

        best_frames: dict[int, int] = {}
        best_rotation = np.eye(3, dtype=np.float64)
        best_translation = np.zeros(3, dtype=np.float64)
        for seed_index, (seed_rotation, seed_translation, _) in enumerate(hypotheses):
            candidate_frames = _inlier_frames_for_seed(
                hypotheses,
                seed_index,
                translation_gate,
                rotation_gate,
            )
            if len(candidate_frames) > len(best_frames):
                best_frames = candidate_frames
                best_rotation = seed_rotation
                best_translation = seed_translation

        if len(best_frames) < settings.min_inliers_per_edge:
            continue

        selected_hypotheses = {
            frame_index: hypotheses[hypothesis_index][:2]
            for frame_index, hypothesis_index in best_frames.items()
        }
        inlier_rotations = [values[0] for values in selected_hypotheses.values()]
        inlier_translations = np.stack([values[1] for values in selected_hypotheses.values()], axis=0)
        best_rotation = _average_rotations(inlier_rotations)
        best_translation = np.mean(inlier_translations, axis=0)
        consensus[pair] = _PairConsensus(
            marker_a=pair[0],
            marker_b=pair[1],
            rotation_ba=best_rotation,
            translation_ba=best_translation,
            inlier_frames=tuple(sorted(selected_hypotheses)),
            inlier_hypotheses=selected_hypotheses,
        )

    connected = _connected_marker_ids(consensus, reference_marker_id)
    missing = sorted(set(expected_ids) - connected)
    if missing:
        return consensus, (
            f"Expected marker IDs are not connected to reference {reference_marker_id}; "
            f"missing {missing}."
        )

    for pair, edge in consensus.items():
        diagnostics = _edge_diagnostics(pair, edge)
        if diagnostics.inlier_count < settings.min_inliers_per_edge:
            return consensus, (
                f"Pair ({pair[0]}, {pair[1]}) has only "
                f"{diagnostics.inlier_count} inlier frames; need "
                f"{settings.min_inliers_per_edge}."
            )
        if diagnostics.translation_rms_m > translation_gate:
            return consensus, (
                f"Pair ({pair[0]}, {pair[1]}) translation RMS "
                f"{diagnostics.translation_rms_m:.4f} m exceeds gate."
            )
        if diagnostics.rotation_rms_deg > rotation_gate:
            return consensus, (
                f"Pair ({pair[0]}, {pair[1]}) rotation RMS "
                f"{diagnostics.rotation_rms_deg:.2f} deg exceeds gate."
            )

    return consensus, None


def _inlier_frames_for_seed(
    hypotheses: list[tuple[np.ndarray, np.ndarray, int]],
    seed_index: int,
    translation_gate: float,
    rotation_gate: float,
) -> dict[int, int]:
    seed_rotation, seed_translation, _ = hypotheses[seed_index]
    inlier_frames: dict[int, int] = {}
    for hypothesis_index, (rotation, translation, frame_index) in enumerate(hypotheses):
        if (
            np.linalg.norm(translation - seed_translation) > translation_gate
            or _rotation_geodesic_deg(rotation, seed_rotation) > rotation_gate
        ):
            continue
        current_index = inlier_frames.get(frame_index)
        if current_index is None:
            inlier_frames[frame_index] = hypothesis_index
            continue
        current_rotation, current_translation, _ = hypotheses[current_index]
        current_cost = (
            np.linalg.norm(current_translation - seed_translation) / max(translation_gate, 1e-9)
            + _rotation_geodesic_deg(current_rotation, seed_rotation) / max(rotation_gate, 1e-9)
        )
        candidate_cost = (
            np.linalg.norm(translation - seed_translation) / max(translation_gate, 1e-9)
            + _rotation_geodesic_deg(rotation, seed_rotation) / max(rotation_gate, 1e-9)
        )
        if candidate_cost < current_cost:
            inlier_frames[frame_index] = hypothesis_index
    return inlier_frames


def _connected_marker_ids(
    pair_consensus: dict[MarkerPair, _PairConsensus],
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


def _missing_from_graph(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
) -> frozenset[int]:
    connected = _connected_marker_ids(pair_consensus, reference_marker_id)
    return frozenset(set(expected_ids) - connected)


def _assign_ippe_candidates(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    settings: CalibrationSettings,
    marker_size_m: float,
) -> tuple[dict[int, dict[int, _MarkerCandidate]], tuple[int, ...]]:
    assigned: dict[int, dict[int, _MarkerCandidate]] = {}
    rejected_frames: list[int] = []
    for frame_index, candidates in frame_candidates:
        marker_ids = sorted(candidates)
        best_holder: dict[str, object] = {"score": float("-inf"), "assignment": None}
        _search_assignments(
            marker_ids,
            candidates,
            pair_consensus,
            settings,
            marker_size_m,
            {},
            0,
            best_holder,
        )
        if best_holder["assignment"] is None:
            rejected_frames.append(frame_index)
            continue
        assigned[frame_index] = best_holder["assignment"]
    return assigned, tuple(rejected_frames)


def _restrict_pair_consensus_to_frames(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    allowed_frames: frozenset[int],
    expected_ids: list[int],
    reference_marker_id: int,
    settings: CalibrationSettings,
) -> tuple[dict[MarkerPair, _PairConsensus], str | None]:
    updated: dict[MarkerPair, _PairConsensus] = {}
    for pair, edge in pair_consensus.items():
        supported_frames = tuple(
            sorted(frame_index for frame_index in edge.inlier_frames if frame_index in allowed_frames)
        )
        if len(supported_frames) < settings.min_inliers_per_edge:
            return updated, (
                f"Pair ({pair[0]}, {pair[1]}) has only {len(supported_frames)} accepted assigned "
                f"frames; need {settings.min_inliers_per_edge}."
            )
        updated[pair] = _PairConsensus(
            marker_a=edge.marker_a,
            marker_b=edge.marker_b,
            rotation_ba=edge.rotation_ba,
            translation_ba=edge.translation_ba,
            inlier_frames=supported_frames,
            inlier_hypotheses={
                frame_index: edge.inlier_hypotheses[frame_index] for frame_index in supported_frames
            },
        )

    connected = _connected_marker_ids(updated, reference_marker_id)
    missing = sorted(set(expected_ids) - connected)
    if missing:
        return updated, (
            f"Expected marker IDs are not connected after rejecting assignment frames; missing {missing}."
        )
    return updated, None


def _markers_in_frame_indices(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    frame_indices: frozenset[int],
) -> set[int]:
    markers: set[int] = set()
    for frame_index, (_, markers_in_frame) in enumerate(observations):
        if frame_index in frame_indices:
            markers.update(markers_in_frame)
    return markers


def _mask_corner_observations_for_frames(
    corner_observations: list[_CornerObservation],
    allowed_frames: frozenset[int],
) -> np.ndarray:
    return np.array(
        [observation.frame_index in allowed_frames for observation in corner_observations],
        dtype=bool,
    )


def _covisible_frame_count(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> int:
    return len(_covisible_frames_from_inliers(corner_observations, inlier_mask))


def _covisible_frames_from_inliers(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> frozenset[int]:
    complete = _complete_markers_per_frame(corner_observations, inlier_mask)
    return frozenset(
        frame_index
        for frame_index, marker_ids in complete.items()
        if len(marker_ids) >= 2
    )


def _search_assignments(
    marker_ids: list[int],
    candidates: dict[int, list[_MarkerCandidate]],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    settings: CalibrationSettings,
    marker_size_m: float,
    current: dict[int, _MarkerCandidate],
    index: int,
    best_holder: dict,
) -> None:
    if index == len(marker_ids):
        score = _score_assignment(current, pair_consensus, settings, marker_size_m)
        if score is None:
            return
        if score > best_holder["score"]:
            best_holder["score"] = score
            best_holder["assignment"] = dict(current)
        return

    marker_id = marker_ids[index]
    for candidate in candidates[marker_id]:
        current[marker_id] = candidate
        _search_assignments(
            marker_ids,
            candidates,
            pair_consensus,
            settings,
            marker_size_m,
            current,
            index + 1,
            best_holder,
        )
    current.pop(marker_id, None)


def _score_assignment(
    assignment: dict[int, _MarkerCandidate],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    settings: CalibrationSettings,
    marker_size_m: float,
) -> float | None:
    translation_gate = _pair_translation_gate(settings, marker_size_m)
    rotation_gate = settings.pair_rotation_rms_gate_deg
    marker_ids = sorted(assignment)
    total_cost = 0.0
    constrained_edges = 0
    for index_a, marker_low in enumerate(marker_ids):
        for marker_high in marker_ids[index_a + 1 :]:
            pair = (marker_low, marker_high)
            edge = pair_consensus.get(pair)
            if edge is None:
                continue
            rotation_ba, translation_ba = _transform_high_in_low(
                assignment[marker_low],
                assignment[marker_high],
            )
            translation_error = float(np.linalg.norm(translation_ba - edge.translation_ba))
            rotation_error = _rotation_geodesic_deg(rotation_ba, edge.rotation_ba)
            if translation_error > translation_gate or rotation_error > rotation_gate:
                return None
            constrained_edges += 1
            total_cost += (translation_error / translation_gate) ** 2 + (
                rotation_error / rotation_gate
            ) ** 2
    if constrained_edges == 0:
        return None
    return -total_cost


def _reference_gauge_pose(marker_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    half = marker_size_m / 2.0
    top_left = np.array([-half, -half, 0.0], dtype=np.float64)
    top_right = np.array([half, -half, 0.0], dtype=np.float64)
    bottom_right = np.array([half, half, 0.0], dtype=np.float64)
    bottom_left = np.array([-half, half, 0.0], dtype=np.float64)
    rotation = footprint_orientation(top_left, top_right, bottom_left, bottom_right)
    translation = marker_origin_on_object(bottom_left, bottom_right)
    return rotation, translation


def _initialize_marker_poses(
    reference_marker_id: int,
    ref_rotation: np.ndarray,
    ref_translation: np.ndarray,
    expected_ids: list[int],
    pair_consensus: dict[MarkerPair, _PairConsensus],
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


def _initialize_frame_poses(
    assigned_candidates: dict[int, dict[int, _MarkerCandidate]],
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
            frame_poses[frame_index] = _average_poses(estimates)
    return frame_poses


def _average_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    rotations = [rotation for rotation, _ in poses]
    translations = np.stack([translation for _, translation in poses], axis=0)
    return _average_rotations(rotations), np.mean(translations, axis=0)


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0].copy()
    quaternions = [_rotation_matrix_to_quaternion(rotation) for rotation in rotations]
    reference = quaternions[0]
    aligned = [quaternion if np.dot(quaternion, reference) >= 0.0 else -quaternion for quaternion in quaternions]
    mean = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 0.0:
        return rotations[0].copy()
    return _quaternion_to_rotation_matrix(mean / norm)


def _build_corner_observations(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    expected_ids: list[int],
) -> list[_CornerObservation]:
    expected_set = set(expected_ids)
    corner_observations: list[_CornerObservation] = []
    for frame_index, (_, markers) in enumerate(observations):
        for marker_id, corners in markers.items():
            if marker_id not in expected_set:
                continue
            for corner_index in range(4):
                corner_observations.append(
                    _CornerObservation(
                        frame_index=frame_index,
                        marker_id=marker_id,
                        corner_index=corner_index,
                        image_point=corners[corner_index],
                    )
                )
    return corner_observations


def _run_bundle_adjustment(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    reference_marker_id: int,
    non_reference_ids: list[int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settings: CalibrationSettings,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray] | None],
    np.ndarray,
    str | None,
]:
    active_frames = sorted(
        {
            observation.frame_index
            for observation, keep in zip(corner_observations, inlier_mask, strict=True)
            if keep
        }
    )
    if not active_frames:
        return marker_poses, frame_poses, inlier_mask, "Bundle adjustment has no active frames."
    if not non_reference_ids:
        return marker_poses, frame_poses, inlier_mask, None

    x0 = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
    if not np.all(np.isfinite(x0)):
        return marker_poses, frame_poses, inlier_mask, "Bundle adjustment initial parameters are non-finite."

    jac_sparsity = _build_jac_sparsity(
        corner_observations,
        inlier_mask,
        non_reference_ids,
        active_frames,
        reference_marker_id,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(params)):
            return np.full(jac_sparsity.shape[0], 1e3, dtype=np.float64)
        marker_state, frame_pose_list = _unpack_parameters(
            params,
            marker_poses,
            frame_poses,
            non_reference_ids,
            active_frames,
            reference_marker_id,
        )
        values: list[float] = []
        for observation, keep in zip(corner_observations, inlier_mask, strict=True):
            if not keep:
                continue
            frame_pose = frame_pose_list[observation.frame_index]
            marker_pose = marker_state.get(observation.marker_id)
            if frame_pose is None or marker_pose is None:
                values.extend([1000.0, 1000.0])
                continue
            projected = _project_corner(
                observation.corner_index,
                marker_pose,
                frame_pose,
                object_points,
                camera_matrix,
                dist_coeffs,
            )
            if not np.all(np.isfinite(projected)):
                values.extend([1000.0, 1000.0])
                continue
            delta = projected - observation.image_point
            values.extend(delta.tolist())
        return np.asarray(values, dtype=np.float64)

    try:
        result = least_squares(
            residuals,
            x0,
            jac_sparsity=jac_sparsity,
            loss="huber",
            f_scale=settings.huber_delta_px,
            max_nfev=max(settings.max_ba_iterations * len(x0), len(x0) + 1),
        )
    except ValueError as exc:
        return marker_poses, frame_poses, inlier_mask, f"Bundle adjustment failed: {exc}"

    if not result.success or not np.all(np.isfinite(result.x)):
        return (
            marker_poses,
            frame_poses,
            inlier_mask,
            f"Bundle adjustment did not converge (status={result.status}).",
        )

    marker_poses, frame_poses = _unpack_parameters(
        result.x,
        marker_poses,
        frame_poses,
        non_reference_ids,
        active_frames,
        reference_marker_id,
    )
    depth_failure = _positive_depth_failure(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points,
    )
    if depth_failure is not None:
        return marker_poses, frame_poses, inlier_mask, depth_failure
    return marker_poses, frame_poses, inlier_mask, None


def _prune_and_refit(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    reference_marker_id: int,
    non_reference_ids: list[int],
    expected_ids: list[int],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    accepted_frames: frozenset[int],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settings: CalibrationSettings,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray] | None],
    np.ndarray,
    dict[MarkerPair, _PairConsensus],
    str | None,
]:
    errors = _corner_errors(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points,
        camera_matrix,
        dist_coeffs,
    )
    pruned = inlier_mask & (errors <= settings.corner_outlier_px)
    pruned = _drop_frames_without_covisibility(corner_observations, pruned)
    pruned = _mask_corner_observations_for_frames(corner_observations, accepted_frames) & pruned
    if int(np.count_nonzero(pruned)) < 8:
        return marker_poses, frame_poses, inlier_mask, pair_consensus, (
            "Too few inlier corners remain after pruning."
        )

    remaining_frames = _covisible_frames_from_inliers(corner_observations, pruned)
    updated_consensus, support_failure = _recheck_pair_support(
        pair_consensus,
        corner_observations,
        pruned,
        expected_ids,
        reference_marker_id,
        settings,
        allowed_frames=remaining_frames,
    )
    if support_failure is not None:
        return marker_poses, frame_poses, pruned, updated_consensus, support_failure

    marker_poses, frame_poses, pruned, ba_failure = _run_bundle_adjustment(
        corner_observations,
        pruned,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_ids,
        object_points,
        camera_matrix,
        dist_coeffs,
        settings,
    )
    if ba_failure is not None:
        return marker_poses, frame_poses, pruned, updated_consensus, ba_failure
    return marker_poses, frame_poses, pruned, updated_consensus, None


def _complete_markers_per_frame(
    corner_observations: list[_CornerObservation],
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


def _drop_frames_without_covisibility(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> np.ndarray:
    updated = inlier_mask.copy()
    while True:
        complete = _complete_markers_per_frame(corner_observations, updated)
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


def _recheck_pair_support(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    expected_ids: list[int],
    reference_marker_id: int,
    settings: CalibrationSettings,
    allowed_frames: frozenset[int] | None = None,
) -> tuple[dict[MarkerPair, _PairConsensus], str | None]:
    complete = _complete_markers_per_frame(corner_observations, inlier_mask)
    updated: dict[MarkerPair, _PairConsensus] = {}
    for pair, edge in pair_consensus.items():
        marker_low, marker_high = pair
        supported_frames = tuple(
            sorted(
                frame_index
                for frame_index in edge.inlier_frames
                if (allowed_frames is None or frame_index in allowed_frames)
                and frame_index in complete
                and marker_low in complete[frame_index]
                and marker_high in complete[frame_index]
            )
        )
        if len(supported_frames) < settings.min_inliers_per_edge:
            return updated, (
                f"Pair ({marker_low}, {marker_high}) has only "
                f"{len(supported_frames)} supported frames after pruning; need "
                f"{settings.min_inliers_per_edge}."
            )
        selected_hypotheses = {
            frame_index: edge.inlier_hypotheses[frame_index]
            for frame_index in supported_frames
        }
        updated[pair] = _PairConsensus(
            marker_a=marker_low,
            marker_b=marker_high,
            rotation_ba=edge.rotation_ba,
            translation_ba=edge.translation_ba,
            inlier_frames=supported_frames,
            inlier_hypotheses=selected_hypotheses,
        )

    connected = _connected_marker_ids(updated, reference_marker_id)
    missing = sorted(set(expected_ids) - connected)
    if missing:
        return updated, (
            f"Expected marker IDs are not connected after pruning; missing {missing}."
        )
    return updated, None


def _build_jac_sparsity(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    non_reference_ids: list[int],
    active_frames: list[int],
    reference_marker_id: int,
) -> lil_matrix:
    marker_param_index = {marker_id: index for index, marker_id in enumerate(non_reference_ids)}
    frame_param_index = {frame_index: index for index, frame_index in enumerate(active_frames)}
    num_marker_params = 6 * len(non_reference_ids)
    num_frame_params = 6 * len(active_frames)
    num_params = num_marker_params + num_frame_params
    num_residuals = 2 * int(np.count_nonzero(inlier_mask))
    sparsity = lil_matrix((num_residuals, num_params), dtype=int)
    row = 0
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        if observation.marker_id != reference_marker_id:
            marker_offset = 6 * marker_param_index[observation.marker_id]
            sparsity[row : row + 2, marker_offset : marker_offset + 6] = 1
        frame_offset = num_marker_params + 6 * frame_param_index[observation.frame_index]
        sparsity[row : row + 2, frame_offset : frame_offset + 6] = 1
        row += 2
    return sparsity


def _positive_depth_failure(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    object_points: np.ndarray,
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
        point_layout = marker_rotation @ object_points[observation.corner_index] + marker_translation
        point_camera = frame_rotation @ point_layout + frame_translation
        if not np.all(np.isfinite(point_camera)) or float(point_camera[2]) <= min_depth_m:
            return (
                f"Bundle adjustment produced non-positive depth for marker "
                f"{observation.marker_id} in frame {observation.frame_index}."
            )
    return None


def _pack_parameters(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    non_reference_ids: list[int],
    active_frames: list[int],
) -> np.ndarray:
    values: list[float] = []
    for marker_id in non_reference_ids:
        rotation, translation = marker_poses[marker_id]
        rvec, _ = cv2.Rodrigues(rotation)
        values.extend(rvec.reshape(3).tolist())
        values.extend(translation.reshape(3).tolist())
    for frame_index in active_frames:
        frame_pose = frame_poses[frame_index]
        if frame_pose is None:
            values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
            continue
        rotation, translation = frame_pose
        rvec, _ = cv2.Rodrigues(rotation)
        values.extend(rvec.reshape(3).tolist())
        values.extend(translation.reshape(3).tolist())
    return np.asarray(values, dtype=np.float64)


def _unpack_parameters(
    params: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    non_reference_ids: list[int],
    active_frames: list[int],
    reference_marker_id: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray] | None]]:
    marker_state = dict(marker_poses)
    offset = 0
    for marker_id in non_reference_ids:
        rvec = params[offset : offset + 3]
        translation = params[offset + 3 : offset + 6]
        offset += 6
        rotation, _ = cv2.Rodrigues(rvec)
        marker_state[marker_id] = (rotation, translation)
    marker_state[reference_marker_id] = marker_poses[reference_marker_id]

    updated_frame_poses = list(frame_poses)
    for frame_index in active_frames:
        rvec = params[offset : offset + 3]
        translation = params[offset + 3 : offset + 6]
        offset += 6
        rotation, _ = cv2.Rodrigues(rvec)
        updated_frame_poses[frame_index] = (rotation, translation)
    return marker_state, updated_frame_poses


def _project_corner(
    corner_index: int,
    marker_pose: tuple[np.ndarray, np.ndarray],
    frame_pose: tuple[np.ndarray, np.ndarray],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    marker_rotation, marker_translation = marker_pose
    frame_rotation, frame_translation = frame_pose
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


def _corner_errors(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    object_points: np.ndarray,
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
        projected = _project_corner(
            observation.corner_index,
            marker_pose,
            frame_pose,
            object_points,
            camera_matrix,
            dist_coeffs,
        )
        errors[index] = float(np.linalg.norm(projected - observation.image_point))
    return errors


def _footprints_from_poses(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    marker_size_m: float,
) -> dict[int, MarkerFootprint]:
    object_points = marker_corner_object_points(marker_size_m)
    footprints: dict[int, MarkerFootprint] = {}
    for marker_id, (rotation, translation) in marker_poses.items():
        payload = {}
        for corner_index, corner_name in enumerate(CORNER_NAMES):
            point = rotation @ object_points[corner_index] + translation
            payload[corner_name] = point.tolist()
        footprints[marker_id] = footprint_from_dict(marker_id, payload)
    return footprints


def _edge_diagnostics(
    pair: MarkerPair,
    edge: _PairConsensus,
) -> EdgeDiagnostics:
    translations: list[float] = []
    rotations: list[float] = []
    for frame_index in edge.inlier_frames:
        rotation, translation = edge.inlier_hypotheses[frame_index]
        translations.append(float(np.linalg.norm(translation - edge.translation_ba)))
        rotations.append(_rotation_geodesic_deg(rotation, edge.rotation_ba))
    return EdgeDiagnostics(
        marker_a=pair[0],
        marker_b=pair[1],
        inlier_count=len(edge.inlier_frames),
        translation_rms_m=float(np.sqrt(np.mean(np.square(translations))) if translations else 0.0),
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(rotations))) if rotations else 0.0),
    )


def _build_quality_report(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
    missing_ids: frozenset[int],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> CalibrationQualityReport:
    errors = _corner_errors(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points,
        camera_matrix,
        dist_coeffs,
    )
    finite_errors = errors[np.isfinite(errors) & inlier_mask]
    per_marker: dict[int, list[float]] = {}
    for observation, error, keep in zip(corner_observations, errors, inlier_mask, strict=True):
        if not keep or not np.isfinite(error):
            continue
        per_marker.setdefault(observation.marker_id, []).append(error)
    per_marker_rms = {
        marker_id: float(np.sqrt(np.mean(np.square(values))))
        for marker_id, values in per_marker.items()
    }
    edge_reports = [_edge_diagnostics(pair, edge) for pair, edge in sorted(pair_consensus.items())]
    connected = frozenset(_connected_marker_ids(pair_consensus, reference_marker_id))
    observed_ids = {
        observation.marker_id
        for observation, keep in zip(corner_observations, inlier_mask, strict=True)
        if keep
    }
    final_frame_count = _covisible_frame_count(corner_observations, inlier_mask)
    return CalibrationQualityReport(
        reprojection_rms_px=float(np.sqrt(np.mean(np.square(finite_errors))) if finite_errors.size else float("inf")),
        per_marker_reprojection_rms_px=per_marker_rms,
        edges=tuple(edge_reports),
        pair_translation_rms_max_m=max((edge.translation_rms_m for edge in edge_reports), default=0.0),
        pair_rotation_rms_max_deg=max((edge.rotation_rms_deg for edge in edge_reports), default=0.0),
        frame_count=final_frame_count,
        observation_count=len(corner_observations),
        inlier_corner_count=int(np.count_nonzero(inlier_mask)),
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        connected_marker_ids=connected,
        missing_expected_ids=missing_ids,
        unused_expected_ids=frozenset(set(expected_ids) - observed_ids),
    )


def _check_quality_gates(
    quality: CalibrationQualityReport,
    settings: CalibrationSettings,
    marker_size_m: float,
    expected_ids: list[int],
) -> str | None:
    if quality.reprojection_rms_px > settings.reprojection_rms_gate_px:
        return (
            f"Global reprojection RMS {quality.reprojection_rms_px:.3f} px exceeds "
            f"{settings.reprojection_rms_gate_px:.3f} px gate."
        )
    for marker_id in expected_ids:
        marker_rms = quality.per_marker_reprojection_rms_px.get(marker_id)
        if marker_rms is None:
            return f"Marker {marker_id} has no inlier reprojection samples after calibration."
        if marker_rms > settings.reprojection_rms_gate_px:
            return (
                f"Marker {marker_id} reprojection RMS {marker_rms:.3f} px exceeds "
                f"{settings.reprojection_rms_gate_px:.3f} px gate."
            )
    translation_gate = settings.pair_translation_rms_gate_ratio * marker_size_m
    if quality.pair_translation_rms_max_m > translation_gate:
        return (
            f"Pair translation RMS {quality.pair_translation_rms_max_m:.4f} m exceeds "
            f"{translation_gate:.4f} m gate."
        )
    if quality.pair_rotation_rms_max_deg > settings.pair_rotation_rms_gate_deg:
        return (
            f"Pair rotation RMS {quality.pair_rotation_rms_max_deg:.2f} deg exceeds "
            f"{settings.pair_rotation_rms_gate_deg:.2f} deg gate."
        )
    if quality.missing_expected_ids:
        return f"Missing expected marker IDs: {sorted(quality.missing_expected_ids)}."
    return None


def _empty_quality(
    missing_expected_ids: frozenset[int],
    connected_marker_ids: frozenset[int],
    input_frame_count: int = 0,
    rejected_frame_count: int = 0,
    accepted_frame_count: int = 0,
) -> CalibrationQualityReport:
    return CalibrationQualityReport(
        reprojection_rms_px=float("inf"),
        per_marker_reprojection_rms_px={},
        edges=(),
        pair_translation_rms_max_m=float("inf"),
        pair_rotation_rms_max_deg=float("inf"),
        frame_count=0,
        observation_count=0,
        inlier_corner_count=0,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        connected_marker_ids=connected_marker_ids,
        missing_expected_ids=missing_expected_ids,
        unused_expected_ids=frozenset(),
    )


def _quality_from_pairs(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
    missing_ids: frozenset[int],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    observation_count: int,
) -> CalibrationQualityReport:
    edge_reports = [_edge_diagnostics(pair, edge) for pair, edge in sorted(pair_consensus.items())]
    return CalibrationQualityReport(
        reprojection_rms_px=float("inf"),
        per_marker_reprojection_rms_px={},
        edges=tuple(edge_reports),
        pair_translation_rms_max_m=max((edge.translation_rms_m for edge in edge_reports), default=0.0),
        pair_rotation_rms_max_deg=max((edge.rotation_rms_deg for edge in edge_reports), default=0.0),
        frame_count=accepted_frame_count,
        observation_count=observation_count,
        inlier_corner_count=0,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        connected_marker_ids=frozenset(_connected_marker_ids(pair_consensus, reference_marker_id)),
        missing_expected_ids=missing_ids,
        unused_expected_ids=frozenset(set(expected_ids) - _connected_marker_ids(pair_consensus, reference_marker_id)),
    )


def _rotation_geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
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


def _synth_marker_corners(
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


def _synth_pair_observations(
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
        markers = _synth_marker_corners(
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


def _self_check() -> None:
    """Minimal synthetic sanity check for import-time regression."""
    marker_size = 0.07
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    object_points = marker_corner_object_points(marker_size)
    settings = CalibrationSettings(min_inliers_per_edge=20)

    ref_rotation, ref_translation = _reference_gauge_pose(marker_size)
    pair_poses = {
        0: (ref_rotation, ref_translation),
        1: (ref_rotation, ref_translation + np.array([0.12, 0.0, -0.05])),
    }

    mostly_good = calibrate_marker_layout(
        _synth_pair_observations(
            25,
            pair_poses,
            object_points,
            camera_matrix,
            dist_coeffs,
            corrupt_frames=frozenset({2, 7, 11, 16, 22}),
            corrupt_offset=np.array([0.20, 0.0, -0.08]),
        ),
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=marker_size,
        settings=settings,
    )
    assert mostly_good.failure_reason is None, mostly_good.failure_reason
    assert mostly_good.layout is not None
    assert mostly_good.quality is not None
    assert mostly_good.quality.input_frame_count == 25
    assert mostly_good.quality.rejected_frame_count == 5
    assert mostly_good.quality.accepted_frame_count == 20

    all_bad = calibrate_marker_layout(
        _synth_pair_observations(
            25,
            pair_poses,
            object_points,
            camera_matrix,
            dist_coeffs,
            corrupt_frames=frozenset(range(25)),
            varying_corrupt=True,
        ),
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=marker_size,
        settings=settings,
    )
    assert all_bad.layout is None
    assert all_bad.failure_reason is not None
    assert all_bad.quality is not None
    assert all_bad.quality.input_frame_count == 25
    assert all_bad.quality.accepted_frame_count == 0

    marker_poses = {
        0: (ref_rotation, ref_translation),
        1: (ref_rotation, ref_translation + np.array([0.12, 0.0, -0.05])),
        2: (ref_rotation, ref_translation + np.array([0.24, 0.0, -0.10])),
        3: (ref_rotation, ref_translation + np.array([0.36, 0.0, -0.15])),
    }

    observations: list[FrameObservation] = []
    chain_pairs = [(0, 1), (1, 2), (2, 3)]
    frame_index = 0
    for marker_a, marker_b in chain_pairs:
        for _ in range(25):
            layout_rotation, _ = cv2.Rodrigues(np.array([0.1, -0.15 + 0.002 * frame_index, 0.05]))
            layout_translation = np.array([0.02, -0.01, 0.6 + 0.002 * frame_index], dtype=np.float64)
            markers = _synth_marker_corners(
                marker_poses,
                (marker_b, marker_a),
                layout_rotation,
                layout_translation,
                object_points,
                camera_matrix,
                dist_coeffs,
            )
            observations.append(FrameObservation(frame_id=frame_index, markers=markers))
            frame_index += 1

    result = calibrate_marker_layout(
        observations,
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1, 2, 3],
        reference_marker_id=0,
        marker_size_m=marker_size,
        settings=settings,
    )
    assert result.failure_reason is None, result.failure_reason
    assert result.layout is not None
    for edge in result.quality.edges if result.quality else ():
        assert edge.inlier_count >= 20
    recovered = result.layout.footprints[3].bottom_left - result.layout.footprints[0].bottom_left
    assert np.linalg.norm(recovered - np.array([0.36, 0.0, -0.15])) < 0.02


def _input_boundary_self_check() -> None:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    valid_obs = [
        FrameObservation(
            frame_id=0,
            markers={
                0: np.zeros((4, 2), dtype=np.float64),
                1: np.ones((4, 2), dtype=np.float64),
            },
        )
    ]

    duplicate_ids = calibrate_marker_layout(
        valid_obs,
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1, 0],
        reference_marker_id=0,
        marker_size_m=0.07,
    )
    assert duplicate_ids.failure_reason is not None
    assert "duplicates" in duplicate_ids.failure_reason

    duplicate_frames = calibrate_marker_layout(
        [valid_obs[0], valid_obs[0]],
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=0.07,
    )
    assert duplicate_frames.failure_reason is not None
    assert "Duplicate FrameObservation.frame_id" in duplicate_frames.failure_reason

    bad_corners = calibrate_marker_layout(
        [
            FrameObservation(
                frame_id=0,
                markers={0: np.zeros((3, 2)), 1: np.ones((4, 2))},
            )
        ],
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=0.07,
    )
    assert bad_corners.failure_reason is not None
    assert "Malformed corners" in bad_corners.failure_reason


if __name__ == "__main__":
    _self_check()
    _input_boundary_self_check()
    print("marker_layout_calibration self-check passed")
