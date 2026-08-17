"""Quality aggregation for marker layout solve states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from object_apriltag.layout import CORNER_NAMES, footprint_from_dict
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CornerObservation,
    MarkerPair,
    PairConsensus,
    connected_marker_ids,
    corner_errors,
    covisible_frame_count,
    rotation_geodesic_deg,
)
from object_apriltag.marker_layout_calibration.types import (
    CalibrationQualityReport,
    EdgeDiagnostics,
    QualityGateFailure,
)
from object_apriltag.pose import marker_corner_object_points


def pair_translation_gate(
    settings: object,
    marker_sizes_m: Mapping[int, float],
    pair: MarkerPair,
) -> float:
    """Compute the size-scaled translation gate for a marker pair.

    Args:
        settings: Calibration settings with ``pair_translation_rms_gate_ratio``.
        marker_sizes_m: Physical edge lengths keyed by marker ID.
        pair: Low-to-high marker ID pair.

    Returns:
        Translation RMS gate in meters: ratio times the smaller marker edge length.
    """
    return settings.pair_translation_rms_gate_ratio * min(
        marker_sizes_m[pair[0]],
        marker_sizes_m[pair[1]],
    )


def edge_diagnostics(
    pair: MarkerPair,
    edge: PairConsensus,
) -> EdgeDiagnostics:
    """Compute RMS deviation of inlier hypotheses from pair consensus.

    Args:
        pair: Low-to-high marker ID pair.
        edge: Pair consensus with per-frame inlier hypotheses.

    Returns:
        Edge diagnostics with inlier count and translation/rotation RMS.
    """
    translations: list[float] = []
    rotations: list[float] = []
    for frame_index in edge.inlier_frames:
        rotation, translation = edge.inlier_hypotheses[frame_index]
        translations.append(float(np.linalg.norm(translation - edge.translation_ba)))
        rotations.append(rotation_geodesic_deg(rotation, edge.rotation_ba))
    return EdgeDiagnostics(
        marker_a=pair[0],
        marker_b=pair[1],
        inlier_count=len(edge.inlier_frames),
        translation_rms_m=float(np.sqrt(np.mean(np.square(translations))) if translations else 0.0),
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(rotations))) if rotations else 0.0),
    )


def footprints_from_poses(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    marker_sizes_m: Mapping[int, float],
) -> dict[int, object]:
    """Build object-frame corner footprints from solved marker poses.

    Args:
        marker_poses: Object-frame marker poses keyed by marker ID.
        marker_sizes_m: Physical edge lengths keyed by marker ID.

    Returns:
        ``MarkerFootprint`` objects keyed by marker ID with corner positions in
        the object frame.
    """
    from object_apriltag.layout import MarkerFootprint

    footprints: dict[int, MarkerFootprint] = {}
    for marker_id, (rotation, translation) in marker_poses.items():
        object_points = marker_corner_object_points(marker_sizes_m[marker_id])
        payload = {}
        for corner_index, corner_name in enumerate(CORNER_NAMES):
            point = rotation @ object_points[corner_index] + translation
            payload[corner_name] = point.tolist()
        footprints[marker_id] = footprint_from_dict(marker_id, payload)
    return footprints


def build_quality_report(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    pair_consensus: dict[MarkerPair, PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
    missing_ids: frozenset[int],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    assignment_rejections: object | None = None,
    assignment_rejection_records: object | None = None,
    fallback_assignment_records: object | None = None,
    dropped_pair_edges: object | None = None,
    restored_pair_edges: object | None = None,
    anchor_core: object | None = None,
) -> CalibrationQualityReport:
    """Build a full calibration quality report from bundle-adjustment state.

    Args:
        corner_observations: Full per-corner observation list.
        inlier_mask: Boolean mask parallel to ``corner_observations``.
        marker_poses: Object-frame marker poses keyed by marker ID.
        frame_poses: Per-frame layout poses.
        pair_consensus: Accepted pair consensus edges.
        expected_ids: Full set of marker IDs targeted by calibration.
        reference_marker_id: Root marker for connectivity analysis.
        missing_ids: Expected IDs still missing from the solved graph.
        input_frame_count: Raw observation frame count before rejection.
        rejected_frame_count: Frames rejected during assignment.
        accepted_frame_count: Frames with accepted assignments.
        object_points_by_marker: Object-frame corner coordinates per marker.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.
        assignment_rejections: Optional aggregated assignment rejection summary.
        assignment_rejection_records: Optional per-frame rejection records.
        fallback_assignment_records: Optional fallback assignment records.
        dropped_pair_edges: Optional dropped pair-edge audit records.
        restored_pair_edges: Optional restored pair-edge audit records.
        anchor_core: Optional anchor-core diagnostics payload.

    Returns:
        ``CalibrationQualityReport`` with reprojection, pair RMS, connectivity,
        and optional diagnostic attachments.
    """
    errors = corner_errors(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points_by_marker,
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
    edge_reports = [edge_diagnostics(pair, edge) for pair, edge in sorted(pair_consensus.items())]
    connected = frozenset(connected_marker_ids(pair_consensus, reference_marker_id))
    observed_ids = {
        observation.marker_id
        for observation, keep in zip(corner_observations, inlier_mask, strict=True)
        if keep
    }
    final_frame_count = covisible_frame_count(corner_observations, inlier_mask)
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
        assignment_rejections=assignment_rejections,
        assignment_rejection_records=assignment_rejection_records,
        fallback_assignment_records=fallback_assignment_records,
        dropped_pair_edges=dropped_pair_edges,
        restored_pair_edges=restored_pair_edges,
        anchor_core=anchor_core,
    )


def quality_from_pairs(
    pair_consensus: dict[MarkerPair, PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
    missing_ids: frozenset[int],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    observation_count: int,
    *,
    assignment_rejections: object | None = None,
    assignment_rejection_records: object | None = None,
    fallback_assignment_records: object | None = None,
    dropped_pair_edges: object | None = None,
    restored_pair_edges: object | None = None,
    anchor_core: object | None = None,
) -> CalibrationQualityReport:
    """Build a pair-only quality report without reprojection metrics.

    Args:
        pair_consensus: Accepted pair consensus edges.
        expected_ids: Full set of marker IDs targeted by calibration.
        reference_marker_id: Root marker for connectivity analysis.
        missing_ids: Expected IDs still missing from the solved graph.
        input_frame_count: Raw observation frame count before rejection.
        rejected_frame_count: Frames rejected during assignment.
        accepted_frame_count: Frames with accepted assignments.
        observation_count: Total corner observation count when BA is unavailable.
        assignment_rejections: Optional aggregated assignment rejection summary.
        assignment_rejection_records: Optional per-frame rejection records.
        fallback_assignment_records: Optional fallback assignment records.
        dropped_pair_edges: Optional dropped pair-edge audit records.
        restored_pair_edges: Optional restored pair-edge audit records.
        anchor_core: Optional anchor-core diagnostics payload.

    Returns:
        ``CalibrationQualityReport`` with pair RMS and connectivity fields;
        reprojection metrics are set to sentinel empty values.
    """
    edge_reports = [edge_diagnostics(pair, edge) for pair, edge in sorted(pair_consensus.items())]
    connected = connected_marker_ids(pair_consensus, reference_marker_id)
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
        connected_marker_ids=frozenset(connected),
        missing_expected_ids=missing_ids,
        unused_expected_ids=frozenset(set(expected_ids) - connected),
        assignment_rejections=assignment_rejections,
        assignment_rejection_records=assignment_rejection_records,
        fallback_assignment_records=fallback_assignment_records,
        dropped_pair_edges=dropped_pair_edges,
        restored_pair_edges=restored_pair_edges,
        anchor_core=anchor_core,
    )


def collect_quality_gate_failures(
    quality: object,
    settings: object,
    marker_sizes_m: Mapping[int, float],
    expected_ids: list[int],
) -> tuple[QualityGateFailure, ...]:
    """Collect strict, data, and connectivity gate failures against settings.

    Args:
        quality: ``CalibrationQualityReport`` to evaluate.
        settings: Calibration gate thresholds.
        marker_sizes_m: Physical edge lengths keyed by marker ID.
        expected_ids: Marker IDs required to have inlier reprojection samples.

    Returns:
        Tuple of gate failure records for exceeded thresholds or missing data.
    """
    failures: list[QualityGateFailure] = []
    if quality.reprojection_rms_px > settings.reprojection_rms_gate_px:
        failures.append(
            QualityGateFailure(
                "strict",
                (
                    f"Global reprojection RMS {quality.reprojection_rms_px:.3f} px exceeds "
                    f"{settings.reprojection_rms_gate_px:.3f} px gate."
                ),
            )
        )
    for marker_id in expected_ids:
        marker_rms = quality.per_marker_reprojection_rms_px.get(marker_id)
        if marker_rms is None:
            failures.append(
                QualityGateFailure(
                    "data",
                    f"Marker {marker_id} has no inlier reprojection samples after calibration.",
                )
            )
            continue
        if marker_rms > settings.reprojection_rms_gate_px:
            failures.append(
                QualityGateFailure(
                    "strict",
                    (
                        f"Marker {marker_id} reprojection RMS {marker_rms:.3f} px exceeds "
                        f"{settings.reprojection_rms_gate_px:.3f} px gate."
                    ),
                )
            )
    for edge in quality.edges:
        pair = (edge.marker_a, edge.marker_b)
        translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
        if edge.translation_rms_m > translation_gate:
            failures.append(
                QualityGateFailure(
                    "strict",
                    (
                        f"Pair ({pair[0]},{pair[1]}) translation RMS {edge.translation_rms_m:.4f} m exceeds "
                        f"{translation_gate:.4f} m gate."
                    ),
                )
            )
    if quality.pair_rotation_rms_max_deg > settings.pair_rotation_rms_gate_deg:
        failures.append(
            QualityGateFailure(
                "strict",
                (
                    f"Pair rotation RMS {quality.pair_rotation_rms_max_deg:.2f} deg exceeds "
                    f"{settings.pair_rotation_rms_gate_deg:.2f} deg gate."
                ),
            )
        )
    if quality.missing_expected_ids:
        failures.append(
            QualityGateFailure(
                "connectivity",
                f"Missing expected marker IDs: {sorted(quality.missing_expected_ids)}.",
            )
        )
    return tuple(failures)
