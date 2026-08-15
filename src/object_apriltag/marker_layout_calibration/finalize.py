"""Quality gates, partial recovery, and accepted result construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from object_apriltag.layout import build_marker_layout

from object_apriltag.marker_layout_calibration.input import object_points_by_marker
from object_apriltag.marker_layout_calibration.solve_primitives import MarkerPair, PairConsensus, connected_marker_ids
from object_apriltag.marker_layout_calibration.solve_quality import (
    build_quality_report,
    collect_quality_gate_failures,
    footprints_from_poses,
    quality_from_pairs,
)
from object_apriltag.marker_layout_calibration.types import (
    CalibrationQualityReport,
    CalibrationResult,
    CalibrationSettings,
    FrameObservation,
    OmittedMarkerDiagnostic,
)

def connectivity_omission_reason(stage: str) -> str:
    if stage == "initial_consensus":
        return "not_connected_to_reference"
    if stage == "assignment_support":
        return "not_connected_after_assignment_support"
    if stage == "post_pruning":
        return "not_connected_after_pruning"
    return "not_connected_to_reference"


def omitted_marker_records(
    requested_marker_ids: Sequence[int],
    emitted_marker_ids: set[int],
    omitted: Mapping[int, str],
) -> tuple[OmittedMarkerDiagnostic, ...]:
    return tuple(
        OmittedMarkerDiagnostic(marker_id=marker_id, reason=omitted[marker_id])
        for marker_id in sorted(requested_marker_ids)
        if marker_id not in emitted_marker_ids
    )


def quality_for_partial_output(
    quality: CalibrationQualityReport,
    *,
    requested_marker_ids: Sequence[int],
    emitted_marker_ids: set[int],
) -> CalibrationQualityReport:
    missing = frozenset(set(requested_marker_ids) - emitted_marker_ids)
    return CalibrationQualityReport(
        reprojection_rms_px=quality.reprojection_rms_px,
        per_marker_reprojection_rms_px={
            marker_id: value
            for marker_id, value in quality.per_marker_reprojection_rms_px.items()
            if marker_id in emitted_marker_ids
        },
        edges=tuple(
            edge
            for edge in quality.edges
            if edge.marker_a in emitted_marker_ids and edge.marker_b in emitted_marker_ids
        ),
        pair_translation_rms_max_m=quality.pair_translation_rms_max_m,
        pair_rotation_rms_max_deg=quality.pair_rotation_rms_max_deg,
        frame_count=quality.frame_count,
        observation_count=quality.observation_count,
        inlier_corner_count=quality.inlier_corner_count,
        input_frame_count=quality.input_frame_count,
        rejected_frame_count=quality.rejected_frame_count,
        accepted_frame_count=quality.accepted_frame_count,
        connected_marker_ids=frozenset(emitted_marker_ids),
        missing_expected_ids=missing,
        unused_expected_ids=frozenset(
            marker_id
            for marker_id in quality.unused_expected_ids
            if marker_id in emitted_marker_ids
        ),
        assignment_rejections=quality.assignment_rejections,
        assignment_rejection_records=quality.assignment_rejection_records,
        fallback_assignment_records=quality.fallback_assignment_records,
        dropped_pair_edges=quality.dropped_pair_edges,
        restored_pair_edges=quality.restored_pair_edges,
        anchor_core=quality.anchor_core,
    )


def wrap_subset_as_partial(
    subset_result: CalibrationResult,
    *,
    requested_marker_ids: Sequence[int],
    emitted_marker_ids: set[int],
    omitted: Mapping[int, str],
) -> CalibrationResult:
    if subset_result.layout is None or subset_result.quality is None:
        return CalibrationResult(
            None,
            subset_result.quality,
            subset_result.failure_reason,
            outcome="refused",
            calibration_policy="best_effort",
        )
    omitted_records = omitted_marker_records(requested_marker_ids, emitted_marker_ids, omitted)
    return CalibrationResult(
        subset_result.layout,
        quality_for_partial_output(
            subset_result.quality,
            requested_marker_ids=requested_marker_ids,
            emitted_marker_ids=emitted_marker_ids,
        ),
        None,
        outcome="partial",
        calibration_policy="best_effort",
        failed_quality_gates=subset_result.failed_quality_gates,
        selected_checkpoint_stage=subset_result.selected_checkpoint_stage,
        failed_refinement_stage=subset_result.failed_refinement_stage,
        omitted_markers=omitted_records,
        partial_output=True,
    )


def emit_partial_calibration_result(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    requested_marker_ids: Sequence[int],
    connected_ids: set[int],
    omitted: Mapping[int, str],
    reference_marker_id: int,
    marker_size_m: float,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    best_effort: bool,
    anchor_marker_ids: Sequence[int] | None = None,
) -> CalibrationResult:
    emitted_ids = sorted(connected_ids & set(requested_marker_ids))
    non_reference = [marker_id for marker_id in emitted_ids if marker_id != reference_marker_id]
    if reference_marker_id not in emitted_ids or not non_reference:
        return CalibrationResult(
            None,
            None,
            (
                "Partial output requires at least one non-reference marker connected "
                f"to reference {reference_marker_id}."
            ),
            outcome="refused",
            calibration_policy="best_effort",
            partial_output=True,
        )

    merged_omitted = dict(omitted)
    for marker_id in requested_marker_ids:
        if marker_id not in connected_ids and marker_id not in merged_omitted:
            merged_omitted[marker_id] = "not_connected_to_reference"

    emitted_sizes = {marker_id: marker_sizes_m[marker_id] for marker_id in emitted_ids}
    filtered_anchors: tuple[int, ...] | None = None
    if anchor_marker_ids is not None:
        emitted_set = set(emitted_ids)
        filtered_anchors = tuple(marker_id for marker_id in anchor_marker_ids if marker_id in emitted_set)
        if not filtered_anchors:
            filtered_anchors = None
    from object_apriltag.marker_layout_calibration.pipeline import calibrate_marker_layout

    subset_result = calibrate_marker_layout(
        observations,
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=emitted_ids,
        reference_marker_id=reference_marker_id,
        marker_size_m=marker_size_m,
        settings=settings,
        anchor_marker_ids=filtered_anchors,
        marker_sizes_m=emitted_sizes,
        best_effort=best_effort,
        partial_output=False,
    )
    if subset_result.layout is None:
        return CalibrationResult(
            None,
            subset_result.quality,
            subset_result.failure_reason,
            outcome="refused",
            calibration_policy="best_effort",
            partial_output=True,
            omitted_markers=omitted_marker_records(
                requested_marker_ids,
                set(emitted_ids),
                merged_omitted,
            ),
        )
    return wrap_subset_as_partial(
        subset_result,
        requested_marker_ids=requested_marker_ids,
        emitted_marker_ids=set(emitted_ids),
        omitted=merged_omitted,
    )


def partial_from_pair_consensus_or_refuse(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pair_consensus: dict[MarkerPair, PairConsensus],
    quality: CalibrationQualityReport,
    failure_message: str,
    *,
    requested_marker_ids: Sequence[int],
    omitted_markers: Mapping[int, str],
    connectivity_stage: str,
    reference_marker_id: int,
    marker_size_m: float,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    best_effort: bool,
    partial_output: bool,
    anchor_marker_ids: Sequence[int] | None,
) -> CalibrationResult:
    if partial_output and best_effort:
        connected = connected_marker_ids(pair_consensus, reference_marker_id)
        merged = dict(omitted_markers)
        for marker_id in requested_marker_ids:
            if marker_id not in connected and marker_id not in merged:
                merged[marker_id] = connectivity_omission_reason(connectivity_stage)
        return emit_partial_calibration_result(
            observations,
            camera_matrix,
            dist_coeffs,
            requested_marker_ids=requested_marker_ids,
            connected_ids=connected,
            omitted=merged,
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            best_effort=best_effort,
            anchor_marker_ids=anchor_marker_ids,
        )
    return CalibrationResult(None, quality, failure_message)


def partial_after_missing_accepted_frames_or_refuse(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pair_consensus: dict[MarkerPair, PairConsensus],
    quality: CalibrationQualityReport,
    failure_message: str,
    *,
    requested_marker_ids: Sequence[int],
    omitted_markers: Mapping[int, str],
    markers_in_accepted_frames: set[int],
    missing_after_rejection: Sequence[int],
    reference_marker_id: int,
    marker_size_m: float,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    best_effort: bool,
    partial_output: bool,
    anchor_marker_ids: Sequence[int] | None,
) -> CalibrationResult:
    if partial_output and best_effort:
        connected = connected_marker_ids(pair_consensus, reference_marker_id) & markers_in_accepted_frames
        merged = dict(omitted_markers)
        for marker_id in missing_after_rejection:
            merged.setdefault(marker_id, "no_accepted_frame_observations")
        return emit_partial_calibration_result(
            observations,
            camera_matrix,
            dist_coeffs,
            requested_marker_ids=requested_marker_ids,
            connected_ids=connected,
            omitted=merged,
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            best_effort=best_effort,
            anchor_marker_ids=anchor_marker_ids,
        )
    return CalibrationResult(None, quality, failure_message)


def maybe_wrap_partial_success(
    result: CalibrationResult,
    *,
    requested_marker_ids: Sequence[int],
    emitted_marker_ids: set[int],
    omitted_markers: Mapping[int, str],
) -> CalibrationResult:
    if not omitted_markers or result.layout is None or result.failure_reason is not None:
        return result
    return wrap_subset_as_partial(
        result,
        requested_marker_ids=requested_marker_ids,
        emitted_marker_ids=emitted_marker_ids,
        omitted=omitted_markers,
    )


def accepted_calibration_result(
    layout,
    quality: CalibrationQualityReport,
    *,
    best_effort: bool,
) -> CalibrationResult:
    return CalibrationResult(
        layout,
        quality,
        None,
        outcome="accepted",
        calibration_policy="best_effort" if best_effort else "strict",
    )


def check_quality_gates(
    quality: CalibrationQualityReport,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    expected_ids: list[int],
) -> str | None:
    failures = collect_quality_gate_failures(quality, settings, marker_sizes_m, expected_ids)
    return failures[0].message if failures else None


def empty_quality(
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


def finalize_solved_calibration(
    marker_poses,
    quality: CalibrationQualityReport,
    settings: CalibrationSettings,
    marker_sizes_m,
    expected_ids: list[int],
    reference_marker_id: int,
    marker_size_m: float,
    missing_ids: frozenset[int],
    *,
    gate_failure: str | None,
    best_effort: bool,
    anchor_marker_ids=None,
) -> CalibrationResult | None:
    from typing import Literal

    policy: Literal["strict", "best_effort"] = "best_effort" if best_effort else "strict"
    gate_failures = collect_quality_gate_failures(
        quality,
        settings,
        marker_sizes_m,
        expected_ids,
    )
    failed_gate_messages = tuple(failure.message for failure in gate_failures)

    if gate_failures:
        if (
            best_effort
            and all(failure.category == "strict" for failure in gate_failures)
            and not missing_ids
        ):
            footprints = footprints_from_poses(marker_poses, marker_sizes_m)
            if set(footprints) != set(expected_ids):
                absent = sorted(set(expected_ids) - set(footprints))
                return CalibrationResult(
                    None,
                    quality,
                    f"Calibration did not produce all expected marker footprints; missing {absent}.",
                    outcome="refused",
                    calibration_policy=policy,
                    failed_quality_gates=failed_gate_messages,
                )
            layout = build_marker_layout(
                reference_marker_id=reference_marker_id,
                marker_size_m=marker_size_m,
                footprints=footprints,
                marker_sizes_m=dict(marker_sizes_m),
                anchor_marker_ids=anchor_marker_ids,
            )
            return CalibrationResult(
                layout,
                quality,
                None,
                outcome="provisional",
                calibration_policy=policy,
                failed_quality_gates=failed_gate_messages,
            )
        return CalibrationResult(
            None,
            quality,
            gate_failure or gate_failures[0].message,
            outcome="refused",
            calibration_policy=policy,
            failed_quality_gates=failed_gate_messages,
        )

    if missing_ids:
        return CalibrationResult(
            None,
            quality,
            f"Expected marker IDs are not connected to reference: {sorted(missing_ids)}.",
            outcome="refused",
            calibration_policy=policy,
        )

    footprints = footprints_from_poses(marker_poses, marker_sizes_m)
    if set(footprints) != set(expected_ids):
        absent = sorted(set(expected_ids) - set(footprints))
        return CalibrationResult(
            None,
            quality,
            f"Calibration did not produce all expected marker footprints; missing {absent}.",
            outcome="refused",
            calibration_policy=policy,
        )
    return None
