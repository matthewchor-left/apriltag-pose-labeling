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
    """Map an internal connectivity stage name to a stable omission reason string.

    Args:
        stage: Internal pipeline stage where connectivity was lost.

    Returns:
        A stable ``reason`` token for ``OmittedMarkerDiagnostic``.
    """
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
    """Build omission diagnostics for requested markers absent from the emitted layout.

    Args:
        requested_marker_ids: Marker IDs the caller asked to calibrate.
        emitted_marker_ids: Marker IDs present in the output layout.
        omitted: Mapping from omitted marker ID to omission reason.

    Returns:
        Sorted diagnostics for requested IDs not present in the emitted layout.
    """
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
    """Trim quality metrics to emitted markers and record which requested IDs were omitted.

    Args:
        quality: Full quality report from the subset solve.
        requested_marker_ids: Marker IDs originally requested by the caller.
        emitted_marker_ids: Marker IDs retained in the partial output.

    Returns:
        A quality report scoped to ``emitted_marker_ids`` with
        ``missing_expected_ids`` set to the requested-but-omitted IDs.
    """
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
    """Re-label a successful subset solve as a partial best-effort calibration result.

    Args:
        subset_result: Result from calibrating the connected marker subset.
        requested_marker_ids: Marker IDs originally requested by the caller.
        emitted_marker_ids: Marker IDs present in the subset layout.
        omitted: Mapping from omitted marker ID to omission reason.

    Returns:
        A ``partial`` best-effort result with trimmed quality and omission records,
        or a ``refused`` result when the subset solve did not produce a layout.
    """
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
    quality: CalibrationQualityReport | None = None,
) -> CalibrationResult:
    """Re-run calibration on the connected subset when partial output is allowed.

    Skips subset re-solve when the reference marker is isolated, returning ``quality``
    when provided so diagnostics can still be exported.

    Args:
        observations: Full multi-frame corner observations.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Camera distortion coefficients.
        requested_marker_ids: Marker IDs originally requested by the caller.
        connected_ids: Marker IDs connected to the reference in pair consensus.
        omitted: Known omission reasons before the subset re-solve.
        reference_marker_id: Layout reference marker ID.
        marker_size_m: Default marker edge length in meters.
        marker_sizes_m: Per-marker physical sizes for the full request.
        settings: Calibration thresholds and iteration limits.
        best_effort: Whether best-effort policy applies to the subset solve.
        anchor_marker_ids: Optional explicit anchor-core marker IDs.
        quality: Optional quality report to preserve when subset re-solve is skipped.

    Returns:
        A ``partial`` result when the subset solve succeeds, or ``refused`` when
        connectivity prerequisites are not met or the subset solve fails.
    """
    emitted_ids = sorted(connected_ids & set(requested_marker_ids))
    non_reference = [marker_id for marker_id in emitted_ids if marker_id != reference_marker_id]
    if reference_marker_id not in emitted_ids or not non_reference:
        merged_omitted = dict(omitted)
        for marker_id in requested_marker_ids:
            if marker_id not in connected_ids and marker_id not in merged_omitted:
                merged_omitted[marker_id] = "not_connected_to_reference"
        return CalibrationResult(
            None,
            quality,
            (
                "Partial subset re-solve skipped: reference marker "
                f"{reference_marker_id} has no connected non-reference markers."
            ),
            outcome="refused",
            calibration_policy="best_effort",
            partial_output=True,
            omitted_markers=omitted_marker_records(
                requested_marker_ids,
                set(connected_ids),
                merged_omitted,
            ),
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
    """Attempt partial recovery after pair-consensus failure, or return a refused result.

    Args:
        observations: Full multi-frame corner observations.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Camera distortion coefficients.
        pair_consensus: Pair-consensus graph at the failure point.
        quality: Quality report accumulated before refusal.
        failure_message: Refusal message when partial recovery is disabled.
        requested_marker_ids: Marker IDs originally requested by the caller.
        omitted_markers: Omission reasons already known before recovery.
        connectivity_stage: Internal stage name for newly disconnected markers.
        reference_marker_id: Layout reference marker ID.
        marker_size_m: Default marker edge length in meters.
        marker_sizes_m: Per-marker physical sizes for the full request.
        settings: Calibration thresholds and iteration limits.
        best_effort: Whether best-effort partial output is enabled.
        partial_output: Whether the caller requested partial output on failure.
        anchor_marker_ids: Optional explicit anchor-core marker IDs.

    Returns:
        A partial calibration result when ``partial_output`` and ``best_effort`` are
        both true; otherwise a refused result carrying ``failure_message``.
    """
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
            quality=quality,
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
    """Attempt partial recovery when some markers lack accepted-frame observations.

    Args:
        observations: Full multi-frame corner observations.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Camera distortion coefficients.
        pair_consensus: Pair-consensus graph at the failure point.
        quality: Quality report accumulated before refusal.
        failure_message: Refusal message when partial recovery is disabled.
        requested_marker_ids: Marker IDs originally requested by the caller.
        omitted_markers: Omission reasons already known before recovery.
        markers_in_accepted_frames: Marker IDs with at least one accepted frame.
        missing_after_rejection: Marker IDs with no accepted-frame observations.
        reference_marker_id: Layout reference marker ID.
        marker_size_m: Default marker edge length in meters.
        marker_sizes_m: Per-marker physical sizes for the full request.
        settings: Calibration thresholds and iteration limits.
        best_effort: Whether best-effort partial output is enabled.
        partial_output: Whether the caller requested partial output on failure.
        anchor_marker_ids: Optional explicit anchor-core marker IDs.

    Returns:
        A partial calibration result when ``partial_output`` and ``best_effort`` are
        both true; otherwise a refused result carrying ``failure_message``.
    """
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
            quality=quality,
        )
    return CalibrationResult(None, quality, failure_message)


def maybe_wrap_partial_success(
    result: CalibrationResult,
    *,
    requested_marker_ids: Sequence[int],
    emitted_marker_ids: set[int],
    omitted_markers: Mapping[int, str],
) -> CalibrationResult:
    """Promote a full solve with explicit omissions to a partial outcome when appropriate.

    Args:
        result: Calibration result from the main solve path.
        requested_marker_ids: Marker IDs originally requested by the caller.
        emitted_marker_ids: Marker IDs present in the solved layout.
        omitted_markers: Mapping from omitted marker ID to omission reason.

    Returns:
        ``result`` unchanged when omissions are absent or the solve failed; otherwise
        a ``partial`` wrapper with trimmed quality and omission records.
    """
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
    """Build a successful calibration result tagged with the active policy.

    Args:
        layout: Solved marker layout.
        quality: Final quality report for the accepted solve.
        best_effort: Whether the solve ran under best-effort policy.

    Returns:
        An ``accepted`` ``CalibrationResult`` with ``calibration_policy`` set from
        ``best_effort``.
    """
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
    """Return the first quality-gate failure message, if any.

    Args:
        quality: Aggregate calibration quality metrics.
        settings: Thresholds used to evaluate gates.
        marker_sizes_m: Per-marker physical sizes for pair translation gates.
        expected_ids: Marker IDs that must be connected and covered.

    Returns:
        The first gate failure message, or ``None`` when all gates pass.
    """
    failures = collect_quality_gate_failures(quality, settings, marker_sizes_m, expected_ids)
    return failures[0].message if failures else None


def empty_quality(
    missing_expected_ids: frozenset[int],
    connected_marker_ids: frozenset[int],
    input_frame_count: int = 0,
    rejected_frame_count: int = 0,
    accepted_frame_count: int = 0,
) -> CalibrationQualityReport:
    """Construct a sentinel quality report for early pipeline failures.

    Args:
        missing_expected_ids: Expected marker IDs not connected to the reference.
        connected_marker_ids: Marker IDs connected at the failure point.
        input_frame_count: Total frames supplied to calibration.
        rejected_frame_count: Frames rejected before pose solving.
        accepted_frame_count: Frames accepted for solving.

    Returns:
        A quality report with infinite RMS metrics and zero observation counts.
    """
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
    """Apply quality gates and footprint checks after pose solving.

    Args:
        marker_poses: Solved object-frame marker poses keyed by marker ID.
        quality: Aggregate quality metrics from the solve.
        settings: Thresholds used to evaluate gates.
        marker_sizes_m: Per-marker physical sizes for layout construction.
        expected_ids: Marker IDs that must appear in the final layout.
        reference_marker_id: Layout reference marker ID.
        marker_size_m: Default marker edge length in meters.
        missing_ids: Expected marker IDs not connected to the reference.
        gate_failure: Precomputed gate failure message, if any.
        best_effort: Whether strict-only gate failures may yield a provisional layout.
        anchor_marker_ids: Optional explicit anchor-core marker IDs.

    Returns:
        ``None`` when the caller should build the layout from solved poses.
        A refused or provisional ``CalibrationResult`` when gates, connectivity, or
        footprint coverage block a strict acceptance.

    Notes:
        Under best-effort, strict-only gate failures with full connectivity may yield
        a provisional result instead of refusing.
    """
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
