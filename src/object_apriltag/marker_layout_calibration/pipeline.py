"""Marker layout calibration pipeline orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from object_apriltag.layout import build_marker_layout
from object_apriltag.pose import marker_corner_object_points

from object_apriltag.marker_layout_calibration.continuous_refinement import (
    ContinuousLayoutRefinement,
    LayoutRefinementContext,
    LayoutSolveState,
)
from object_apriltag.marker_layout_calibration import discrete_graph
from object_apriltag.marker_layout_calibration.discrete_graph import (
    collect_assignment_pair_hypotheses,
    connectivity_failure_message,
    connected_marker_ids_from_pairs,
    estimate_frame_candidates,
    estimate_pair_consensus,
    largest_connected_component_from_pairs,
    normalize_observations,
    raw_covisible_pair_counts,
)
from object_apriltag.marker_layout_calibration.finalize import (
    accepted_calibration_result,
    check_quality_gates,
    emit_partial_calibration_result,
    empty_quality,
    finalize_solved_calibration,
    maybe_wrap_partial_success,
    partial_after_missing_accepted_frames_or_refuse,
    partial_from_pair_consensus_or_refuse,
)
from object_apriltag.marker_layout_calibration.input import (
    object_points_by_marker as build_object_points_by_marker,
    parse_expected_marker_ids,
    uniform_marker_sizes,
    validate_camera_inputs,
    validate_marker_size,
    validate_marker_sizes,
    validate_observations,
    validate_settings,
)
from object_apriltag.marker_layout_calibration.reference_selection import (
    select_reference_marker,
)
from object_apriltag.marker_layout_calibration.rotation_consistent_assignment import (
    assign_frames_rotation_consistent,
    build_assignment_rejection_records,
    summarize_assignment_rejection_records,
)
from object_apriltag.marker_layout_calibration.pose_initialization import (
    build_corner_observations,
    initialize_frame_poses,
    initialize_marker_poses,
    markers_in_frame_indices,
    reference_gauge_pose,
    restrict_pair_consensus_to_frames,
    synth_marker_corners,
    synth_pair_observations,
)
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CalibrationSolveDiagnostics,
    PairConsensus,
    connected_marker_ids,
    covisible_frame_count,
    mask_corner_observations_for_frames,
    missing_from_graph,
    timed_solve_stage,
)
from object_apriltag.marker_layout_calibration.solve_quality import (
    build_quality_report,
    footprints_from_poses,
    quality_from_pairs,
)
from object_apriltag.marker_layout_calibration.types import (
    AssignmentRejectionSummary,
    DroppedPairEdge,
    FrameAssignmentRejectionRecord,
    CalibrationResult,
    CalibrationSettings,
    FrameAssignmentRejection,
    FrameObservation,
    RestoredPairEdge,
)


def _auto_select_reference_marker(
    pairs: Sequence[tuple[int, int]],
    expected_ids: Sequence[int],
    keypoint_sources: Mapping[str, tuple[int, str, float]] | None,
) -> int:
    """Select a reference marker from pair connectivity and keypoint coverage."""
    return select_reference_marker(pairs, expected_ids, keypoint_sources or {})


def _reconcile_auto_reference_after_consensus(
    *,
    auto_reference: bool,
    pair_consensus: Mapping[tuple[int, int], PairConsensus],
    expected_ids: Sequence[int],
    reference_marker_id: int,
    keypoint_sources: Mapping[str, tuple[int, str, float]] | None,
    connectivity_stage: str,
    connectivity_failure: str | None,
) -> tuple[int, str | None]:
    """Re-select an automatic reference and refresh connectivity failure state."""
    if not auto_reference or not pair_consensus:
        return reference_marker_id, connectivity_failure
    selected = _auto_select_reference_marker(
        list(pair_consensus),
        expected_ids,
        keypoint_sources,
    )
    if connectivity_failure is None:
        return selected, None
    missing = sorted(missing_from_graph(pair_consensus, list(expected_ids), selected))
    refreshed_failure = (
        connectivity_failure_message(connectivity_stage, selected, missing)
        if missing
        else None
    )
    return selected, refreshed_failure


def calibrate_marker_layout(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    expected_marker_ids: Sequence[int],
    reference_marker_id: int | None,
    marker_size_m: float,
    settings: CalibrationSettings | None = None,
    marker_sizes_m: Mapping[int, float] | None = None,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
    keypoint_sources: Mapping[str, tuple[int, str, float]] | None = None,
    *,
    _subset_resolve: bool = False,
) -> CalibrationResult:
    """Run the rotation-consistent marker-layout calibration pipeline.

    Validates inputs, assigns IPPE candidates with cross-frame rotation agreement,
    builds pair consensus from those assignments, and refines poses via continuous
    bundle adjustment with partial-output recovery.

    Args:
        observations: Per-frame detected marker corner observations.
        camera_matrix: 3×3 camera intrinsics matrix.
        dist_coeffs: Lens distortion coefficients.
        expected_marker_ids: Marker IDs expected in the layout.
        reference_marker_id: Gauge-fixed reference marker ID, or ``None`` to
            auto-select from pair connectivity and keypoint-source coverage.
        marker_size_m: Default physical marker edge length in meters.
        settings: Calibration thresholds and optimizer options; defaults apply if omitted.
        marker_sizes_m: Per-marker edge lengths; uniform default when omitted.
        solve_diagnostics: Optional mutable container for per-stage timing and optimizer stats.
        keypoint_sources: Object-model keypoint derivation map used for automatic
            reference selection when ``reference_marker_id`` is omitted.
        _subset_resolve: Internal flag for connected-subset re-solves that must not
            emit another partial layout.

    Returns:
        ``CalibrationResult`` with layout, quality report, and outcome metadata on success,
        refusal, partial, or provisional paths.
    """
    best_effort = True
    partial_output = not _subset_resolve

    settings = settings or CalibrationSettings()
    settings_failure = validate_settings(settings)
    if settings_failure is not None:
        return CalibrationResult(None, None, settings_failure)

    auto_reference = reference_marker_id is None
    expected_ids, expected_failure = parse_expected_marker_ids(
        expected_marker_ids,
        reference_marker_id,
    )
    if expected_failure is not None:
        return CalibrationResult(None, None, expected_failure)

    requested_marker_ids = list(expected_ids)
    omitted_markers: dict[int, str] = {}

    marker_size_failure = validate_marker_size(marker_size_m)
    if marker_size_failure is not None:
        return CalibrationResult(None, None, marker_size_failure)

    if marker_sizes_m is None:
        marker_sizes_m = uniform_marker_sizes(expected_ids, marker_size_m)
    else:
        sizes_failure = validate_marker_sizes(marker_sizes_m, expected_ids)
        if sizes_failure is not None:
            return CalibrationResult(None, None, sizes_failure)

    camera_matrix, dist_coeffs, camera_failure = validate_camera_inputs(
        camera_matrix,
        dist_coeffs,
    )
    if camera_failure is not None:
        return CalibrationResult(None, None, camera_failure)

    observations_failure = validate_observations(observations, expected_ids)
    if observations_failure is not None:
        return CalibrationResult(None, None, observations_failure)

    object_points_by_marker = build_object_points_by_marker(marker_sizes_m)

    normalized_observations = normalize_observations(observations, expected_ids)
    if not normalized_observations:
        missing = frozenset(expected_ids)
        return CalibrationResult(
            None,
            empty_quality(missing, frozenset(), input_frame_count=len(observations)),
            f"No usable observations for expected marker IDs; missing {sorted(missing)}.",
        )

    observed_ids = {
        marker_id
        for _, markers in normalized_observations
        for marker_id in markers
    }
    never_observed = sorted(set(expected_ids) - observed_ids)
    if never_observed:
        if partial_output and best_effort:
            for marker_id in never_observed:
                omitted_markers[marker_id] = "never_observed"
            expected_ids = [marker_id for marker_id in expected_ids if marker_id in observed_ids]
            if not auto_reference and reference_marker_id not in expected_ids:
                return CalibrationResult(
                    None,
                    empty_quality(
                        frozenset(never_observed),
                        frozenset(observed_ids),
                        input_frame_count=len(normalized_observations),
                    ),
                    f"Reference marker {reference_marker_id} was never observed.",
                    outcome="refused",
                    calibration_policy="best_effort",
                    partial_output=True,
                )
            if len(expected_ids) < 2:
                largest = largest_connected_component_from_pairs(
                    raw_covisible_pair_counts(normalized_observations).keys(),
                    observed_ids,
                )
                if len(largest) >= 2:
                    for marker_id in set(requested_marker_ids) - largest:
                        omitted_markers.setdefault(marker_id, "never_observed")
                    expected_ids = sorted(largest)
                elif len(expected_ids) == 1:
                    pass
        else:
            return CalibrationResult(
                None,
                empty_quality(
                    frozenset(never_observed),
                    frozenset(observed_ids),
                    input_frame_count=len(normalized_observations),
                ),
                f"Expected marker IDs never observed: {never_observed}.",
            )

    with timed_solve_stage(solve_diagnostics, "ippe_candidate_generation"):
        frame_candidates = estimate_frame_candidates(
            normalized_observations,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
        )
    if not frame_candidates:
        return CalibrationResult(
            None,
            empty_quality(
                frozenset(expected_ids),
                frozenset(),
                input_frame_count=len(normalized_observations),
            ),
            "No valid IPPE marker poses found in any frame.",
        )


    raw_pair_counts = raw_covisible_pair_counts(normalized_observations)
    if auto_reference:
        reference_marker_id = _auto_select_reference_marker(
            list(raw_pair_counts.keys()),
            expected_ids,
            keypoint_sources,
        )
    raw_connected = connected_marker_ids_from_pairs(raw_pair_counts.keys(), reference_marker_id)
    raw_missing = sorted(set(expected_ids) - raw_connected)
    if raw_missing:
        if partial_output and best_effort:
            for marker_id in raw_missing:
                omitted_markers[marker_id] = "not_connected_in_raw_observations"
            expected_ids = [marker_id for marker_id in expected_ids if marker_id in raw_connected]
            non_reference = [
                marker_id for marker_id in expected_ids if marker_id != reference_marker_id
            ]
            if not non_reference:
                largest = largest_connected_component_from_pairs(
                    raw_pair_counts.keys(),
                    (
                        marker_id
                        for marker_id in requested_marker_ids
                        if marker_id in observed_ids
                    ),
                )
                if len(largest) >= 2:
                    for marker_id in set(expected_ids) - largest:
                        if marker_id not in omitted_markers:
                            omitted_markers[marker_id] = "not_connected_in_raw_observations"
                    expected_ids = sorted(largest)
                    if auto_reference:
                        reference_marker_id = _auto_select_reference_marker(
                            list(raw_pair_counts.keys()),
                            expected_ids,
                            keypoint_sources,
                        )
        else:
            return CalibrationResult(
                None,
                quality_from_pairs(
                    {},
                    expected_ids,
                    reference_marker_id,
                    frozenset(raw_missing),
                    input_frame_count=len(normalized_observations),
                    rejected_frame_count=0,
                    accepted_frame_count=0,
                    observation_count=0,
                ),
                (
                    f"Expected marker IDs are not connected in raw observations; "
                    f"missing {raw_missing}."
                ),
            )

    restored_pair_edges: list[RestoredPairEdge] = []
    with timed_solve_stage(solve_diagnostics, "rotation_consistent_assignment"):
        rotation_result = assign_frames_rotation_consistent(
            frame_candidates,
            reference_marker_id,
            settings,
        )
    rotation_consistent_assignments = rotation_result.assigned
    assignment_pair_hypotheses = collect_assignment_pair_hypotheses(
        rotation_consistent_assignments,
        frozenset(expected_ids),
    )
    with timed_solve_stage(solve_diagnostics, "initial_pair_consensus"):
        pair_consensus, pair_failure, dropped_pair_edges = estimate_pair_consensus(
            assignment_pair_hypotheses,
            expected_ids,
            reference_marker_id,
            marker_sizes_m,
            settings,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )
    dropped_edges = list(dropped_pair_edges)
    reference_marker_id, pair_failure = _reconcile_auto_reference_after_consensus(
        auto_reference=auto_reference,
        pair_consensus=pair_consensus,
        expected_ids=expected_ids,
        reference_marker_id=reference_marker_id,
        keypoint_sources=keypoint_sources,
        connectivity_stage="initial_consensus",
        connectivity_failure=pair_failure,
    )
    if pair_failure is not None:
        missing = missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
        return partial_from_pair_consensus_or_refuse(
            observations,
            camera_matrix,
            dist_coeffs,
            pair_consensus,
            quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing,
                input_frame_count=len(normalized_observations),
                rejected_frame_count=0,
                accepted_frame_count=0,
                observation_count=0,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            pair_failure,
            requested_marker_ids=requested_marker_ids,
            omitted_markers=omitted_markers,
            connectivity_stage="initial_consensus",
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            solve_diagnostics=solve_diagnostics,
        )

    rejected_frames = rotation_result.rejected_frames
    assignment_rejections = tuple(
        FrameAssignmentRejection(reason="rotation_inconsistent")
        for _ in rejected_frames
    )
    assigned_candidates = rotation_consistent_assignments
    assignment_rejection_records = build_assignment_rejection_records(
        normalized_observations,
        rejected_frames,
        assignment_rejections,
    )
    assignment_rejection_summary = summarize_assignment_rejection_records(assignment_rejection_records)
    input_frame_count = len(normalized_observations)
    rejected_frame_count = len(rejected_frames)
    accepted_frames = frozenset(assigned_candidates)
    accepted_frame_count = len(accepted_frames)
    if accepted_frame_count == 0:
        return CalibrationResult(
            None,
            quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=0,
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            "No frames with assignable IPPE candidates remain after rejecting inconsistent samples.",
        )

    pair_consensus, assignment_support_failure, assignment_drops = restrict_pair_consensus_to_frames(
        pair_consensus,
        accepted_frames,
        expected_ids,
        reference_marker_id,
        settings,
        marker_sizes_m=marker_sizes_m,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    dropped_edges.extend(assignment_drops)
    reference_marker_id, assignment_support_failure = _reconcile_auto_reference_after_consensus(
        auto_reference=auto_reference,
        pair_consensus=pair_consensus,
        expected_ids=expected_ids,
        reference_marker_id=reference_marker_id,
        keypoint_sources=keypoint_sources,
        connectivity_stage="assignment_support",
        connectivity_failure=assignment_support_failure,
    )
    if assignment_support_failure is not None:
        return partial_from_pair_consensus_or_refuse(
            observations,
            camera_matrix,
            dist_coeffs,
            pair_consensus,
            quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            assignment_support_failure,
            requested_marker_ids=requested_marker_ids,
            omitted_markers=omitted_markers,
            connectivity_stage="assignment_support",
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            solve_diagnostics=solve_diagnostics,
        )

    markers_in_accepted_frames = markers_in_frame_indices(normalized_observations, accepted_frames)
    missing_after_rejection = sorted(set(expected_ids) - markers_in_accepted_frames)
    if missing_after_rejection:
        return partial_after_missing_accepted_frames_or_refuse(
            observations,
            camera_matrix,
            dist_coeffs,
            pair_consensus,
            quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                frozenset(missing_after_rejection),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            f"Expected marker IDs have no accepted-frame observations after rejection: {missing_after_rejection}.",
            requested_marker_ids=requested_marker_ids,
            omitted_markers=omitted_markers,
            markers_in_accepted_frames=markers_in_accepted_frames,
            missing_after_rejection=missing_after_rejection,
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            solve_diagnostics=solve_diagnostics,
        )

    ref_rotation, ref_translation = reference_gauge_pose(marker_sizes_m[reference_marker_id])
    marker_poses = initialize_marker_poses(
        reference_marker_id,
        ref_rotation,
        ref_translation,
        expected_ids,
        pair_consensus,
    )
    frame_poses = initialize_frame_poses(
        assigned_candidates,
        marker_poses,
        len(normalized_observations),
    )

    corner_observations = build_corner_observations(normalized_observations, expected_ids)
    inlier_mask = mask_corner_observations_for_frames(corner_observations, accepted_frames)
    non_reference_ids = [marker_id for marker_id in expected_ids if marker_id != reference_marker_id]

    refinement_context = build_layout_refinement_context(
        reference_marker_id=reference_marker_id,
        non_reference_ids=non_reference_ids,
        expected_ids=expected_ids,
        accepted_frames=accepted_frames,
        object_points_by_marker=object_points_by_marker,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        settings=settings,
        marker_sizes_m=marker_sizes_m,
        marker_size_m=marker_size_m,
        restored_pair_edges=restored_pair_edges,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        assignment_rejection_summary=assignment_rejection_summary,
        assignment_rejection_records=assignment_rejection_records,
        dropped_edges=dropped_edges,
        solve_diagnostics=solve_diagnostics,
    )

    return _run_continuous_refinement(
        refinement_context=refinement_context,
        corner_observations=corner_observations,
        marker_poses=marker_poses,
        frame_poses=frame_poses,
        inlier_mask=inlier_mask,
        pair_consensus=pair_consensus,
        dropped_edges=dropped_edges,
        observations=observations,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        requested_marker_ids=requested_marker_ids,
        omitted_markers=omitted_markers,
        reference_marker_id=reference_marker_id,
        marker_size_m=marker_size_m,
        marker_sizes_m=marker_sizes_m,
        settings=settings,
        partial_output=partial_output,
        expected_ids=expected_ids,
        object_points_by_marker=object_points_by_marker,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
    )



def _run_continuous_refinement(
    *,
    refinement_context: LayoutRefinementContext,
    corner_observations,
    marker_poses,
    frame_poses,
    inlier_mask,
    pair_consensus,
    dropped_edges: list[DroppedPairEdge],
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    requested_marker_ids: list[int],
    omitted_markers: dict[int, str],
    reference_marker_id: int,
    marker_size_m: float,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    partial_output: bool,
    expected_ids: list[int],
    object_points_by_marker: dict[int, np.ndarray],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
) -> CalibrationResult:
    """Run continuous refinement and finalize into a ``CalibrationResult``.

    Invokes ``ContinuousLayoutRefinement``, builds the quality report, applies
    quality gates, and emits full, partial, or refused outcomes per policy flags.

    Args:
        refinement_context: Settings, diagnostics, and hooks for continuous refinement.
        corner_observations: Flattened corner observations for bundle adjustment.
        marker_poses: Seed marker poses in the object frame.
        frame_poses: Seed per-frame camera poses.
        inlier_mask: Boolean mask selecting active corner observations.
        pair_consensus: Pair-edge consensus after discrete graph solving.
        dropped_edges: Mutable list of dropped pair edges; refinement pruning drops are appended.
        observations: Original frame observations for partial-output emission.
        camera_matrix: 3×3 camera intrinsics matrix.
        dist_coeffs: Lens distortion coefficients.
        requested_marker_ids: Marker IDs requested before best-effort omissions.
        omitted_markers: Markers already omitted with reason codes.
        reference_marker_id: Gauge-fixed reference marker ID.
        marker_size_m: Default physical marker edge length in meters.
        marker_sizes_m: Per-marker edge lengths.
        settings: Calibration thresholds and optimizer options.
        partial_output: Whether to emit partial layouts for markers disconnected after pruning.
        expected_ids: Marker IDs still expected after upstream omissions.
        object_points_by_marker: Object-frame corner coordinates per marker.
        input_frame_count: Total input frames before rejection.
        rejected_frame_count: Frames rejected during assignment.
        accepted_frame_count: Frames accepted into bundle adjustment.
    Returns:
        Final ``CalibrationResult`` after refinement, quality gating, and optional partial wrapping.

    Notes:
        Pruning drops from refinement are merged into ``dropped_edges``.
    """
    refinement_outcome = ContinuousLayoutRefinement(refinement_context).run(
        LayoutSolveState(
            corner_observations=corner_observations,
            marker_poses=marker_poses,
            frame_poses=frame_poses,
            inlier_mask=inlier_mask,
            pair_consensus=pair_consensus,
        )
    )
    marker_poses = refinement_outcome.state.marker_poses
    frame_poses = refinement_outcome.state.frame_poses
    inlier_mask = refinement_outcome.state.inlier_mask
    pair_consensus = refinement_outcome.state.pair_consensus
    dropped_edges.extend(refinement_outcome.pruning_drops)
    if refinement_outcome.early_result is not None:
        return refinement_outcome.early_result

    connected_ids = connected_marker_ids(pair_consensus, reference_marker_id)
    missing_ids = frozenset(set(expected_ids) - connected_ids)
    final_accepted_frame_count = covisible_frame_count(corner_observations, inlier_mask)
    quality = build_quality_report(
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
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        assignment_rejections=refinement_context.assignment_rejection_summary,
        assignment_rejection_records=refinement_context.assignment_rejection_records,
        dropped_pair_edges=tuple(dropped_edges),
        restored_pair_edges=refinement_context.restored_pair_edges,
    )
    gate_failure = check_quality_gates(quality, settings, marker_sizes_m, expected_ids)
    if missing_ids and partial_output:
        merged = dict(omitted_markers)
        for marker_id in missing_ids:
            merged.setdefault(marker_id, "not_connected_after_pruning")
        return emit_partial_calibration_result(
            observations,
            camera_matrix,
            dist_coeffs,
            requested_marker_ids=requested_marker_ids,
            connected_ids=set(connected_ids),
            omitted=merged,
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            quality=quality,
            solve_diagnostics=refinement_context.solve_diagnostics,
        )
    finalized = finalize_solved_calibration(
        marker_poses,
        quality,
        settings,
        marker_sizes_m,
        expected_ids,
        reference_marker_id,
        marker_size_m,
        missing_ids,
        gate_failure=gate_failure,
    )
    if finalized is not None:
        return finalized
    emitted_footprints = footprints_from_poses(marker_poses, marker_sizes_m)
    emitted_sizes = {marker_id: marker_sizes_m[marker_id] for marker_id in emitted_footprints}
    layout = build_marker_layout(
        reference_marker_id=reference_marker_id,
        marker_size_m=marker_size_m,
        footprints=emitted_footprints,
        marker_sizes_m=emitted_sizes,
    )
    return maybe_wrap_partial_success(
        accepted_calibration_result(layout, quality),
        requested_marker_ids=requested_marker_ids,
        emitted_marker_ids=set(expected_ids),
        omitted_markers=omitted_markers,
    )


def build_layout_refinement_context(
    *,
    reference_marker_id: int,
    non_reference_ids: list[int],
    expected_ids: list[int],
    accepted_frames: frozenset[int],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    marker_size_m: float,
    restored_pair_edges: list[RestoredPairEdge] | None,
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    assignment_rejection_summary: AssignmentRejectionSummary | None,
    assignment_rejection_records: tuple[FrameAssignmentRejectionRecord, ...] | None,
    dropped_edges: list[DroppedPairEdge],
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
) -> LayoutRefinementContext:
    """Assemble ``LayoutRefinementContext`` for the continuous refinement stage."""
    return LayoutRefinementContext(
        reference_marker_id=reference_marker_id,
        non_reference_ids=non_reference_ids,
        expected_ids=expected_ids,
        accepted_frames=accepted_frames,
        object_points_by_marker=object_points_by_marker,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        settings=settings,
        marker_sizes_m=marker_sizes_m,
        marker_size_m=marker_size_m,
        restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        assignment_rejection_summary=assignment_rejection_summary,
        assignment_rejection_records=assignment_rejection_records,
        dropped_edges=dropped_edges,
        restore_weak_connectivity=discrete_graph.maybe_restore_weak_connectivity,
        refresh_pair_consensus_after_initial_ba=None,
        solve_diagnostics=solve_diagnostics,
    )




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

    ref_rotation, ref_translation = reference_gauge_pose(marker_size)
    pair_poses = {
        0: (ref_rotation, ref_translation),
        1: (ref_rotation, ref_translation + np.array([0.12, 0.0, -0.05])),
    }

    mostly_good = calibrate_marker_layout(
        synth_pair_observations(
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
    assert mostly_good.quality.rejected_frame_count == 0
    assert mostly_good.quality.accepted_frame_count == 20

    all_bad = calibrate_marker_layout(
        synth_pair_observations(
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
            markers = synth_marker_corners(
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
    """Assert validation failures for malformed or duplicate input observations."""
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
