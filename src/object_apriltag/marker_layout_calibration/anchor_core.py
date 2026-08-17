"""Anchor Core bootstrap and hierarchical marker expansion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from object_apriltag.marker_layout_calibration.assignment import assign_ippe_candidates
from object_apriltag.marker_layout_calibration.continuous_refinement import run_bundle_adjustment
from object_apriltag.marker_layout_calibration.discrete_graph import (
    best_pair_consensus,
    collect_pair_hypotheses,
    estimate_pair_consensus,
    transform_high_in_low,
)
from object_apriltag.marker_layout_calibration.pose_initialization import (
    build_corner_observations,
    initialize_frame_poses,
    initialize_marker_poses,
    reference_gauge_pose,
)
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CalibrationSolveDiagnostics,
    MarkerCandidate,
    MarkerPair,
    PairConsensus,
    average_poses,
    mask_corner_observations_for_frames,
    timed_solve_stage,
)
from object_apriltag.marker_layout_calibration.solve_quality import edge_diagnostics, pair_translation_gate
from object_apriltag.marker_layout_calibration.types import (
    AnchorCoreBootstrapDiagnostics,
    AnchorCoreDiagnostics,
    CalibrationSettings,
    DroppedPairEdge,
    FrameAssignmentRejection,
    FrameFallbackAssignment,
    MarkerExpansionRecord,
    RestoredPairEdge,
)

@dataclass
class MarkerPoseHypothesis:
    """Object-frame marker pose from a frame layout pose and one IPPE candidate.

    Attributes:
        rotation: Marker rotation in object frame, shape ``(3, 3)``.
        translation: Marker translation in object frame, shape ``(3,)``.
        frame_index: Observation frame index supporting this hypothesis.
        candidate: Camera-frame IPPE candidate used to derive the pose.
    """

    rotation: np.ndarray
    translation: np.ndarray
    frame_index: int
    candidate: MarkerCandidate


def filter_pair_hypotheses_to_markers(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    marker_ids: frozenset[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    """Restrict pair hypotheses to edges inside a marker subset.

    Args:
        pair_hypotheses: Per-pair relative-transform hypotheses keyed by low-to-high
            marker ID.
        marker_ids: Marker IDs that must both be present on an edge.

    Returns:
        Filtered copy of ``pair_hypotheses`` containing only edges whose endpoints
        lie in ``marker_ids``.
    """
    return {
        pair: hypotheses
        for pair, hypotheses in pair_hypotheses.items()
        if pair[0] in marker_ids and pair[1] in marker_ids
    }


def relative_pose_high_in_low(
    low_rotation: np.ndarray,
    low_translation: np.ndarray,
    high_rotation: np.ndarray,
    high_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express the high marker pose in the low marker frame.

    Args:
        low_rotation: Low-marker rotation in the shared parent frame, shape ``(3, 3)``.
        low_translation: Low-marker translation in the shared parent frame, shape ``(3,)``.
        high_rotation: High-marker rotation in the shared parent frame, shape ``(3, 3)``.
        high_translation: High-marker translation in the shared parent frame, shape ``(3,)``.

    Returns:
        Tuple ``(rotation, translation)`` mapping points from the low marker frame to
        the high marker frame via ``p_high = rotation @ p_low + translation``.

    Notes:
        Parent frame is object frame for marker poses or camera frame for IPPE
        candidates; both inputs must use the same parent.
    """
    rotation = low_rotation.T @ high_rotation
    translation = low_rotation.T @ (high_translation - low_translation)
    return rotation, translation


def pair_consensus_from_assignment_hypotheses(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[MarkerPair, PairConsensus]:
    """Build pair consensus from assignment hypotheses with pose fallback.

    Args:
        pair_hypotheses: Per-pair relative-transform hypotheses from assigned IPPE
            candidates.
        settings: Calibration gates for inlier count and pair RMS.
        marker_sizes_m: Physical edge lengths keyed by marker ID.
        marker_poses: Optional solved object-frame marker poses used when robust
            consensus fails RMS gates.

    Returns:
        Pair consensus edges that pass gates, or pose-derived edges when robust
        consensus is rejected but marker poses are available.

    Notes:
        Fallback edges reuse inlier frame sets from hypotheses but replace
        rotation and translation with poses propagated through
        ``relative_pose_high_in_low``.
    """
    rotation_gate = settings.pair_rotation_rms_gate_deg
    consensus: dict[MarkerPair, PairConsensus] = {}
    for pair, hypotheses in pair_hypotheses.items():
        translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
        edge = best_pair_consensus(pair, hypotheses, translation_gate, rotation_gate)
        if edge is not None and len(edge.inlier_frames) >= settings.min_inliers_per_edge:
            diagnostics = edge_diagnostics(pair, edge)
            if diagnostics.translation_rms_m <= translation_gate and diagnostics.rotation_rms_deg <= rotation_gate:
                consensus[pair] = edge
                continue
        if marker_poses is None:
            continue
        marker_low, marker_high = pair
        if marker_low not in marker_poses or marker_high not in marker_poses:
            continue
        inlier_frames = tuple(sorted({frame_index for _, _, frame_index in hypotheses}))
        if len(inlier_frames) < settings.min_inliers_per_edge:
            continue
        low_rotation, low_translation = marker_poses[marker_low]
        high_rotation, high_translation = marker_poses[marker_high]
        rotation_ba, translation_ba = relative_pose_high_in_low(
            low_rotation,
            low_translation,
            high_rotation,
            high_translation,
        )
        hypotheses_by_frame = {
            frame_index: (rotation, translation)
            for rotation, translation, frame_index in hypotheses
        }
        consensus[pair] = PairConsensus(
            marker_a=marker_low,
            marker_b=marker_high,
            rotation_ba=rotation_ba,
            translation_ba=translation_ba,
            inlier_frames=inlier_frames,
            inlier_hypotheses={
                frame_index: hypotheses_by_frame[frame_index]
                for frame_index in inlier_frames
                if frame_index in hypotheses_by_frame
            },
        )
    return consensus


def collect_assignment_pair_hypotheses(
    assigned_candidates: dict[int, dict[int, MarkerCandidate]],
    marker_ids: frozenset[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    """Collect per-frame pair hypotheses from frozen IPPE assignments.

    Args:
        assigned_candidates: Per-frame marker-to-IPPE-candidate assignments.
        marker_ids: Marker IDs to include when forming co-visible pairs.

    Returns:
        Per-pair lists of ``(rotation_ba, translation_ba, frame_index)`` relative
        transforms from low to high marker, one entry per co-visible frame.
    """
    hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    for frame_index, assignment in assigned_candidates.items():
        visible = sorted(marker_id for marker_id in assignment if marker_id in marker_ids)
        for index_a, marker_low in enumerate(visible):
            for marker_high in visible[index_a + 1 :]:
                pair = (marker_low, marker_high)
                rotation_ba, translation_ba = transform_high_in_low(
                    assignment[marker_low],
                    assignment[marker_high],
                )
                hypotheses.setdefault(pair, []).append(
                    (rotation_ba, translation_ba, frame_index)
                )
    return hypotheses


def marker_pose_from_frame_and_candidate(
    layout_rotation: np.ndarray,
    layout_translation: np.ndarray,
    candidate: MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a camera-frame IPPE candidate into the object frame.

    Args:
        layout_rotation: Object-to-camera layout rotation, shape ``(3, 3)``.
        layout_translation: Object-to-camera layout translation, shape ``(3,)``.
        candidate: Camera-frame IPPE pose for the marker.

    Returns:
        Tuple ``(rotation, translation)`` of the marker pose in the object frame.

    Notes:
        Layout pose maps object points to the camera; this applies the inverse
        transform to the candidate camera-frame marker pose.
    """
    marker_rotation = layout_rotation.T @ candidate.rotation
    marker_translation = layout_rotation.T @ (candidate.tvec.reshape(3) - layout_translation)
    return marker_rotation, marker_translation


def frame_pose_from_solved_assignment(
    assignment: dict[int, MarkerCandidate],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    solved_ids: frozenset[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Average layout pose from assignment candidates on solved markers.

    Args:
        assignment: Per-marker IPPE candidates for one frame.
        marker_poses: Object-frame marker poses for solved markers.
        solved_ids: Marker IDs treated as already solved in the expansion graph.

    Returns:
        Mean object-to-camera layout pose, or ``None`` when no solved markers
        appear in the assignment.
    """
    estimates: list[tuple[np.ndarray, np.ndarray]] = []
    for marker_id, candidate in assignment.items():
        if marker_id not in solved_ids or marker_id not in marker_poses:
            continue
        marker_rotation, marker_translation = marker_poses[marker_id]
        layout_rotation = candidate.rotation @ marker_rotation.T
        layout_translation = candidate.tvec.reshape(3) - layout_rotation @ marker_translation
        estimates.append((layout_rotation, layout_translation))
    if len(estimates) < 1:
        return None
    return average_poses(estimates)


def frame_pose_from_known_marker_candidates(
    candidates: dict[int, list[MarkerCandidate]],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    solved_ids: frozenset[int],
) -> tuple[tuple[np.ndarray, np.ndarray], dict[int, MarkerCandidate]] | None:
    """Infer layout pose from lowest-RMS IPPE picks on solved markers.

    Args:
        candidates: Per-marker IPPE candidate lists for one frame.
        marker_poses: Object-frame marker poses for solved markers.
        solved_ids: Marker IDs treated as already solved in the expansion graph.

    Returns:
        Tuple of mean layout pose and the selected low-RMS candidates per solved
        marker, or ``None`` when no solved markers have candidates in the frame.
    """
    selected: dict[int, MarkerCandidate] = {}
    estimates: list[tuple[np.ndarray, np.ndarray]] = []
    for marker_id in sorted(candidates):
        if marker_id not in solved_ids or marker_id not in marker_poses:
            continue
        marker_rotation, marker_translation = marker_poses[marker_id]
        best_candidate = min(candidates[marker_id], key=lambda candidate: candidate.reprojection_rms_px)
        layout_rotation = best_candidate.rotation @ marker_rotation.T
        layout_translation = best_candidate.tvec.reshape(3) - layout_rotation @ marker_translation
        selected[marker_id] = best_candidate
        estimates.append((layout_rotation, layout_translation))
    if not estimates:
        return None
    return average_poses(estimates), selected


def expand_markers_hierarchically(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
    assigned_candidates: dict[int, dict[int, MarkerCandidate]],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    solved_ids: frozenset[int],
    expected_ids: list[int],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, dict[int, MarkerCandidate]],
    tuple[MarkerExpansionRecord, ...],
    frozenset[int],
]:
    """Grow the solved marker set via hierarchical self-pair consensus.

    Args:
        frame_candidates: Per-frame IPPE candidate pools.
        assigned_candidates: Current per-frame IPPE assignments.
        marker_poses: Object-frame marker poses for markers already solved.
        solved_ids: Marker IDs currently treated as solved (typically anchor core).
        expected_ids: Full set of marker IDs targeted by calibration.
        settings: Calibration gates for pair support and RMS.
        marker_sizes_m: Physical edge lengths keyed by marker ID.

    Returns:
        Tuple of updated marker poses, updated assignments, expansion audit records,
        and marker IDs still unresolved after hierarchical expansion.

    Notes:
        Each unsolved marker is tested with a synthetic self-pair
        ``(marker_id, marker_id)`` built from layout poses inferred from solved
        neighbors; accepted markers are added iteratively until no progress.
    """
    rotation_gate = settings.pair_rotation_rms_gate_deg
    poses = dict(marker_poses)
    assignments = {
        frame_index: dict(assignment)
        for frame_index, assignment in assigned_candidates.items()
    }
    expansion_records: list[MarkerExpansionRecord] = []
    solved = set(solved_ids)
    expected_set = set(expected_ids)

    while True:
        unsolved = sorted(
            marker_id
            for marker_id in (expected_set - solved)
            if not any(
                record.marker_id == marker_id and record.status == "rejected"
                for record in expansion_records
            )
        )
        if not unsolved:
            break
        progress = False
        candidates_by_frame = {frame_index: candidates for frame_index, candidates in frame_candidates}
        for marker_id in unsolved:
            hypotheses: list[MarkerPoseHypothesis] = []
            for frame_index, candidates in candidates_by_frame.items():
                if marker_id not in candidates:
                    continue
                frame_pose_result = frame_pose_from_known_marker_candidates(
                    candidates,
                    poses,
                    frozenset(solved),
                )
                if frame_pose_result is None:
                    continue
                layout_rotation, layout_translation = frame_pose_result[0]
                selected_solved = frame_pose_result[1]
                assignments.setdefault(frame_index, {}).update(selected_solved)
                for candidate in candidates[marker_id]:
                    rotation, translation = marker_pose_from_frame_and_candidate(
                        layout_rotation,
                        layout_translation,
                        candidate,
                    )
                    hypotheses.append(
                        MarkerPoseHypothesis(
                            rotation=rotation,
                            translation=translation,
                            frame_index=frame_index,
                            candidate=candidate,
                        )
                    )
            if not hypotheses:
                continue
            translation_gate = pair_translation_gate(
                settings,
                marker_sizes_m,
                (marker_id, marker_id),
            )
            relative_hypotheses = [
                (hypothesis.rotation, hypothesis.translation, hypothesis.frame_index)
                for hypothesis in hypotheses
            ]
            edge = best_pair_consensus(
                (marker_id, marker_id),
                relative_hypotheses,
                translation_gate,
                rotation_gate,
            )
            if edge is None or len(edge.inlier_frames) < settings.min_inliers_per_edge:
                expansion_records.append(
                    MarkerExpansionRecord(
                        marker_id=marker_id,
                        status="rejected",
                        support_frames=len(edge.inlier_frames) if edge is not None else 0,
                        reason="insufficient_support",
                    )
                )
                continue
            diagnostics = edge_diagnostics((marker_id, marker_id), edge)
            if diagnostics.translation_rms_m > translation_gate:
                expansion_records.append(
                    MarkerExpansionRecord(
                        marker_id=marker_id,
                        status="rejected",
                        support_frames=diagnostics.inlier_count,
                        reason="translation_rms_gate",
                    )
                )
                continue
            if diagnostics.rotation_rms_deg > rotation_gate:
                expansion_records.append(
                    MarkerExpansionRecord(
                        marker_id=marker_id,
                        status="rejected",
                        support_frames=diagnostics.inlier_count,
                        reason="rotation_rms_gate",
                    )
                )
                continue
            poses[marker_id] = (edge.rotation_ba.copy(), edge.translation_ba.copy())
            solved.add(marker_id)
            progress = True
            expansion_records.append(
                MarkerExpansionRecord(
                    marker_id=marker_id,
                    status="accepted",
                    support_frames=diagnostics.inlier_count,
                )
            )
            hypotheses_by_frame = {
                hypothesis.frame_index: hypothesis for hypothesis in hypotheses
            }
            for frame_index in edge.inlier_frames:
                hypothesis = hypotheses_by_frame.get(frame_index)
                if hypothesis is None:
                    continue
                assignments.setdefault(frame_index, {})[marker_id] = hypothesis.candidate
        if not progress:
            break

    unresolved = frozenset(expected_set - solved)
    for marker_id in sorted(unresolved):
        if not any(record.marker_id == marker_id for record in expansion_records):
            expansion_records.append(
                MarkerExpansionRecord(
                    marker_id=marker_id,
                    status="rejected",
                    support_frames=0,
                    reason="unreachable_from_anchor_core",
                )
            )
    return poses, assignments, tuple(expansion_records), unresolved


def freeze_assigned_frame_candidates(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
    assigned_candidates: dict[int, dict[int, MarkerCandidate]],
) -> list[tuple[int, dict[int, list[MarkerCandidate]]]]:
    """Freeze IPPE assignments to a single candidate per assigned marker.

    Args:
        frame_candidates: Per-frame IPPE candidate pools.
        assigned_candidates: Per-frame marker-to-candidate assignments to freeze.

    Returns:
        Frames whose frozen candidate dict retains at least two markers; frames
        with fewer than two assigned markers are dropped.
    """
    frozen: list[tuple[int, dict[int, list[MarkerCandidate]]]] = []
    for frame_index, candidates in frame_candidates:
        assignment = assigned_candidates.get(frame_index)
        if assignment is None:
            continue
        frozen_candidates = {
            marker_id: [assignment[marker_id]]
            for marker_id in assignment
            if marker_id in candidates
        }
        if len(frozen_candidates) >= 2:
            frozen.append((frame_index, frozen_candidates))
    return frozen


def assign_and_initialize_anchor_core(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    normalized_observations: list[tuple[str | int, dict[int, np.ndarray]]],
    expected_ids: list[int],
    anchor_ids: tuple[int, ...],
    reference_marker_id: int,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    stop_after_expansion: bool = False,
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
) -> tuple[
    dict[int, dict[int, MarkerCandidate]] | None,
    tuple[int, ...],
    tuple[FrameAssignmentRejection, ...],
    tuple[FrameFallbackAssignment, ...],
    dict[MarkerPair, PairConsensus] | None,
    dict[int, tuple[np.ndarray, np.ndarray]] | None,
    list[DroppedPairEdge],
    AnchorCoreDiagnostics,
    str | None,
]:
    """Run anchor-core bootstrap, mini-BA, expansion, and full-graph consensus.

    Args:
        frame_candidates: Per-frame IPPE candidate pools.
        pair_hypotheses: Initial per-pair relative-transform hypotheses.
        normalized_observations: Parsed corner observations indexed by frame.
        expected_ids: Full set of marker IDs targeted by calibration.
        anchor_ids: Marker IDs forming the anchor core subgraph.
        reference_marker_id: Gauge reference marker fixed during mini bundle adjustment.
        marker_sizes_m: Physical edge lengths keyed by marker ID.
        settings: Calibration gates and optimizer settings.
        object_points_by_marker: Object-frame corner coordinates per marker.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.
        stop_after_expansion: When true, return after hierarchical expansion without
            full-graph re-assignment.
        best_effort: Allow weak-edge restoration and fallback IPPE assignment.
        restored_pair_edges: Optional list extended with weak-edge restoration records.
        solve_diagnostics: Optional collector for stage timings and optimizer runs.

    Returns:
        Tuple of assigned candidates (or ``None`` on early bootstrap failure),
        rejected frame indices, assignment rejections, fallback assignments,
        pair consensus (or ``None`` when expansion leaves unresolved markers),
        marker poses (or ``None`` before pose initialization succeeds),
        dropped pair-edge records, anchor-core diagnostics, and an optional
        human-readable failure message.

    Notes:
        Mini bundle adjustment runs on anchor markers only; expansion propagates
        poses along self-pair consensus before optional full-graph IPPE search
        and consensus on frozen assignments.
    """
    anchor_set = frozenset(anchor_ids)
    dropped_edges: list[DroppedPairEdge] = []
    anchor_hypotheses = filter_pair_hypotheses_to_markers(pair_hypotheses, anchor_set)
    with timed_solve_stage(solve_diagnostics, "initial_pair_consensus"):
        anchor_consensus, anchor_pair_failure, anchor_drops = estimate_pair_consensus(
            anchor_hypotheses,
            expected_ids,
            reference_marker_id,
            marker_sizes_m,
            settings,
            connectivity_ids=anchor_ids,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )
    dropped_edges.extend(anchor_drops)
    bootstrap = AnchorCoreBootstrapDiagnostics(
        status="failed",
        frames_considered=len(frame_candidates),
        frames_accepted=0,
        failure_reason=anchor_pair_failure,
    )
    anchor_core = AnchorCoreDiagnostics(
        mode="anchor_core",
        configured_anchor_ids=anchor_ids,
        bootstrap=bootstrap,
        expansion=(),
        final_solved_ids=frozenset(),
        unresolved_ids=frozenset(expected_ids),
    )
    if anchor_pair_failure is not None:
        return None, (), (), (), None, None, dropped_edges, anchor_core, (
            f"Anchor core bootstrap failed: {anchor_pair_failure}"
        )

    assigned_candidates, rejected_frames, assignment_rejections, bootstrap_fallback = (
        assign_ippe_candidates(
            frame_candidates,
            anchor_consensus,
            settings,
            marker_sizes_m,
            search_marker_ids=anchor_set,
            best_effort=best_effort,
            solve_diagnostics=solve_diagnostics,
        )
    )
    accepted_frames = frozenset(assigned_candidates)
    bootstrap = AnchorCoreBootstrapDiagnostics(
        status="failed" if not accepted_frames else "ok",
        frames_considered=len(frame_candidates),
        frames_accepted=len(accepted_frames),
        failure_reason=None if accepted_frames else "no_anchor_assignable_frames",
    )
    anchor_core = AnchorCoreDiagnostics(
        mode="anchor_core",
        configured_anchor_ids=anchor_ids,
        bootstrap=bootstrap,
        expansion=(),
        final_solved_ids=frozenset(anchor_set) if accepted_frames else frozenset(),
        unresolved_ids=frozenset(set(expected_ids) - anchor_set),
    )
    if not accepted_frames:
        return (
            assigned_candidates,
            rejected_frames,
            assignment_rejections,
            bootstrap_fallback,
            anchor_consensus,
            None,
            dropped_edges,
            anchor_core,
            "No frames with assignable anchor IPPE candidates remain after bootstrap.",
        )

    ref_rotation, ref_translation = reference_gauge_pose(marker_sizes_m[reference_marker_id])
    marker_poses = initialize_marker_poses(
        reference_marker_id,
        ref_rotation,
        ref_translation,
        list(anchor_ids),
        anchor_consensus,
    )
    frame_poses = initialize_frame_poses(
        assigned_candidates,
        marker_poses,
        len(normalized_observations),
    )
    anchor_corner_observations = build_corner_observations(
        normalized_observations,
        list(anchor_ids),
    )
    inlier_mask = mask_corner_observations_for_frames(anchor_corner_observations, accepted_frames)
    non_reference_anchors = [
        marker_id for marker_id in anchor_ids if marker_id != reference_marker_id
    ]

    marker_poses, frame_poses, inlier_mask, ba_failure = run_bundle_adjustment(
        anchor_corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_anchors,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        settings,
        solve_diagnostics=solve_diagnostics,
        stage_name="initial_bundle_adjustment",
    )
    if ba_failure is not None:
        bootstrap = AnchorCoreBootstrapDiagnostics(
            status="failed",
            frames_considered=len(frame_candidates),
            frames_accepted=len(accepted_frames),
            failure_reason=ba_failure,
        )
        anchor_core = AnchorCoreDiagnostics(
            mode="anchor_core",
            configured_anchor_ids=anchor_ids,
            bootstrap=bootstrap,
            expansion=(),
            final_solved_ids=frozenset(anchor_set),
            unresolved_ids=frozenset(set(expected_ids) - anchor_set),
        )
        return (
            assigned_candidates,
            rejected_frames,
            assignment_rejections,
            bootstrap_fallback,
            anchor_consensus,
            marker_poses,
            dropped_edges,
            anchor_core,
            f"Anchor core mini bundle adjustment failed: {ba_failure}",
        )

    marker_poses, assignments, expansion_records, unresolved = expand_markers_hierarchically(
        frame_candidates,
        assigned_candidates,
        marker_poses,
        anchor_set,
        expected_ids,
        settings,
        marker_sizes_m,
    )
    if unresolved:
        anchor_core = AnchorCoreDiagnostics(
            mode="anchor_core",
            configured_anchor_ids=anchor_ids,
            bootstrap=AnchorCoreBootstrapDiagnostics(
                status="ok",
                frames_considered=len(frame_candidates),
                frames_accepted=len(accepted_frames),
            ),
            expansion=expansion_records,
            final_solved_ids=frozenset(marker_poses),
            unresolved_ids=unresolved,
        )
        return (
            assignments,
            rejected_frames,
            assignment_rejections,
            bootstrap_fallback,
            None,
            marker_poses,
            dropped_edges,
            anchor_core,
            f"Anchor core expansion could not solve all expected markers; missing {sorted(unresolved)}.",
        )

    if stop_after_expansion:
        expected_set = frozenset(expected_ids)
        pair_consensus = pair_consensus_from_assignment_hypotheses(
            collect_assignment_pair_hypotheses(assignments, expected_set),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )
        anchor_core = AnchorCoreDiagnostics(
            mode="anchor_core",
            configured_anchor_ids=anchor_ids,
            bootstrap=AnchorCoreBootstrapDiagnostics(
                status="ok",
                frames_considered=len(frame_candidates),
                frames_accepted=len(accepted_frames),
            ),
            expansion=expansion_records,
            final_solved_ids=frozenset(marker_poses),
            unresolved_ids=frozenset(),
            stopped_after_expansion=True,
        )
        return (
            assignments,
            rejected_frames,
            assignment_rejections,
            bootstrap_fallback,
            pair_consensus,
            marker_poses,
            dropped_edges,
            anchor_core,
            None,
        )

    expected_set = frozenset(expected_ids)
    seed_frames = freeze_assigned_frame_candidates(frame_candidates, assignments)
    seed_hypotheses = collect_pair_hypotheses(seed_frames, expected_ids)
    with timed_solve_stage(solve_diagnostics, "initial_pair_consensus"):
        seed_consensus, seed_failure, seed_drops = estimate_pair_consensus(
            seed_hypotheses,
            expected_ids,
            reference_marker_id,
            marker_sizes_m,
            settings,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )
    dropped_edges.extend(seed_drops)
    if seed_failure is not None:
        seed_consensus = pair_consensus_from_assignment_hypotheses(
            collect_assignment_pair_hypotheses(assignments, expected_set),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )

    assignments, rejected_frames, assignment_rejections, fallback_assignments = (
        assign_ippe_candidates(
            frame_candidates,
            seed_consensus,
            settings,
            marker_sizes_m,
            search_marker_ids=expected_set,
            best_effort=best_effort,
            solve_diagnostics=solve_diagnostics,
        )
    )

    frozen_frames = freeze_assigned_frame_candidates(frame_candidates, assignments)
    frozen_hypotheses = collect_pair_hypotheses(frozen_frames, expected_ids)
    with timed_solve_stage(solve_diagnostics, "initial_pair_consensus"):
        pair_consensus, pair_failure, post_drops = estimate_pair_consensus(
            frozen_hypotheses,
            expected_ids,
            reference_marker_id,
            marker_sizes_m,
            settings,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )
    dropped_edges.extend(post_drops)
    if pair_failure is not None:
        pair_consensus = pair_consensus_from_assignment_hypotheses(
            collect_assignment_pair_hypotheses(assignments, expected_set),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )
        connected = connected_marker_ids(pair_consensus, reference_marker_id)
        missing = sorted(set(expected_ids) - connected)
        if missing:
            anchor_core = AnchorCoreDiagnostics(
                mode="anchor_core",
                configured_anchor_ids=anchor_ids,
                bootstrap=AnchorCoreBootstrapDiagnostics(
                    status="ok",
                    frames_considered=len(frame_candidates),
                    frames_accepted=len(accepted_frames),
                ),
                expansion=expansion_records,
                final_solved_ids=frozenset(marker_poses),
                unresolved_ids=frozenset(missing),
            )
            return (
                assignments,
                rejected_frames,
                assignment_rejections,
                fallback_assignments,
                pair_consensus,
                marker_poses,
                dropped_edges,
                anchor_core,
                (
                    f"Expected marker IDs are not connected after anchor-core expansion; "
                    f"missing {missing}."
                ),
            )

    anchor_core = AnchorCoreDiagnostics(
        mode="anchor_core",
        configured_anchor_ids=anchor_ids,
        bootstrap=AnchorCoreBootstrapDiagnostics(
            status="ok",
            frames_considered=len(frame_candidates),
            frames_accepted=len(accepted_frames),
        ),
        expansion=expansion_records,
        final_solved_ids=frozenset(marker_poses),
        unresolved_ids=frozenset(),
    )
    return (
        assignments,
        rejected_frames,
        assignment_rejections,
        fallback_assignments,
        pair_consensus,
        marker_poses,
        dropped_edges,
        anchor_core,
        None,
    )
