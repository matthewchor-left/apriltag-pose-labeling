"""Continuous corner bundle adjustment, pruning, and checkpoint recovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from object_apriltag.layout import build_marker_layout
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CalibrationSolveDiagnostics,
    CornerObservation,
    MarkerCandidate,
    MarkerPair,
    OptimizationCheckpointStage,
    PairConsensus,
    copy_frame_poses,
    copy_marker_poses,
    covisible_frame_count,
    covisible_frames_from_inliers,
    corner_errors,
    drop_frames_without_covisibility,
    mask_corner_observations_for_frames,
    missing_from_graph,
    positive_depth_failure,
    poses_are_finite,
    project_corner,
    record_optimizer_run,
    snapshot_pair_consensus,
    timed_solve_stage,
    connected_marker_ids,
    complete_markers_per_frame,
)
from object_apriltag.marker_layout_calibration.solve_quality import (
    build_quality_report,
    collect_quality_gate_failures,
    edge_diagnostics,
    footprints_from_poses,
    pair_translation_gate,
    quality_from_pairs,
)

if TYPE_CHECKING:
    from object_apriltag.marker_layout_calibration.types import (
        AnchorCoreDiagnostics,
        AssignmentRejectionSummary,
        CalibrationQualityReport,
        CalibrationResult,
        CalibrationSettings,
        DroppedPairEdge,
        FrameAssignmentRejectionRecord,
        FrameFallbackAssignmentRecord,
        RestoredPairEdge,
    )

WeakConnectivityRestore = Callable[
    [
        dict[MarkerPair, PairConsensus],
        dict[MarkerPair, PairConsensus],
        list,
        list[int],
        int,
        str,
    ],
    str | None,
]

PairConsensusRefresh = Callable[
    [dict[MarkerPair, PairConsensus], dict[int, tuple[np.ndarray, np.ndarray]]],
    dict[MarkerPair, PairConsensus],
]


@dataclass(frozen=True)
class OptimizationCheckpoint:
    stage: OptimizationCheckpointStage
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]]
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None]
    inlier_mask: np.ndarray
    pair_consensus: dict[MarkerPair, PairConsensus]


@dataclass
class LayoutSolveState:
    corner_observations: list[CornerObservation]
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]]
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None]
    inlier_mask: np.ndarray
    pair_consensus: dict[MarkerPair, PairConsensus]

    def with_poses(
        self,
        marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
        frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
        inlier_mask: np.ndarray,
        pair_consensus: dict[MarkerPair, PairConsensus],
    ) -> LayoutSolveState:
        return LayoutSolveState(
            self.corner_observations,
            marker_poses,
            frame_poses,
            inlier_mask,
            pair_consensus,
        )

    def checkpoint_snapshot(self, stage: OptimizationCheckpointStage) -> OptimizationCheckpoint:
        return OptimizationCheckpoint(
            stage=stage,
            marker_poses=copy_marker_poses(self.marker_poses),
            frame_poses=copy_frame_poses(self.frame_poses),
            inlier_mask=self.inlier_mask.copy(),
            pair_consensus=snapshot_pair_consensus(self.pair_consensus),
        )


@dataclass
class LayoutRefinementContext:
    reference_marker_id: int
    non_reference_ids: list[int]
    expected_ids: list[int]
    accepted_frames: frozenset[int]
    object_points_by_marker: dict[int, np.ndarray]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    settings: CalibrationSettings
    marker_sizes_m: Mapping[int, float]
    marker_size_m: float
    best_effort: bool
    restored_pair_edges: tuple[RestoredPairEdge, ...] | None
    input_frame_count: int
    rejected_frame_count: int
    accepted_frame_count: int
    assignment_rejection_summary: AssignmentRejectionSummary | None
    assignment_rejection_records: tuple[FrameAssignmentRejectionRecord, ...] | None
    fallback_assignment_records: tuple[FrameFallbackAssignmentRecord, ...] | None
    dropped_edges: list[DroppedPairEdge]
    anchor_core_diagnostics: AnchorCoreDiagnostics | None
    restore_weak_connectivity: WeakConnectivityRestore
    refresh_pair_consensus_after_initial_ba: PairConsensusRefresh | None = None
    anchor_marker_ids: Sequence[int] | None = None
    solve_diagnostics: CalibrationSolveDiagnostics | None = None


@dataclass(frozen=True)
class LayoutRefinementOutcome:
    state: LayoutSolveState
    pruning_drops: tuple[DroppedPairEdge, ...]
    early_result: CalibrationResult | None


class ContinuousLayoutRefinement:
    def __init__(self, context: LayoutRefinementContext) -> None:
        self._ctx = context
        self._checkpoints: list[OptimizationCheckpoint] = []

    def run(self, state: LayoutSolveState) -> LayoutRefinementOutcome:
        ctx = self._ctx
        self._maybe_record_checkpoint(state, "graph_initialization")

        marker_poses, frame_poses, inlier_mask, ba_failure = run_bundle_adjustment(
            state.corner_observations,
            state.inlier_mask,
            state.marker_poses,
            state.frame_poses,
            ctx.reference_marker_id,
            ctx.non_reference_ids,
            ctx.object_points_by_marker,
            ctx.camera_matrix,
            ctx.dist_coeffs,
            ctx.settings,
            solve_diagnostics=ctx.solve_diagnostics,
            stage_name="initial_bundle_adjustment",
        )
        if ba_failure is not None:
            return self._failure_outcome(
                state,
                marker_poses,
                frame_poses,
                inlier_mask,
                state.pair_consensus,
                "initial_bundle_adjustment",
                ba_failure,
                accepted_frame_count=ctx.accepted_frame_count,
                observation_count=len(state.corner_observations),
            )

        pair_consensus = state.pair_consensus
        if ctx.refresh_pair_consensus_after_initial_ba is not None:
            pair_consensus = ctx.refresh_pair_consensus_after_initial_ba(
                pair_consensus,
                marker_poses,
            )
        state = state.with_poses(marker_poses, frame_poses, inlier_mask, pair_consensus)
        self._maybe_record_checkpoint(state, "initial_bundle_adjustment")

        marker_poses, frame_poses, inlier_mask, pair_consensus, prune_failure, pruning_drops = (
            _prune_and_refit(state, ctx)
        )
        if prune_failure is not None:
            ctx.dropped_edges.extend(pruning_drops)
            return self._failure_outcome(
                state,
                marker_poses,
                frame_poses,
                inlier_mask,
                pair_consensus,
                "post_pruning_refit",
                prune_failure,
                accepted_frame_count=covisible_frame_count(
                    state.corner_observations,
                    inlier_mask,
                ),
                observation_count=int(np.count_nonzero(inlier_mask)),
            )

        state = state.with_poses(marker_poses, frame_poses, inlier_mask, pair_consensus)
        self._maybe_record_checkpoint(state, "post_pruning_refit")
        return LayoutRefinementOutcome(state, pruning_drops, None)

    def _maybe_record_checkpoint(
        self,
        state: LayoutSolveState,
        stage: OptimizationCheckpointStage,
    ) -> None:
        checkpoint = state.checkpoint_snapshot(stage)
        if _is_valid_complete_checkpoint(
            checkpoint,
            expected_ids=self._ctx.expected_ids,
            reference_marker_id=self._ctx.reference_marker_id,
            corner_observations=state.corner_observations,
            object_points_by_marker=self._ctx.object_points_by_marker,
        ):
            self._checkpoints.append(checkpoint)

    def _failure_outcome(
        self,
        state: LayoutSolveState,
        marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
        frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
        inlier_mask: np.ndarray,
        pair_consensus: dict[MarkerPair, PairConsensus],
        failed_stage: OptimizationCheckpointStage,
        failure_message: str,
        *,
        accepted_frame_count: int,
        observation_count: int,
    ) -> LayoutRefinementOutcome:
        ctx = self._ctx
        recovered = self._maybe_recover_from_checkpoints(
            state.corner_observations,
            failed_stage,
            tuple(ctx.dropped_edges),
        )
        if recovered is not None:
            return LayoutRefinementOutcome(
                state.with_poses(marker_poses, frame_poses, inlier_mask, pair_consensus),
                (),
                recovered,
            )
        from object_apriltag.marker_layout_calibration.types import CalibrationResult

        quality = quality_from_pairs(
            pair_consensus,
            ctx.expected_ids,
            ctx.reference_marker_id,
            missing_from_graph(pair_consensus, ctx.expected_ids, ctx.reference_marker_id),
            input_frame_count=ctx.input_frame_count,
            rejected_frame_count=ctx.rejected_frame_count,
            accepted_frame_count=accepted_frame_count,
            observation_count=observation_count,
            assignment_rejections=ctx.assignment_rejection_summary,
            assignment_rejection_records=ctx.assignment_rejection_records,
            fallback_assignment_records=ctx.fallback_assignment_records,
            dropped_pair_edges=tuple(ctx.dropped_edges),
            restored_pair_edges=ctx.restored_pair_edges,
            anchor_core=ctx.anchor_core_diagnostics,
        )
        return LayoutRefinementOutcome(
            state.with_poses(marker_poses, frame_poses, inlier_mask, pair_consensus),
            (),
            CalibrationResult(None, quality, failure_message),
        )

    def _maybe_recover_from_checkpoints(
        self,
        corner_observations: list[CornerObservation],
        failed_stage: OptimizationCheckpointStage,
        dropped_pair_edges: tuple[DroppedPairEdge, ...],
    ) -> CalibrationResult | None:
        ctx = self._ctx
        if not ctx.best_effort:
            return None
        checkpoint = _latest_valid_optimization_checkpoint(
            self._checkpoints,
            expected_ids=ctx.expected_ids,
            reference_marker_id=ctx.reference_marker_id,
            corner_observations=corner_observations,
            object_points_by_marker=ctx.object_points_by_marker,
        )
        if checkpoint is None:
            return None
        return _provisional_result_from_checkpoint(
            checkpoint,
            failed_refinement_stage=failed_stage,
            context=ctx,
            corner_observations=corner_observations,
            dropped_pair_edges=dropped_pair_edges,
        )


def run_bundle_adjustment(
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    reference_marker_id: int,
    non_reference_ids: list[int],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settings: CalibrationSettings,
    *,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
    stage_name: str | None = None,
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
            projected = project_corner(
                observation.corner_index,
                observation.marker_id,
                marker_pose,
                frame_pose,
                object_points_by_marker,
                camera_matrix,
                dist_coeffs,
            )
            if not np.all(np.isfinite(projected)):
                values.extend([1000.0, 1000.0])
                continue
            delta = projected - observation.image_point
            values.extend(delta.tolist())
        return np.asarray(values, dtype=np.float64)

    inlier_corner_count = int(np.count_nonzero(inlier_mask))
    active_frame_count = len(active_frames)
    stage_timer = (
        timed_solve_stage(solve_diagnostics, stage_name)
        if stage_name is not None
        else nullcontext()
    )
    with stage_timer:
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

    record_optimizer_run(
        solve_diagnostics,
        stage_name=stage_name,
        active_frame_count=active_frame_count,
        inlier_corner_count=inlier_corner_count,
        result=result,
    )

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
    depth_failure = positive_depth_failure(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points_by_marker,
    )
    if depth_failure is not None:
        return marker_poses, frame_poses, inlier_mask, depth_failure
    return marker_poses, frame_poses, inlier_mask, None


def recheck_pair_support(
    pair_consensus: dict[MarkerPair, PairConsensus],
    corner_observations: Sequence[CornerObservation],
    inlier_mask: np.ndarray,
    expected_ids: list[int],
    reference_marker_id: int,
    settings: CalibrationSettings,
    allowed_frames: frozenset[int] | None = None,
    *,
    marker_sizes_m: Mapping[int, float],
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
    restore_weak_connectivity: WeakConnectivityRestore | None = None,
) -> tuple[dict[MarkerPair, PairConsensus], str | None, tuple[DroppedPairEdge, ...]]:
    from object_apriltag.marker_layout_calibration.types import DroppedPairEdge

    rotation_gate = settings.pair_rotation_rms_gate_deg
    complete = complete_markers_per_frame(corner_observations, inlier_mask)
    updated: dict[MarkerPair, PairConsensus] = {}
    weak_pool: dict[MarkerPair, PairConsensus] = {}
    dropped: list[DroppedPairEdge] = []
    for pair, edge in pair_consensus.items():
        translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
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
            if supported_frames:
                selected_hypotheses = {
                    frame_index: edge.inlier_hypotheses[frame_index]
                    for frame_index in supported_frames
                }
                weak_pool[pair] = PairConsensus(
                    marker_a=marker_low,
                    marker_b=marker_high,
                    rotation_ba=edge.rotation_ba,
                    translation_ba=edge.translation_ba,
                    inlier_frames=supported_frames,
                    inlier_hypotheses=selected_hypotheses,
                )
            dropped.append(
                _make_pruning_dropped_edge(
                    pair,
                    observed_count=len(edge.inlier_frames),
                    supported_count=len(supported_frames),
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        selected_hypotheses = {
            frame_index: edge.inlier_hypotheses[frame_index]
            for frame_index in supported_frames
        }
        updated[pair] = PairConsensus(
            marker_a=marker_low,
            marker_b=marker_high,
            rotation_ba=edge.rotation_ba,
            translation_ba=edge.translation_ba,
            inlier_frames=supported_frames,
            inlier_hypotheses=selected_hypotheses,
        )

    if restore_weak_connectivity is None:
        from object_apriltag.marker_layout_calibration.discrete_graph import maybe_restore_weak_connectivity

        restore_weak_connectivity = maybe_restore_weak_connectivity

    failure = restore_weak_connectivity(
        updated,
        weak_pool,
        dropped,
        expected_ids,
        reference_marker_id,
        "post_pruning",
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    if failure is not None:
        return updated, failure, tuple(dropped)
    return updated, None, tuple(dropped)


def _provisional_result_from_checkpoint(
    checkpoint: OptimizationCheckpoint,
    *,
    failed_refinement_stage: OptimizationCheckpointStage,
    context: LayoutRefinementContext,
    corner_observations: list[CornerObservation],
    dropped_pair_edges: tuple[DroppedPairEdge, ...],
) -> CalibrationResult:
    from object_apriltag.marker_layout_calibration.types import CalibrationResult

    ctx = context
    marker_poses = checkpoint.marker_poses
    frame_poses = checkpoint.frame_poses
    inlier_mask = checkpoint.inlier_mask
    pair_consensus = checkpoint.pair_consensus
    connected_ids = connected_marker_ids(pair_consensus, ctx.reference_marker_id)
    missing_ids = frozenset(set(ctx.expected_ids) - connected_ids)
    accepted_frame_count = covisible_frame_count(corner_observations, inlier_mask)
    quality = build_quality_report(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        pair_consensus,
        ctx.expected_ids,
        ctx.reference_marker_id,
        missing_ids,
        ctx.input_frame_count,
        ctx.rejected_frame_count,
        accepted_frame_count,
        ctx.object_points_by_marker,
        ctx.camera_matrix,
        ctx.dist_coeffs,
        assignment_rejections=ctx.assignment_rejection_summary,
        assignment_rejection_records=ctx.assignment_rejection_records,
        fallback_assignment_records=ctx.fallback_assignment_records,
        dropped_pair_edges=dropped_pair_edges,
        restored_pair_edges=ctx.restored_pair_edges,
        anchor_core=ctx.anchor_core_diagnostics,
    )
    gate_failures = collect_quality_gate_failures(
        quality,
        ctx.settings,
        ctx.marker_sizes_m,
        ctx.expected_ids,
    )
    failed_gate_messages = tuple(failure.message for failure in gate_failures)
    layout = build_marker_layout(
        reference_marker_id=ctx.reference_marker_id,
        marker_size_m=ctx.marker_size_m,
        footprints=footprints_from_poses(marker_poses, ctx.marker_sizes_m),
        marker_sizes_m=dict(ctx.marker_sizes_m),
        anchor_marker_ids=ctx.anchor_marker_ids,
    )
    return CalibrationResult(
        layout,
        quality,
        None,
        outcome="provisional",
        calibration_policy="best_effort",
        failed_quality_gates=failed_gate_messages,
        selected_checkpoint_stage=checkpoint.stage,
        failed_refinement_stage=failed_refinement_stage,
    )


def _make_pruning_dropped_edge(
    pair: MarkerPair,
    *,
    observed_count: int,
    supported_count: int,
    required_count: int,
    translation_gate: float,
    rotation_gate: float,
    edge: PairConsensus | None = None,
) -> DroppedPairEdge:
    from object_apriltag.marker_layout_calibration.types import DroppedPairEdge

    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    if edge is not None:
        diagnostics = edge_diagnostics(pair, edge)
        translation_rms_m = float(diagnostics.translation_rms_m)
        rotation_rms_deg = float(diagnostics.rotation_rms_deg)
    return DroppedPairEdge(
        marker_a=pair[0],
        marker_b=pair[1],
        stage="post_pruning",
        reason="insufficient_support",
        observed_count=observed_count,
        supported_count=supported_count,
        required_count=required_count,
        translation_rms_m=translation_rms_m,
        rotation_rms_deg=rotation_rms_deg,
        translation_gate_m=float(translation_gate),
        rotation_gate_deg=float(rotation_gate),
    )


def _is_valid_complete_checkpoint(
    checkpoint: OptimizationCheckpoint,
    *,
    expected_ids: list[int],
    reference_marker_id: int,
    corner_observations: Sequence[CornerObservation],
    object_points_by_marker: dict[int, np.ndarray],
) -> bool:
    if set(checkpoint.marker_poses) != set(expected_ids):
        return False
    connected = connected_marker_ids(checkpoint.pair_consensus, reference_marker_id)
    if not set(expected_ids).issubset(connected):
        return False
    if not poses_are_finite(checkpoint.marker_poses, checkpoint.frame_poses):
        return False
    return positive_depth_failure(
        corner_observations,
        checkpoint.inlier_mask,
        checkpoint.marker_poses,
        checkpoint.frame_poses,
        object_points_by_marker,
    ) is None


def _latest_valid_optimization_checkpoint(
    checkpoints: Sequence[OptimizationCheckpoint],
    *,
    expected_ids: list[int],
    reference_marker_id: int,
    corner_observations: Sequence[CornerObservation],
    object_points_by_marker: dict[int, np.ndarray],
) -> OptimizationCheckpoint | None:
    for checkpoint in reversed(checkpoints):
        if _is_valid_complete_checkpoint(
            checkpoint,
            expected_ids=expected_ids,
            reference_marker_id=reference_marker_id,
            corner_observations=corner_observations,
            object_points_by_marker=object_points_by_marker,
        ):
            return checkpoint
    return None


def _prune_and_refit(
    state: LayoutSolveState,
    ctx: LayoutRefinementContext,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray] | None],
    np.ndarray,
    dict[MarkerPair, PairConsensus],
    str | None,
    tuple[DroppedPairEdge, ...],
]:
    corner_observations = state.corner_observations
    marker_poses = state.marker_poses
    frame_poses = state.frame_poses
    inlier_mask = state.inlier_mask
    pair_consensus = state.pair_consensus

    with timed_solve_stage(ctx.solve_diagnostics, "pruning"):
        errors = corner_errors(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            ctx.object_points_by_marker,
            ctx.camera_matrix,
            ctx.dist_coeffs,
        )
        pruned = inlier_mask & (errors <= ctx.settings.corner_outlier_px)
        pruned = drop_frames_without_covisibility(corner_observations, pruned)
        pruned = mask_corner_observations_for_frames(corner_observations, ctx.accepted_frames) & pruned
        if int(np.count_nonzero(pruned)) < 8:
            return marker_poses, frame_poses, inlier_mask, pair_consensus, (
                "Too few inlier corners remain after pruning."
            ), ()

        remaining_frames = covisible_frames_from_inliers(corner_observations, pruned)
        updated_consensus, support_failure, dropped_edges = recheck_pair_support(
            pair_consensus,
            corner_observations,
            pruned,
            ctx.expected_ids,
            ctx.reference_marker_id,
            ctx.settings,
            allowed_frames=remaining_frames,
            marker_sizes_m=ctx.marker_sizes_m,
            best_effort=ctx.best_effort,
            restored_pair_edges=list(ctx.restored_pair_edges) if ctx.restored_pair_edges else None,
            restore_weak_connectivity=ctx.restore_weak_connectivity,
        )
        if support_failure is not None:
            return marker_poses, frame_poses, pruned, updated_consensus, support_failure, dropped_edges

    marker_poses, frame_poses, pruned, ba_failure = run_bundle_adjustment(
        corner_observations,
        pruned,
        marker_poses,
        frame_poses,
        ctx.reference_marker_id,
        ctx.non_reference_ids,
        ctx.object_points_by_marker,
        ctx.camera_matrix,
        ctx.dist_coeffs,
        ctx.settings,
        solve_diagnostics=ctx.solve_diagnostics,
        stage_name="post_pruning_refit",
    )
    if ba_failure is not None:
        return marker_poses, frame_poses, pruned, updated_consensus, ba_failure, dropped_edges
    return marker_poses, frame_poses, pruned, updated_consensus, None, dropped_edges


def _build_jac_sparsity(
    corner_observations: Sequence[CornerObservation],
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
