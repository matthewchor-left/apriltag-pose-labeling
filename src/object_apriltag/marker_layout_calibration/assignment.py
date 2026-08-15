"""Constrained IPPE assignment and rejection/fallback diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from object_apriltag.marker_layout_calibration.discrete_graph import (
    best_pair_consensus,
    transform_high_in_low,
)
from object_apriltag.marker_layout_calibration.solve_quality import pair_translation_gate
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CalibrationSolveDiagnostics,
    MarkerCandidate,
    MarkerPair,
    PairConsensus,
    timed_solve_stage,
    rotation_geodesic_deg,
)
from object_apriltag.marker_layout_calibration.types import (
    AssignmentRejectionCauseCount,
    AssignmentRejectionCauseStats,
    AssignmentRejectionSummary,
    CalibrationSettings,
    FrameAssignmentRejection,
    FrameAssignmentRejectionRecord,
    FrameAssignmentResult,
    FrameFallbackAssignment,
    FrameFallbackAssignmentRecord,
    FrameObservation,
    MeasurementDistribution,
)


def json_safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def measurement_distribution(values: Sequence[float]) -> MeasurementDistribution | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return MeasurementDistribution(
        min=json_safe_float(float(np.min(finite))),
        median=json_safe_float(float(np.median(finite))),
        p95=json_safe_float(float(np.percentile(finite, 95))),
        max=json_safe_float(float(np.max(finite))),
    )


def assign_ippe_candidates(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
    pair_consensus: dict[MarkerPair, PairConsensus],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    search_marker_ids: frozenset[int] | None = None,
    best_effort: bool = False,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
) -> tuple[
    dict[int, dict[int, MarkerCandidate]],
    tuple[int, ...],
    tuple[FrameAssignmentRejection, ...],
    tuple[FrameFallbackAssignment, ...],
]:
    assigned: dict[int, dict[int, MarkerCandidate]] = {}
    rejected_frames: list[int] = []
    rejections: list[FrameAssignmentRejection] = []
    fallback_assignments: list[FrameFallbackAssignment] = []
    for frame_index, candidates in frame_candidates:
        with timed_solve_stage(solve_diagnostics, "strict_assignment"):
            result = resolve_frame_ippe_assignment(
                candidates,
                pair_consensus,
                settings,
                marker_sizes_m,
                search_marker_ids=search_marker_ids,
            )
        if result.assignment is not None:
            assigned[frame_index] = result.assignment
            continue
        if best_effort:
            with timed_solve_stage(solve_diagnostics, "fallback_assignment"):
                fallback = resolve_frame_ippe_fallback_assignment(
                    candidates,
                    pair_consensus,
                    settings,
                    marker_sizes_m,
                    search_marker_ids=search_marker_ids,
                )
            if fallback.assignment is not None:
                assigned[frame_index] = fallback.assignment
                fallback_assignments.append(
                    FrameFallbackAssignment(
                        frame_index=frame_index,
                        disagreement_cost=fallback.disagreement_cost,
                        marker_pair=fallback.worst_pair,
                        translation_error_m=fallback.worst_translation_error_m,
                        rotation_error_deg=fallback.worst_rotation_error_deg,
                    )
                )
                continue
        rejected_frames.append(frame_index)
        rejections.append(
            result.rejection
            or FrameAssignmentRejection(reason="no_constrained_pair")
        )
    return assigned, tuple(rejected_frames), tuple(rejections), tuple(fallback_assignments)


@dataclass(frozen=True)
class FrameFallbackAssignmentResult:
    assignment: dict[int, MarkerCandidate] | None
    disagreement_cost: float
    worst_pair: MarkerPair | None
    worst_translation_error_m: float
    worst_rotation_error_deg: float


def resolve_frame_ippe_fallback_assignment(
    candidates: dict[int, list[MarkerCandidate]],
    pair_consensus: dict[MarkerPair, PairConsensus],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    search_marker_ids: frozenset[int] | None = None,
) -> FrameFallbackAssignmentResult:
    if search_marker_ids is not None:
        marker_ids = sorted(marker_id for marker_id in candidates if marker_id in search_marker_ids)
    else:
        marker_ids = sorted(candidates)
    if len(marker_ids) < 2:
        return FrameFallbackAssignmentResult(
            assignment=None,
            disagreement_cost=float("inf"),
            worst_pair=None,
            worst_translation_error_m=0.0,
            worst_rotation_error_deg=0.0,
        )
    holder = new_fallback_assignment_search_holder(settings, marker_sizes_m)
    search_assignments(
        marker_ids,
        candidates,
        pair_consensus,
        {},
        0,
        holder,
        assignment_mode="fallback",
    )
    assignment = holder.get("assignment")
    if assignment is None:
        return FrameFallbackAssignmentResult(
            assignment=None,
            disagreement_cost=float("inf"),
            worst_pair=None,
            worst_translation_error_m=0.0,
            worst_rotation_error_deg=0.0,
        )
    return FrameFallbackAssignmentResult(
        assignment=dict(assignment),  # type: ignore[arg-type]
        disagreement_cost=float(holder["best_cost"]),
        worst_pair=holder.get("worst_pair"),  # type: ignore[arg-type]
        worst_translation_error_m=float(holder["worst_translation_error"]),
        worst_rotation_error_deg=float(holder["worst_rotation_error"]),
    )


def resolve_frame_ippe_assignment(
    candidates: dict[int, list[MarkerCandidate]],
    pair_consensus: dict[MarkerPair, PairConsensus],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    search_marker_ids: frozenset[int] | None = None,
) -> FrameAssignmentResult:
    if search_marker_ids is not None:
        marker_ids = sorted(marker_id for marker_id in candidates if marker_id in search_marker_ids)
    else:
        marker_ids = sorted(candidates)
    if len(marker_ids) < 2:
        return FrameAssignmentResult(
            assignment=None,
            rejection=FrameAssignmentRejection(reason="insufficient_anchors_visible"),
        )
    holder = new_assignment_search_holder(settings, marker_sizes_m)
    search_assignments(marker_ids, candidates, pair_consensus, {}, 0, holder)
    assignment = holder["assignment"]
    if assignment is not None:
        return FrameAssignmentResult(assignment=dict(assignment), rejection=None)
    return FrameAssignmentResult(
        assignment=None,
        rejection=rejection_from_assignment_search_holder(holder),
    )


def new_fallback_assignment_search_holder(
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
) -> dict[str, object]:
    return {
        "best_cost": float("inf"),
        "assignment": None,
        "worst_pair": None,
        "worst_translation_error": 0.0,
        "worst_rotation_error": 0.0,
        "marker_sizes_m": dict(marker_sizes_m),
        "settings": settings,
        "rotation_gate": settings.pair_rotation_rms_gate_deg,
    }


def new_assignment_search_holder(
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
) -> dict[str, object]:
    return {
        "score": float("-inf"),
        "assignment": None,
        "saw_constrained_pair": False,
        "worst_key": None,
        "worst_pair": None,
        "worst_translation_error": 0.0,
        "worst_rotation_error": 0.0,
        "worst_translation_fail": False,
        "worst_rotation_fail": False,
        "worst_translation_gate": 0.0,
        "marker_sizes_m": dict(marker_sizes_m),
        "settings": settings,
        "rotation_gate": settings.pair_rotation_rms_gate_deg,
    }


def rejection_from_assignment_search_holder(holder: dict[str, object]) -> FrameAssignmentRejection:
    if not holder["saw_constrained_pair"]:
        return FrameAssignmentRejection(reason="no_constrained_pair")

    worst_pair = holder["worst_pair"]
    if worst_pair is None:
        return FrameAssignmentRejection(reason="no_constrained_pair")

    marker_sizes_m = holder["marker_sizes_m"]
    assert isinstance(marker_sizes_m, dict)
    translation_gate = float(holder.get("worst_translation_gate", 0.0))
    rotation_gate = float(holder["rotation_gate"])
    worst_translation_error = float(holder["worst_translation_error"])
    worst_rotation_error = float(holder["worst_rotation_error"])
    worst_translation_fail = bool(holder["worst_translation_fail"])
    worst_rotation_fail = bool(holder["worst_rotation_fail"])
    if worst_translation_fail and worst_rotation_fail:
        primary_reason = (
            "translation_gate"
            if (worst_translation_error / translation_gate)
            >= (worst_rotation_error / rotation_gate)
            else "rotation_gate"
        )
    elif worst_translation_fail:
        primary_reason = "translation_gate"
    else:
        primary_reason = "rotation_gate"
    return FrameAssignmentRejection(
        reason=primary_reason,
        marker_pair=worst_pair,
        translation_error_m=worst_translation_error,
        rotation_error_deg=worst_rotation_error,
        translation_gate_m=translation_gate,
        rotation_gate_deg=rotation_gate,
    )


def evaluate_fallback_assignment(
    assignment: dict[int, MarkerCandidate],
    marker_ids: list[int],
    pair_consensus: dict[MarkerPair, PairConsensus],
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    rotation_gate: float,
) -> tuple[float | None, MarkerPair | None, float, float]:
    constrained_edges = 0
    total_cost = 0.0
    worst_exceedance = -1.0
    worst_pair: MarkerPair | None = None
    worst_translation_error = 0.0
    worst_rotation_error = 0.0

    for index_a, marker_low in enumerate(marker_ids):
        for marker_high in marker_ids[index_a + 1 :]:
            pair = (marker_low, marker_high)
            edge = pair_consensus.get(pair)
            if edge is None:
                continue
            translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
            constrained_edges += 1
            rotation_ba, translation_ba = transform_high_in_low(
                assignment[marker_low],
                assignment[marker_high],
            )
            translation_error = float(np.linalg.norm(translation_ba - edge.translation_ba))
            rotation_error = rotation_geodesic_deg(rotation_ba, edge.rotation_ba)
            if not np.isfinite(translation_error) or not np.isfinite(rotation_error):
                return None, None, 0.0, 0.0
            if translation_gate <= 0.0 or rotation_gate <= 0.0:
                return None, None, 0.0, 0.0
            total_cost += (translation_error / translation_gate) ** 2 + (
                rotation_error / rotation_gate
            ) ** 2
            exceedance = max(translation_error / translation_gate, rotation_error / rotation_gate)
            if exceedance > worst_exceedance:
                worst_exceedance = exceedance
                worst_pair = pair
                worst_translation_error = translation_error
                worst_rotation_error = rotation_error

    if constrained_edges == 0:
        return None, None, 0.0, 0.0
    return total_cost, worst_pair, worst_translation_error, worst_rotation_error


def evaluate_complete_assignment(
    assignment: dict[int, MarkerCandidate],
    marker_ids: list[int],
    pair_consensus: dict[MarkerPair, PairConsensus],
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    rotation_gate: float,
) -> tuple[
    float | None,
    bool,
    tuple[float, MarkerPair, int] | None,
    MarkerPair | None,
    float,
    float,
    bool,
    bool,
    float,
]:
    constrained_edges = 0
    total_cost = 0.0
    assignment_valid = True
    worst_key: tuple[float, MarkerPair, int] | None = None
    worst_pair: MarkerPair | None = None
    worst_translation_error = 0.0
    worst_rotation_error = 0.0
    worst_translation_fail = False
    worst_rotation_fail = False
    worst_translation_gate = 0.0

    for index_a, marker_low in enumerate(marker_ids):
        for marker_high in marker_ids[index_a + 1 :]:
            pair = (marker_low, marker_high)
            edge = pair_consensus.get(pair)
            if edge is None:
                continue
            translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
            constrained_edges += 1
            rotation_ba, translation_ba = transform_high_in_low(
                assignment[marker_low],
                assignment[marker_high],
            )
            translation_error = float(np.linalg.norm(translation_ba - edge.translation_ba))
            rotation_error = rotation_geodesic_deg(rotation_ba, edge.rotation_ba)
            translation_fail = translation_error > translation_gate
            rotation_fail = rotation_error > rotation_gate
            if translation_fail or rotation_fail:
                assignment_valid = False
                translation_exceedance = translation_error / translation_gate
                rotation_exceedance = rotation_error / rotation_gate
                exceedance = max(translation_exceedance, rotation_exceedance)
                reason_rank = 0 if translation_exceedance >= rotation_exceedance else 1
                candidate_key = (exceedance, pair, reason_rank)
                if worst_key is None or candidate_key > worst_key:
                    worst_key = candidate_key
                    worst_pair = pair
                    worst_translation_error = translation_error
                    worst_rotation_error = rotation_error
                    worst_translation_fail = translation_fail
                    worst_rotation_fail = rotation_fail
                    worst_translation_gate = translation_gate
                continue
            total_cost += (translation_error / translation_gate) ** 2 + (
                rotation_error / rotation_gate
            ) ** 2

    if constrained_edges == 0 or not assignment_valid:
        return (
            None,
            constrained_edges > 0,
            worst_key,
            worst_pair,
            worst_translation_error,
            worst_rotation_error,
            worst_translation_fail,
            worst_rotation_fail,
            worst_translation_gate,
        )
    return (
        -total_cost,
        True,
        worst_key,
        worst_pair,
        worst_translation_error,
        worst_rotation_error,
        worst_translation_fail,
        worst_rotation_fail,
        worst_translation_gate,
    )


def merge_assignment_violation_into_holder(
    holder: dict[str, object],
    has_constrained_pair: bool,
    worst_key: tuple[float, MarkerPair, int] | None,
    worst_pair: MarkerPair | None,
    worst_translation_error: float,
    worst_rotation_error: float,
    worst_translation_fail: bool,
    worst_rotation_fail: bool,
    worst_translation_gate: float,
) -> None:
    if has_constrained_pair:
        holder["saw_constrained_pair"] = True
    if worst_key is None:
        return
    current_worst_key = holder["worst_key"]
    if current_worst_key is None or worst_key > current_worst_key:
        holder["worst_key"] = worst_key
        holder["worst_pair"] = worst_pair
        holder["worst_translation_error"] = worst_translation_error
        holder["worst_rotation_error"] = worst_rotation_error
        holder["worst_translation_fail"] = worst_translation_fail
        holder["worst_rotation_fail"] = worst_rotation_fail
        holder["worst_translation_gate"] = worst_translation_gate


def summarize_assignment_rejections(
    rejections: Sequence[FrameAssignmentRejection],
) -> AssignmentRejectionSummary:
    if rejections and not isinstance(rejections[0], FrameAssignmentRejection):
        raise TypeError(
            "summarize_assignment_rejections expects FrameAssignmentRejection inputs; "
            "use summarize_assignment_rejection_records for FrameAssignmentRejectionRecord."
        )
    by_reason: dict[str, int] = {}
    by_pair: dict[MarkerPair, int] = {}
    cause_counts: dict[tuple[str, MarkerPair | None], int] = {}
    for rejection in rejections:
        by_reason[rejection.reason] = by_reason.get(rejection.reason, 0) + 1
        if rejection.marker_pair is not None:
            by_pair[rejection.marker_pair] = by_pair.get(rejection.marker_pair, 0) + 1
        cause_key = (rejection.reason, rejection.marker_pair)
        cause_counts[cause_key] = cause_counts.get(cause_key, 0) + 1
    top_causes = tuple(
        AssignmentRejectionCauseCount(reason=reason, marker_pair=pair, count=count)
        for (reason, pair), count in sorted(
            cause_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1] or (-1, -1)),
        )
    )
    return AssignmentRejectionSummary(
        total_rejected=len(rejections),
        by_reason=tuple(sorted(by_reason.items())),
        by_pair=tuple(sorted(by_pair.items())),
        top_causes=top_causes,
        by_cause=(),
    )


def summarize_assignment_rejection_records(
    records: Sequence[FrameAssignmentRejectionRecord],
) -> AssignmentRejectionSummary:
    if records and not isinstance(records[0], FrameAssignmentRejectionRecord):
        raise TypeError(
            "summarize_assignment_rejection_records expects FrameAssignmentRejectionRecord inputs; "
            "use summarize_assignment_rejections for FrameAssignmentRejection."
        )
    by_reason: dict[str, int] = {}
    by_pair: dict[MarkerPair, int] = {}
    cause_counts: dict[tuple[str, MarkerPair | None], int] = {}
    cause_groups: dict[tuple[str, MarkerPair | None], list[FrameAssignmentRejectionRecord]] = {}
    for record in records:
        by_reason[record.reason] = by_reason.get(record.reason, 0) + 1
        if record.marker_pair is not None:
            by_pair[record.marker_pair] = by_pair.get(record.marker_pair, 0) + 1
        cause_key = (record.reason, record.marker_pair)
        cause_counts[cause_key] = cause_counts.get(cause_key, 0) + 1
        cause_groups.setdefault(cause_key, []).append(record)
    top_causes = tuple(
        AssignmentRejectionCauseCount(reason=reason, marker_pair=pair, count=count)
        for (reason, pair), count in sorted(
            cause_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1] or (-1, -1)),
        )
    )
    by_cause: list[AssignmentRejectionCauseStats] = []
    for (reason, pair), group in sorted(
        cause_groups.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1] or (-1, -1)),
    ):
        sorted_group = sorted(group, key=lambda record: record.frame_index)
        translation_errors = [
            value
            for value in (record.translation_error_m for record in group)
            if value is not None
        ]
        rotation_errors = [
            value
            for value in (record.rotation_error_deg for record in group)
            if value is not None
        ]
        translation_gate = next(
            (record.translation_gate_m for record in group if record.translation_gate_m is not None),
            None,
        )
        rotation_gate = next(
            (record.rotation_gate_deg for record in group if record.rotation_gate_deg is not None),
            None,
        )
        translation_ratios = [
            error / translation_gate
            for error in translation_errors
            if translation_gate is not None and translation_gate > 0.0
        ]
        rotation_ratios = [
            error / rotation_gate
            for error in rotation_errors
            if rotation_gate is not None and rotation_gate > 0.0
        ]
        by_cause.append(
            AssignmentRejectionCauseStats(
                reason=reason,
                marker_pair=pair,
                count=len(group),
                sample_frame_ids=tuple(
                    record.frame_id for record in sorted_group[:_ASSIGNMENT_REJECTION_SAMPLE_FRAME_IDS]
                ),
                translation_error_m=measurement_distribution(translation_errors),
                rotation_error_deg=measurement_distribution(rotation_errors),
                translation_gate_m=json_safe_float(translation_gate),
                rotation_gate_deg=json_safe_float(rotation_gate),
                translation_error_ratio=measurement_distribution(translation_ratios),
                rotation_error_ratio=measurement_distribution(rotation_ratios),
            )
        )
    return AssignmentRejectionSummary(
        total_rejected=len(records),
        by_reason=tuple(sorted(by_reason.items())),
        by_pair=tuple(sorted(by_pair.items())),
        top_causes=top_causes,
        by_cause=tuple(by_cause),
    )


_ASSIGNMENT_REJECTION_SAMPLE_FRAME_IDS = 10


def build_fallback_assignment_records(
    normalized_observations: Sequence[tuple[str | int, dict[int, np.ndarray]]],
    fallback_assignments: Sequence[FrameFallbackAssignment],
) -> tuple[FrameFallbackAssignmentRecord, ...]:
    records: list[FrameFallbackAssignmentRecord] = []
    for fallback in fallback_assignments:
        frame_id, markers = normalized_observations[fallback.frame_index]
        visible_marker_ids = tuple(sorted(int(marker_id) for marker_id in markers))
        records.append(
            FrameFallbackAssignmentRecord(
                frame_index=fallback.frame_index,
                frame_id=frame_id,
                visible_marker_ids=visible_marker_ids,
                disagreement_cost=json_safe_float(fallback.disagreement_cost) or float("inf"),
                marker_pair=fallback.marker_pair,
                translation_error_m=json_safe_float(fallback.translation_error_m),
                rotation_error_deg=json_safe_float(fallback.rotation_error_deg),
            )
        )
    return tuple(records)


def build_assignment_rejection_records(
    normalized_observations: Sequence[tuple[str | int, dict[int, np.ndarray]]],
    rejected_frame_indices: Sequence[int],
    rejections: Sequence[FrameAssignmentRejection],
) -> tuple[FrameAssignmentRejectionRecord, ...]:
    records: list[FrameAssignmentRejectionRecord] = []
    for frame_index, rejection in zip(rejected_frame_indices, rejections, strict=True):
        frame_id, markers = normalized_observations[frame_index]
        visible_marker_ids = tuple(sorted(int(marker_id) for marker_id in markers))
        records.append(
            FrameAssignmentRejectionRecord(
                frame_index=frame_index,
                frame_id=frame_id,
                visible_marker_ids=visible_marker_ids,
                reason=rejection.reason,
                marker_pair=rejection.marker_pair,
                translation_error_m=json_safe_float(rejection.translation_error_m),
                rotation_error_deg=json_safe_float(rejection.rotation_error_deg),
                translation_gate_m=json_safe_float(rejection.translation_gate_m),
                rotation_gate_deg=json_safe_float(rejection.rotation_gate_deg),
            )
        )
    return tuple(records)




def search_assignments(
    marker_ids: list[int],
    candidates: dict[int, list[MarkerCandidate]],
    pair_consensus: dict[MarkerPair, PairConsensus],
    current: dict[int, MarkerCandidate],
    index: int,
    holder: dict[str, object],
    *,
    assignment_mode: Literal["strict", "fallback"] = "strict",
) -> None:
    if index == len(marker_ids):
        if assignment_mode == "fallback":
            cost, worst_pair, worst_translation_error, worst_rotation_error = (
                evaluate_fallback_assignment(
                    current,
                    marker_ids,
                    pair_consensus,
                    holder["marker_sizes_m"],  # type: ignore[arg-type]
                    holder["settings"],  # type: ignore[arg-type]
                    float(holder["rotation_gate"]),
                )
            )
            if cost is not None and cost < float(holder["best_cost"]):
                holder["best_cost"] = cost
                holder["assignment"] = dict(current)
                holder["worst_pair"] = worst_pair
                holder["worst_translation_error"] = worst_translation_error
                holder["worst_rotation_error"] = worst_rotation_error
            return
        (
            score,
            has_constrained_pair,
            worst_key,
            worst_pair,
            worst_translation_error,
            worst_rotation_error,
            worst_translation_fail,
            worst_rotation_fail,
            worst_translation_gate,
        ) = evaluate_complete_assignment(
            current,
            marker_ids,
            pair_consensus,
            holder["marker_sizes_m"],  # type: ignore[arg-type]
            holder["settings"],  # type: ignore[arg-type]
            float(holder["rotation_gate"]),
        )
        merge_assignment_violation_into_holder(
            holder,
            has_constrained_pair,
            worst_key,
            worst_pair,
            worst_translation_error,
            worst_rotation_error,
            worst_translation_fail,
            worst_rotation_fail,
            worst_translation_gate,
        )
        if score is not None and score > float(holder["score"]):
            holder["score"] = score
            holder["assignment"] = dict(current)
        return

    marker_id = marker_ids[index]
    for candidate in candidates[marker_id]:
        current[marker_id] = candidate
        search_assignments(
            marker_ids,
            candidates,
            pair_consensus,
            current,
            index + 1,
            holder,
            assignment_mode=assignment_mode,
        )
    current.pop(marker_id, None)

