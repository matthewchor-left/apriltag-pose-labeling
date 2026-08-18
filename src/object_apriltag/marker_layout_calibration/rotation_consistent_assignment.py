"""Rotation-consistent global IPPE assignment before pair consensus."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from object_apriltag.marker_layout_calibration.discrete_graph import transform_high_in_low
from object_apriltag.marker_layout_calibration.solve_primitives import (
    MarkerCandidate,
    MarkerPair,
    rotation_geodesic_deg,
)
from object_apriltag.marker_layout_calibration.types import (
    AssignmentRejectionCauseCount,
    AssignmentRejectionCauseStats,
    AssignmentRejectionSummary,
    CalibrationSettings,
    FrameAssignmentRejection,
    FrameAssignmentRejectionRecord,
    MeasurementDistribution,
)


@dataclass(frozen=True)
class RotationConsistentAssignmentResult:
    """Outcome of rotation-consistent frame assignment."""

    assigned: dict[int, dict[int, MarkerCandidate]]
    rejected_frames: tuple[int, ...]
    seed_frame_index: int | None


def assign_frames_rotation_consistent(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
    reference_marker_id: int,
    settings: CalibrationSettings,
) -> RotationConsistentAssignmentResult:
    """Assign one IPPE branch per marker per frame with cross-frame rotation agreement.

    Args:
        frame_candidates: Per-frame IPPE candidate pools.
        reference_marker_id: Reference marker used for branch majority filtering.
        settings: Calibration gates; ``pair_rotation_rms_gate_deg`` sets cross-frame
            shared-pair agreement.

    Returns:
        Accepted per-frame assignments and rejected frame indices.
    """
    if not frame_candidates:
        return RotationConsistentAssignmentResult({}, (), None)

    candidates_by_frame = {
        frame_index: candidates for frame_index, candidates in frame_candidates
    }
    rotation_gate = settings.pair_rotation_rms_gate_deg
    seed_frame_index = _select_seed_frame_index(frame_candidates)
    if seed_frame_index is None:
        rejected = tuple(frame_index for frame_index, _ in frame_candidates)
        return RotationConsistentAssignmentResult({}, rejected, None)

    seed_candidates = candidates_by_frame[seed_frame_index]
    seed_assignment = _select_lowest_reprojection_assignment(seed_candidates)
    if seed_assignment is None:
        rejected = tuple(frame_index for frame_index, _ in frame_candidates)
        return RotationConsistentAssignmentResult({}, rejected, seed_frame_index)

    assigned: dict[int, dict[int, MarkerCandidate]] = {seed_frame_index: seed_assignment}
    remaining = [
        frame_index
        for frame_index, candidates in frame_candidates
        if frame_index != seed_frame_index and len(candidates) >= 2
    ]
    remaining.sort(
        key=lambda frame_index: (
            -_shared_marker_count(seed_assignment, candidates_by_frame[frame_index]),
            frame_index,
        )
    )

    for frame_index in remaining:
        neighbors = [
            (neighbor_index, neighbor_assignment)
            for neighbor_index, neighbor_assignment in assigned.items()
            if _shared_marker_count(neighbor_assignment, candidates_by_frame[frame_index]) >= 2
        ]
        if not neighbors:
            neighbors = [
                (neighbor_index, neighbor_assignment)
                for neighbor_index, neighbor_assignment in assigned.items()
                if _shared_marker_ids(neighbor_assignment, candidates_by_frame[frame_index])
            ]
        compatible = _compatible_assignments(
            candidates_by_frame[frame_index],
            neighbors,
            rotation_gate,
        )
        if not compatible:
            continue
        assigned[frame_index] = _pick_lowest_reprojection_assignment(compatible)

    assigned, reference_rejected = _filter_reference_branch_outliers(
        assigned,
        candidates_by_frame,
        reference_marker_id,
    )
    assigned_indices = set(assigned)
    rejected_frames = sorted(
        frame_index
        for frame_index, candidates in frame_candidates
        if frame_index not in assigned_indices
    )
    rejected_frames = tuple(sorted(set(rejected_frames) | set(reference_rejected)))
    return RotationConsistentAssignmentResult(
        assigned=assigned,
        rejected_frames=rejected_frames,
        seed_frame_index=seed_frame_index,
    )


def _select_seed_frame_index(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
) -> int | None:
    best_frame: int | None = None
    best_marker_count = -1
    best_mean_rms = float("inf")
    for frame_index, candidates in frame_candidates:
        marker_count = len(candidates)
        if marker_count < 2:
            continue
        mean_rms = float(
            np.mean([candidate.reprojection_rms_px for group in candidates.values() for candidate in group])
        )
        if marker_count > best_marker_count or (
            marker_count == best_marker_count and mean_rms < best_mean_rms
        ):
            best_frame = frame_index
            best_marker_count = marker_count
            best_mean_rms = mean_rms
    return best_frame


def _enumerate_assignments(
    candidates: dict[int, list[MarkerCandidate]],
) -> list[dict[int, MarkerCandidate]]:
    marker_ids = sorted(candidates)
    if len(marker_ids) < 2:
        return []
    ranges = [range(len(candidates[marker_id])) for marker_id in marker_ids]
    assignments: list[dict[int, MarkerCandidate]] = []
    for indices in itertools.product(*ranges):
        assignments.append(
            {
                marker_id: candidates[marker_id][candidate_index]
                for marker_id, candidate_index in zip(marker_ids, indices, strict=True)
            }
        )
    return assignments


def _mean_reprojection_rms(assignment: dict[int, MarkerCandidate]) -> float:
    return float(np.mean([candidate.reprojection_rms_px for candidate in assignment.values()]))


def _select_lowest_reprojection_assignment(
    candidates: dict[int, list[MarkerCandidate]],
) -> dict[int, MarkerCandidate] | None:
    compatible = _enumerate_assignments(candidates)
    return _pick_lowest_reprojection_assignment(compatible)


def _pick_lowest_reprojection_assignment(
    assignments: list[dict[int, MarkerCandidate]],
) -> dict[int, MarkerCandidate] | None:
    if not assignments:
        return None
    return min(assignments, key=_mean_reprojection_rms)


def _shared_marker_ids(
    assignment: dict[int, MarkerCandidate],
    candidates: dict[int, list[MarkerCandidate]],
) -> frozenset[int]:
    return frozenset(assignment) & frozenset(candidates)


def _shared_marker_count(
    assignment: dict[int, MarkerCandidate],
    candidates: dict[int, list[MarkerCandidate]],
) -> int:
    return len(_shared_marker_ids(assignment, candidates))


def _assignments_compatible(
    left: dict[int, MarkerCandidate],
    right: dict[int, MarkerCandidate],
    rotation_gate: float,
) -> bool:
    shared = sorted(_shared_marker_ids(left, right))
    for index_a, marker_low in enumerate(shared):
        for marker_high in shared[index_a + 1 :]:
            rotation_left, _ = transform_high_in_low(left[marker_low], left[marker_high])
            rotation_right, _ = transform_high_in_low(right[marker_low], right[marker_high])
            if rotation_geodesic_deg(rotation_left, rotation_right) > rotation_gate:
                return False
    return True


def _compatible_assignments(
    candidates: dict[int, list[MarkerCandidate]],
    neighbors: list[tuple[int, dict[int, MarkerCandidate]]],
    rotation_gate: float,
) -> list[dict[int, MarkerCandidate]]:
    if not neighbors:
        return _enumerate_assignments(candidates)
    compatible: list[dict[int, MarkerCandidate]] = []
    for assignment in _enumerate_assignments(candidates):
        if all(
            _assignments_compatible(assignment, neighbor_assignment, rotation_gate)
            for _, neighbor_assignment in neighbors
        ):
            compatible.append(assignment)
    return compatible


def _candidate_branch_index(
    candidates: list[MarkerCandidate],
    chosen: MarkerCandidate,
) -> int:
    for index, candidate in enumerate(candidates):
        if (
            np.allclose(candidate.rotation, chosen.rotation, atol=1e-9, rtol=1e-9)
            and np.allclose(candidate.tvec, chosen.tvec, atol=1e-9, rtol=1e-9)
        ):
            return index
    raise ValueError("Chosen IPPE candidate is not present in the candidate list.")


def _filter_reference_branch_outliers(
    assigned: dict[int, dict[int, MarkerCandidate]],
    candidates_by_frame: dict[int, dict[int, list[MarkerCandidate]]],
    reference_marker_id: int,
) -> tuple[dict[int, dict[int, MarkerCandidate]], tuple[int, ...]]:
    votes = {0: 0, 1: 0}
    for frame_index, assignment in assigned.items():
        if reference_marker_id not in assignment:
            continue
        candidates = candidates_by_frame[frame_index][reference_marker_id]
        if len(candidates) < 2:
            continue
        branch = _candidate_branch_index(candidates, assignment[reference_marker_id])
        votes[branch] += 1
    if votes[0] + votes[1] == 0:
        return assigned, ()

    majority_branch = 0 if votes[0] >= votes[1] else 1
    filtered: dict[int, dict[int, MarkerCandidate]] = {}
    rejected: list[int] = []
    for frame_index, assignment in assigned.items():
        if reference_marker_id not in assignment:
            filtered[frame_index] = assignment
            continue
        candidates = candidates_by_frame[frame_index][reference_marker_id]
        branch = _candidate_branch_index(candidates, assignment[reference_marker_id])
        if branch == majority_branch:
            filtered[frame_index] = assignment
        else:
            rejected.append(frame_index)
    return filtered, tuple(rejected)

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


_ASSIGNMENT_REJECTION_SAMPLE_FRAME_IDS = 10


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
        translation_errors = [v for v in (r.translation_error_m for r in group) if v is not None]
        rotation_errors = [v for v in (r.rotation_error_deg for r in group) if v is not None]
        translation_gate = next((r.translation_gate_m for r in group if r.translation_gate_m is not None), None)
        rotation_gate = next((r.rotation_gate_deg for r in group if r.rotation_gate_deg is not None), None)
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

