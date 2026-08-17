"""Rotation-consistent global IPPE assignment before pair consensus."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from object_apriltag.marker_layout_calibration.discrete_graph import transform_high_in_low
from object_apriltag.marker_layout_calibration.solve_primitives import (
    MarkerCandidate,
    rotation_geodesic_deg,
)
from object_apriltag.marker_layout_calibration.types import CalibrationSettings


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
