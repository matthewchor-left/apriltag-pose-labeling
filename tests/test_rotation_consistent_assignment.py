"""Tests for rotation-consistent IPPE frame assignment."""

from __future__ import annotations

import itertools
import unittest

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration.rotation_consistent_assignment import (
    _compatible_assignments,
    _pick_lowest_reprojection_assignment,
    _search_assignments,
    _select_lowest_reprojection_assignment,
    assign_frames_rotation_consistent,
)
from object_apriltag.marker_layout_calibration.solve_primitives import MarkerCandidate
from object_apriltag.marker_layout_calibration.types import CalibrationSettings


def _make_candidate(
    translation: np.ndarray,
    *,
    rotation: np.ndarray | None = None,
    reprojection_rms_px: float = 0.5,
) -> MarkerCandidate:
    rotation = np.eye(3) if rotation is None else rotation
    rvec, _ = cv2.Rodrigues(rotation)
    return MarkerCandidate(
        rvec=rvec.reshape(3),
        tvec=translation.astype(np.float64),
        rotation=rotation.astype(np.float64),
        reprojection_rms_px=reprojection_rms_px,
    )


def _flip_rotation() -> np.ndarray:
    return cv2.Rodrigues(np.array([np.pi, 0.0, 0.0], dtype=np.float64))[0]


class RotationConsistentAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = CalibrationSettings(
            min_inliers_per_edge=2,
            pair_rotation_rms_gate_deg=5.0,
        )

    def test_assignments_agree_on_shared_pair_rotation_across_frames(self) -> None:
        """Independent greedy consensus can flip branches; global assignment should not."""
        identity = np.eye(3)
        flipped = _flip_rotation()
        frame_candidates = [
            (
                0,
                {
                    0: [
                        _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=identity),
                        _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=flipped),
                    ],
                    1: [
                        _make_candidate(np.array([0.2, 0.0, 0.5]), rotation=identity),
                        _make_candidate(np.array([0.2, 0.0, 0.5]), rotation=flipped),
                    ],
                },
            ),
            (
                1,
                {
                    0: [
                        _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=identity),
                        _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=flipped),
                    ],
                    1: [
                        _make_candidate(np.array([0.2, 0.0, 0.5]), rotation=identity),
                        _make_candidate(np.array([0.2, 0.0, 0.5]), rotation=flipped, reprojection_rms_px=0.1),
                    ],
                },
            ),
        ]
        result = assign_frames_rotation_consistent(
            frame_candidates,
            reference_marker_id=0,
            settings=self.settings,
        )
        self.assertEqual(set(result.assigned), {0, 1})
        branch_frame_0 = result.assigned[0][0].rotation
        branch_frame_1 = result.assigned[1][0].rotation
        self.assertTrue(np.allclose(branch_frame_0, branch_frame_1, atol=1e-6))

    def test_rejects_reference_branch_outliers(self) -> None:
        from object_apriltag.marker_layout_calibration.rotation_consistent_assignment import (
            _filter_reference_branch_outliers,
        )

        identity = np.eye(3)
        flipped = _flip_rotation()
        candidates_by_frame = {
            0: {
                0: [
                    _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=identity),
                    _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=flipped),
                ],
            },
            1: {
                0: [
                    _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=identity),
                    _make_candidate(np.array([0.0, 0.0, 0.5]), rotation=flipped),
                ],
            },
        }
        assigned = {
            0: {0: candidates_by_frame[0][0][0]},
            1: {0: candidates_by_frame[1][0][1]},
        }
        filtered, rejected = _filter_reference_branch_outliers(
            assigned,
            candidates_by_frame,
            reference_marker_id=0,
        )
        self.assertEqual(set(filtered), {0})
        self.assertEqual(rejected, (1,))

    def test_rejects_frame_without_compatible_assignment(self) -> None:
        identity = np.eye(3)
        flipped = _flip_rotation()
        frame_candidates = [
            (
                0,
                {
                    0: [_make_candidate(np.array([0.0, 0.0, 0.5]), rotation=identity)],
                    1: [_make_candidate(np.array([0.2, 0.0, 0.5]), rotation=identity)],
                },
            ),
            (
                1,
                {
                    0: [_make_candidate(np.array([0.0, 0.0, 0.5]), rotation=flipped)],
                    1: [_make_candidate(np.array([0.2, 0.0, 0.5]), rotation=identity)],
                },
            ),
        ]
        result = assign_frames_rotation_consistent(
            frame_candidates,
            reference_marker_id=0,
            settings=self.settings,
        )
        self.assertIn(0, result.assigned)
        self.assertIn(1, result.rejected_frames)


def _enumerate_assignments_bruteforce(
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


def _assignment_signature(assignment: dict[int, MarkerCandidate]) -> tuple[tuple[int, float, float, float], ...]:
    return tuple(
        sorted(
            (
                marker_id,
                float(candidate.tvec[0]),
                float(candidate.tvec[1]),
                float(candidate.tvec[2]),
            )
            for marker_id, candidate in assignment.items()
        )
    )


class AssignmentSearchTests(unittest.TestCase):
    def _random_candidates(self, marker_count: int) -> dict[int, list[MarkerCandidate]]:
        rng = np.random.default_rng(7)
        candidates: dict[int, list[MarkerCandidate]] = {}
        for marker_id in range(marker_count):
            branch_candidates = []
            for branch in range(2):
                translation = rng.normal(size=3) * 0.05 + np.array([0.0, 0.0, 0.5])
                rotation = cv2.Rodrigues(rng.normal(size=3) * 0.2)[0]
                branch_candidates.append(
                    _make_candidate(
                        translation,
                        rotation=rotation,
                        reprojection_rms_px=float(branch + marker_id) * 0.1,
                    )
                )
            candidates[marker_id] = branch_candidates
        return candidates

    def test_search_matches_bruteforce_without_neighbors(self) -> None:
        candidates = self._random_candidates(marker_count=5)
        expected = _enumerate_assignments_bruteforce(candidates)
        actual = _search_assignments(candidates, [], rotation_gate=5.0)
        self.assertEqual(
            {_assignment_signature(assignment) for assignment in expected},
            {_assignment_signature(assignment) for assignment in actual},
        )

    def test_search_matches_bruteforce_with_neighbor_constraints(self) -> None:
        from object_apriltag.marker_layout_calibration.rotation_consistent_assignment import (
            _assignments_compatible,
        )

        candidates = self._random_candidates(marker_count=6)
        neighbor_assignment = _enumerate_assignments_bruteforce(candidates)[0]
        neighbors = [(0, neighbor_assignment)]
        expected = [
            assignment
            for assignment in _enumerate_assignments_bruteforce(candidates)
            if _assignments_compatible(assignment, neighbor_assignment, 5.0)
        ]
        actual = _search_assignments(candidates, neighbors, rotation_gate=5.0)
        self.assertEqual(
            {_assignment_signature(assignment) for assignment in expected},
            {_assignment_signature(assignment) for assignment in actual},
        )

    def test_lowest_reprojection_assignment_matches_bruteforce(self) -> None:
        candidates = self._random_candidates(marker_count=5)
        expected = _pick_lowest_reprojection_assignment(_enumerate_assignments_bruteforce(candidates))
        actual = _select_lowest_reprojection_assignment(candidates)
        assert expected is not None and actual is not None
        self.assertEqual(_assignment_signature(expected), _assignment_signature(actual))

    def test_compatible_assignments_wrapper_matches_search(self) -> None:
        candidates = self._random_candidates(marker_count=4)
        neighbor_assignment = _enumerate_assignments_bruteforce(candidates)[3]
        neighbors = [(2, neighbor_assignment)]
        self.assertEqual(
            _compatible_assignments(candidates, neighbors, rotation_gate=5.0),
            _search_assignments(candidates, neighbors, rotation_gate=5.0),
        )


if __name__ == "__main__":
    unittest.main()
