"""Tests for rotation-consistent IPPE frame assignment."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration.rotation_consistent_assignment import (
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


if __name__ == "__main__":
    unittest.main()
