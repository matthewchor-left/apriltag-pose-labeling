"""Behavior tests for IPPE assignment rejection diagnostics."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    FrameAssignmentRejection,
    FrameObservation,
    _MarkerCandidate,
    _PairConsensus,
    _collect_pair_hypotheses,
    _estimate_frame_candidates,
    _estimate_pair_consensus,
    _normalize_observations,
    calibrate_marker_layout,
    resolve_frame_ippe_assignment,
    summarize_assignment_rejections,
)
from object_apriltag.pose import marker_corner_object_points
from tests.test_marker_layout_calibration import (
    _default_camera,
    _pair_poses,
    _rotate_marker_corners,
    _synth_pair_with_corrupt_frames,
    synthesize_observations,
)


def _make_candidate(
    translation: np.ndarray,
    *,
    rotation: np.ndarray | None = None,
) -> _MarkerCandidate:
    rotation = np.eye(3) if rotation is None else rotation
    rvec, _ = cv2.Rodrigues(rotation)
    return _MarkerCandidate(
        rvec=rvec.reshape(3),
        tvec=translation.astype(np.float64),
        rotation=rotation.astype(np.float64),
        reprojection_rms_px=0.0,
    )


def _identity_consensus(
    marker_a: int,
    marker_b: int,
    translation_ba: np.ndarray,
    *,
    rotation_ba: np.ndarray | None = None,
) -> _PairConsensus:
    rotation = np.eye(3) if rotation_ba is None else rotation_ba
    return _PairConsensus(
        marker_a=marker_a,
        marker_b=marker_b,
        rotation_ba=rotation,
        translation_ba=translation_ba.astype(np.float64),
        inlier_frames=(0,),
        inlier_hypotheses={0: (rotation, translation_ba.astype(np.float64))},
    )


class FrameAssignmentRejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)
        self.expected_ids = [0, 1]

    def _frame_assignment_inputs(
        self,
        observations: list[FrameObservation],
        *,
        corrupt_frame_index: int | None = None,
    ):
        if corrupt_frame_index is not None:
            observations = [
                FrameObservation(frame_id=obs.frame_id, markers={k: v.copy() for k, v in obs.markers.items()})
                for obs in observations
            ]
            _rotate_marker_corners(observations, [corrupt_frame_index], marker_id=1)

        normalized = _normalize_observations(observations, self.expected_ids)
        object_points = marker_corner_object_points(self.marker_size_m).astype(np.float64)
        frame_candidates = _estimate_frame_candidates(
            normalized,
            object_points,
            self.camera_matrix,
            self.dist_coeffs,
        )
        pair_hypotheses = _collect_pair_hypotheses(frame_candidates, self.expected_ids)
        pair_consensus, pair_failure = _estimate_pair_consensus(
            pair_hypotheses,
            self.expected_ids,
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(pair_failure)
        return frame_candidates, pair_consensus

    def test_rejects_frame_with_translation_pair_conflict(self) -> None:
        observations = _synth_pair_with_corrupt_frames(25, frozenset({7}))
        frame_candidates, pair_consensus = self._frame_assignment_inputs(observations)
        corrupt_index = 7
        _, candidates = next(item for item in frame_candidates if item[0] == corrupt_index)

        result = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        self.assertIsNone(result.assignment)
        self.assertIsNotNone(result.rejection)
        assert result.rejection is not None
        self.assertEqual(result.rejection.reason, "translation_gate")
        self.assertEqual(result.rejection.marker_pair, (0, 1))
        self.assertIsNotNone(result.rejection.translation_error_m)
        self.assertIsNotNone(result.rejection.translation_gate_m)
        self.assertGreater(result.rejection.translation_error_m, result.rejection.translation_gate_m)

    def test_rejects_frame_with_rotation_pair_conflict(self) -> None:
        observations = synthesize_observations(
            _pair_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        frame_candidates, pair_consensus = self._frame_assignment_inputs(
            observations,
            corrupt_frame_index=10,
        )
        _, candidates = next(item for item in frame_candidates if item[0] == 10)

        result = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        self.assertIsNone(result.assignment)
        assert result.rejection is not None
        self.assertEqual(result.rejection.reason, "rotation_gate")
        self.assertEqual(result.rejection.marker_pair, (0, 1))
        self.assertIsNotNone(result.rejection.rotation_error_deg)
        self.assertIsNotNone(result.rejection.rotation_gate_deg)
        self.assertGreater(result.rejection.rotation_error_deg, result.rejection.rotation_gate_deg)

    def test_rejects_frame_with_no_constrained_pair(self) -> None:
        translation_01 = np.array([0.12, 0.0, -0.05], dtype=np.float64)
        candidates = {
            0: [_make_candidate(np.zeros(3))],
            2: [_make_candidate(np.array([0.24, 0.0, -0.10]))],
        }
        pair_consensus = {
            (0, 1): _identity_consensus(0, 1, translation_01),
        }

        result = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        self.assertIsNone(result.assignment)
        assert result.rejection is not None
        self.assertEqual(result.rejection.reason, "no_constrained_pair")
        self.assertIsNone(result.rejection.marker_pair)

    def test_accepts_consistent_frame_without_changing_assignment(self) -> None:
        observations = synthesize_observations(
            _pair_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        frame_candidates, pair_consensus = self._frame_assignment_inputs(observations)
        frame_index, candidates = frame_candidates[0]

        before = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )
        after = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        self.assertIsNotNone(before.assignment)
        self.assertIsNone(before.rejection)
        self.assertEqual(before.assignment, after.assignment)

        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=self.expected_ids,
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(result.failure_reason)
        assert result.quality is not None
        self.assertEqual(result.quality.rejected_frame_count, 0)
        assert result.quality.assignment_rejections is not None
        self.assertEqual(result.quality.assignment_rejections.total_rejected, 0)

    def test_chooses_worst_pair_deterministically_when_multiple_pairs_conflict(self) -> None:
        translation_01 = np.array([0.12, 0.0, -0.05], dtype=np.float64)
        translation_12 = np.array([0.12, 0.0, -0.05], dtype=np.float64)
        candidates = {
            0: [_make_candidate(np.zeros(3))],
            1: [_make_candidate(translation_01)],
            2: [_make_candidate(translation_01 + translation_12 + np.array([0.20, 0.0, 0.0]))],
        }
        pair_consensus = {
            (0, 1): _identity_consensus(0, 1, translation_01),
            (1, 2): _identity_consensus(1, 2, translation_12),
        }

        result = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        assert result.rejection is not None
        self.assertEqual(result.rejection.marker_pair, (1, 2))
        self.assertEqual(result.rejection.reason, "translation_gate")


class AssignmentRejectionAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def test_calibration_aggregates_rejection_causes_and_pairs(self) -> None:
        observations = _synth_pair_with_corrupt_frames(
            25,
            frozenset({2, 7, 11, 16, 22}),
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

        self.assertIsNone(result.failure_reason)
        assert result.quality is not None
        summary = result.quality.assignment_rejections
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary.total_rejected, 5)
        self.assertEqual(dict(summary.by_reason), {"translation_gate": 5})
        self.assertEqual(dict(summary.by_pair), {(0, 1): 5})
        self.assertEqual(len(summary.top_causes), 1)
        self.assertEqual(summary.top_causes[0].count, 5)

    def test_assignment_rejections_absent_when_assignment_has_not_run(self) -> None:
        observations = synthesize_observations(
            _pair_poses(self.marker_size_m),
            frame_count=19,
            marker_size_m=self.marker_size_m,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

        self.assertIsNone(result.layout)
        assert result.quality is not None
        self.assertIsNone(result.quality.assignment_rejections)

    def test_primary_reason_prefers_larger_normalized_gate_exceedance(self) -> None:
        translation_01 = np.array([0.12, 0.0, -0.05], dtype=np.float64)
        bad_rotation, _ = cv2.Rodrigues(np.array([0.0, 0.0, 2.5]))
        candidates = {
            0: [_make_candidate(np.zeros(3))],
            1: [
                _make_candidate(
                    translation_01 + np.array([0.05, 0.0, 0.0]),
                    rotation=bad_rotation,
                )
            ],
        }
        pair_consensus = {
            (0, 1): _identity_consensus(0, 1, translation_01),
        }

        result = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        assert result.rejection is not None
        self.assertEqual(result.rejection.marker_pair, (0, 1))
        self.assertEqual(result.rejection.reason, "rotation_gate")
        self.assertGreater(result.rejection.rotation_error_deg, result.rejection.rotation_gate_deg)
        self.assertGreater(result.rejection.translation_error_m, result.rejection.translation_gate_m)


class AssignmentSearchTraversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def test_rejected_frame_evaluates_each_complete_assignment_once(self) -> None:
        from math import prod
        from unittest import mock

        import object_apriltag.marker_layout_calibration as calibration

        translation_01 = np.array([0.12, 0.0, -0.05], dtype=np.float64)
        bad_translation = translation_01 + np.array([0.20, 0.0, 0.0], dtype=np.float64)
        candidates = {
            0: [_make_candidate(np.zeros(3)), _make_candidate(np.array([0.01, 0.0, 0.0]))],
            1: [
                _make_candidate(bad_translation),
                _make_candidate(bad_translation + np.array([0.01, 0.0, 0.0])),
            ],
        }
        pair_consensus = {(0, 1): _identity_consensus(0, 1, translation_01)}
        expected_evaluations = prod(len(options) for options in candidates.values())

        with mock.patch.object(
            calibration,
            "_evaluate_complete_assignment",
            wraps=calibration._evaluate_complete_assignment,
        ) as evaluate_mock:
            result = resolve_frame_ippe_assignment(
                candidates,
                pair_consensus,
                self.settings,
                self.marker_size_m,
            )

        self.assertIsNone(result.assignment)
        self.assertIsNotNone(result.rejection)
        self.assertEqual(evaluate_mock.call_count, expected_evaluations)

    def test_rejected_frames_record_one_rejection_per_rejected_frame(self) -> None:
        from object_apriltag.marker_layout_calibration import _assign_ippe_candidates

        observations = _synth_pair_with_corrupt_frames(25, frozenset({2, 7, 11, 16, 22}))
        normalized = _normalize_observations(observations, [0, 1])
        object_points = marker_corner_object_points(self.marker_size_m).astype(np.float64)
        camera_matrix, dist_coeffs = _default_camera()
        frame_candidates = _estimate_frame_candidates(
            normalized,
            object_points,
            camera_matrix,
            dist_coeffs,
        )
        pair_hypotheses = _collect_pair_hypotheses(frame_candidates, [0, 1])
        pair_consensus, pair_failure = _estimate_pair_consensus(
            pair_hypotheses,
            [0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(pair_failure)

        _, rejected_frames, rejections = _assign_ippe_candidates(
            frame_candidates,
            pair_consensus,
            self.settings,
            self.marker_size_m,
        )

        self.assertEqual(len(rejected_frames), len(rejections))
        self.assertEqual(len(rejected_frames), 5)


class AssignmentRejectionCliSummaryTests(unittest.TestCase):
    def test_format_assignment_rejection_summary_is_compact(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import format_assignment_rejection_summary

        summary = summarize_assignment_rejections(
            [
                FrameAssignmentRejection(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    translation_error_m=0.02,
                    translation_gate_m=0.007,
                ),
                FrameAssignmentRejection(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    translation_error_m=0.03,
                    translation_gate_m=0.007,
                ),
                FrameAssignmentRejection(
                    reason="rotation_gate",
                    marker_pair=(1, 2),
                    rotation_error_deg=12.0,
                    rotation_gate_deg=5.0,
                ),
                FrameAssignmentRejection(reason="no_constrained_pair"),
            ]
        )
        lines = format_assignment_rejection_summary(summary, max_lines=2)

        self.assertEqual(len(lines), 2)
        self.assertIn("translation_gate", lines[0])
        self.assertIn("(0,1)", lines[0])
        self.assertIn("x2", lines[0])

    def test_print_refusal_includes_assignment_rejection_summary(self) -> None:
        from io import StringIO
        from unittest import mock

        from object_apriltag.cli.calibrate_marker_model import print_refusal
        from object_apriltag.marker_layout_calibration import CalibrationQualityReport, CalibrationResult

        summary = summarize_assignment_rejections(
            [
                FrameAssignmentRejection(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    translation_error_m=0.02,
                    translation_gate_m=0.007,
                )
            ]
        )
        quality = CalibrationQualityReport(
            reprojection_rms_px=float("inf"),
            per_marker_reprojection_rms_px={},
            edges=(),
            pair_translation_rms_max_m=0.0,
            pair_rotation_rms_max_deg=0.0,
            frame_count=0,
            observation_count=0,
            inlier_corner_count=0,
            input_frame_count=25,
            rejected_frame_count=1,
            accepted_frame_count=0,
            connected_marker_ids=frozenset({0, 1}),
            missing_expected_ids=frozenset(),
            unused_expected_ids=frozenset(),
            assignment_rejections=summary,
        )
        buffer = StringIO()
        with mock.patch("builtins.print", side_effect=lambda *args, **kwargs: buffer.write(" ".join(str(a) for a in args) + "\n")):
            print_refusal(
                CalibrationResult(
                    layout=None,
                    quality=quality,
                    failure_reason="No frames with assignable IPPE candidates remain after rejecting inconsistent samples.",
                )
            )
        output = buffer.getvalue()
        self.assertIn("assignment rejections:", output)
        self.assertIn("translation_gate", output)
        self.assertIn("(0,1)", output)
