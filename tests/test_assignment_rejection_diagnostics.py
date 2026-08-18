"""Behavior tests for rotation-consistent assignment rejection diagnostics."""

from __future__ import annotations

import unittest

from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    FrameAssignmentRejection,
    FrameAssignmentRejectionRecord,
    FrameObservation,
    build_assignment_rejection_records,
    calibrate_marker_layout,
    summarize_assignment_rejection_records,
    summarize_assignment_rejections,
)
from tests.test_marker_layout_calibration import (
    _default_camera,
    _pair_poses,
    _rotate_marker_corners,
    synthesize_observations,
)


class AssignmentRejectionAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def test_calibration_aggregates_rotation_inconsistent_rejections(self) -> None:
        observations = synthesize_observations(
            _pair_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        _rotate_marker_corners(observations, [2, 7, 11], marker_id=1)
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
        self.assertEqual(summary.total_rejected, 3)
        self.assertEqual(dict(summary.by_reason), {"rotation_inconsistent": 3})
        self.assertEqual(dict(summary.by_pair), {})
        self.assertEqual(len(summary.top_causes), 1)
        self.assertEqual(summary.top_causes[0].count, 3)

    def test_assignment_rejections_recorded_when_frames_are_rejected(self) -> None:
        observations = synthesize_observations(
            _pair_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        _rotate_marker_corners(observations, [2, 7, 11], marker_id=1)
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

        self.assertIsNotNone(result.layout)
        assert result.quality is not None
        self.assertIsNotNone(result.quality.assignment_rejections)
        self.assertEqual(result.quality.assignment_rejections.total_rejected, 3)
        assert result.quality.assignment_rejection_records is not None
        self.assertEqual(len(result.quality.assignment_rejection_records), 3)


class AssignmentRejectionRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def test_rejected_frames_preserve_identity_in_quality_report(self) -> None:
        observations = [
            FrameObservation(frame_id=f"capture-{index}", markers=obs.markers)
            for index, obs in enumerate(
                synthesize_observations(
                    _pair_poses(self.marker_size_m),
                    frame_count=25,
                    marker_size_m=self.marker_size_m,
                )
            )
        ]
        _rotate_marker_corners(observations, [2, 7, 11], marker_id=1)
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
        records = result.quality.assignment_rejection_records
        self.assertIsNotNone(records)
        assert records is not None
        self.assertEqual(len(records), 3)
        corrupt_records = [record for record in records if record.frame_index in {2, 7, 11}]
        self.assertEqual(len(corrupt_records), 3)
        for record in corrupt_records:
            self.assertEqual(record.frame_id, f"capture-{record.frame_index}")
            self.assertEqual(record.visible_marker_ids, (0, 1))
            self.assertEqual(record.reason, "rotation_inconsistent")
            self.assertIsNone(record.marker_pair)


class AssignmentRejectionDistributionTests(unittest.TestCase):
    def test_summarize_reports_deterministic_error_distributions(self) -> None:
        records = (
            FrameAssignmentRejectionRecord(
                frame_index=0,
                frame_id="a",
                visible_marker_ids=(0, 1),
                reason="translation_gate",
                marker_pair=(0, 1),
                translation_error_m=0.02,
                rotation_error_deg=1.0,
                translation_gate_m=0.007,
                rotation_gate_deg=5.0,
            ),
            FrameAssignmentRejectionRecord(
                frame_index=1,
                frame_id="b",
                visible_marker_ids=(0, 1),
                reason="translation_gate",
                marker_pair=(0, 1),
                translation_error_m=0.03,
                rotation_error_deg=2.0,
                translation_gate_m=0.007,
                rotation_gate_deg=5.0,
            ),
            FrameAssignmentRejectionRecord(
                frame_index=2,
                frame_id="c",
                visible_marker_ids=(0, 1),
                reason="translation_gate",
                marker_pair=(0, 1),
                translation_error_m=0.04,
                rotation_error_deg=3.0,
                translation_gate_m=0.007,
                rotation_gate_deg=5.0,
            ),
        )
        summary = summarize_assignment_rejection_records(records)
        self.assertEqual(summary.total_rejected, 3)
        self.assertEqual(len(summary.by_cause), 1)
        cause = summary.by_cause[0]
        self.assertEqual(cause.count, 3)
        self.assertEqual(cause.sample_frame_ids, ("a", "b", "c"))
        assert cause.translation_error_m is not None
        self.assertAlmostEqual(cause.translation_error_m.min, 0.02)
        self.assertAlmostEqual(cause.translation_error_m.median, 0.03)
        self.assertAlmostEqual(cause.translation_error_m.max, 0.04)
        assert cause.translation_error_ratio is not None
        self.assertAlmostEqual(cause.translation_error_ratio.min, 0.02 / 0.007, places=5)
        self.assertAlmostEqual(cause.translation_error_ratio.median, 0.03 / 0.007, places=5)

    def test_build_records_from_assignment_results(self) -> None:
        from object_apriltag.marker_layout_calibration.discrete_graph import normalize_observations

        observations = synthesize_observations(
            _pair_poses(0.07),
            frame_count=5,
            marker_size_m=0.07,
        )
        normalized = normalize_observations(observations, [0, 1])
        rejections = (
            FrameAssignmentRejection(
                reason="rotation_inconsistent",
            ),
        )
        records = build_assignment_rejection_records(normalized, (1,), rejections)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].frame_index, 1)
        self.assertEqual(records[0].frame_id, 1)
        self.assertEqual(records[0].visible_marker_ids, (0, 1))

    def test_summarize_assignment_rejections_accepts_legacy_rejection_inputs(self) -> None:
        summary = summarize_assignment_rejections(
            [
                FrameAssignmentRejection(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    translation_error_m=0.02,
                    translation_gate_m=0.007,
                ),
                FrameAssignmentRejection(
                    reason="rotation_gate",
                    marker_pair=(1, 2),
                    rotation_error_deg=12.0,
                    rotation_gate_deg=5.0,
                ),
            ]
        )
        self.assertEqual(summary.total_rejected, 2)
        self.assertEqual(dict(summary.by_reason), {"rotation_gate": 1, "translation_gate": 1})
        self.assertEqual(summary.by_cause, ())

    def test_summarize_assignment_rejections_rejects_record_inputs(self) -> None:
        with self.assertRaises(TypeError):
            summarize_assignment_rejections(
                [
                    FrameAssignmentRejectionRecord(
                        frame_index=0,
                        frame_id=0,
                        visible_marker_ids=(0, 1),
                        reason="translation_gate",
                    )
                ]
            )

    def test_summarize_assignment_rejection_records_rejects_legacy_inputs(self) -> None:
        with self.assertRaises(TypeError):
            summarize_assignment_rejection_records(
                [FrameAssignmentRejection(reason="translation_gate")]
            )
