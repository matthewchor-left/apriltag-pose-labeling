"""Tests for marker calibration CLI diagnostics formatting and JSON export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from object_apriltag.cli.calibration_diagnostics import (
    build_calibration_diagnostics_document,
    format_assignment_rejection_cause_detail,
    format_dropped_pair_edge,
    format_quality_diagnostics_lines,
    format_reprojection_rms_px,
    save_calibration_diagnostics,
    serialize_calibration_diagnostics_document,
)
from object_apriltag.marker_layout_calibration import (
    AssignmentRejectionCauseCount,
    AssignmentRejectionCauseStats,
    AssignmentRejectionSummary,
    CalibrationQualityReport,
    CalibrationResult,
    DroppedPairEdge,
    EdgeDiagnostics,
    FrameAssignmentRejectionRecord,
    MeasurementDistribution,
)


def _distribution(**values: float) -> MeasurementDistribution:
    return MeasurementDistribution(
        min=values.get("min"),
        median=values.get("median"),
        p95=values.get("p95"),
        max=values.get("max"),
    )


def _quality_report(**overrides: object) -> CalibrationQualityReport:
    values: dict[str, object] = {
        "reprojection_rms_px": 0.42,
        "per_marker_reprojection_rms_px": {0: 0.4, 1: 0.44},
        "edges": (
            EdgeDiagnostics(
                marker_a=0,
                marker_b=1,
                inlier_count=25,
                translation_rms_m=0.01,
                rotation_rms_deg=1.2,
            ),
        ),
        "pair_translation_rms_max_m": 0.01,
        "pair_rotation_rms_max_deg": 1.2,
        "frame_count": 20,
        "observation_count": 160,
        "inlier_corner_count": 160,
        "input_frame_count": 25,
        "rejected_frame_count": 5,
        "accepted_frame_count": 20,
        "connected_marker_ids": frozenset({0, 1}),
        "missing_expected_ids": frozenset(),
        "unused_expected_ids": frozenset(),
        "assignment_rejections": None,
        "assignment_rejection_records": None,
        "dropped_pair_edges": None,
    }
    values.update(overrides)
    return CalibrationQualityReport(**values)


class ReprojectionRmsFormattingTests(unittest.TestCase):
    def test_finite_reprojection_rms_keeps_px_format(self) -> None:
        self.assertEqual(format_reprojection_rms_px(0.42), "0.420 px")
        self.assertEqual(format_reprojection_rms_px(2.0), "2.000 px")

    def test_non_finite_reprojection_rms_uses_na(self) -> None:
        self.assertEqual(format_reprojection_rms_px(float("inf")), "N/A")
        self.assertEqual(format_reprojection_rms_px(float("nan")), "N/A")


class DiagnosticsSafeFloatFormattingTests(unittest.TestCase):
    def test_accepted_pair_edge_uses_na_for_non_finite_rms(self) -> None:
        lines = format_quality_diagnostics_lines(
            _quality_report(
                edges=(
                    EdgeDiagnostics(
                        marker_a=0,
                        marker_b=1,
                        inlier_count=20,
                        translation_rms_m=float("nan"),
                        rotation_rms_deg=float("inf"),
                    ),
                ),
            )
        )
        pair_line = next(line for line in lines if line.startswith("pair "))
        self.assertIn("trans_rms=N/A m", pair_line)
        self.assertIn("rot_rms=N/A deg", pair_line)
        self.assertNotIn("nan", pair_line.lower())
        self.assertNotIn("inf", pair_line.lower())

    def test_dropped_edge_uses_na_for_non_finite_rms_and_gates(self) -> None:
        line = format_dropped_pair_edge(
            DroppedPairEdge(
                marker_a=0,
                marker_b=2,
                stage="initial_consensus",
                reason="translation_rms_gate",
                observed_count=22,
                supported_count=22,
                required_count=20,
                translation_rms_m=float("inf"),
                rotation_rms_deg=float("nan"),
                translation_gate_m=float("inf"),
                rotation_gate_deg=5.0,
            )
        )
        self.assertIn("tr_rms=N/Am", line)
        self.assertIn("rot_rms=N/Adeg", line)
        self.assertIn("tr_gate=N/Am", line)
        self.assertIn("rot_gate=5.0deg", line)
        self.assertNotIn("nan", line.lower())
        self.assertNotIn("inf", line.lower())

    def test_distribution_uses_na_for_non_finite_members(self) -> None:
        line = format_assignment_rejection_cause_detail(
            AssignmentRejectionCauseStats(
                reason="translation_gate",
                marker_pair=(0, 1),
                count=1,
                sample_frame_ids=(0,),
                translation_error_m=_distribution(
                    min=0.02,
                    median=float("nan"),
                    p95=0.04,
                    max=float("inf"),
                ),
                rotation_error_deg=None,
                translation_gate_m=None,
                rotation_gate_deg=None,
                translation_error_ratio=None,
                rotation_error_ratio=None,
            )
        )
        self.assertIn("tr_err_m min/med/p95/max=0.020/N/A/0.040/N/A", line)
        self.assertNotIn("nan", line.lower())
        self.assertNotIn("inf", line.lower())


class AssignmentRejectionDetailFormattingTests(unittest.TestCase):
    def test_cause_detail_includes_full_distribution_and_sample_frames(self) -> None:
        cause = AssignmentRejectionCauseStats(
            reason="translation_gate",
            marker_pair=(0, 1),
            count=3,
            sample_frame_ids=("frame-2", "frame-7", "frame-11"),
            translation_error_m=_distribution(min=0.02, median=0.03, p95=0.039, max=0.04),
            rotation_error_deg=_distribution(min=1.0, median=2.0, p95=2.9, max=3.0),
            translation_gate_m=0.007,
            rotation_gate_deg=5.0,
            translation_error_ratio=_distribution(min=2.8, median=4.2, p95=5.5, max=5.7),
            rotation_error_ratio=None,
        )
        line = format_assignment_rejection_cause_detail(cause)
        self.assertIn("translation_gate", line)
        self.assertIn("pair=(0,1)", line)
        self.assertIn("x3", line)
        self.assertIn("tr_err_m min/med/p95/max=0.020/0.030/0.039/0.040", line)
        self.assertIn("rot_err_deg min/med/p95/max=1.0/2.0/2.9/3.0", line)
        self.assertIn("tr_gate_m=0.007", line)
        self.assertIn("rot_gate_deg=5.0", line)
        self.assertIn("tr_ratio min/med/p95/max=2.800/4.200/5.500/5.700", line)
        self.assertIn("sample_frames=[frame-2, frame-7, frame-11]", line)

    def test_quality_diagnostics_prints_every_cause_not_only_top_three(self) -> None:
        summary = AssignmentRejectionSummary(
            total_rejected=4,
            by_reason=(("translation_gate", 2), ("rotation_gate", 2)),
            by_pair=(((0, 1), 2), ((1, 2), 2)),
            top_causes=(
                AssignmentRejectionCauseCount(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    count=2,
                ),
            ),
            by_cause=(
                AssignmentRejectionCauseStats(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    count=2,
                    sample_frame_ids=(0, 1),
                    translation_error_m=None,
                    rotation_error_deg=None,
                    translation_gate_m=None,
                    rotation_gate_deg=None,
                    translation_error_ratio=None,
                    rotation_error_ratio=None,
                ),
                AssignmentRejectionCauseStats(
                    reason="rotation_gate",
                    marker_pair=(1, 2),
                    count=2,
                    sample_frame_ids=(2, 3),
                    translation_error_m=None,
                    rotation_error_deg=None,
                    translation_gate_m=None,
                    rotation_gate_deg=None,
                    translation_error_ratio=None,
                    rotation_error_ratio=None,
                ),
            ),
        )
        lines = format_quality_diagnostics_lines(
            _quality_report(assignment_rejections=summary),
        )
        cause_lines = [
            line
            for line in lines
            if line.startswith("assignment ") and not line.startswith("assignment rejections:")
        ]
        self.assertEqual(len(cause_lines), 2)
        self.assertIn("translation_gate", cause_lines[0])
        self.assertIn("rotation_gate", cause_lines[1])


class DroppedPairEdgeFormattingTests(unittest.TestCase):
    def test_dropped_edge_line_includes_stage_support_and_gates(self) -> None:
        edge = DroppedPairEdge(
            marker_a=0,
            marker_b=2,
            stage="initial_consensus",
            reason="insufficient_support",
            observed_count=15,
            supported_count=5,
            required_count=20,
            translation_rms_m=0.012,
            rotation_rms_deg=3.2,
            translation_gate_m=0.007,
            rotation_gate_deg=5.0,
        )
        line = format_dropped_pair_edge(edge)
        self.assertIn("dropped pair (0,2)", line)
        self.assertIn("stage=initial_consensus", line)
        self.assertIn("reason=insufficient_support", line)
        self.assertIn("support=5/20", line)
        self.assertIn("observed=15", line)
        self.assertIn("tr_rms=0.012m", line)
        self.assertIn("rot_rms=3.2deg", line)
        self.assertIn("tr_gate=0.007m", line)
        self.assertIn("rot_gate=5.0deg", line)

    def test_empty_dropped_edge_list_prints_no_dropped_lines(self) -> None:
        lines = format_quality_diagnostics_lines(_quality_report(dropped_pair_edges=()))
        self.assertFalse(any(line.startswith("dropped pair") for line in lines))

    def test_quality_diagnostics_prints_every_dropped_edge(self) -> None:
        edges = (
            DroppedPairEdge(
                marker_a=0,
                marker_b=2,
                stage="assignment_support",
                reason="insufficient_support",
                observed_count=10,
                supported_count=4,
                required_count=20,
            ),
            DroppedPairEdge(
                marker_a=1,
                marker_b=3,
                stage="post_pruning",
                reason="translation_rms_gate",
                observed_count=22,
                supported_count=22,
                required_count=20,
                translation_rms_m=0.02,
                translation_gate_m=0.007,
            ),
        )
        lines = format_quality_diagnostics_lines(_quality_report(dropped_pair_edges=edges))
        dropped_lines = [line for line in lines if line.startswith("dropped pair")]
        self.assertEqual(len(dropped_lines), 2)


class CalibrationDiagnosticsDocumentTests(unittest.TestCase):
    TOP_LEVEL_KEYS = (
        "version",
        "succeeded",
        "failure_reason",
        "calibration_policy",
        "outcome",
        "failed_quality_gates",
        "quality",
        "assignment_rejections",
        "assignment_rejection_records",
        "dropped_pair_edges",
        "anchor_core",
    )
    QUALITY_KEYS = (
        "reprojection_rms_px",
        "per_marker_reprojection_rms_px",
        "edges",
        "pair_translation_rms_max_m",
        "pair_rotation_rms_max_deg",
        "frame_count",
        "observation_count",
        "inlier_corner_count",
        "input_frame_count",
        "rejected_frame_count",
        "accepted_frame_count",
        "connected_marker_ids",
        "missing_expected_ids",
        "unused_expected_ids",
    )
    RECORD_KEYS = (
        "frame_index",
        "frame_id",
        "visible_marker_ids",
        "reason",
        "marker_pair",
        "translation_error_m",
        "rotation_error_deg",
        "translation_gate_m",
        "rotation_gate_deg",
    )
    DROPPED_EDGE_KEYS = (
        "marker_a",
        "marker_b",
        "stage",
        "reason",
        "observed_count",
        "supported_count",
        "required_count",
        "translation_rms_m",
        "rotation_rms_deg",
        "translation_gate_m",
        "rotation_gate_deg",
    )

    def test_unrun_stages_serialize_as_null_not_empty_lists(self) -> None:
        document = build_calibration_diagnostics_document(
            _quality_report(
                assignment_rejection_records=None,
                dropped_pair_edges=None,
            ),
            succeeded=False,
            failure_reason="too few frames",
        )
        payload = json.loads(serialize_calibration_diagnostics_document(document))
        self.assertIsNone(payload["assignment_rejection_records"])
        self.assertIsNone(payload["dropped_pair_edges"])

    def test_run_stages_with_no_issues_serialize_as_empty_lists(self) -> None:
        document = build_calibration_diagnostics_document(
            _quality_report(
                assignment_rejections=AssignmentRejectionSummary(
                    total_rejected=0,
                    by_reason=(),
                    by_pair=(),
                    top_causes=(),
                    by_cause=(),
                ),
                assignment_rejection_records=(),
                dropped_pair_edges=(),
            ),
            succeeded=True,
            failure_reason=None,
        )
        payload = json.loads(serialize_calibration_diagnostics_document(document))
        self.assertEqual(payload["assignment_rejection_records"], [])
        self.assertEqual(payload["dropped_pair_edges"], [])

    def test_document_serializes_complete_records_with_null_non_finite(self) -> None:
        summary = AssignmentRejectionSummary(
            total_rejected=1,
            by_reason=(("translation_gate", 1),),
            by_pair=(((0, 1), 1),),
            top_causes=(),
            by_cause=(
                AssignmentRejectionCauseStats(
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    count=1,
                    sample_frame_ids=("capture-1",),
                    translation_error_m=_distribution(min=0.02, median=0.02, p95=0.02, max=0.02),
                    rotation_error_deg=None,
                    translation_gate_m=0.007,
                    rotation_gate_deg=5.0,
                    translation_error_ratio=_distribution(min=2.8, median=2.8, p95=2.8, max=2.8),
                    rotation_error_ratio=None,
                ),
            ),
        )
        records = (
            FrameAssignmentRejectionRecord(
                frame_index=1,
                frame_id="capture-1",
                visible_marker_ids=(0, 1),
                reason="translation_gate",
                marker_pair=(0, 1),
                translation_error_m=0.02,
                rotation_error_deg=float("nan"),
                translation_gate_m=0.007,
                rotation_gate_deg=5.0,
            ),
        )
        quality = _quality_report(
            reprojection_rms_px=float("inf"),
            assignment_rejections=summary,
            assignment_rejection_records=records,
            dropped_pair_edges=(
                DroppedPairEdge(
                    marker_a=0,
                    marker_b=2,
                    stage="initial_consensus",
                    reason="insufficient_support",
                    observed_count=10,
                    supported_count=4,
                    required_count=20,
                    translation_rms_m=float("inf"),
                ),
            ),
        )
        document = build_calibration_diagnostics_document(
            quality,
            succeeded=False,
            failure_reason="refused",
        )
        text = serialize_calibration_diagnostics_document(document)
        payload = json.loads(text)
        self.assertEqual(list(payload.keys()), list(self.TOP_LEVEL_KEYS))
        self.assertEqual(list(payload["quality"].keys()), list(self.QUALITY_KEYS))
        self.assertEqual(list(payload["assignment_rejection_records"][0].keys()), list(self.RECORD_KEYS))
        self.assertEqual(list(payload["dropped_pair_edges"][0].keys()), list(self.DROPPED_EDGE_KEYS))
        self.assertEqual(payload["version"], 3)
        self.assertFalse(payload["succeeded"])
        self.assertEqual(payload["failure_reason"], "refused")
        self.assertIsNone(payload["quality"]["reprojection_rms_px"])
        self.assertIsNone(payload["assignment_rejection_records"][0]["rotation_error_deg"])
        self.assertIsNone(payload["dropped_pair_edges"][0]["translation_rms_m"])
        self.assertEqual(payload["assignment_rejection_records"][0]["frame_index"], 1)
        self.assertEqual(payload["assignment_rejections"]["by_cause"][0]["count"], 1)
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_document_key_order_is_deterministic(self) -> None:
        document = build_calibration_diagnostics_document(
            _quality_report(),
            succeeded=True,
            failure_reason=None,
        )
        first = serialize_calibration_diagnostics_document(document)
        second = serialize_calibration_diagnostics_document(document)
        self.assertEqual(first, second)
        self.assertTrue(first.index('"version"') < first.index('"succeeded"') < first.index('"quality"'))

    def test_atomic_write_reports_path_and_preserves_existing_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "diagnostics.json"
            existing = {"keep": True}
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(existing), encoding="utf-8")
            result = CalibrationResult(
                layout=None,
                quality=_quality_report(),
                failure_reason="refused",
            )

            def broken_serialize(_document: dict[str, object]) -> str:
                raise OSError("disk full")

            with self.assertRaises(RuntimeError):
                save_calibration_diagnostics(path, result, serialize_fn=broken_serialize)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), existing)

            save_calibration_diagnostics(path, result)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(payload["succeeded"])
            self.assertEqual(payload["failure_reason"], "refused")

    def test_provisional_result_records_policy_outcome_and_failed_gates(self) -> None:
        failed_gates = (
            "Global reprojection RMS 0.500 px exceeds 0.150 px gate.",
            "Pair rotation RMS 6.00 deg exceeds 5.00 deg gate.",
        )
        result = CalibrationResult(
            layout=mock.Mock(),
            quality=_quality_report(reprojection_rms_px=0.5),
            failure_reason=None,
            outcome="provisional",
            calibration_policy="best_effort",
            failed_quality_gates=failed_gates,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "diagnostics.json"
            save_calibration_diagnostics(path, result)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["succeeded"])
        self.assertIsNone(payload["failure_reason"])
        self.assertEqual(payload["calibration_policy"], "best_effort")
        self.assertEqual(payload["outcome"], "provisional")
        self.assertEqual(payload["failed_quality_gates"], list(failed_gates))


if __name__ == "__main__":
    unittest.main()
