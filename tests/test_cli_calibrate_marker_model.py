"""CLI tests for live marker layout calibration and inspector."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = REPO_ROOT / "config/Camera/nexplaygroundcam/intrinsics.json"
MARKER_MODEL = REPO_ROOT / "config/Model/remote1/marker_model.json"


def _run_cli_help(command: str) -> str:
    result = subprocess.run(
        ["uv", "run", command, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _write_intrinsics(path: Path, *, width: int = 640, height: int = 480) -> None:
    payload = {
        "calibration_source": "test",
        "image_size": [width, height],
        "camera_matrix": [[900.0, 0.0, width / 2.0], [0.0, 900.0, height / 2.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _marker_corners(marker_id: int) -> np.ndarray:
    base = 120.0 + marker_id * 40.0
    return np.array(
        [
            [base, base],
            [base + 30.0, base],
            [base + 30.0, base + 30.0],
            [base, base + 30.0],
        ],
        dtype=np.float64,
    )


def _quality_report_mock(**overrides: object) -> mock.Mock:
    values: dict[str, object] = {
        "frame_count": 20,
        "inlier_corner_count": 160,
        "reprojection_rms_px": 0.1,
        "connected_marker_ids": {0, 1},
        "missing_expected_ids": frozenset(),
        "edges": [],
        "input_frame_count": 25,
        "accepted_frame_count": 20,
        "rejected_frame_count": 5,
        "assignment_rejections": None,
        "dropped_pair_edges": None,
        "anchor_core": None,
    }
    values.update(overrides)
    return mock.Mock(**values)


class CliHelpTests(unittest.TestCase):
    def test_calibrate_marker_model_help_lists_controls_and_sampling(self) -> None:
        help_text = _run_cli_help("object-calibrate-marker-model")
        self.assertIn("--marker-ids", help_text)
        self.assertIn("--anchor-marker-ids", help_text)
        self.assertIn("--anchor-stop-after-expansion", help_text)
        self.assertIn("--sample-rate-hz", help_text)
        self.assertIn("--diagnostics-output", help_text)
        self.assertIn("S  solve", help_text)
        self.assertIn("Q  quit", help_text)
        self.assertIn("2 Hz", help_text) if "2 Hz" in help_text else self.assertIn("sample-rate-hz", help_text)

    def test_inspect_marker_model_help_lists_visualize_option(self) -> None:
        help_text = _run_cli_help("object-inspect-marker-model")
        self.assertIn("--marker-model", help_text)
        self.assertIn("--visualize", help_text)
        self.assertIn("marker model diagram", help_text)


class CalibrateMarkerModelValidationTests(unittest.TestCase):
    def test_existing_output_requires_force(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calibration = tmp_path / "intrinsics.json"
            output = tmp_path / "marker_model.json"
            _write_intrinsics(calibration)
            output.write_text("{}", encoding="utf-8")
            args = mock.Mock(
                calibration=calibration,
                output=output,
                force=False,
                marker_ids=["0", "1"],
                reference_marker_id=0,
                marker_size=0.07,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--force", str(ctx.exception))

    def test_duplicate_marker_ids_are_rejected(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            args = mock.Mock(
                calibration=calibration,
                output=Path(tmp_dir) / "out.json",
                force=True,
                marker_ids=["0", "0", "1"],
                reference_marker_id=0,
                marker_size=0.07,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("duplicates", str(ctx.exception))

    def test_marker_id_ranges_expand_in_validation(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            args = mock.Mock(
                calibration=calibration,
                output=Path(tmp_dir) / "out.json",
                force=True,
                marker_ids=["0", "1", "3-5"],
                reference_marker_id=0,
                marker_size=0.07,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=["0", "3-4"],
                anchor_stop_after_expansion=False,
            )
            expected_ids, _, anchor_ids, stop_after = validate_args(args)
            self.assertEqual(expected_ids, [0, 1, 3, 4, 5])
            self.assertEqual(anchor_ids, (0, 3, 4))
            self.assertFalse(stop_after)

    def test_anchor_marker_ids_must_include_reference(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            args = mock.Mock(
                calibration=calibration,
                output=Path(tmp_dir) / "out.json",
                force=True,
                marker_ids=["0", "1", "2"],
                reference_marker_id=0,
                marker_size=0.07,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=["1", "2"],
                anchor_stop_after_expansion=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("reference_marker_id", str(ctx.exception))

    def test_anchor_stop_after_expansion_requires_anchor_marker_ids(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            args = mock.Mock(
                calibration=calibration,
                output=Path(tmp_dir) / "out.json",
                force=True,
                marker_ids=["0", "1", "2"],
                reference_marker_id=0,
                marker_size=0.07,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=True,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--anchor-stop-after-expansion", str(ctx.exception))


class CalibrateMarkerModelCaptureTests(unittest.TestCase):
    def _run_capture(
        self,
        *,
        wait_keys: list[int],
        monotonic_values: list[float],
        visible_by_frame: list[dict[int, np.ndarray]],
        calibration_size: tuple[int, int] = (640, 480),
        calibrate_result: object | None = None,
        calibrate_side_effect: object | None = None,
        diagnostics_output: Path | None = None,
    ) -> tuple[list, Path, mock.Mock, mock.MagicMock, mock.Mock, bool]:
        from object_apriltag.cli.calibrate_marker_model import run_capture

        width, height = calibration_size
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calibration = tmp_path / "intrinsics.json"
            output = tmp_path / "marker_model.json"
            _write_intrinsics(calibration, width=width, height=height)

            args = mock.Mock(
                camera=0,
                calibration=calibration,
                dictionary="36h11",
                detection_sensitivity="default",
                marker_size=0.07,
                marker_ids=["0", "1"],
                reference_marker_id=0,
                output=output,
                force=True,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=diagnostics_output,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
            )

            frame = np.zeros((height, width, 3), dtype=np.uint8)
            capture = mock.MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, frame.copy())

            detector = mock.MagicMock()

            def detect_markers(_gray):
                frame_index = detector.detectMarkers.call_count - 1
                visible = visible_by_frame[frame_index % len(visible_by_frame)]
                if not visible:
                    return [], None, None
                corners = [corner.reshape(1, 4, 2) for corner in visible.values()]
                ids = np.array([[marker_id] for marker_id in visible], dtype=np.int32)
                return corners, ids, None

            detector.detectMarkers.side_effect = detect_markers

            monotonic_iter = iter(monotonic_values)

            def monotonic_side_effect() -> float:
                try:
                    return next(monotonic_iter)
                except StopIteration:
                    return monotonic_values[-1] + 1.0

            wait_iter = iter(wait_keys)

            accepted_layout = mock.Mock()
            accepted_layout.marker_ids = {0, 1}
            if calibrate_side_effect is not None:
                calibrate_mock = mock.Mock(side_effect=calibrate_side_effect)
            else:
                calibrate_mock = mock.Mock(
                    return_value=calibrate_result
                    if calibrate_result is not None
                    else mock.Mock(
                        layout=accepted_layout,
                        failure_reason=None,
                        quality=_quality_report_mock(),
                    )
                )

            with (
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.VideoCapture", return_value=capture),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.build_apriltag_detector",
                    return_value=detector,
                ),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.time.monotonic",
                    side_effect=monotonic_side_effect,
                ),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.cv2.waitKey",
                    side_effect=lambda _delay: next(wait_iter, 0),
                ),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.imshow"),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.destroyAllWindows") as destroy_mock,
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.calibrate_marker_layout",
                    calibrate_mock,
                ),
                mock.patch("object_apriltag.cli.calibrate_marker_model.save_marker_model") as save_mock,
            ):
                saved = run_capture(args)
            return calibrate_mock.call_args_list, output, save_mock, capture, destroy_mock, saved

    def test_samples_refresh_live_pair_readiness(self) -> None:
        visible_by_frame = [
            {0: _marker_corners(0)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
        ]
        monotonic_values = [0.0, *(index * 0.25 for index in range(1, 10))]

        captured_lengths: list[int] = []

        def track_readiness(observations, *_args, **_kwargs):
            captured_lengths.append(len(observations))
            return mock.Mock(
                pairs=(),
                connected_marker_ids=frozenset({0}),
                missing_marker_ids=frozenset({1}),
                sample_count=len(observations),
                failure_reason=None,
            )

        with mock.patch(
            "object_apriltag.cli.calibrate_marker_model.compute_live_pair_readiness",
            side_effect=track_readiness,
        ):
            self._run_capture(
                wait_keys=[0] * 6 + [ord("q"), ord("q")],
                monotonic_values=monotonic_values,
                visible_by_frame=visible_by_frame,
            )

        self.assertGreaterEqual(len(captured_lengths), 1)
        self.assertGreater(captured_lengths[-1], 0)

    def test_capture_continues_while_pair_readiness_is_slow(self) -> None:
        gate = threading.Event()
        compute_calls = 0

        def slow_readiness(observations, *_args, **_kwargs):
            nonlocal compute_calls
            compute_calls += 1
            if compute_calls == 1:
                gate.wait(timeout=2.0)
            return mock.Mock(
                pairs=(),
                connected_marker_ids=frozenset({0}),
                missing_marker_ids=frozenset({1}),
                sample_count=len(observations),
                failure_reason=None,
            )

        visible_by_frame = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 8
        monotonic_values = [0.0, *(index * 0.25 for index in range(1, 20))]

        with mock.patch(
            "object_apriltag.cli.calibrate_marker_model.compute_live_pair_readiness",
            side_effect=slow_readiness,
        ):
            calibrate_calls, _, _, capture, _, _ = self._run_capture(
                wait_keys=[0] * 10 + [ord("q"), ord("q")],
                monotonic_values=monotonic_values,
                visible_by_frame=visible_by_frame,
            )

        gate.set()
        self.assertEqual(len(calibrate_calls), 0)
        self.assertGreaterEqual(capture.read.call_count, 5)
        self.assertGreaterEqual(compute_calls, 1)

    def test_samples_at_two_hz_and_ignores_single_marker_frames(self) -> None:
        visible_by_frame = [
            {0: _marker_corners(0)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
            {1: _marker_corners(1)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
            {0: _marker_corners(0), 1: _marker_corners(1)},
        ]
        monotonic_values = [0.0, *(index * 0.25 for index in range(1, 20))]

        observations_holder: list = []

        def capture_observations(observations, *_args, **_kwargs):
            observations_holder.extend(observations)
            return mock.Mock(layout=None, failure_reason="refused", quality=None)

        self._run_capture(
            wait_keys=[0] * 8 + [ord("S"), ord("q"), ord("q")],
            monotonic_values=monotonic_values,
            visible_by_frame=visible_by_frame,
            calibrate_side_effect=capture_observations,
        )
        self.assertGreaterEqual(len(observations_holder), 2)
        for observation in observations_holder[:2]:
            self.assertEqual(sorted(observation.markers), [0, 1])

    def test_solve_refusal_prints_frame_counts_and_continues_without_writing(self) -> None:
        refused = mock.Mock(
            layout=None,
            failure_reason="refused",
            quality=_quality_report_mock(
                frame_count=20,
                inlier_corner_count=0,
                reprojection_rms_px=5.0,
                connected_marker_ids={0},
                missing_expected_ids={1},
            ),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 3
        monotonic = [0.0, 0.0, 0.6, 0.6, 1.2, 1.2, 1.8, 1.8]
        with mock.patch("builtins.print") as print_mock:
            calibrate_calls, output_path, save_mock, capture, destroy_mock, saved = self._run_capture(
                wait_keys=[ord("s"), ord("q"), ord("q")],
                monotonic_values=monotonic,
                visible_by_frame=visible,
                calibrate_result=refused,
            )
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertEqual(len(calibrate_calls), 1)
        save_mock.assert_not_called()
        self.assertFalse(output_path.exists())
        self.assertFalse(saved)
        self.assertIn("frames input/accepted/rejected: 25/20/5", printed)
        capture.release.assert_called_once()
        destroy_mock.assert_called_once()

    def test_solve_acceptance_prints_frame_counts_writes_and_exits(self) -> None:
        accepted_layout = mock.Mock()
        accepted_layout.marker_ids = {0, 1}
        accepted = mock.Mock(
            layout=accepted_layout,
            failure_reason=None,
            quality=_quality_report_mock(),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with mock.patch("builtins.print") as print_mock:
            calibrate_calls, output_path, save_mock, capture, destroy_mock, saved = self._run_capture(
                wait_keys=[ord("S")],
                monotonic_values=monotonic,
                visible_by_frame=visible,
                calibrate_result=accepted,
            )
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertEqual(len(calibrate_calls), 1)
        save_mock.assert_called_once()
        self.assertEqual(save_mock.call_args[0][0], output_path)
        self.assertTrue(saved)
        self.assertIn("frames input/accepted/rejected: 25/20/5", printed)
        capture.release.assert_called_once()
        destroy_mock.assert_called_once()

    def test_q_cancels_without_writing(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}]
        calibrate_calls, output_path, _, capture, destroy_mock, saved = self._run_capture(
            wait_keys=[ord("Q")],
            monotonic_values=[0.0, 0.0],
            visible_by_frame=visible,
        )
        self.assertEqual(len(calibrate_calls), 0)
        self.assertFalse(output_path.exists())
        self.assertFalse(saved)
        capture.release.assert_called_once()
        destroy_mock.assert_called_once()

    def test_diagnostics_output_writes_json_on_refusal_and_success(self) -> None:
        from object_apriltag.marker_layout_calibration import (
            CalibrationQualityReport,
            CalibrationResult,
            EdgeDiagnostics,
        )

        refused_quality = CalibrationQualityReport(
            reprojection_rms_px=float("inf"),
            per_marker_reprojection_rms_px={},
            edges=(),
            pair_translation_rms_max_m=0.0,
            pair_rotation_rms_max_deg=0.0,
            frame_count=0,
            observation_count=0,
            inlier_corner_count=0,
            input_frame_count=25,
            rejected_frame_count=5,
            accepted_frame_count=0,
            connected_marker_ids=frozenset({0, 1}),
            missing_expected_ids=frozenset(),
            unused_expected_ids=frozenset(),
            assignment_rejections=None,
            assignment_rejection_records=(),
            dropped_pair_edges=(),
        )
        accepted_layout = mock.Mock()
        accepted_layout.marker_ids = {0, 1}
        accepted_quality = CalibrationQualityReport(
            reprojection_rms_px=0.1,
            per_marker_reprojection_rms_px={0: 0.1, 1: 0.1},
            edges=(
                EdgeDiagnostics(
                    marker_a=0,
                    marker_b=1,
                    inlier_count=20,
                    translation_rms_m=0.01,
                    rotation_rms_deg=1.0,
                ),
            ),
            pair_translation_rms_max_m=0.01,
            pair_rotation_rms_max_deg=1.0,
            frame_count=20,
            observation_count=160,
            inlier_corner_count=160,
            input_frame_count=25,
            rejected_frame_count=5,
            accepted_frame_count=20,
            connected_marker_ids=frozenset({0, 1}),
            missing_expected_ids=frozenset(),
            unused_expected_ids=frozenset(),
            assignment_rejections=None,
            assignment_rejection_records=(),
            dropped_pair_edges=(),
        )
        refused = CalibrationResult(
            layout=None,
            quality=refused_quality,
            failure_reason="refused",
        )
        accepted = CalibrationResult(
            layout=accepted_layout,
            quality=accepted_quality,
            failure_reason=None,
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6, 0.6, 1.2]

        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics_path = Path(tmp_dir) / "diagnostics.json"
            with mock.patch("builtins.print"):
                self._run_capture(
                    wait_keys=[ord("s"), ord("q")],
                    monotonic_values=monotonic[:3],
                    visible_by_frame=visible,
                    calibrate_result=refused,
                    diagnostics_output=diagnostics_path,
                )
            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["succeeded"])
            self.assertEqual(payload["failure_reason"], "refused")
            self.assertIsNone(payload["quality"]["reprojection_rms_px"])

            with mock.patch("builtins.print"):
                self._run_capture(
                    wait_keys=[ord("s")],
                    monotonic_values=monotonic[:3],
                    visible_by_frame=visible,
                    calibrate_result=accepted,
                    diagnostics_output=diagnostics_path,
                )
            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["succeeded"])
            self.assertIsNone(payload["failure_reason"])
            self.assertAlmostEqual(payload["quality"]["reprojection_rms_px"], 0.1)

    def test_diagnostics_output_writes_on_refusal_and_success_only_when_requested(self) -> None:
        refused = mock.Mock(
            layout=None,
            failure_reason="refused",
            quality=_quality_report_mock(
                reprojection_rms_px=float("inf"),
            ),
        )
        accepted_layout = mock.Mock()
        accepted_layout.marker_ids = {0, 1}
        accepted = mock.Mock(
            layout=accepted_layout,
            failure_reason=None,
            quality=_quality_report_mock(),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6, 0.6, 1.2]

        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics_path = Path(tmp_dir) / "diagnostics.json"

            with mock.patch("builtins.print"), mock.patch(
                "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
            ) as save_mock:
                self._run_capture(
                    wait_keys=[ord("s"), ord("s"), ord("q")],
                    monotonic_values=monotonic,
                    visible_by_frame=visible,
                    calibrate_side_effect=[refused, accepted],
                    diagnostics_output=diagnostics_path,
                )
            self.assertEqual(save_mock.call_count, 2)
            self.assertEqual(save_mock.call_args_list[0].args[0], diagnostics_path)
            self.assertIs(save_mock.call_args_list[0].args[1], refused)
            self.assertIs(save_mock.call_args_list[1].args[1], accepted)

        with mock.patch("builtins.print"), mock.patch(
            "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
        ) as save_mock:
            self._run_capture(
                wait_keys=[ord("s"), ord("q")],
                monotonic_values=monotonic[:3],
                visible_by_frame=visible,
                calibrate_result=accepted,
            )
        save_mock.assert_not_called()

    def test_diagnostics_output_writes_bundle_adjustment_failure_with_preserved_diagnostics(self) -> None:
        from object_apriltag.marker_layout_calibration import (
            AssignmentRejectionSummary,
            CalibrationQualityReport,
            CalibrationResult,
            DroppedPairEdge,
            EdgeDiagnostics,
            FrameAssignmentRejectionRecord,
        )

        ba_failure_quality = CalibrationQualityReport(
            reprojection_rms_px=float("inf"),
            per_marker_reprojection_rms_px={},
            edges=(
                EdgeDiagnostics(
                    marker_a=0,
                    marker_b=1,
                    inlier_count=20,
                    translation_rms_m=0.01,
                    rotation_rms_deg=1.0,
                ),
            ),
            pair_translation_rms_max_m=0.01,
            pair_rotation_rms_max_deg=1.0,
            frame_count=20,
            observation_count=160,
            inlier_corner_count=0,
            input_frame_count=25,
            rejected_frame_count=2,
            accepted_frame_count=20,
            connected_marker_ids=frozenset({0, 1}),
            missing_expected_ids=frozenset(),
            unused_expected_ids=frozenset(),
            assignment_rejections=AssignmentRejectionSummary(
                total_rejected=2,
                by_reason=(("translation_gate", 2),),
                by_pair=(((0, 1), 2),),
                top_causes=(),
                by_cause=(),
            ),
            assignment_rejection_records=(
                FrameAssignmentRejectionRecord(
                    frame_index=2,
                    frame_id="capture-2",
                    visible_marker_ids=(0, 1),
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    translation_error_m=0.02,
                    translation_gate_m=0.007,
                ),
                FrameAssignmentRejectionRecord(
                    frame_index=7,
                    frame_id="capture-7",
                    visible_marker_ids=(0, 1),
                    reason="translation_gate",
                    marker_pair=(0, 1),
                    translation_error_m=0.03,
                    translation_gate_m=0.007,
                ),
            ),
            dropped_pair_edges=(
                DroppedPairEdge(
                    marker_a=0,
                    marker_b=2,
                    stage="assignment_support",
                    reason="insufficient_support",
                    observed_count=10,
                    supported_count=4,
                    required_count=20,
                ),
            ),
        )
        ba_failure = CalibrationResult(
            layout=None,
            quality=ba_failure_quality,
            failure_reason="Bundle adjustment failed: singular matrix",
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics_path = Path(tmp_dir) / "diagnostics.json"
            with mock.patch("builtins.print"):
                self._run_capture(
                    wait_keys=[ord("s"), ord("q")],
                    monotonic_values=monotonic,
                    visible_by_frame=visible,
                    calibrate_result=ba_failure,
                    diagnostics_output=diagnostics_path,
                )
            payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["succeeded"])
            self.assertEqual(payload["failure_reason"], "Bundle adjustment failed: singular matrix")
            self.assertIsNone(payload["quality"]["reprojection_rms_px"])
            self.assertEqual(len(payload["assignment_rejection_records"]), 2)
            self.assertEqual(
                {record["frame_id"] for record in payload["assignment_rejection_records"]},
                {"capture-2", "capture-7"},
            )
            self.assertEqual(len(payload["dropped_pair_edges"]), 1)
            self.assertEqual(payload["dropped_pair_edges"][0]["stage"], "assignment_support")

    def test_resolution_mismatch_releases_capture_and_exits(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calibration = tmp_path / "intrinsics.json"
            output = tmp_path / "marker_model.json"
            _write_intrinsics(calibration, width=640, height=480)
            argv = [
                "object-calibrate-marker-model",
                "--camera",
                "0",
                "--calibration",
                str(calibration),
                "--dictionary",
                "36h11",
                "--detection-sensitivity",
                "default",
                "--marker-size",
                "0.07",
                "--marker-ids",
                "0",
                "1",
                "--reference-marker-id",
                "0",
                "--output",
                str(output),
                "--force",
            ]
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            capture = mock.MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, frame)

            with (
                mock.patch("sys.argv", argv),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.VideoCapture", return_value=capture),
                mock.patch("object_apriltag.cli.calibrate_marker_model.build_apriltag_detector"),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.waitKey", return_value=ord("q")),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.imshow"),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.destroyAllWindows") as destroy_mock,
            ):
                with self.assertRaises(SystemExit) as ctx:
                    main()
                self.assertEqual(ctx.exception.code, 1)
            capture.release.assert_called_once()
            destroy_mock.assert_called_once()
            self.assertFalse(output.exists())


class CalibrationHudTests(unittest.TestCase):
    def test_hud_uses_single_white_text_pass_on_black_panel(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import (
            build_pair_readiness_hud_lines,
            draw_calibration_hud,
        )
        from object_apriltag.cli.live_pair_readiness_worker import LivePairReadinessView
        from object_apriltag.marker_layout_calibration import (
            LivePairReadinessDiagnostics,
            PairReadinessEdge,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        readiness_view = LivePairReadinessView(
            diagnostics=LivePairReadinessDiagnostics(
                pairs=(
                    PairReadinessEdge(
                        marker_a=0,
                        marker_b=1,
                        raw_covisible_frames=3,
                        robust_inlier_count=3,
                        translation_rms_m=None,
                        rotation_rms_deg=None,
                        status="weak",
                    ),
                ),
                connected_marker_ids=frozenset({0}),
                missing_marker_ids=frozenset({1}),
                sample_count=3,
            ),
            represented_sample_count=3,
            is_computing=False,
        )
        hud_lines = build_pair_readiness_hud_lines(
            expected_ids=[0, 1],
            visible_ids=[0, 1],
            current_sample_count=3,
            readiness_view=readiness_view,
            reference_marker_id=0,
        )
        with (
            mock.patch("object_apriltag.viz.overlay.cv2.rectangle") as rectangle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock,
        ):
            draw_calibration_hud(frame, hud_lines=hud_lines, last_solve_quality=None)

        rectangle_mock.assert_called_once()
        self.assertEqual(rectangle_mock.call_args.args[4], -1)
        self.assertTrue(text_mock.call_args_list)
        self.assertTrue(
            all(call.args[5] == (255, 255, 255) for call in text_mock.call_args_list)
        )


class CliFrameCountFormattingTests(unittest.TestCase):
    def test_format_solve_frame_counts_uses_input_accepted_rejected(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import format_solve_frame_counts

        text = format_solve_frame_counts(_quality_report_mock())
        self.assertEqual(text, "frames input/accepted/rejected: 25/20/5")


class InspectMarkerModelRegressionTests(unittest.TestCase):
    def test_inspector_prints_reference_marker_without_visualize(self) -> None:
        from object_apriltag.cli.inspect_marker_model import main

        argv = [
            "object-inspect-marker-model",
            "--marker-model",
            str(MARKER_MODEL),
            "--no-visualize",
        ]
        with mock.patch("sys.argv", argv), mock.patch("builtins.print") as print_mock:
            main()
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("Reference marker id: 0", printed)
        self.assertIn("Marker 0", printed)


if __name__ == "__main__":
    unittest.main()
