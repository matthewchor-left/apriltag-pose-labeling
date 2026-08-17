"""CLI tests for live marker layout calibration and inspector."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import cv2

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
        self.assertIn("--config", help_text)
        self.assertIn("--source", help_text)
        self.assertIn("--marker-ids", help_text)
        self.assertIn("--marker-size-for", help_text)
        self.assertIn("--anchor-marker-ids", help_text)
        self.assertIn("--anchor-stop-after-expansion", help_text)
        self.assertIn("--best-effort", help_text)
        self.assertIn("--partial-output", help_text)
        self.assertIn("--auto", help_text)
        self.assertIn("--sample-rate-hz", help_text)
        self.assertIn("ignored in manual mode", help_text)
        self.assertIn("--diagnostics-output", help_text)
        self.assertIn("--benchmark", help_text)
        self.assertIn("--benchmark-frame-selection", help_text)
        self.assertIn("Selecting sharpest", help_text)
        self.assertIn("--object-model", help_text)
        self.assertIn("--overlay-object-model", help_text)
        self.assertIn("keypoint_sources", help_text)
        self.assertIn("C  capture", help_text)
        self.assertIn("S  solve", help_text)
        self.assertIn("Q  quit", help_text)

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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=["0", "3-4"],
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
            )
            expected_ids, marker_sizes_m, _, anchor_ids, stop_after, _, _ = validate_args(args)
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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=["1", "2"],
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("reference_marker_id", str(ctx.exception))

    def test_marker_size_for_resolves_overrides(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            args = mock.Mock(
                calibration=calibration,
                output=Path(tmp_dir) / "out.json",
                force=True,
                marker_ids=["0", "1", "4", "10", "11"],
                reference_marker_id=0,
                marker_size=0.07,
                marker_size_for=["4:0.03", "10-11:0.025"],
                sample_rate_hz=2.0,
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                object_model=None,
                overlay_object_model=False,
            )
            expected_ids, marker_sizes_m, _, _, _, _, _ = validate_args(args)
            self.assertEqual(expected_ids, [0, 1, 4, 10, 11])
            self.assertEqual(marker_sizes_m[4], 0.03)
            self.assertEqual(marker_sizes_m[10], 0.025)
            self.assertEqual(marker_sizes_m[0], 0.07)

    def test_marker_size_for_accepts_repeatable_flag_groups(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            args = mock.Mock(
                calibration=calibration,
                output=Path(tmp_dir) / "out.json",
                force=True,
                marker_ids=["0", "1", "4", "10", "11"],
                reference_marker_id=0,
                marker_size=0.07,
                marker_size_for=[["4:0.03"], ["10-11:0.025"]],
                sample_rate_hz=2.0,
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                object_model=None,
                overlay_object_model=False,
            )
            _, marker_sizes_m, _, _, _, _, _ = validate_args(args)
            self.assertEqual(marker_sizes_m[4], 0.03)
            self.assertEqual(marker_sizes_m[11], 0.025)

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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=True,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--anchor-stop-after-expansion", str(ctx.exception))

    def test_best_effort_rejects_anchor_stop_after_expansion(self) -> None:
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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=["0", "1"],
                anchor_stop_after_expansion=True,
                best_effort=True,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--best-effort", str(ctx.exception))
            self.assertIn("--anchor-stop-after-expansion", str(ctx.exception))

    def test_partial_output_requires_best_effort(self) -> None:
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
                auto=False,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=["0", "1"],
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=True,
                marker_size_for=None,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--partial-output", str(ctx.exception))
            self.assertIn("--best-effort", str(ctx.exception))

    def test_sample_rate_hz_must_be_finite_and_positive(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            calibration = Path(tmp_dir) / "intrinsics.json"
            _write_intrinsics(calibration)
            for sample_rate_hz in (0.0, -1.0, float("inf"), float("nan")):
                with self.subTest(sample_rate_hz=sample_rate_hz):
                    args = mock.Mock(
                        calibration=calibration,
                        output=Path(tmp_dir) / "out.json",
                        force=True,
                        marker_ids=["0", "1"],
                        reference_marker_id=0,
                        marker_size=0.07,
                        sample_rate_hz=sample_rate_hz,
                        auto=False,
                        min_pair_inliers=20,
                        reprojection_rms_gate_px=2.0,
                        pair_translation_rms_gate_ratio=0.10,
                        pair_rotation_rms_gate_deg=5.0,
                        diagnostics_output=None,
                        anchor_marker_ids=None,
                        anchor_stop_after_expansion=False,
                        best_effort=False,
                        partial_output=False,
                        marker_size_for=None,
                        object_model=None,
                        overlay_object_model=False,
                    )
                    with self.assertRaises(RuntimeError) as ctx:
                        validate_args(args)
                    self.assertIn("--sample-rate-hz", str(ctx.exception))


class CalibrateMarkerModelBenchmarkValidationTests(unittest.TestCase):
    def _validation_args(self, tmp_dir: str, **overrides: object) -> mock.Mock:
        calibration = Path(tmp_dir) / "intrinsics.json"
        _write_intrinsics(calibration)
        video = Path(tmp_dir) / "clip.mov"
        video.write_bytes(b"\x00" * 64)
        args = mock.Mock(
            calibration=calibration,
            output=Path(tmp_dir) / "out.json",
            force=True,
            marker_ids=["0", "1"],
            reference_marker_id=0,
            marker_size=0.07,
            sample_rate_hz=2.0,
            auto=False,
            min_pair_inliers=20,
            reprojection_rms_gate_px=2.0,
            pair_translation_rms_gate_ratio=0.10,
            pair_rotation_rms_gate_deg=5.0,
            diagnostics_output=Path(tmp_dir) / "diagnostics.json",
            anchor_marker_ids=None,
            anchor_stop_after_expansion=False,
            best_effort=False,
            partial_output=False,
            marker_size_for=None,
            object_model=None,
            overlay_object_model=False,
            source=video,
            benchmark=True,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        if not hasattr(args, "benchmark_frame_selection"):
            args.benchmark_frame_selection = "uniform"
        return args

    def test_benchmark_frame_selection_requires_benchmark(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._validation_args(
                tmp_dir,
                benchmark=False,
                benchmark_frame_selection="sharpest",
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--benchmark-frame-selection", str(ctx.exception))
            self.assertIn("--benchmark", str(ctx.exception))

    def test_benchmark_requires_diagnostics_output(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._validation_args(tmp_dir, diagnostics_output=None)
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--diagnostics-output", str(ctx.exception))

    def test_benchmark_rejects_camera_source(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._validation_args(tmp_dir, source=0)
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("video file", str(ctx.exception))

    def test_benchmark_rejects_overlay_object_model(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(
                object_model,
                keypoint_sources={"top": {"marker_id": 1, "corner": "top_left"}},
            )
            args = self._validation_args(
                tmp_dir,
                object_model=object_model,
                overlay_object_model=True,
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--overlay-object-model", str(ctx.exception))


class CalibrateMarkerModelBenchmarkTests(unittest.TestCase):
    def _run_benchmark(
        self,
        *,
        visible_by_frame: list[dict[int, np.ndarray]],
        calibrate_result: object | None = None,
        reported_fps: float = 10.0,
        sample_rate_hz: float = 2.0,
        diagnostics_output: Path | None = None,
        benchmark_frame_selection: str = "uniform",
        sharpness_by_frame: list[float] | None = None,
    ) -> tuple[mock.Mock, mock.Mock, list, mock.Mock, mock.Mock, mock.Mock, mock.Mock, bool, mock.Mock, mock.MagicMock]:
        from object_apriltag.cli.calibrate_marker_model import run_benchmark

        width, height = 640, 480
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calibration = tmp_path / "intrinsics.json"
            output = tmp_path / "marker_model.json"
            video = tmp_path / "clip.mov"
            video.write_bytes(b"\x00" * 128)
            _write_intrinsics(calibration, width=width, height=height)
            if diagnostics_output is None:
                diagnostics_output = tmp_path / "diagnostics.json"

            args = mock.Mock(
                source=video,
                benchmark=True,
                calibration=calibration,
                dictionary="36h11",
                detection_sensitivity="default",
                marker_size=0.07,
                marker_ids=["0", "1"],
                reference_marker_id=0,
                output=output,
                force=True,
                auto=False,
                sample_rate_hz=sample_rate_hz,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=diagnostics_output,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
                benchmark_frame_selection=benchmark_frame_selection,
            )

            frame = np.zeros((height, width, 3), dtype=np.uint8)
            decoded_frame_index = {"value": -1}

            def read_frame_from_fixture() -> tuple[bool, np.ndarray | None]:
                decoded_frame_index["value"] += 1
                if decoded_frame_index["value"] < len(visible_by_frame):
                    encoded = frame.copy()
                    index = decoded_frame_index["value"]
                    encoded[0, 0, :] = index
                    return True, encoded
                return False, None

            capture = mock.MagicMock()
            capture.isOpened.return_value = True
            capture.read.side_effect = read_frame_from_fixture

            def capture_get(prop: float) -> float:
                if prop == cv2.CAP_PROP_FPS:
                    return reported_fps
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return float(len(visible_by_frame))
                return 0.0

            capture.get.side_effect = capture_get

            detector = mock.MagicMock()

            def detect_markers(_gray):
                frame_index = int(_gray[0, 0])
                visible = visible_by_frame[frame_index]
                if not visible:
                    return [], None, None
                corners = [corner.reshape(1, 4, 2) for corner in visible.values()]
                ids = np.array([[marker_id] for marker_id in visible], dtype=np.int32)
                return corners, ids, None

            detector.detectMarkers.side_effect = detect_markers

            sharpness_ctx = (
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model._frame_sharpness_score",
                    side_effect=lambda frame: float(
                        sharpness_by_frame[int(frame[0, 0, 0])]
                    ),
                )
                if sharpness_by_frame is not None
                else contextlib.nullcontext()
            )

            accepted_layout = mock.Mock()
            accepted_layout.marker_ids = {0, 1}
            calibrate_mock = mock.Mock(
                return_value=calibrate_result
                if calibrate_result is not None
                else mock.Mock(
                    layout=accepted_layout,
                    failure_reason=None,
                    outcome="accepted",
                    quality=_quality_report_mock(),
                )
            )

            with (
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.VideoCapture", return_value=capture),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.build_apriltag_detector",
                    return_value=detector,
                ),
                sharpness_ctx,
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.calibrate_marker_layout",
                    calibrate_mock,
                ),
                mock.patch("object_apriltag.cli.calibrate_marker_model.save_marker_model") as save_mock,
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                    return_value=diagnostics_output,
                ) as save_diagnostics_mock,
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.LivePairReadinessWorker",
                ) as worker_cls,
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.imshow") as imshow_mock,
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.waitKey") as wait_key_mock,
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.cv2.destroyAllWindows",
                ) as destroy_mock,
                mock.patch("builtins.print"),
            ):
                saved = run_benchmark(args)
            return (
                calibrate_mock,
                save_mock,
                calibrate_mock.call_args_list[0].args[0],
                worker_cls,
                imshow_mock,
                wait_key_mock,
                destroy_mock,
                saved,
                save_diagnostics_mock,
                detector,
            )

    def test_benchmark_never_uses_gui_or_readiness_worker(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 4
        (
            _,
            _,
            _,
            worker_cls,
            imshow_mock,
            wait_key_mock,
            destroy_mock,
            _,
            _,
            _,
        ) = self._run_benchmark(visible_by_frame=visible)
        worker_cls.assert_not_called()
        imshow_mock.assert_not_called()
        wait_key_mock.assert_not_called()
        destroy_mock.assert_not_called()

    def test_benchmark_rejects_invalid_reported_fps(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import run_benchmark

        width, height = 640, 480
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calibration = tmp_path / "intrinsics.json"
            output = tmp_path / "marker_model.json"
            video = tmp_path / "clip.mov"
            video.write_bytes(b"\x00" * 128)
            _write_intrinsics(calibration, width=width, height=height)
            args = mock.Mock(
                source=video,
                benchmark=True,
                calibration=calibration,
                dictionary="36h11",
                detection_sensitivity="default",
                marker_size=0.07,
                marker_ids=["0", "1"],
                reference_marker_id=0,
                output=output,
                force=True,
                auto=False,
                sample_rate_hz=2.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=tmp_path / "diagnostics.json",
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=None,
                overlay_object_model=False,
            )
            capture = mock.MagicMock()
            capture.isOpened.return_value = True
            capture.get.return_value = 0.0
            with (
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.VideoCapture", return_value=capture),
                mock.patch("object_apriltag.cli.calibrate_marker_model.build_apriltag_detector"),
                mock.patch("builtins.print"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    run_benchmark(args)
            self.assertIn("FPS", str(ctx.exception))
            capture.release.assert_called_once()

    def test_benchmark_samples_deterministically_by_video_time(self) -> None:
        visible_by_frame = (
            [{0: _marker_corners(0)}] * 5
            + [{0: _marker_corners(0), 1: _marker_corners(1)}] * 11
        )
        _, _, observations, *_ = self._run_benchmark(
            visible_by_frame=visible_by_frame,
            reported_fps=10.0,
        )
        self.assertEqual(len(observations), 3)
        for observation in observations:
            self.assertEqual(sorted(observation.markers), [0, 1])

    def test_benchmark_solves_exactly_once_at_eof(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 6
        calibrate_mock, save_mock, _, _, _, _, _, saved, _, _ = self._run_benchmark(
            visible_by_frame=visible,
        )
        calibrate_mock.assert_called_once()
        save_mock.assert_called_once()
        self.assertTrue(saved)

    def test_benchmark_refusal_writes_diagnostics_with_payload(self) -> None:
        refused = mock.Mock(
            layout=None,
            failure_reason="refused",
            quality=_quality_report_mock(
                frame_count=0,
                inlier_corner_count=0,
                reprojection_rms_px=float("inf"),
            ),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 6

        with tempfile.TemporaryDirectory() as tmp_dir:
            diagnostics_path = Path(tmp_dir) / "diagnostics.json"
            calibrate_mock, save_mock, observations, _, _, _, _, _, save_diagnostics_mock, _ = (
                self._run_benchmark(
                    visible_by_frame=visible,
                    calibrate_result=refused,
                    diagnostics_output=diagnostics_path,
                )
            )
            self.assertEqual(len(observations), 2)
            calibrate_mock.assert_called_once()
            save_mock.assert_not_called()
            save_diagnostics_mock.assert_called_once()
            benchmark_payload = save_diagnostics_mock.call_args.kwargs["benchmark"]
            self.assertIn("source", benchmark_payload)
            self.assertIn("counts", benchmark_payload)
            self.assertIn("timing_seconds", benchmark_payload)
            self.assertIn("throughput", benchmark_payload)
            self.assertIn("environment", benchmark_payload)
            self.assertEqual(benchmark_payload["counts"]["sampled_observations"], 2)

    def test_benchmark_passes_solve_diagnostics_collector(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 6
        calibrate_mock, *_ = self._run_benchmark(visible_by_frame=visible)
        self.assertIn("solve_diagnostics", calibrate_mock.call_args.kwargs)
        diagnostics = calibrate_mock.call_args.kwargs["solve_diagnostics"]
        self.assertIsNotNone(diagnostics)
        self.assertEqual(diagnostics.solve_stages_seconds, {})
        self.assertEqual(diagnostics.optimizer_runs, [])

    def test_build_benchmark_payload_includes_solve_diagnostics(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import _build_benchmark_payload
        from object_apriltag.marker_layout_calibration import CalibrationSolveDiagnostics

        diagnostics = CalibrationSolveDiagnostics(
            solve_stages_seconds={
                "ippe_candidate_generation": 0.5,
                "initial_bundle_adjustment": 1.25,
            },
            optimizer_runs=[
                {
                    "stage": "initial_bundle_adjustment",
                    "nfev": 42,
                    "njev": None,
                    "status": 1,
                    "cost": 0.12,
                    "active_frame_count": 20,
                    "inlier_corner_count": 160,
                    "timing_seconds": {
                        "setup": 0.001,
                        "least_squares": 0.5,
                        "post": 0.0001,
                        "residual_callback_total": 0.4,
                        "residual_unpack": 0.05,
                        "projection_loop": 0.34,
                        "residual_callback_other": 0.01,
                        "least_squares_overhead": 0.1,
                    },
                    "counts": {
                        "parameter_count": 126,
                        "residual_count": 320,
                        "residual_callback_invocations": 42,
                        "projection_calls": 6720,
                        "opencv_projectpoints_invocations": 42,
                        "batched_corner_count": 6720,
                    },
                }
            ],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "clip.mov"
            source.write_bytes(b"\x00" * 64)
            payload = _build_benchmark_payload(
                source_path=source,
                reported_fps=30.0,
                reported_frame_count=100,
                image_size=(640, 480),
                decoded_frames=100,
                detector_invocations=10,
                frames_skipped_before_detection=90,
                frames_with_expected_markers=9,
                covisible_frames=8,
                sampled_observations=10,
                detected_markers=20,
                open_source_ns=1_000_000,
                decode_ns=2_000_000,
                detection_ns=3_000_000,
                ingest_total_ns=6_000_000,
                calibration_solve_ns=7_000_000,
                total_through_solve_ns=13_000_000,
                solve_diagnostics=diagnostics,
            )
        self.assertEqual(
            payload["timing_seconds"]["solve_stages"],
            diagnostics.solve_stages_seconds,
        )
        self.assertEqual(payload["optimizer_runs"], diagnostics.optimizer_runs)
        self.assertEqual(payload["frame_selection"], "uniform")
        self.assertNotIn("solve_stages", payload)
        self.assertEqual(
            set(payload["counts"]),
            {
                "decoded_frames",
                "detector_invocations",
                "frames_skipped_before_detection",
                "frames_with_expected_markers",
                "covisible_frames",
                "sampled_observations",
                "detected_markers",
            },
        )
        self.assertIn("detector_invocations_per_second", payload["throughput"])

    def test_benchmark_60fps_10hz_invokes_detection_on_scheduled_frames(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 60
        _, _, _, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=60.0,
            sample_rate_hz=10.0,
        )
        self.assertEqual(detector.detectMarkers.call_count, 10)
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(counts["detector_invocations"], 10)
        self.assertEqual(counts["frames_skipped_before_detection"], 50)

    def test_benchmark_missed_sample_retries_at_sample_interval_not_every_frame(self) -> None:
        visible_by_frame = (
            [{0: _marker_corners(0)}]
            + [{0: _marker_corners(0), 1: _marker_corners(1)}] * 17
        )
        _, _, observations, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible_by_frame,
            reported_fps=60.0,
            sample_rate_hz=10.0,
        )
        self.assertEqual(detector.detectMarkers.call_count, 3)
        self.assertEqual(len(observations), 2)
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(counts["frames_skipped_before_detection"], 15)

    def test_benchmark_sampling_is_deterministic_across_runs(self) -> None:
        visible_by_frame = (
            [{0: _marker_corners(0)}] * 5
            + [{0: _marker_corners(0), 1: _marker_corners(1)}] * 11
        )

        def run_once() -> tuple[int, int, list]:
            _, _, observations, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
                visible_by_frame=visible_by_frame,
                reported_fps=10.0,
                sample_rate_hz=2.0,
            )
            counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
            return (
                detector.detectMarkers.call_count,
                len(observations),
                [sorted(observation.markers) for observation in observations],
            )

        self.assertEqual(run_once(), run_once())

    def test_benchmark_counts_reconcile_decoded_frames(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 12
        _, _, _, _, _, _, _, _, save_diagnostics_mock, _ = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=10.0,
            sample_rate_hz=2.0,
        )
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(
            counts["detector_invocations"] + counts["frames_skipped_before_detection"],
            counts["decoded_frames"],
        )

    def test_benchmark_sharpest_picks_highest_sharpness_in_window(self) -> None:
        visible_by_frame = [{0: _marker_corners(0)}] * 5
        visible_by_frame[2] = {0: _marker_corners(0), 1: _marker_corners(1)}
        sharpness = [1.0, 2.0, 50.0, 49.0, 3.0]
        _, _, observations, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible_by_frame,
            reported_fps=10.0,
            sample_rate_hz=1.0,
            benchmark_frame_selection="sharpest",
            sharpness_by_frame=sharpness,
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(detector.detectMarkers.call_count, 1)
        benchmark = save_diagnostics_mock.call_args.kwargs["benchmark"]
        self.assertEqual(benchmark["frame_selection"], "sharpest")
        self.assertIn("sharpness_scoring", benchmark["timing_seconds"])

    def test_benchmark_sharpest_one_detection_per_completed_window(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 60
        _, _, _, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=60.0,
            sample_rate_hz=10.0,
            benchmark_frame_selection="sharpest",
            sharpness_by_frame=[float(index) for index in range(60)],
        )
        self.assertEqual(detector.detectMarkers.call_count, 10)
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(counts["detector_invocations"], 10)
        self.assertEqual(counts["frames_skipped_before_detection"], 50)

    def test_benchmark_sharpest_starts_new_window_on_exact_decimal_boundary(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 61
        *_, detector = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=60.0,
            sample_rate_hz=10.0,
            benchmark_frame_selection="sharpest",
            sharpness_by_frame=[float(index) for index in range(61)],
        )
        self.assertEqual(detector.detectMarkers.call_count, 11)

    def test_benchmark_sharpest_tie_breaks_to_earliest_frame(self) -> None:
        visible_by_frame = [{0: _marker_corners(0)}] * 6
        visible_by_frame[1] = {0: _marker_corners(0), 1: _marker_corners(1)}
        visible_by_frame[4] = {0: _marker_corners(0), 1: _marker_corners(1)}
        sharpness = [5.0, 30.0, 30.0, 10.0, 30.0, 1.0]
        detected_indices: list[int] = []

        from object_apriltag.cli import calibrate_marker_model as cli_module

        original_detect = cli_module.detect_expected_markers

        def record_detect(detector, frame, expected_ids):
            detected_indices.append(int(frame[0, 0, 0]))
            return original_detect(detector, frame, expected_ids)

        with mock.patch.object(cli_module, "detect_expected_markers", side_effect=record_detect):
            self._run_benchmark(
                visible_by_frame=visible_by_frame,
                reported_fps=10.0,
                sample_rate_hz=1.0,
                benchmark_frame_selection="sharpest",
                sharpness_by_frame=sharpness,
            )
        self.assertEqual(detected_indices, [1])

    def test_benchmark_sharpest_flushes_final_partial_window(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 7
        _, _, observations, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=10.0,
            sample_rate_hz=2.0,
            benchmark_frame_selection="sharpest",
            sharpness_by_frame=[float(index) for index in range(7)],
        )
        self.assertEqual(detector.detectMarkers.call_count, 2)
        self.assertEqual(len(observations), 2)
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(
            counts["detector_invocations"] + counts["frames_skipped_before_detection"],
            counts["decoded_frames"],
        )

    def test_benchmark_uniform_frame_selection_unchanged_by_default(self) -> None:
        visible_by_frame = (
            [{0: _marker_corners(0)}] * 5
            + [{0: _marker_corners(0), 1: _marker_corners(1)}] * 11
        )
        _, _, observations, _, _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible_by_frame,
            reported_fps=10.0,
            sample_rate_hz=2.0,
            benchmark_frame_selection="uniform",
        )
        self.assertEqual(len(observations), 3)
        benchmark = save_diagnostics_mock.call_args.kwargs["benchmark"]
        self.assertEqual(benchmark["frame_selection"], "uniform")
        self.assertNotIn("sharpness_scoring", benchmark["timing_seconds"])
        counts = benchmark["counts"]
        self.assertEqual(
            counts["detector_invocations"] + counts["frames_skipped_before_detection"],
            counts["decoded_frames"],
        )


def _write_object_model(path: Path, *, keypoint_sources: dict | None) -> None:
    payload = {
        "units": "meters",
        "coordinate_frame": "marker_model",
        "keypoints": {
            "top": [0.0, 0.0, 0.0],
            "bottom": [0.0, 0.1, 0.0],
        },
        "skeleton": [["top", "bottom"]],
    }
    if keypoint_sources is not None:
        payload["keypoint_sources"] = keypoint_sources
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _synthetic_marker_layout():
    from object_apriltag.layout import build_marker_layout, footprint_from_dict

    marker_size = 0.04
    half = marker_size / 2.0
    square = {
        "top_left": [-half, -half, 0.0],
        "top_right": [half, -half, 0.0],
        "bottom_right": [half, half, 0.0],
        "bottom_left": [-half, half, 0.0],
    }
    footprints = {
        0: footprint_from_dict(0, square),
        1: footprint_from_dict(
            1,
            {
                "top_left": [0.06, -half, 0.0],
                "top_right": [0.06 + marker_size, -half, 0.0],
                "bottom_right": [0.06 + marker_size, half, 0.0],
                "bottom_left": [0.06, half, 0.0],
            },
        ),
    }
    return build_marker_layout(0, marker_size, footprints)


class CalibrateMarkerModelObjectModelValidationTests(unittest.TestCase):
    def _validation_args(self, tmp_dir: str, object_model: Path | None) -> mock.Mock:
        calibration = Path(tmp_dir) / "intrinsics.json"
        _write_intrinsics(calibration)
        return mock.Mock(
            calibration=calibration,
            output=Path(tmp_dir) / "out.json",
            force=True,
            marker_ids=["0", "1"],
            reference_marker_id=0,
            marker_size=0.07,
            sample_rate_hz=2.0,
            auto=False,
            min_pair_inliers=20,
            reprojection_rms_gate_px=2.0,
            pair_translation_rms_gate_ratio=0.10,
            pair_rotation_rms_gate_deg=5.0,
            diagnostics_output=None,
            anchor_marker_ids=None,
            anchor_stop_after_expansion=False,
            best_effort=False,
            partial_output=False,
            marker_size_for=None,
            object_model=object_model,
            overlay_object_model=False,
        )

    def test_validate_args_rejects_missing_object_model_file(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing_object_model.json"
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(self._validation_args(tmp_dir, missing))
            self.assertIn("Object model file not found", str(ctx.exception))

    def test_validate_args_rejects_empty_keypoint_sources(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(object_model, keypoint_sources={})
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(self._validation_args(tmp_dir, object_model))
            self.assertIn("keypoint_sources", str(ctx.exception))

    def test_validate_args_rejects_invalid_corner_in_keypoint_sources(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(
                object_model,
                keypoint_sources={"top": {"marker_id": 1, "corner": "center"}},
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(self._validation_args(tmp_dir, object_model))
            self.assertIn("corner", str(ctx.exception))

    def test_validate_args_rejects_source_marker_outside_marker_ids(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(
                object_model,
                keypoint_sources={"top": {"marker_id": 9, "corner": "top_left"}},
            )
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(self._validation_args(tmp_dir, object_model))
            self.assertIn("must appear in --marker-ids", str(ctx.exception))
            self.assertIn("9", str(ctx.exception))

    def test_validate_args_rejects_invalid_padding_mm_in_keypoint_sources(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        invalid_values = {
            "bool": True,
            "string": "3.0",
            "null": None,
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -2.0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            for label, padding_mm in invalid_values.items():
                object_model = Path(tmp_dir) / f"object_model_{label}.json"
                _write_object_model(
                    object_model,
                    keypoint_sources={
                        "top": {
                            "marker_id": 1,
                            "corner": "top_left",
                            "padding_mm": padding_mm,
                        }
                    },
                )
                with self.subTest(label=label):
                    with self.assertRaises(RuntimeError) as ctx:
                        validate_args(self._validation_args(tmp_dir, object_model))
                    self.assertIn("padding_mm", str(ctx.exception))


    def test_validate_args_rejects_overlay_without_object_model(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._validation_args(tmp_dir, None)
            args.overlay_object_model = True
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--object-model is required", str(ctx.exception))


class CalibrateMarkerModelObjectModelCaptureTests(unittest.TestCase):
    def test_solve_success_updates_object_model_keypoints(self) -> None:
        layout = _synthetic_marker_layout()
        accepted = mock.Mock(
            layout=layout,
            failure_reason=None,
            outcome="accepted",
            quality=_quality_report_mock(),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(
                object_model,
                keypoint_sources={
                    "top": {"marker_id": 1, "corner": "top_left"},
                    "bottom": {"marker_id": "01", "corner": "bottom_right"},
                },
            )
            original = json.loads(object_model.read_text(encoding="utf-8"))
            capture_tests = CalibrateMarkerModelCaptureTests()
            with mock.patch("builtins.print"):
                _, _, save_mock, _, _, saved = capture_tests._run_capture(
                    wait_keys=[ord("S")],
                    monotonic_values=monotonic,
                    visible_by_frame=visible,
                    calibrate_result=accepted,
                    object_model=object_model,
                )
            self.assertTrue(saved)
            save_mock.assert_called_once()
            updated = json.loads(object_model.read_text(encoding="utf-8"))
            np.testing.assert_allclose(
                updated["keypoints"]["top"],
                layout.footprints[1].top_left.tolist(),
                atol=1e-12,
            )
            np.testing.assert_allclose(
                updated["keypoints"]["bottom"],
                layout.footprints[1].bottom_right.tolist(),
                atol=1e-12,
            )
            self.assertEqual(updated["keypoint_sources"], original["keypoint_sources"])
            self.assertEqual(updated["skeleton"], original["skeleton"])

    def test_solve_success_applies_padding_mm_to_object_model_keypoints(self) -> None:
        from object_apriltag.layout import footprint_corner_with_padding

        layout = _synthetic_marker_layout()
        padding_mm = 3.0
        accepted = mock.Mock(
            layout=layout,
            failure_reason=None,
            outcome="accepted",
            quality=_quality_report_mock(),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            keypoint_sources = {
                "top": {
                    "marker_id": 1,
                    "corner": "top_left",
                    "padding_mm": padding_mm,
                },
            }
            _write_object_model(object_model, keypoint_sources=keypoint_sources)
            original = json.loads(object_model.read_text(encoding="utf-8"))
            capture_tests = CalibrateMarkerModelCaptureTests()
            with mock.patch("builtins.print"):
                _, _, save_mock, _, _, saved = capture_tests._run_capture(
                    wait_keys=[ord("S")],
                    monotonic_values=monotonic,
                    visible_by_frame=visible,
                    calibrate_result=accepted,
                    object_model=object_model,
                )
            self.assertTrue(saved)
            save_mock.assert_called_once()
            updated = json.loads(object_model.read_text(encoding="utf-8"))
            expected = footprint_corner_with_padding(
                layout.footprints[1],
                "top_left",
                padding_mm / 1000.0,
            )
            np.testing.assert_allclose(updated["keypoints"]["top"], expected.tolist(), atol=1e-12)
            self.assertFalse(
                np.allclose(
                    updated["keypoints"]["top"],
                    layout.footprints[1].top_left.tolist(),
                    atol=1e-12,
                )
            )
            self.assertEqual(updated["keypoint_sources"], original["keypoint_sources"])

    def test_solve_refusal_leaves_object_model_unchanged(self) -> None:
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
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6, 0.6, 1.2, 1.2]

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(
                object_model,
                keypoint_sources={"top": {"marker_id": 1, "corner": "top_left"}},
            )
            original_text = object_model.read_text(encoding="utf-8")
            capture_tests = CalibrateMarkerModelCaptureTests()
            with mock.patch("builtins.print"):
                _, _, save_mock, _, _, saved = capture_tests._run_capture(
                    wait_keys=[ord("s"), ord("q")],
                    monotonic_values=monotonic,
                    visible_by_frame=visible,
                    calibrate_result=refused,
                    object_model=object_model,
                )
            self.assertFalse(saved)
            save_mock.assert_not_called()
            self.assertEqual(object_model.read_text(encoding="utf-8"), original_text)

    def test_solve_success_with_missing_source_marker_reports_failure(self) -> None:
        from object_apriltag.layout import build_marker_layout

        full_layout = _synthetic_marker_layout()
        layout = build_marker_layout(
            0,
            full_layout.marker_size_m,
            {0: full_layout.footprints[0]},
        )
        accepted = mock.Mock(
            layout=layout,
            failure_reason=None,
            outcome="accepted",
            quality=_quality_report_mock(),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            object_model = Path(tmp_dir) / "object_model.json"
            _write_object_model(
                object_model,
                keypoint_sources={"top": {"marker_id": 1, "corner": "top_left"}},
            )
            original_text = object_model.read_text(encoding="utf-8")
            capture_tests = CalibrateMarkerModelCaptureTests()
            with self.assertRaises(RuntimeError) as ctx:
                capture_tests._run_capture(
                    wait_keys=[ord("S")],
                    monotonic_values=monotonic,
                    visible_by_frame=visible,
                    calibrate_result=accepted,
                    object_model=object_model,
                )
            self.assertEqual(object_model.read_text(encoding="utf-8"), original_text)
            self.assertIn("Marker model saved", str(ctx.exception))
            self.assertIn("object model update failed", str(ctx.exception))
            self.assertIn("marker 1", str(ctx.exception))


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
        auto: bool = False,
        sample_rate_hz: float = 10.0,
        object_model: Path | None = None,
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
                auto=auto,
                sample_rate_hz=sample_rate_hz,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=diagnostics_output,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                best_effort=False,
                partial_output=False,
                marker_size_for=None,
                object_model=object_model,
                overlay_object_model=False,
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
                wait_keys=[0, ord("c"), ord("c"), 0, 0, ord("q")],
                monotonic_values=monotonic_values,
                visible_by_frame=visible_by_frame,
            )

        self.assertGreaterEqual(len(captured_lengths), 1)
        self.assertGreater(captured_lengths[-1], 0)

    def test_manual_mode_does_not_auto_capture_without_c(self) -> None:
        visible_by_frame = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 8
        monotonic_values = [0.0, *(index * 0.25 for index in range(1, 20))]

        observations_holder: list = []

        def capture_observations(observations, *_args, **_kwargs):
            observations_holder.extend(observations)
            return mock.Mock(layout=None, failure_reason="refused", quality=None)

        self._run_capture(
            wait_keys=[0] * 8 + [ord("S"), ord("q")],
            monotonic_values=monotonic_values,
            visible_by_frame=visible_by_frame,
            calibrate_side_effect=capture_observations,
            auto=False,
        )
        self.assertEqual(observations_holder, [])

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
                wait_keys=[ord("c")] + [0] * 9 + [ord("q")],
                monotonic_values=monotonic_values,
                visible_by_frame=visible_by_frame,
            )

        gate.set()
        self.assertEqual(len(calibrate_calls), 0)
        self.assertGreaterEqual(capture.read.call_count, 5)
        self.assertGreaterEqual(compute_calls, 1)

    def test_manual_capture_records_only_frames_with_two_markers(self) -> None:
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
            wait_keys=[
                ord("c"),
                ord("c"),
                ord("c"),
                ord("c"),
                ord("S"),
                ord("q"),
            ],
            monotonic_values=monotonic_values,
            visible_by_frame=visible_by_frame,
            calibrate_side_effect=capture_observations,
        )
        self.assertGreaterEqual(len(observations_holder), 2)
        for observation in observations_holder[:2]:
            self.assertEqual(sorted(observation.markers), [0, 1])

    def test_auto_mode_samples_at_configured_rate_and_ignores_single_marker_frames(self) -> None:
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
            wait_keys=[0] * 8 + [ord("S"), ord("q")],
            monotonic_values=monotonic_values,
            visible_by_frame=visible_by_frame,
            calibrate_side_effect=capture_observations,
            auto=True,
            sample_rate_hz=2.0,
        )
        self.assertGreaterEqual(len(observations_holder), 2)
        for observation in observations_holder:
            self.assertEqual(sorted(observation.markers), [0, 1])

    def test_auto_mode_does_not_double_capture_when_timer_and_c_coincide(self) -> None:
        visible = {0: _marker_corners(0), 1: _marker_corners(1)}
        visible_by_frame = [visible]
        monotonic_values = [0.0, 0.0, 0.0]

        observations_holder: list = []

        def capture_observations(observations, *_args, **_kwargs):
            observations_holder.extend(observations)
            return mock.Mock(layout=None, failure_reason="refused", quality=None)

        self._run_capture(
            wait_keys=[ord("c"), ord("S"), ord("q")],
            monotonic_values=monotonic_values,
            visible_by_frame=visible_by_frame,
            calibrate_side_effect=capture_observations,
            auto=True,
            sample_rate_hz=2.0,
        )
        self.assertEqual(len(observations_holder), 1)

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
            outcome="accepted",
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

    def test_solve_provisional_warns_writes_and_exits_successfully(self) -> None:
        provisional_layout = mock.Mock()
        provisional_layout.marker_ids = {0, 1}
        provisional = mock.Mock(
            layout=provisional_layout,
            failure_reason=None,
            outcome="provisional",
            calibration_policy="best_effort",
            failed_quality_gates=("Global reprojection RMS 0.500 px exceeds 0.150 px gate.",),
            quality=_quality_report_mock(reprojection_rms_px=0.5),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with mock.patch("builtins.print") as print_mock:
            calibrate_calls, output_path, save_mock, capture, destroy_mock, saved = self._run_capture(
                wait_keys=[ord("S")],
                monotonic_values=monotonic,
                visible_by_frame=visible,
                calibrate_result=provisional,
            )
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertEqual(len(calibrate_calls), 1)
        save_mock.assert_called_once()
        self.assertTrue(saved)
        self.assertIn("provisional", printed.lower())
        self.assertIn("Saved marker model", printed)
        capture.release.assert_called_once()
        destroy_mock.assert_called_once()

    def test_solve_partial_writes_model_and_prints_omitted_markers(self) -> None:
        from object_apriltag.marker_layout_calibration import OmittedMarkerDiagnostic

        partial_layout = mock.Mock()
        partial_layout.marker_ids = {0, 1}
        partial = mock.Mock(
            layout=partial_layout,
            failure_reason=None,
            outcome="partial",
            calibration_policy="best_effort",
            partial_output=True,
            omitted_markers=(
                OmittedMarkerDiagnostic(2, "no_accepted_frame_observations"),
                OmittedMarkerDiagnostic(3, "not_connected_in_raw_observations"),
            ),
            quality=_quality_report_mock(),
        )
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 2
        monotonic = [0.0, 0.0, 0.6]

        with mock.patch("builtins.print") as print_mock:
            calibrate_calls, output_path, save_mock, capture, destroy_mock, saved = self._run_capture(
                wait_keys=[ord("S")],
                monotonic_values=monotonic,
                visible_by_frame=visible,
                calibrate_result=partial,
            )
        printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertEqual(len(calibrate_calls), 1)
        save_mock.assert_called_once()
        self.assertTrue(saved)
        self.assertIn("partial", printed.lower())
        self.assertIn("omitted marker 2: no_accepted_frame_observations", printed)
        self.assertIn("omitted marker 3: not_connected_in_raw_observations", printed)
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
                "--source",
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


class KeypointSourceOverlayTests(unittest.TestCase):
    @staticmethod
    def _frontal_marker_geometry(
        marker_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        half = marker_size / 2.0
        center = 320.0
        top = center - half * 900.0
        bottom = center + half * 900.0
        left = center - half * 900.0
        right = center + half * 900.0
        corners = np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float64,
        )
        camera_matrix = np.array(
            [[900.0, 0.0, center], [0.0, 900.0, center], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        return corners, camera_matrix, dist_coeffs

    def test_project_keypoint_source_on_marker_returns_image_points(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import project_keypoint_source_on_marker

        marker_size = 0.02
        corners, camera_matrix, dist_coeffs = self._frontal_marker_geometry(marker_size)

        projected = project_keypoint_source_on_marker(
            corners,
            marker_id=19,
            marker_size_m=marker_size,
            corner_name="top_left",
            padding_m=0.002,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self.assertIsNotNone(projected)
        raw_image, target_image = projected
        self.assertTrue(np.all(np.isfinite(raw_image)))
        self.assertTrue(np.all(np.isfinite(target_image)))
        self.assertGreater(np.linalg.norm(target_image - raw_image), 0.0)

    def test_project_keypoint_source_overlay_uses_miter_local_point_not_radial_bisector(self) -> None:
        import cv2

        from object_apriltag.cli.calibrate_marker_model import (
            marker_frame_footprint,
            project_keypoint_source_on_marker,
        )
        from object_apriltag.layout import CORNER_NAMES, footprint_corner_with_padding, rectangle_center
        from object_apriltag.pose import estimate_marker_pose
        from object_apriltag.viz.projection import project_camera_point

        marker_size = 0.04
        padding_m = 0.003
        corners, camera_matrix, dist_coeffs = self._frontal_marker_geometry(marker_size)
        corner_name = "top_left"

        projected = project_keypoint_source_on_marker(
            corners,
            marker_id=19,
            marker_size_m=marker_size,
            corner_name=corner_name,
            padding_m=padding_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self.assertIsNotNone(projected)
        raw_image, target_image = projected

        rvec, tvec = estimate_marker_pose(corners, marker_size, camera_matrix, dist_coeffs)
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        footprint = marker_frame_footprint(19, marker_size)
        corner_point = footprint.corners_by_name()[corner_name]
        miter_point = footprint_corner_with_padding(footprint, corner_name, padding_m)
        expected_raw_image = project_camera_point(
            rotation @ corner_point + translation,
            camera_matrix,
            dist_coeffs,
        )
        expected_target_image = project_camera_point(
            rotation @ miter_point + translation,
            camera_matrix,
            dist_coeffs,
        )
        np.testing.assert_allclose(raw_image, expected_raw_image, atol=1e-6)
        np.testing.assert_allclose(target_image, expected_target_image, atol=1e-6)

        corner_index = CORNER_NAMES.index(corner_name)
        prev_corner = footprint.corners_by_name()[CORNER_NAMES[(corner_index - 1) % 4]]
        next_corner = footprint.corners_by_name()[CORNER_NAMES[(corner_index + 1) % 4]]
        z_axis = footprint.orientation[:, 2]
        edge_a = next_corner - corner_point
        edge_b = corner_point - prev_corner
        inward_hint = rectangle_center(*footprint.corners()) - corner_point
        n1 = np.cross(z_axis, edge_a)
        n1 /= np.linalg.norm(n1)
        if float(np.dot(n1, inward_hint)) > 0.0:
            n1 = -n1
        n2 = np.cross(z_axis, edge_b)
        n2 /= np.linalg.norm(n2)
        if float(np.dot(n2, inward_hint)) > 0.0:
            n2 = -n2
        radial_point = corner_point + padding_m * (n1 + n2) / np.linalg.norm(n1 + n2)
        radial_image = project_camera_point(
            rotation @ radial_point + translation,
            camera_matrix,
            dist_coeffs,
        )
        self.assertFalse(np.allclose(target_image, radial_image, atol=0.5))
        self.assertGreater(
            float(np.linalg.norm(miter_point - corner_point)),
            padding_m,
        )

    def test_project_keypoint_source_overlay_matches_padded_point_for_rotated_pose(self) -> None:
        import cv2

        from object_apriltag.cli.calibrate_marker_model import (
            marker_frame_footprint,
            project_keypoint_source_on_marker,
        )
        from object_apriltag.layout import footprint_corner_with_padding
        from object_apriltag.pose import marker_corner_object_points
        from object_apriltag.viz.projection import project_camera_point

        marker_size = 0.04
        padding_m = 0.003
        center = 320.0
        camera_matrix = np.array(
            [[900.0, 0.0, center], [0.0, 900.0, center], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        rvec_true = np.array([0.35, -0.25, 0.4], dtype=np.float64)
        tvec_true = np.array([[0.04], [-0.03], [0.65]], dtype=np.float64)
        object_points = marker_corner_object_points(marker_size)
        image_points, _ = cv2.projectPoints(
            object_points,
            rvec_true,
            tvec_true,
            camera_matrix,
            dist_coeffs,
        )
        corners = image_points.reshape(4, 2).astype(np.float64)
        corner_name = "bottom_right"

        projected = project_keypoint_source_on_marker(
            corners,
            marker_id=7,
            marker_size_m=marker_size,
            corner_name=corner_name,
            padding_m=padding_m,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        self.assertIsNotNone(projected)
        _, target_image = projected

        rotation, _ = cv2.Rodrigues(rvec_true)
        translation = tvec_true.reshape(3)
        footprint = marker_frame_footprint(7, marker_size)
        miter_point = footprint_corner_with_padding(footprint, corner_name, padding_m)
        expected_target_image = project_camera_point(
            rotation @ miter_point + translation,
            camera_matrix,
            dist_coeffs,
        )
        np.testing.assert_allclose(target_image, expected_target_image, atol=1e-6)

    def test_draw_keypoint_source_overlays_marks_visible_source_tag(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import draw_keypoint_source_overlays

        marker_size = 0.02
        half = marker_size / 2.0
        center = 320.0
        top = center - half * 900.0
        bottom = center + half * 900.0
        left = center - half * 900.0
        right = center + half * 900.0
        corners = np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float64,
        )
        camera_matrix = np.array(
            [[900.0, 0.0, center], [0.0, 900.0, center], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoint_sources = {"logo": (19, "top_left", 0.002)}

        with mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.circle") as circle_mock:
            draw_keypoint_source_overlays(
                frame,
                visible={19: corners},
                keypoint_sources=keypoint_sources,
                marker_sizes_m={19: marker_size},
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
            )

        self.assertGreaterEqual(circle_mock.call_count, 2)


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


def _write_workspace_recipe(
    workspace: Path,
    *,
    execution: dict | None = None,
    solver: dict | None = None,
) -> Path:
    intrinsics = workspace / "intrinsics.json"
    _write_intrinsics(intrinsics)
    (workspace / "clip.mov").write_text("", encoding="utf-8")
    payload = {
        "config_version": 1,
        "inputs": {"source": "clip.mov", "intrinsics": "intrinsics.json"},
        "detector": {"dictionary": "36h11", "sensitivity": "default"},
        "markers": {
            "reference_marker_id": 0,
            "anchor_marker_ids": [0, 1],
            "groups": [{"ids": [0, 1], "size_m": 0.07}],
        },
        "execution": execution
        or {
            "mode": "benchmark",
            "sample_rate_hz": 10.0,
            "frame_selection": "uniform",
        },
        "solver": solver
        or {
            "policy": "strict",
            "anchor_stop_after_expansion": False,
            "partial_output": False,
            "min_inliers_per_edge": 20,
            "reprojection_rms_gate_px": 2.0,
            "pair_translation_rms_gate_ratio": 0.1,
            "pair_rotation_rms_gate_deg": 5.0,
            "huber_delta_px": 1.25,
            "corner_outlier_px": 3.0,
            "max_ba_iterations": 50,
        },
        "object_model": {
            "keypoint_sources": {
                "a": {"marker_id": 0, "corner": "top_left"},
                "b": {"marker_id": 1, "corner": "top_right"},
            },
            "skeleton": [["a", "b"]],
        },
    }
    config_path = workspace / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


class ConfigModeCliTests(unittest.TestCase):
    def test_config_mode_rejects_legacy_flags(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import _parse_config_mode_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = _write_workspace_recipe(Path(tmp_dir))
            argv = [
                "object-calibrate-marker-model",
                "--config",
                str(config_path),
                "--source",
                "0",
            ]
            with mock.patch("sys.argv", argv):
                with self.assertRaises(RuntimeError) as ctx:
                    _parse_config_mode_args()
            self.assertIn("cannot be mixed", str(ctx.exception))

    def test_config_mode_requires_force_for_existing_outputs(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import namespace_from_recipe, validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            (workspace / "marker_model.json").write_text("{}", encoding="utf-8")
            args = namespace_from_recipe(config_path, force=False)
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("--force", str(ctx.exception))

    def test_config_mode_force_allows_existing_outputs(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import namespace_from_recipe, validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            (workspace / "marker_model.json").write_text("{}", encoding="utf-8")
            args = namespace_from_recipe(config_path, force=True)
            expected_ids, _, settings, _, _, _, _ = validate_args(args)
            self.assertEqual(expected_ids, [0, 1])
            self.assertEqual(settings.huber_delta_px, 1.25)

    def test_config_mode_requires_force_for_existing_diagnostics(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import namespace_from_recipe, validate_args

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            (workspace / "diagnostics.json").write_text("{}", encoding="utf-8")
            args = namespace_from_recipe(config_path, force=False)
            with self.assertRaises(RuntimeError) as ctx:
                validate_args(args)
            self.assertIn("diagnostics.json", str(ctx.exception))

    def test_legacy_mode_emits_deprecation_warning(self) -> None:
        from object_apriltag.cli import calibrate_marker_model as cli_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            calibration = workspace / "intrinsics.json"
            _write_intrinsics(calibration)
            output = workspace / "marker_model.json"
            argv = [
                "object-calibrate-marker-model",
                "--source",
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
            ]
            with mock.patch("sys.argv", argv):
                with mock.patch.object(cli_module, "run_capture", return_value=False):
                    with self.assertWarns(FutureWarning):
                        cli_module.main()

    def test_config_mode_publishes_paired_outputs(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import apply_calibration_result, namespace_from_recipe
        from object_apriltag.layout import build_marker_layout, footprint_from_dict

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            args = namespace_from_recipe(config_path, force=True)
            half = 0.035
            footprints = {
                0: footprint_from_dict(0, _square_corners(half)),
                1: footprint_from_dict(
                    1,
                    {
                        "top_left": [0.06, -half, 0.0],
                        "top_right": [0.06 + 0.07, -half, 0.0],
                        "bottom_right": [0.06 + 0.07, half, 0.0],
                        "bottom_left": [0.06, half, 0.0],
                    },
                ),
            }
            layout = build_marker_layout(0, 0.07, footprints)
            result = mock.Mock(
                layout=layout,
                quality=_quality_report_mock(),
                failure_reason=None,
                outcome="accepted",
                omitted_markers=(),
                failed_quality_gates=(),
                failed_refinement_stage=None,
            )
            with mock.patch(
                "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                return_value=workspace / "diagnostics.json",
            ):
                saved = apply_calibration_result(args, result)
            self.assertTrue(saved)
            self.assertTrue((workspace / "marker_model.json").exists())
            self.assertTrue((workspace / "object_model.json").exists())
            object_model = json.loads((workspace / "object_model.json").read_text(encoding="utf-8"))
            self.assertIn("a", object_model["keypoints"])
            self.assertNotIn("note", object_model)

    def test_config_mode_missing_source_marker_writes_diagnostics_only(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import apply_calibration_result, namespace_from_recipe
        from object_apriltag.layout import build_marker_layout, footprint_from_dict

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            args = namespace_from_recipe(config_path, force=True)
            half = 0.035
            footprints = {
                0: footprint_from_dict(0, _square_corners(half)),
            }
            layout = build_marker_layout(0, 0.07, footprints)
            result = mock.Mock(
                layout=layout,
                quality=_quality_report_mock(),
                failure_reason=None,
                outcome="partial",
                omitted_markers=(),
                failed_quality_gates=(),
                failed_refinement_stage=None,
            )
            with mock.patch(
                "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                return_value=workspace / "diagnostics.json",
            ) as diagnostics_mock:
                saved = apply_calibration_result(args, result)
            self.assertFalse(saved)
            self.assertFalse((workspace / "marker_model.json").exists())
            self.assertFalse((workspace / "object_model.json").exists())
            diagnostics_mock.assert_called_once()


def _square_corners(half: float) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, 0.0],
        "top_right": [half, -half, 0.0],
        "bottom_right": [half, half, 0.0],
        "bottom_left": [-half, half, 0.0],
    }


if __name__ == "__main__":
    unittest.main()
