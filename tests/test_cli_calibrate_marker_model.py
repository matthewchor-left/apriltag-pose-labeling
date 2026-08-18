"""CLI tests for config-driven marker layout calibration."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER_MODEL = REPO_ROOT / "config/Model/playground/setup1/calibration_01/marker_model.json"


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
    }
    values.update(overrides)
    return mock.Mock(**values)


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
        "detector": {"dictionary": "36h11", "sensitivity": "relaxed"},
        "markers": {
            "reference_marker_id": None,
            "anchor_marker_ids": None,
            "groups": [{"ids": [0, 1], "size_m": 0.07}],
        },
        "execution": execution
        or {
            "mode": "benchmark",
            "sample_rate_hz": 10.0,
            "frame_selection": "sharpest",
        },
        "solver": solver
        or {
            "policy": "best_effort",
            "discrete_method": "rotation_consistent",
            "anchor_stop_after_expansion": False,
            "partial_output": True,
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


def _square_corners(half: float) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, 0.0],
        "top_right": [half, -half, 0.0],
        "bottom_right": [half, half, 0.0],
        "bottom_left": [-half, half, 0.0],
    }


class CliHelpTests(unittest.TestCase):
    def test_calibrate_marker_model_help_lists_config_flags(self) -> None:
        help_text = _run_cli_help("object-calibrate-marker-model")
        self.assertIn("--config", help_text)
        self.assertIn("--force", help_text)
        self.assertNotIn("--source", help_text)
        self.assertNotIn("--benchmark", help_text)

    def test_inspect_marker_model_help_lists_visualize_option(self) -> None:
        help_text = _run_cli_help("object-inspect-marker-model")
        self.assertIn("--marker-model", help_text)
        self.assertIn("--visualize", help_text)
        self.assertIn("marker model diagram", help_text)


class CalibrateMarkerModelBenchmarkTests(unittest.TestCase):
    def _run_benchmark(
        self,
        *,
        visible_by_frame: list[dict[int, np.ndarray]],
        calibrate_result: object | None = None,
        reported_fps: float = 10.0,
        sample_rate_hz: float = 2.0,
        sharpness_by_frame: list[float] | None = None,
    ) -> tuple[mock.Mock, mock.Mock, list, bool, mock.Mock, mock.MagicMock]:
        from object_apriltag.cli.calibrate_marker_model import run_benchmark
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        width, height = 640, 480
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = _write_workspace_recipe(
                tmp_path,
                execution={
                    "mode": "benchmark",
                    "sample_rate_hz": sample_rate_hz,
                    "frame_selection": "sharpest",
                },
            )
            recipe = load_calibration_recipe(config_path)
            video = tmp_path / "clip.mov"
            video.write_bytes(b"\x00" * 128)

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
            accepted_layout.footprints = {0: mock.Mock(), 1: mock.Mock()}
            calibrate_mock = mock.Mock(
                return_value=calibrate_result
                if calibrate_result is not None
                else mock.Mock(
                    layout=accepted_layout,
                    failure_reason=None,
                    outcome="accepted",
                    quality=_quality_report_mock(),
                    omitted_markers=(),
                    failed_quality_gates=(),
                    failed_refinement_stage=None,
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
                    "object_apriltag.cli.calibrate_marker_model.build_object_model_document_from_layout",
                    return_value={"keypoints": {}, "skeleton": []},
                ),
                mock.patch("object_apriltag.cli.calibrate_marker_model.save_object_model_document"),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                    return_value=tmp_path / "diagnostics.json",
                ) as save_diagnostics_mock,
                mock.patch("builtins.print"),
            ):
                saved = run_benchmark(recipe)
            return (
                calibrate_mock,
                save_mock,
                calibrate_mock.call_args_list[0].args[0],
                saved,
                save_diagnostics_mock,
                detector,
            )

    def test_benchmark_rejects_invalid_reported_fps(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import run_benchmark
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        width, height = 640, 480
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = _write_workspace_recipe(tmp_path)
            recipe = load_calibration_recipe(config_path)
            capture = mock.MagicMock()
            capture.isOpened.return_value = True
            capture.get.return_value = 0.0
            with (
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.VideoCapture", return_value=capture),
                mock.patch("object_apriltag.cli.calibrate_marker_model.build_apriltag_detector"),
                mock.patch("builtins.print"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    run_benchmark(recipe)
            self.assertIn("FPS", str(ctx.exception))
            capture.release.assert_called_once()

    def test_benchmark_solves_exactly_once_at_eof(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 6
        calibrate_mock, save_mock, _, saved, _, _ = self._run_benchmark(visible_by_frame=visible)
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
        calibrate_mock, save_mock, observations, _, save_diagnostics_mock, _ = self._run_benchmark(
            visible_by_frame=visible,
            calibrate_result=refused,
        )
        self.assertGreater(len(observations), 0)
        calibrate_mock.assert_called_once()
        save_mock.assert_not_called()
        save_diagnostics_mock.assert_called_once()
        benchmark_payload = save_diagnostics_mock.call_args.kwargs["benchmark"]
        self.assertIn("source", benchmark_payload)
        self.assertIn("counts", benchmark_payload)
        self.assertIn("timing_seconds", benchmark_payload)
        self.assertIn("throughput", benchmark_payload)
        self.assertIn("environment", benchmark_payload)
        self.assertEqual(benchmark_payload["frame_selection"], "sharpest")
        self.assertIn("sharpness_scoring", benchmark_payload["timing_seconds"])

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
                    "timing_seconds": {"least_squares": 0.5},
                    "counts": {"parameter_count": 126},
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
                ingest_total_ns=10_000_000,
                calibration_solve_ns=4_000_000,
                total_through_solve_ns=14_000_000,
                solve_diagnostics=diagnostics,
                sharpness_scoring_ns=500_000,
            )
            self.assertEqual(payload["frame_selection"], "sharpest")
            self.assertIn("solve_stages", payload["timing_seconds"])
            self.assertEqual(len(payload["optimizer_runs"]), 1)

    def test_benchmark_sharpest_picks_highest_sharpness_in_window(self) -> None:
        visible_by_frame = [{0: _marker_corners(0)}] * 5
        visible_by_frame[2] = {0: _marker_corners(0), 1: _marker_corners(1)}
        sharpness = [1.0, 2.0, 50.0, 49.0, 3.0]
        _, _, observations, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible_by_frame,
            reported_fps=10.0,
            sample_rate_hz=1.0,
            sharpness_by_frame=sharpness,
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(detector.detectMarkers.call_count, 1)
        benchmark = save_diagnostics_mock.call_args.kwargs["benchmark"]
        self.assertEqual(benchmark["frame_selection"], "sharpest")

    def test_benchmark_sharpest_one_detection_per_completed_window(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 60
        _, _, _, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=60.0,
            sample_rate_hz=10.0,
            sharpness_by_frame=[float(index) for index in range(60)],
        )
        self.assertEqual(detector.detectMarkers.call_count, 10)
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(counts["detector_invocations"], 10)
        self.assertEqual(counts["frames_skipped_before_detection"], 50)

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
                sharpness_by_frame=sharpness,
            )
        self.assertEqual(detected_indices, [1])

    def test_benchmark_sharpest_flushes_final_partial_window(self) -> None:
        visible = [{0: _marker_corners(0), 1: _marker_corners(1)}] * 7
        _, _, observations, _, save_diagnostics_mock, detector = self._run_benchmark(
            visible_by_frame=visible,
            reported_fps=10.0,
            sample_rate_hz=2.0,
            sharpness_by_frame=[float(index) for index in range(7)],
        )
        self.assertEqual(detector.detectMarkers.call_count, 2)
        self.assertEqual(len(observations), 2)
        counts = save_diagnostics_mock.call_args.kwargs["benchmark"]["counts"]
        self.assertEqual(
            counts["detector_invocations"] + counts["frames_skipped_before_detection"],
            counts["decoded_frames"],
        )


class CliFrameCountFormattingTests(unittest.TestCase):
    def test_format_solve_frame_counts_uses_input_accepted_rejected(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import format_solve_frame_counts

        text = format_solve_frame_counts(_quality_report_mock())
        self.assertEqual(text, "frames input/accepted/rejected: 25/20/5")


class ConfigModeCliTests(unittest.TestCase):
    def test_parse_args_rejects_legacy_flags(self) -> None:
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
                from object_apriltag.cli.calibrate_marker_model import parse_args

                with self.assertRaises(SystemExit):
                    parse_args()

    def test_validate_recipe_requires_force_for_existing_outputs(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_recipe
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            (workspace / "marker_model.json").write_text("{}", encoding="utf-8")
            recipe = load_calibration_recipe(config_path)
            with self.assertRaises(RuntimeError) as ctx:
                validate_recipe(recipe, force=False)
            self.assertIn("--force", str(ctx.exception))

    def test_validate_recipe_force_allows_existing_outputs(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_recipe
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            (workspace / "marker_model.json").write_text("{}", encoding="utf-8")
            recipe = load_calibration_recipe(config_path)
            validate_recipe(recipe, force=True)
            self.assertEqual(recipe.settings.huber_delta_px, 1.25)

    def test_validate_recipe_requires_force_for_existing_diagnostics(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import validate_recipe
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            (workspace / "diagnostics.json").write_text("{}", encoding="utf-8")
            recipe = load_calibration_recipe(config_path)
            with self.assertRaises(RuntimeError) as ctx:
                validate_recipe(recipe, force=False)
            self.assertIn("diagnostics.json", str(ctx.exception))

    def test_config_mode_publishes_paired_outputs(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import apply_calibration_result
        from object_apriltag.layout import build_marker_layout, footprint_from_dict
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            recipe = load_calibration_recipe(config_path)
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
                saved = apply_calibration_result(recipe, result)
            self.assertTrue(saved)
            self.assertTrue((workspace / "marker_model.json").exists())
            self.assertTrue((workspace / "object_model.json").exists())
            object_model = json.loads((workspace / "object_model.json").read_text(encoding="utf-8"))
            self.assertIn("a", object_model["keypoints"])
            self.assertNotIn("note", object_model)

    def test_config_mode_partial_missing_source_marker_publishes_subset(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import apply_calibration_result
        from object_apriltag.layout import build_marker_layout, footprint_from_dict
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe
        from object_apriltag.marker_layout_calibration.types import CalibrationResult

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            recipe = load_calibration_recipe(config_path)
            half = 0.035
            footprints = {0: footprint_from_dict(0, _square_corners(half))}
            layout = build_marker_layout(0, 0.07, footprints)
            result = CalibrationResult(
                layout=layout,
                quality=_quality_report_mock(),
                failure_reason=None,
                outcome="partial",
                partial_output=True,
            )
            with mock.patch(
                "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                return_value=workspace / "diagnostics.json",
            ) as diagnostics_mock:
                saved = apply_calibration_result(recipe, result)
            self.assertTrue(saved)
            self.assertTrue((workspace / "marker_model.json").exists())
            object_model = json.loads((workspace / "object_model.json").read_text(encoding="utf-8"))
            self.assertEqual(set(object_model["keypoints"]), {"a"})
            self.assertEqual(object_model["skeleton"], [])
            self.assertNotIn("b", object_model["keypoint_sources"])
            diagnostics_mock.assert_called_once()

    def test_config_mode_strict_missing_source_marker_writes_diagnostics_only(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import apply_calibration_result
        from object_apriltag.layout import build_marker_layout, footprint_from_dict
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            recipe = load_calibration_recipe(config_path)
            half = 0.035
            footprints = {0: footprint_from_dict(0, _square_corners(half))}
            layout = build_marker_layout(0, 0.07, footprints)
            with mock.patch(
                "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                return_value=workspace / "diagnostics.json",
            ) as diagnostics_mock:
                for outcome, partial_output in (
                    ("accepted", False),
                    ("partial", False),
                ):
                    with self.subTest(
                        outcome=outcome,
                        partial_output=partial_output,
                    ):
                        result = mock.Mock(
                            layout=layout,
                            quality=_quality_report_mock(),
                            failure_reason=None,
                            outcome=outcome,
                            partial_output=partial_output,
                            omitted_markers=(),
                            failed_quality_gates=(),
                            failed_refinement_stage=None,
                        )
                        saved = apply_calibration_result(recipe, result)
                        self.assertFalse(saved)
                        self.assertFalse((workspace / "marker_model.json").exists())
                        self.assertFalse((workspace / "object_model.json").exists())
            self.assertEqual(diagnostics_mock.call_count, 2)

    def test_config_mode_partial_no_source_markers_writes_diagnostics_only(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import apply_calibration_result
        from object_apriltag.layout import build_marker_layout, footprint_from_dict
        from object_apriltag.marker_layout_calibration.recipe import load_calibration_recipe

        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir)
            config_path = _write_workspace_recipe(workspace)
            recipe = load_calibration_recipe(config_path)
            half = 0.035
            footprints = {9: footprint_from_dict(9, _square_corners(half))}
            layout = build_marker_layout(9, 0.07, footprints)
            result = mock.Mock(
                layout=layout,
                quality=_quality_report_mock(),
                failure_reason=None,
                outcome="partial",
                partial_output=True,
                omitted_markers=(),
                failed_quality_gates=(),
                failed_refinement_stage=None,
            )
            with mock.patch(
                "object_apriltag.cli.calibrate_marker_model.save_calibration_diagnostics",
                return_value=workspace / "diagnostics.json",
            ) as diagnostics_mock:
                saved = apply_calibration_result(recipe, result)
            self.assertFalse(saved)
            self.assertFalse((workspace / "marker_model.json").exists())
            self.assertFalse((workspace / "object_model.json").exists())
            diagnostics_mock.assert_called_once()


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
        self.assertIn("Reference marker id: 19", printed)
        self.assertIn("Marker 0", printed)

    def test_visualize_opens_interactive_marker_model_plot(self) -> None:
        from object_apriltag.cli.inspect_marker_model import main

        argv = [
            "object-inspect-marker-model",
            "--marker-model",
            str(MARKER_MODEL),
            "--visualize",
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("builtins.print"),
            mock.patch("object_apriltag.cli.inspect_marker_model.show_marker_model_plot") as show_mock,
        ):
            main()
        show_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
