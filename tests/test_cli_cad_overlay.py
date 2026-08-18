"""CLI tests for object-detect CAD overlay integration."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = REPO_ROOT / "config/Camera/nexplaygroundcam/intrinsics.json"
MARKER_MODEL = REPO_ROOT / "config/Model/remote/marker_model.json"
OBJECT_MODEL = REPO_ROOT / "config/Model/remote/object_model.json"
REMOTE_MARKER_MODEL = MARKER_MODEL
REMOTE_OBJECT_MODEL = OBJECT_MODEL


def _run_cli_help(command: str) -> str:
    result = subprocess.run(
        ["uv", "run", command, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _base_detect_argv(*extra: str) -> list[str]:
    return [
        "object-detect",
        "--source",
        "0",
        "--calibration",
        str(CALIBRATION),
        "--marker-model",
        str(MARKER_MODEL),
        "--dictionary",
        "36h11",
        "--detection-sensitivity",
        "relaxed",
        *extra,
    ]


def _write_cad_assets(directory: Path) -> tuple[Path, Path]:
    cad_model = directory / "model.glb"
    cad_model.write_bytes(b"glb")
    registration = directory / "cad_registration.json"
    registration.write_text(json.dumps({"version": 1}), encoding="utf-8")
    return cad_model, registration


def _remote_detect_argv(*extra: str) -> list[str]:
    return [
        "object-detect",
        "--source",
        "0",
        "--calibration",
        str(CALIBRATION),
        "--marker-model",
        str(REMOTE_MARKER_MODEL),
        "--dictionary",
        "36h11",
        "--detection-sensitivity",
        "relaxed",
        *extra,
    ]


class CliCadOverlayHelpTests(unittest.TestCase):
    def test_object_detect_help_lists_cad_overlay_options(self) -> None:
        help_text = _run_cli_help("object-detect")
        self.assertIn("--overlay-cad-model", help_text)
        self.assertIn("--side2side-cad-model", help_text)
        self.assertIn("--cad-model", help_text)
        self.assertIn("cad_registration.json", help_text)
        self.assertIn("--object-model", help_text)


class CliCadOverlayContractTests(unittest.TestCase):
    def test_object_detect_requires_cad_model_when_overlay_enabled(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        argv = _base_detect_argv("--overlay-cad-model")
        with mock.patch("sys.argv", argv):
            with self.assertRaises(RuntimeError) as ctx:
                detect_main()
        self.assertIn("--cad-model", str(ctx.exception))

    def test_missing_registration_requires_object_model_for_generation(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model = Path(tmp_dir) / "model.glb"
            cad_model.write_bytes(b"glb")
            argv = _base_detect_argv("--overlay-cad-model", "--cad-model", str(cad_model))
            with mock.patch("sys.argv", argv):
                with self.assertRaises(RuntimeError) as ctx:
                    detect_main()
        self.assertIn("--object-model", str(ctx.exception))

    def test_missing_registration_is_fitted_before_opening_frame_source(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            calibration = directory / "intrinsics.json"
            marker_model_path = directory / "marker_model.json"
            object_model_path = directory / "object_model.json"
            cad_model_path = directory / "model.glb"
            for path in (
                calibration,
                marker_model_path,
                object_model_path,
                cad_model_path,
            ):
                path.write_bytes(b"fixture")

            marker_layout = mock.Mock(name="marker_layout", marker_ids={0, 1, 2})
            detector = mock.Mock(marker_model=marker_layout, marker_size_m=0.07)
            fitted_registration = mock.Mock(name="fitted_registration")
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_landmarks=mock.Mock(return_value=mock.Mock(name="cad_landmarks")),
                load_cad_registration=mock.Mock(),
            )
            cad_geometry_module = mock.Mock(
                fit_cad_registration=mock.Mock(return_value=fitted_registration)
            )
            draw_cad_only_landmarks = mock.Mock()
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                draw_cad_only_landmarks=draw_cad_only_landmarks,
                object_model_landmark_names=mock.Mock(return_value=frozenset()),
            )
            load_object_model_document = mock.Mock(
                return_value=(mock.Mock(), {"keypoint_sources": {}}),
            )
            argv = [
                "object-detect",
                "--source",
                "0",
                "--calibration",
                str(calibration),
                "--marker-model",
                str(marker_model_path),
                "--dictionary",
                "36h11",
                "--detection-sensitivity",
                "relaxed",
                "--object-model",
                str(object_model_path),
                "--overlay-cad-model",
                "--cad-model",
                str(cad_model_path),
            ]

            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "object_apriltag.cad": cad_module,
                        "object_apriltag.evaluation.cad_geometry": cad_geometry_module,
                        "object_apriltag.viz.cad_overlay": cad_overlay_module,
                    },
                ),
                mock.patch("sys.argv", argv),
                mock.patch(
                    "object_apriltag.cli.detect.load_intrinsics",
                    return_value=(
                        np.eye(3),
                        np.zeros(5),
                        640,
                        480,
                        "test",
                    ),
                ),
                mock.patch(
                    "object_apriltag.cli.detect.require_calibration_image_size",
                    return_value=(640, 480),
                ),
                mock.patch(
                    "object_apriltag.cli.detect.ObjectDetector",
                    return_value=detector,
                ),
                mock.patch(
                    "object_apriltag.cli.detect.load_object_model_document",
                    load_object_model_document,
                ),
                mock.patch(
                    "object_apriltag.cli.detect.open_frame_source",
                    side_effect=RuntimeError("stop-after-startup"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-startup"):
                    detect_main()

        cad_module.load_cad_registration.assert_not_called()
        cad_module.load_cad_landmarks.assert_called_once_with(cad_model_path)
        cad_geometry_module.fit_cad_registration.assert_called_once_with(
            cad_module.load_cad_landmarks.return_value,
            {"keypoint_sources": {}},
            marker_layout,
        )
        load_object_model_document.assert_called_once_with(object_model_path)

    def test_object_model_document_loaded_once_for_in_memory_fit_and_cad_only_overlay(
        self,
    ) -> None:
        from object_apriltag.cli.detect import main as detect_main
        from object_apriltag.viz import cad_overlay as real_cad_overlay

        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            calibration = directory / "intrinsics.json"
            marker_model_path = directory / "marker_model.json"
            object_model_path = directory / "object_model.json"
            cad_model_path = directory / "model.glb"
            for path in (
                calibration,
                marker_model_path,
                object_model_path,
                cad_model_path,
            ):
                path.write_bytes(b"fixture")

            marker_layout = mock.Mock(name="marker_layout", marker_ids={0, 1, 2})
            detector = mock.Mock(marker_model=marker_layout, marker_size_m=0.07)
            fitted_registration = mock.Mock(name="fitted_registration")
            cad_landmarks = mock.Mock(name="cad_landmarks")
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_landmarks=mock.Mock(return_value=cad_landmarks),
                load_cad_registration=mock.Mock(),
            )
            cad_geometry_module = mock.Mock(
                fit_cad_registration=mock.Mock(return_value=fitted_registration)
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                draw_cad_only_landmarks=mock.Mock(),
                object_model_landmark_names=real_cad_overlay.object_model_landmark_names,
            )
            object_model_document = {
                "keypoint_sources": {"shared": {}},
                "keypoints": {"shared": [0.0, 0.0, 0.0]},
            }
            load_object_model_document = mock.Mock(
                return_value=(mock.Mock(), object_model_document),
            )
            argv = [
                "object-detect",
                "--source",
                "0",
                "--calibration",
                str(calibration),
                "--marker-model",
                str(marker_model_path),
                "--dictionary",
                "36h11",
                "--detection-sensitivity",
                "relaxed",
                "--object-model",
                str(object_model_path),
                "--overlay-cad-model",
                "--cad-model",
                str(cad_model_path),
            ]

            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "object_apriltag.cad": cad_module,
                        "object_apriltag.evaluation.cad_geometry": cad_geometry_module,
                        "object_apriltag.viz.cad_overlay": cad_overlay_module,
                    },
                ),
                mock.patch("sys.argv", argv),
                mock.patch(
                    "object_apriltag.cli.detect.load_intrinsics",
                    return_value=(
                        np.eye(3),
                        np.zeros(5),
                        640,
                        480,
                        "test",
                    ),
                ),
                mock.patch(
                    "object_apriltag.cli.detect.require_calibration_image_size",
                    return_value=(640, 480),
                ),
                mock.patch(
                    "object_apriltag.cli.detect.ObjectDetector",
                    return_value=detector,
                ),
                mock.patch(
                    "object_apriltag.cli.detect.load_object_model_document",
                    load_object_model_document,
                ),
                mock.patch(
                    "object_apriltag.cli.detect.open_frame_source",
                    side_effect=RuntimeError("stop-after-startup"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop-after-startup"):
                    detect_main()

        load_object_model_document.assert_called_once_with(object_model_path)
        cad_geometry_module.fit_cad_registration.assert_called_once_with(
            cad_landmarks,
            object_model_document,
            marker_layout,
        )
        cad_overlay_module.draw_cad_only_landmarks.assert_not_called()

    def test_cad_overlay_is_allowed_with_pose_projection_overlay(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv(
                "--overlay-object-model",
                "--object-model",
                str(OBJECT_MODEL),
                "--overlay-cad-model",
                "--cad-model",
                str(cad_model),
            )
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_registration=mock.Mock(return_value=mock.Mock(name="cad_registration")),
            )
            cad_overlay_module = mock.Mock(draw_cad_model_overlay=mock.Mock())
            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch("object_apriltag.cli.detect.open_frame_source") as open_source:
                        open_source.side_effect = RuntimeError("stop-after-validation")
                        with self.assertRaises(RuntimeError) as ctx:
                            detect_main()
        self.assertNotIn("Only one pose projection overlay", str(ctx.exception))
        self.assertEqual(str(ctx.exception), "stop-after-validation")

    def test_cad_assets_load_before_opening_frame_source(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, registration = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv("--overlay-cad-model", "--cad-model", str(cad_model))
            call_order: list[str] = []
            loaded_cad = mock.Mock(name="cad_model")
            loaded_registration = mock.Mock(name="cad_registration")

            def _load_cad_model(path: Path) -> mock.Mock:
                call_order.append("load_cad_model")
                self.assertEqual(path, cad_model)
                return loaded_cad

            def _load_cad_registration(path: Path) -> mock.Mock:
                call_order.append("load_cad_registration")
                self.assertEqual(path, registration)
                return loaded_registration

            def _open_frame_source(*_args: object, **_kwargs: object) -> mock.Mock:
                call_order.append("open_frame_source")
                raise RuntimeError("stop-after-startup")

            cad_module = mock.Mock(
                load_cad_model=mock.Mock(side_effect=_load_cad_model),
                load_cad_registration=mock.Mock(side_effect=_load_cad_registration),
            )
            cad_overlay_module = mock.Mock(draw_cad_model_overlay=mock.Mock())

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        side_effect=_open_frame_source,
                    ):
                        with self.assertRaises(RuntimeError) as ctx:
                            detect_main()

        self.assertEqual(str(ctx.exception), "stop-after-startup")
        self.assertEqual(
            call_order,
            ["load_cad_model", "load_cad_registration", "open_frame_source"],
        )


class CliSide2SideCadModelTests(unittest.TestCase):
    def test_object_detect_requires_cad_model_when_side2side_enabled(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        argv = _base_detect_argv("--side2side-cad-model")
        with mock.patch("sys.argv", argv):
            with self.assertRaises(RuntimeError) as ctx:
                detect_main()
        self.assertIn("--cad-model", str(ctx.exception))

    def test_object_detect_rejects_side2side_without_preview(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv(
                "--no-preview",
                "--plot-graph",
                "--object-model",
                str(OBJECT_MODEL),
                "--side2side-cad-model",
                "--cad-model",
                str(cad_model),
            )
            with mock.patch("sys.argv", argv):
                with self.assertRaises(RuntimeError) as ctx:
                    detect_main()
        self.assertIn("--side2side-cad-model requires --preview", str(ctx.exception))

    def test_side2side_missing_registration_requires_object_model_for_generation(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model = Path(tmp_dir) / "model.glb"
            cad_model.write_bytes(b"glb")
            argv = _base_detect_argv("--side2side-cad-model", "--cad-model", str(cad_model))
            with mock.patch("sys.argv", argv):
                with self.assertRaises(RuntimeError) as ctx:
                    detect_main()
        self.assertIn("--object-model", str(ctx.exception))

    def test_cad_assets_load_once_when_both_cad_flags_enabled(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, registration = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv(
                "--overlay-cad-model",
                "--side2side-cad-model",
                "--cad-model",
                str(cad_model),
            )
            call_order: list[str] = []

            def _load_cad_model(path: Path) -> mock.Mock:
                call_order.append("load_cad_model")
                self.assertEqual(path, cad_model)
                return mock.Mock(name="cad_model")

            def _load_cad_registration(path: Path) -> mock.Mock:
                call_order.append("load_cad_registration")
                self.assertEqual(path, registration)
                return mock.Mock(name="cad_registration")

            cad_module = mock.Mock(
                load_cad_model=mock.Mock(side_effect=_load_cad_model),
                load_cad_registration=mock.Mock(side_effect=_load_cad_registration),
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                render_cad_model_view=mock.Mock(),
            )

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        side_effect=RuntimeError("stop-after-startup"),
                    ):
                        with self.assertRaises(RuntimeError) as ctx:
                            detect_main()

        self.assertEqual(str(ctx.exception), "stop-after-startup")
        self.assertEqual(call_order, ["load_cad_model", "load_cad_registration"])

    def test_side2side_coexists_with_overlay_cad_model(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv(
                "--overlay-cad-model",
                "--side2side-cad-model",
                "--cad-model",
                str(cad_model),
            )
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_registration=mock.Mock(return_value=mock.Mock(name="cad_registration")),
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                render_cad_model_view=mock.Mock(),
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        side_effect=RuntimeError("stop-after-validation"),
                    ):
                        with self.assertRaises(RuntimeError) as ctx:
                            detect_main()
        self.assertEqual(str(ctx.exception), "stop-after-validation")

    def test_side2side_renders_cad_pane_and_composes_camera_cad_graph(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv(
                "--side2side-cad-model",
                "--cad-model",
                str(cad_model),
                "--plot-graph",
                "--object-model",
                str(OBJECT_MODEL),
            )
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            fake_cad_view = np.full((480, 640, 3), 42, dtype=np.uint8)
            fake_plot = np.full((200, 300, 3), 99, dtype=np.uint8)
            fake_pose = mock.Mock(name="pose", rotation=mock.Mock(), origin=mock.Mock())
            loaded_cad = mock.Mock(name="cad_model")
            loaded_registration = mock.Mock(name="cad_registration")

            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=loaded_cad),
                load_cad_registration=mock.Mock(return_value=loaded_registration),
            )
            render_cad_model_view = mock.Mock(return_value=fake_cad_view)
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                render_cad_model_view=render_cad_model_view,
            )
            side_by_side_calls: list[tuple[str, str]] = []

            def _make_side_by_side(left: np.ndarray, right: np.ndarray, target_height: int) -> np.ndarray:
                side_by_side_calls.append((str(left.shape), str(right.shape)))
                return np.zeros((target_height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        return_value=mock.Mock(),
                    ):
                        with mock.patch(
                            "object_apriltag.cli.detect.read_frame",
                            side_effect=[(True, fake_frame), (False, None)],
                        ):
                            with mock.patch(
                                "object_apriltag.cli.detect.ObjectDetector"
                            ) as detector_cls:
                                detector = detector_cls.return_value
                                detector.marker_model = mock.Mock(
                                    marker_ids=[1],
                                    reference_marker_id=1,
                                )
                                detector.marker_size_m = 0.05
                                detector.find_markers.return_value = []
                                detector.fuse.return_value = fake_pose
                                with mock.patch(
                                    "object_apriltag.cli.detect.reference_marker_camera_position",
                                    return_value=None,
                                ):
                                    with mock.patch(
                                        "object_apriltag.cli.detect.layout_reprojection_errors",
                                        return_value=(0.1, 0.2),
                                    ):
                                        with mock.patch(
                                            "object_apriltag.cli.detect.object_world_points_from_pose",
                                            return_value={},
                                        ):
                                            with mock.patch(
                                                "object_apriltag.cli.detect.render_pose_plots",
                                                return_value=fake_plot,
                                            ):
                                                with mock.patch(
                                                    "object_apriltag.cli.detect.make_side_by_side",
                                                    side_effect=_make_side_by_side,
                                                ):
                                                    with mock.patch("object_apriltag.cli.detect.cv2.imshow"):
                                                        with mock.patch(
                                                            "object_apriltag.cli.detect.cv2.waitKey",
                                                            return_value=ord("q"),
                                                        ):
                                                            detect_main()

        render_cad_model_view.assert_called_once()
        render_args, render_kwargs = render_cad_model_view.call_args
        self.assertEqual(render_args[0], fake_frame.shape[:2])
        self.assertIs(render_args[1], fake_pose)
        self.assertEqual(len(side_by_side_calls), 2)
        self.assertEqual(side_by_side_calls[0][1], str(fake_cad_view.shape))
        self.assertEqual(side_by_side_calls[1][1], str(fake_plot.shape))

    def test_side2side_uses_black_pane_when_pose_unavailable(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv("--side2side-cad-model", "--cad-model", str(cad_model))
            fake_frame = np.ones((100, 120, 3), dtype=np.uint8) * 200
            render_cad_model_view = mock.Mock()
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_registration=mock.Mock(return_value=mock.Mock(name="cad_registration")),
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                render_cad_model_view=render_cad_model_view,
            )
            composed_right: np.ndarray | None = None

            def _make_side_by_side(left: np.ndarray, right: np.ndarray, _target_height: int) -> np.ndarray:
                nonlocal composed_right
                composed_right = right
                return np.hstack([left, right])

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        return_value=mock.Mock(),
                    ):
                        with mock.patch(
                            "object_apriltag.cli.detect.read_frame",
                            side_effect=[(True, fake_frame), (False, None)],
                        ):
                            with mock.patch(
                                "object_apriltag.cli.detect.ObjectDetector"
                            ) as detector_cls:
                                detector = detector_cls.return_value
                                detector.marker_model = mock.Mock(
                                    marker_ids=[1],
                                    reference_marker_id=1,
                                )
                                detector.marker_size_m = 0.05
                                detector.find_markers.return_value = []
                                detector.fuse.return_value = None
                                with mock.patch(
                                    "object_apriltag.cli.detect.make_side_by_side",
                                    side_effect=_make_side_by_side,
                                ):
                                    with mock.patch("object_apriltag.cli.detect.cv2.imshow"):
                                        with mock.patch(
                                            "object_apriltag.cli.detect.cv2.waitKey",
                                            return_value=ord("q"),
                                        ):
                                            detect_main()

        render_cad_model_view.assert_not_called()
        self.assertIsNotNone(composed_right)
        assert composed_right is not None
        self.assertTrue(np.all(composed_right == 0))

    def test_side2side_renders_under_no_visualize(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _base_detect_argv(
                "--no-visualize",
                "--side2side-cad-model",
                "--cad-model",
                str(cad_model),
            )
            fake_frame = np.ones((80, 100, 3), dtype=np.uint8) * 150
            fake_cad_view = np.full((80, 100, 3), 17, dtype=np.uint8)
            fake_pose = mock.Mock(name="pose", rotation=mock.Mock(), origin=mock.Mock())
            render_cad_model_view = mock.Mock(return_value=fake_cad_view)
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_registration=mock.Mock(return_value=mock.Mock(name="cad_registration")),
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=mock.Mock(),
                render_cad_model_view=render_cad_model_view,
            )

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        return_value=mock.Mock(),
                    ):
                        with mock.patch(
                            "object_apriltag.cli.detect.read_frame",
                            side_effect=[(True, fake_frame), (False, None)],
                        ):
                            with mock.patch(
                                "object_apriltag.cli.detect.ObjectDetector"
                            ) as detector_cls:
                                detector = detector_cls.return_value
                                detector.marker_model = mock.Mock(
                                    marker_ids=[1],
                                    reference_marker_id=1,
                                )
                                detector.marker_size_m = 0.05
                                detector.find_markers.return_value = []
                                detector.fuse.return_value = fake_pose
                                with mock.patch(
                                    "object_apriltag.cli.detect.reference_marker_camera_position",
                                    return_value=None,
                                ):
                                    with mock.patch(
                                        "object_apriltag.cli.detect.layout_reprojection_errors",
                                        return_value=(0.1, 0.2),
                                    ):
                                        with mock.patch(
                                            "object_apriltag.cli.detect.make_side_by_side"
                                        ) as side_by_side:
                                            side_by_side.side_effect = lambda left, right, _h: np.hstack(
                                                [left, right]
                                            )
                                            with mock.patch("object_apriltag.cli.detect.cv2.imshow"):
                                                with mock.patch(
                                                    "object_apriltag.cli.detect.cv2.waitKey",
                                                    return_value=ord("q"),
                                                ):
                                                    detect_main()

        render_cad_model_view.assert_called_once()
        cad_overlay_module.draw_cad_model_overlay.assert_not_called()


class CliCadOnlyLandmarkOverlayTests(unittest.TestCase):
    def test_overlay_draws_cad_only_landmarks_when_object_model_supplied(self) -> None:
        if not REMOTE_MARKER_MODEL.exists() or not REMOTE_OBJECT_MODEL.exists():
            self.skipTest("remote model fixtures are not available")
        from object_apriltag.cli.detect import main as detect_main
        from object_apriltag.viz import cad_overlay as real_cad_overlay

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _remote_detect_argv(
                "--overlay-cad-model",
                "--cad-model",
                str(cad_model),
                "--object-model",
                str(REMOTE_OBJECT_MODEL),
            )
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            fake_pose = mock.Mock(name="pose", rotation=mock.Mock(), origin=mock.Mock())
            loaded_cad = mock.Mock(name="cad_model")
            loaded_registration = mock.Mock(name="cad_registration")
            loaded_landmarks = mock.Mock(name="cad_landmarks")
            draw_cad_model_overlay = mock.Mock()
            draw_cad_only_landmarks = mock.Mock()
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=loaded_cad),
                load_cad_registration=mock.Mock(return_value=loaded_registration),
                load_cad_landmarks=mock.Mock(return_value=loaded_landmarks),
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=draw_cad_model_overlay,
                draw_cad_only_landmarks=draw_cad_only_landmarks,
                object_model_landmark_names=real_cad_overlay.object_model_landmark_names,
            )

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        return_value=mock.Mock(),
                    ):
                        with mock.patch(
                            "object_apriltag.cli.detect.read_frame",
                            side_effect=[(True, fake_frame), (False, None)],
                        ):
                            with mock.patch(
                                "object_apriltag.cli.detect.ObjectDetector"
                            ) as detector_cls:
                                detector = detector_cls.return_value
                                detector.marker_model = mock.Mock(
                                    marker_ids=[1],
                                    reference_marker_id=1,
                                )
                                detector.marker_size_m = 0.05
                                detector.find_markers.return_value = []
                                detector.fuse.return_value = fake_pose
                                with mock.patch(
                                    "object_apriltag.cli.detect.reference_marker_camera_position",
                                    return_value=None,
                                ):
                                    with mock.patch(
                                        "object_apriltag.cli.detect.layout_reprojection_errors",
                                        return_value=(0.1, 0.2),
                                    ):
                                        with mock.patch("object_apriltag.cli.detect.cv2.imshow"):
                                            with mock.patch(
                                                "object_apriltag.cli.detect.cv2.waitKey",
                                                return_value=ord("q"),
                                            ):
                                                detect_main()

        cad_module.load_cad_landmarks.assert_called_once_with(cad_model)
        draw_cad_model_overlay.assert_called_once()
        draw_cad_only_landmarks.assert_called_once()
        draw_args, draw_kwargs = draw_cad_only_landmarks.call_args
        self.assertEqual(draw_args[0].shape, fake_frame.shape)
        self.assertIs(draw_args[1], fake_pose)
        self.assertIs(draw_args[5], loaded_landmarks)
        self.assertIs(draw_args[6], loaded_registration)
        self.assertIsInstance(draw_args[7], frozenset)
        self.assertTrue(draw_args[7])

    def test_overlay_skips_cad_only_landmarks_without_object_model(self) -> None:
        if not REMOTE_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        from object_apriltag.cli.detect import main as detect_main

        with tempfile.TemporaryDirectory() as tmp_dir:
            cad_model, _ = _write_cad_assets(Path(tmp_dir))
            argv = _remote_detect_argv(
                "--overlay-cad-model",
                "--cad-model",
                str(cad_model),
            )
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            fake_pose = mock.Mock(name="pose", rotation=mock.Mock(), origin=mock.Mock())
            draw_cad_model_overlay = mock.Mock()
            draw_cad_only_landmarks = mock.Mock()
            cad_module = mock.Mock(
                load_cad_model=mock.Mock(return_value=mock.Mock(name="cad_model")),
                load_cad_registration=mock.Mock(return_value=mock.Mock(name="cad_registration")),
                load_cad_landmarks=mock.Mock(),
            )
            cad_overlay_module = mock.Mock(
                draw_cad_model_overlay=draw_cad_model_overlay,
                draw_cad_only_landmarks=draw_cad_only_landmarks,
            )

            with mock.patch.dict(
                sys.modules,
                {
                    "object_apriltag.cad": cad_module,
                    "object_apriltag.viz.cad_overlay": cad_overlay_module,
                },
            ):
                with mock.patch("sys.argv", argv):
                    with mock.patch(
                        "object_apriltag.cli.detect.open_frame_source",
                        return_value=mock.Mock(),
                    ):
                        with mock.patch(
                            "object_apriltag.cli.detect.read_frame",
                            side_effect=[(True, fake_frame), (False, None)],
                        ):
                            with mock.patch(
                                "object_apriltag.cli.detect.ObjectDetector"
                            ) as detector_cls:
                                detector = detector_cls.return_value
                                detector.marker_model = mock.Mock(
                                    marker_ids=[1],
                                    reference_marker_id=1,
                                )
                                detector.marker_size_m = 0.05
                                detector.find_markers.return_value = []
                                detector.fuse.return_value = fake_pose
                                with mock.patch(
                                    "object_apriltag.cli.detect.reference_marker_camera_position",
                                    return_value=None,
                                ):
                                    with mock.patch(
                                        "object_apriltag.cli.detect.layout_reprojection_errors",
                                        return_value=(0.1, 0.2),
                                    ):
                                        with mock.patch("object_apriltag.cli.detect.cv2.imshow"):
                                            with mock.patch(
                                                "object_apriltag.cli.detect.cv2.waitKey",
                                                return_value=ord("q"),
                                            ):
                                                detect_main()

        cad_module.load_cad_landmarks.assert_not_called()
        draw_cad_model_overlay.assert_called_once()
        draw_cad_only_landmarks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
