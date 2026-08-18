"""Tests for eraser-plane tag erasure and annotation-tool CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.eraser import (
    EraserPlane,
    clip_polygon_to_rect,
    erase_with_mask,
    erase_with_planes,
    eraser_offset_to_model_point,
    load_eraser_model,
    plane_from_dict,
    project_eraser_plane,
    project_eraser_planes,
)
from object_apriltag.layout import load_marker_model

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MARKER_MODEL = REPO_ROOT / "config/Model/remote/marker_model.json"

REMOTE1_ERASER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/eraser_model.json"
)


class EraserOffsetTests(unittest.TestCase):
    def setUp(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        self.marker_model = load_marker_model(TEST_MARKER_MODEL)

    def test_offset_maps_to_reference_marker_center(self) -> None:
        point = eraser_offset_to_model_point(np.array([0.01, -0.02, 0.0]), self.marker_model)
        np.testing.assert_allclose(point, [0.01, -0.02, 0.0])


class EraseWithMaskTests(unittest.TestCase):
    def test_pastes_plate_pixels_inside_mask(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        plate = np.zeros_like(frame)
        plate[:, :] = (0, 255, 0)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array([[20, 20], [60, 20], [60, 60], [20, 60]]), 255)

        erased = erase_with_mask(frame, plate, mask)

        self.assertEqual(tuple(erased[30, 30]), (0, 255, 0))
        self.assertEqual(tuple(erased[0, 0]), (255, 255, 255))

    def test_requires_matching_frame_and_plate_shapes(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        plate = np.zeros((30, 30, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        with self.assertRaises(ValueError):
            erase_with_mask(frame, plate, mask)


class EraseWithPlanesTests(unittest.TestCase):
    def test_unions_multiple_plane_polygons(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        plate = np.zeros_like(frame)
        polygons = [
            np.array([[10, 10], [40, 10], [40, 40]], dtype=np.float64),
            np.array([[60, 60], [90, 60], [90, 90]], dtype=np.float64),
        ]

        erased = erase_with_planes(frame, plate, polygons)

        self.assertEqual(tuple(erased[20, 20]), (0, 0, 0))
        self.assertEqual(tuple(erased[70, 70]), (0, 0, 0))
        self.assertEqual(tuple(erased[0, 0]), (255, 255, 255))

    def test_returns_copy_when_no_polygons(self) -> None:
        frame = np.full((40, 40, 3), 127, dtype=np.uint8)
        plate = np.zeros_like(frame)
        erased = erase_with_planes(frame, plate, [])
        np.testing.assert_array_equal(erased, frame)


class ClipPolygonToRectTests(unittest.TestCase):
    def test_clips_polygon_at_left_image_edge(self) -> None:
        polygon = np.array([[-20.0, 40.0], [60.0, 40.0], [60.0, 80.0], [-20.0, 80.0]], dtype=np.float32)
        clipped = clip_polygon_to_rect(polygon, width=100, height=100)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertGreaterEqual(clipped[:, 0].min(), 0.0)
        self.assertAlmostEqual(clipped[:, 0].min(), 0.0, places=6)

    def test_returns_none_when_polygon_is_fully_outside(self) -> None:
        polygon = np.array([[-50.0, -50.0], [-40.0, -50.0], [-40.0, -40.0]], dtype=np.float32)
        self.assertIsNone(clip_polygon_to_rect(polygon, width=100, height=100))


class ProjectEraserPlaneTests(unittest.TestCase):
    def setUp(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        self.marker_model = load_marker_model(TEST_MARKER_MODEL)
        self.camera_matrix = np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def test_projects_plane_for_pose_facing_camera(self) -> None:
        plane = EraserPlane(
            plane_id="front",
            top_left=np.array([-0.024, -0.024, 0.0]),
            top_right=np.array([0.024, -0.024, 0.0]),
            bottom_right=np.array([0.024, 0.024, 0.0]),
            bottom_left=np.array([-0.024, 0.024, 0.0]),
        )
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.8], dtype=np.float64)

        polygon = project_eraser_plane(
            plane.corners(),
            rotation,
            origin,
            self.marker_model,
            self.camera_matrix,
            self.dist_coeffs,
            image_width=640,
            image_height=480,
        )

        self.assertIsNotNone(polygon)
        assert polygon is not None
        self.assertGreaterEqual(len(polygon), 3)

    def test_returns_none_when_all_corners_are_behind_camera(self) -> None:
        plane = EraserPlane(
            plane_id="front",
            top_left=np.array([-0.024, -0.024, 0.0]),
            top_right=np.array([0.024, -0.024, 0.0]),
            bottom_right=np.array([0.024, 0.024, 0.0]),
            bottom_left=np.array([-0.024, 0.024, 0.0]),
        )
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([0.0, 0.0, -0.5], dtype=np.float64)

        polygon = project_eraser_plane(
            plane.corners(),
            rotation,
            origin,
            self.marker_model,
            self.camera_matrix,
            self.dist_coeffs,
            image_width=640,
            image_height=480,
        )

        self.assertIsNone(polygon)

    def test_off_screen_corners_are_clipped_to_image_edge(self) -> None:
        plane = EraserPlane(
            plane_id="front",
            top_left=np.array([-0.024, -0.024, 0.0]),
            top_right=np.array([0.024, -0.024, 0.0]),
            bottom_right=np.array([0.024, 0.024, 0.0]),
            bottom_left=np.array([-0.024, 0.024, 0.0]),
        )
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([-0.15, 0.0, 0.5], dtype=np.float64)

        polygon = project_eraser_plane(
            plane.corners(),
            rotation,
            origin,
            self.marker_model,
            self.camera_matrix,
            self.dist_coeffs,
            image_width=640,
            image_height=480,
        )

        self.assertIsNotNone(polygon)
        assert polygon is not None
        self.assertGreaterEqual(polygon[:, 0].min(), 0.0)
        self.assertGreaterEqual(polygon[:, 1].min(), 0.0)
        self.assertLessEqual(polygon[:, 0].max(), 640.0)
        self.assertLessEqual(polygon[:, 1].max(), 480.0)


class LoadEraserModelTests(unittest.TestCase):
    def test_loads_default_eraser_model(self) -> None:
        eraser_path = REPO_ROOT / "config/Model/remote/eraser_model.json"
        if not eraser_path.exists():
            self.skipTest("remote eraser model fixture is not available")
        model = load_eraser_model(eraser_path)
        self.assertEqual(model.origin, "reference_marker_center")
        self.assertGreater(len(model.planes), 0)

    def test_remote1_loads_plane_id_names(self) -> None:
        if not REMOTE1_ERASER_MODEL_PATH.exists():
            self.skipTest("remote1 eraser model fixture is not available")
        model = load_eraser_model(REMOTE1_ERASER_MODEL_PATH)
        self.assertEqual(
            [plane.plane_id for plane in model.planes],
            ["0", "1", "2", "3", "stick1", "stick2"],
        )

    def test_plane_from_dict_prefers_plane_id_over_legacy_id(self) -> None:
        plane = plane_from_dict(
            {
                "plane_id": "named",
                "id": "legacy",
                "top_left": [-0.01, -0.01, 0.0],
                "top_right": [0.01, -0.01, 0.0],
                "bottom_right": [0.01, 0.01, 0.0],
                "bottom_left": [-0.01, 0.01, 0.0],
            },
            0,
        )
        self.assertEqual(plane.plane_id, "named")

    def test_plane_from_dict_falls_back_to_legacy_id(self) -> None:
        plane = plane_from_dict(
            {
                "id": "legacy",
                "top_left": [-0.01, -0.01, 0.0],
                "top_right": [0.01, -0.01, 0.0],
                "bottom_right": [0.01, 0.01, 0.0],
                "bottom_left": [-0.01, 0.01, 0.0],
            },
            0,
        )
        self.assertEqual(plane.plane_id, "legacy")

    def test_projects_all_planes_from_model(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        eraser_path = REPO_ROOT / "config/Model/remote/eraser_model.json"
        if not eraser_path.exists():
            self.skipTest("remote eraser model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        model = load_eraser_model(eraser_path)
        camera_matrix = np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.8], dtype=np.float64)

        polygons = project_eraser_planes(
            model,
            rotation,
            origin,
            marker_model,
            camera_matrix,
            dist_coeffs,
            image_width=640,
            image_height=480,
        )

        self.assertEqual(len(polygons), len(model.planes))


class AnnotationToolCliTests(unittest.TestCase):
    def test_requires_object_model_when_registration_is_missing(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "intrinsic.json"
            calibration.write_text(
                json.dumps(
                    {
                        "camera_matrix": [[900, 0, 320], [0, 900, 240], [0, 0, 1]],
                        "dist_coeffs": [[0], [0], [0], [0], [0]],
                        "image_width": 640,
                        "image_height": 480,
                    }
                ),
                encoding="utf-8",
            )
            marker_model = root / "marker_model.json"
            marker_model.write_text(
                TEST_MARKER_MODEL.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            cad_model = root / "model.glb"
            cad_model.write_bytes(b"placeholder")
            argv = [
                "annotation-tool",
                "--source",
                "0",
                "--calibration",
                str(calibration),
                "--marker-model",
                str(marker_model),
                "--cad-model",
                str(cad_model),
                "--output",
                str(root / "dataset"),
                "--split",
                "train",
                "--run-name",
                "cli_run",
                "--sample-rate-hz",
                "1",
                "--dictionary",
                "36h11",
                "--detection-sensitivity",
                "relaxed",
            ]
            with mock.patch("sys.argv", argv):
                from object_apriltag.cli import annotation_tool

                with self.assertRaisesRegex(RuntimeError, "cad_registration.json"):
                    annotation_tool.main()

    def test_fits_registration_when_object_model_is_provided(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        nodes = []
        for index, name in enumerate(
            (
                "back-center",
                "back-left-center",
                "back-right-center",
                "front-center",
                "front-left-center",
                "front-right-center",
                "left-center",
                "right-center",
                "top-back-center",
                "top-back-left",
                "top-back-right",
                "top-center",
                "top-front-center",
                "top-front-left",
                "top-front-right",
                "top-left-center",
                "top-right-center",
            )
        ):
            nodes.append({"name": name, "translation": [0.01 * index, 0.0, 0.0]})
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "scenes": [{"nodes": list(range(len(nodes)))}],
        }
        json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_padding = (4 - (len(json_chunk) % 4)) % 4
        json_chunk += b" " * json_padding
        header = b"glTF" + (2).to_bytes(4, "little") + (12 + 8 + len(json_chunk) + 8).to_bytes(4, "little")
        json_header = len(json_chunk).to_bytes(4, "little") + b"JSON"
        bin_header = (0).to_bytes(4, "little") + b"BIN\x00"
        glb_bytes = header + json_header + json_chunk + bin_header

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / "intrinsic.json"
            calibration.write_text(
                json.dumps(
                    {
                        "camera_matrix": [[900, 0, 320], [0, 900, 240], [0, 0, 1]],
                        "dist_coeffs": [[0], [0], [0], [0], [0]],
                        "image_width": 640,
                        "image_height": 480,
                    }
                ),
                encoding="utf-8",
            )
            marker_model = root / "marker_model.json"
            marker_model.write_text(
                TEST_MARKER_MODEL.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            object_model = root / "object_model.json"
            object_model.write_text("{}", encoding="utf-8")
            cad_model = root / "model.glb"
            cad_model.write_bytes(glb_bytes)
            (root / "clip.mov").write_bytes(b"\x00")
            argv = [
                "annotation-tool",
                "--source",
                str(root / "clip.mov"),
                "--calibration",
                str(calibration),
                "--marker-model",
                str(marker_model),
                "--cad-model",
                str(cad_model),
                "--object-model",
                str(object_model),
                "--output",
                str(root / "dataset"),
                "--split",
                "train",
                "--run-name",
                "cli_fit_run",
                "--sample-rate-hz",
                "1",
                "--dictionary",
                "36h11",
                "--detection-sensitivity",
                "relaxed",
            ]
            fitted_registration = mock.Mock(name="cad_registration")
            with (
                mock.patch("sys.argv", argv),
                mock.patch(
                    "object_apriltag.cli.annotation_tool.load_cad_model",
                    return_value=mock.Mock(name="cad_model"),
                ),
                mock.patch(
                    "object_apriltag.cli.annotation_tool.load_required_yolo_landmarks",
                    return_value=mock.Mock(landmarks={"back-center": np.zeros(3)}),
                ),
                mock.patch(
                    "object_apriltag.cli.annotation_tool.load_object_model_document",
                    return_value=(mock.Mock(), {"keypoint_sources": {"back-center": {}}}),
                ),
                mock.patch(
                    "object_apriltag.evaluation.cad_geometry.fit_cad_registration",
                    return_value=fitted_registration,
                ) as fit_mock,
                mock.patch(
                    "object_apriltag.cli.annotation_tool.generate_dataset_from_source",
                    return_value=mock.Mock(
                        frames_processed=0,
                        samples_saved=0,
                        rejections=mock.Mock(no_pose=0, landmarks=0, bbox=0),
                    ),
                ) as generate_mock,
            ):
                from object_apriltag.cli import annotation_tool

                annotation_tool.main()

            fit_mock.assert_called_once()
            generate_mock.assert_called_once()
            self.assertIs(generate_mock.call_args.kwargs["registration"], fitted_registration)


if __name__ == "__main__":
    unittest.main()
