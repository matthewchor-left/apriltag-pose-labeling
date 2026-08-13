"""Tests for board coordinate overlay helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag.board_pose import BoardPoseEstimate
from object_apriltag.detector import ObjectPose
from object_apriltag.eraser import load_eraser_model
from object_apriltag.layout import CORNER_LABELS, CORNER_NAMES, load_marker_model
from object_apriltag.viz.overlay import (
    draw_board_coordinates_hud,
    draw_eraser_model_board_coordinate_labels,
    draw_marker_model_board_coordinate_labels,
    draw_marker_model_footprints,
    draw_object_model_board_coordinate_labels,
    draw_object_pose,
    draw_racket_keypoints,
    format_board_coordinate_hud_row,
    format_board_coordinate_mm,
)
from object_apriltag.viz.skeleton import load_object_model

REMOTE1_MARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/marker_model.json"
)
REMOTE1_OBJECT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/object_model.json"
)
REMOTE1_ERASER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/eraser_model.json"
)


def synthetic_camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def synthetic_board_pose() -> BoardPoseEstimate:
    return BoardPoseEstimate(
        rotation=np.eye(3, dtype=np.float64),
        origin=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        reprojection_rms_px=0.0,
        detected_intersections=10,
        total_intersections=40,
    )


def synthetic_object_pose() -> ObjectPose:
    return ObjectPose(
        origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
        rotation=np.eye(3, dtype=np.float64),
    )


def unique_texts(texts: list[str]) -> list[str]:
    return list(dict.fromkeys(texts))


def semantic_lines(texts: list[str]) -> tuple[list[str], list[str]]:
    unique = unique_texts(texts)
    hud_rows = [line for line in unique if line.endswith(" mm") and ": (" in line]
    identities = [row.rsplit(": ", 1)[0] for row in hud_rows]
    coordinates = [row.rsplit(": ", 1)[1] for row in hud_rows]
    return identities, coordinates


class BoardCoordinateFormatTests(unittest.TestCase):
    def test_format_board_coordinate_mm_one_decimal(self) -> None:
        self.assertEqual(
            format_board_coordinate_mm(np.array([0.0123, -0.0456, 0.0789])),
            "(12.3, -45.6, 78.9) mm",
        )

    def test_format_board_coordinate_mm_zero(self) -> None:
        self.assertEqual(format_board_coordinate_mm(np.zeros(3)), "(0.0, 0.0, 0.0) mm")


class BoardCoordinateHudTests(unittest.TestCase):
    def test_format_board_coordinate_hud_row(self) -> None:
        self.assertEqual(
            format_board_coordinate_hud_row("center", np.array([0.01, 0.02, 0.03])),
            "center: (10.0, 20.0, 30.0) mm",
        )

    def test_draw_board_coordinates_hud_writes_title_and_rows(self) -> None:
        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        texts: list[str] = []

        def capture_put_text(
            _image: np.ndarray,
            text: str,
            _org: tuple[int, int],
            *_args: object,
            **_kwargs: object,
        ) -> None:
            texts.append(text)

        with mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=capture_put_text):
            draw_board_coordinates_hud(
                frame,
                [
                    ("center", np.array([0.01, 0.02, 0.03])),
                    ("top", np.array([0.0, 0.0, 0.0])),
                ],
            )

        unique = unique_texts(texts)
        self.assertIn("Board coordinates", unique)
        self.assertIn("center: (10.0, 20.0, 30.0) mm", unique)
        self.assertIn("top: (0.0, 0.0, 0.0) mm", unique)

    def test_draw_board_coordinates_hud_truncates_when_panel_overflows(self) -> None:
        frame = np.zeros((40, 320, 3), dtype=np.uint8)
        texts: list[str] = []

        def capture_put_text(
            _image: np.ndarray,
            text: str,
            _org: tuple[int, int],
            *_args: object,
            **_kwargs: object,
        ) -> None:
            texts.append(text)

        entries = [(f"p{i}", np.zeros(3, dtype=np.float64)) for i in range(8)]
        with mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=capture_put_text):
            draw_board_coordinates_hud(frame, entries)

        unique = unique_texts(texts)
        self.assertTrue(any(text.startswith("... +") for text in unique))
        self.assertNotIn("p7: (0.0, 0.0, 0.0) mm", unique)


class BoardCoordinateOverlayFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = synthetic_camera()
        self.board_pose = synthetic_board_pose()
        self.object_pose = synthetic_object_pose()
        self.marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        self.object_model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)
        self.eraser_model = load_eraser_model(REMOTE1_ERASER_MODEL_PATH)

    def _draw_and_collect_texts(self, draw_callable: object) -> list[str]:
        frame = np.zeros((900, 640, 3), dtype=np.uint8)
        texts: list[str] = []

        def capture_put_text(
            _image: np.ndarray,
            text: str,
            _org: tuple[int, int],
            *_args: object,
            **_kwargs: object,
        ) -> None:
            texts.append(text)

        with mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=capture_put_text):
            draw_callable(frame)

        return texts

    def test_object_model_overlay_emits_keypoint_identities_and_mm_coordinates(self) -> None:
        texts = self._draw_and_collect_texts(
            lambda frame: draw_object_model_board_coordinate_labels(
                frame,
                self.object_pose,
                self.board_pose,
                self.marker_model,
                self.object_model,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )
        identities, coordinates = semantic_lines(texts)
        self.assertEqual(set(identities), set(self.object_model.keypoint_names))
        self.assertEqual(len(coordinates), len(self.object_model.keypoint_names))
        self.assertTrue(all(text.endswith(" mm") for text in coordinates))

    def test_marker_model_overlay_emits_marker_corner_identities_and_mm_coordinates(self) -> None:
        texts = self._draw_and_collect_texts(
            lambda frame: draw_marker_model_board_coordinate_labels(
                frame,
                self.object_pose,
                self.board_pose,
                self.marker_model,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )
        expected_identities = {
            f"{marker_id}:{CORNER_LABELS[corner_name]}"
            for marker_id in sorted(self.marker_model.footprints)
            for corner_name in self.marker_model.footprints[marker_id].corners_by_name()
        }
        identities, coordinates = semantic_lines(texts)
        self.assertEqual(set(identities), expected_identities)
        self.assertEqual(len(coordinates), len(expected_identities))
        self.assertIn("0:tl", identities)
        self.assertIn("1:br", identities)

    def test_eraser_model_overlay_emits_plane_corner_identities_and_mm_coordinates(self) -> None:
        texts = self._draw_and_collect_texts(
            lambda frame: draw_eraser_model_board_coordinate_labels(
                frame,
                self.object_pose,
                self.board_pose,
                self.eraser_model,
                self.marker_model,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )
        expected_identities = {
            f"{plane.plane_id if plane.plane_id is not None else str(index)}:{CORNER_LABELS[corner_name]}"
            for index, plane in enumerate(self.eraser_model.planes)
            for corner_name in CORNER_NAMES
        }
        identities, coordinates = semantic_lines(texts)
        self.assertEqual(set(identities), expected_identities)
        self.assertTrue(all(text.endswith(" mm") for text in coordinates))
        self.assertEqual(
            sum(1 for text in unique_texts(texts) if text.endswith(" mm")),
            len(expected_identities),
        )
        self.assertIn("stick1:tl", identities)
        self.assertIn("stick2:br", identities)

    def test_hud_lists_all_keypoints_regardless_of_image_size(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        texts: list[str] = []

        def capture_put_text(
            _image: np.ndarray,
            text: str,
            _org: tuple[int, int],
            *_args: object,
            **_kwargs: object,
        ) -> None:
            texts.append(text)

        with mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=capture_put_text):
            draw_object_model_board_coordinate_labels(
                frame,
                self.object_pose,
                self.board_pose,
                self.marker_model,
                self.object_model,
                self.camera_matrix,
                self.dist_coeffs,
            )
        identities, coordinates = semantic_lines(texts)
        self.assertEqual(set(identities), set(self.object_model.keypoint_names))
        self.assertEqual(len(coordinates), len(self.object_model.keypoint_names))


class OrdinaryLabelSuppressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = synthetic_camera()
        self.object_pose = synthetic_object_pose()
        self.marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        self.object_model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)

    def test_draw_racket_keypoints_suppresses_names_but_draws_geometry(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image_points = np.full((len(self.object_model.keypoint_names), 2), 200.0, dtype=np.float32)
        texts: list[str] = []

        with (
            mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=lambda _i, t, *_a, **_k: texts.append(t)),
            mock.patch("object_apriltag.viz.overlay.cv2.circle") as circle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.line") as line_mock,
        ):
            draw_racket_keypoints(frame, image_points, self.object_model, draw_point_labels=False)

        self.assertFalse(any(name in texts for name in self.object_model.keypoint_names))
        self.assertGreater(circle_mock.call_count, 0)
        self.assertGreater(line_mock.call_count, 0)

    def test_draw_object_pose_suppresses_keypoint_names_but_draws_geometry(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        texts: list[str] = []

        with (
            mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=lambda _i, t, *_a, **_k: texts.append(t)),
            mock.patch("object_apriltag.viz.overlay.cv2.circle") as circle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.arrowedLine") as arrow_mock,
        ):
            draw_object_pose(
                frame,
                self.object_pose,
                self.camera_matrix,
                self.dist_coeffs,
                marker_size_m=0.07,
                model=self.object_model,
                marker_model=self.marker_model,
                draw_point_labels=False,
            )

        self.assertFalse(any(name in texts for name in self.object_model.keypoint_names))
        self.assertGreater(circle_mock.call_count, 0)
        self.assertGreater(arrow_mock.call_count, 0)

    def test_draw_marker_model_footprints_suppresses_corner_labels_but_draws_geometry(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        texts: list[str] = []
        expected_corner_labels = {
            f"{marker_id}:{CORNER_LABELS[corner_name]}"
            for marker_id in sorted(self.marker_model.footprints)
            for corner_name in self.marker_model.footprints[marker_id].corners_by_name()
        }

        with (
            mock.patch("object_apriltag.viz.overlay.cv2.putText", side_effect=lambda _i, t, *_a, **_k: texts.append(t)),
            mock.patch("object_apriltag.viz.overlay.cv2.circle") as circle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.polylines") as polyline_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.arrowedLine") as arrow_mock,
        ):
            draw_marker_model_footprints(
                frame,
                self.object_pose,
                self.camera_matrix,
                self.dist_coeffs,
                self.marker_model,
                draw_point_labels=False,
            )

        self.assertFalse(any(label in texts for label in expected_corner_labels))
        self.assertGreater(circle_mock.call_count, 0)
        self.assertGreater(polyline_mock.call_count, 0)
        self.assertGreaterEqual(arrow_mock.call_count, 3)
        self.assertEqual(set(texts), {"X", "Y", "Z"})


if __name__ == "__main__":
    unittest.main()
