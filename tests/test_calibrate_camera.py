"""Tests for board calibration helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.cli.charuco import (
    A4_HEIGHT_M,
    A4_WIDTH_M,
    build_charuco_board,
    checkerboard_object_points,
    detect_charuco,
    detect_checkerboard,
    meters_to_pixels,
    save_charuco_board_a4,
    write_intrinsics_json,
)


class CalibrateCameraHelperTests(unittest.TestCase):
    def test_checkerboard_object_points_grid(self) -> None:
        points = checkerboard_object_points(3, 2, 0.05)
        self.assertEqual(points.shape, (6, 3))
        np.testing.assert_allclose(points[1], [0.05, 0.0, 0.0])

    def test_charuco_detection_on_generated_board(self) -> None:
        board = build_charuco_board(10, 7, 0.024, 0.018, "4x4_50")
        detector = cv2.aruco.CharucoDetector(board)
        image = board.generateImage((1200, 1200))
        gray = image.copy()
        detection = detect_charuco(gray, board, detector)
        self.assertIsNotNone(detection)
        object_points, image_points = detection
        self.assertGreaterEqual(len(object_points), 4)
        self.assertEqual(object_points.shape[1], 3)
        self.assertEqual(image_points.shape[1], 2)

    def test_checkerboard_detection_on_synthetic_grid(self) -> None:
        layout_width, layout_height = 9, 6
        square_px = 40
        width = layout_width * square_px + square_px
        height = layout_height * square_px + square_px
        image = np.zeros((height, width), dtype=np.uint8)
        for row in range(layout_height + 1):
            for col in range(layout_width + 1):
                if (row + col) % 2 == 0:
                    y0 = row * square_px
                    x0 = col * square_px
                    image[y0:y0 + square_px, x0:x0 + square_px] = 255
        detection = detect_checkerboard(image, layout_width, layout_height, 0.024)
        self.assertIsNotNone(detection)
        object_points, image_points = detection
        self.assertEqual(len(object_points), layout_width * layout_height)
        self.assertEqual(len(image_points), layout_width * layout_height)

    def test_write_intrinsics_json_matches_loader_fields(self) -> None:
        camera_matrix = np.eye(3, dtype=np.float64)
        dist_coeffs = np.zeros(5, dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intrinsics.json"
            write_intrinsics_json(
                path,
                calibration_source="charuco",
                image_width=640,
                image_height=480,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                mean_reprojection_error_px=0.5,
                layout_width=10,
                layout_height=7,
                square_size=0.024,
                captured_frames=12,
                marker_size=0.018,
                dictionary="4x4_50",
            )
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["calibration_source"], "charuco")
        self.assertEqual(data["image_size"], [640, 480])
        self.assertEqual(data["squares_x"], 10)
        self.assertEqual(data["squares_y"], 7)
        self.assertEqual(data["marker_size"], 0.018)

    def test_save_board_a4_true_scale(self) -> None:
        dpi = 300.0
        square_size = 0.025
        layout_width, layout_height = 9, 6
        board = build_charuco_board(layout_width, layout_height, square_size, 0.02, "4x4_50")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            save_charuco_board_a4(
                board,
                path,
                layout_width=layout_width,
                layout_height=layout_height,
                square_size=square_size,
                dpi=dpi,
            )
            from PIL import Image

            with Image.open(path) as image:
                dpi_x, dpi_y = image.info.get("dpi", (0.0, 0.0))
                self.assertAlmostEqual(dpi_x, dpi, delta=0.01)
                self.assertAlmostEqual(dpi_y, dpi, delta=0.01)
                page_w = meters_to_pixels(A4_WIDTH_M, dpi)
                page_h = meters_to_pixels(A4_HEIGHT_M, dpi)
                self.assertEqual(image.size, (page_w, page_h))
        board_w_px = meters_to_pixels(layout_width * square_size, dpi)
        self.assertEqual(board_w_px, meters_to_pixels(9 * 0.025, dpi))


if __name__ == "__main__":
    unittest.main()
