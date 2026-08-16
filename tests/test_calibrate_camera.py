"""Tests for board calibration helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.cli.charuco import (
    A4_HEIGHT_M,
    A4_WIDTH_M,
    build_charuco_board,
    capture_from_video,
    checkerboard_object_points,
    detect_charuco,
    detect_checkerboard,
    meters_to_pixels,
    planar_calibration_view_usable,
    require_positive_sample_rate,
    save_charuco_board_a4,
    solve_and_write_intrinsics,
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


def _expected_video_sample_count(frame_count: int, fps: float, sample_rate_hz: float) -> int:
    sample_interval = 1.0 / sample_rate_hz
    next_sample_time = 0.0
    count = 0
    for frame_index in range(frame_count):
        video_time = frame_index / fps
        if video_time >= next_sample_time:
            count += 1
            next_sample_time = video_time + sample_interval
    return count


class VideoCaptureTests(unittest.TestCase):
    def test_sample_rate_must_be_finite_and_positive(self) -> None:
        for sample_rate_hz in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(sample_rate_hz=sample_rate_hz):
                with self.assertRaisesRegex(RuntimeError, "--sample-rate-hz"):
                    require_positive_sample_rate(sample_rate_hz)

    def test_video_capture_samples_detected_frames_without_preview(self) -> None:
        board = build_charuco_board(10, 7, 0.024, 0.018, "4x4_50")
        detector = cv2.aruco.CharucoDetector(board)
        image = board.generateImage((1200, 1200))
        frame = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        fps = 10.0
        frame_count = 20
        sample_rate_hz = 2.0
        frames = [frame] * frame_count
        fake_capture = mock.Mock()
        fake_capture.get.return_value = fps
        read_state = {"index": 0}

        def fake_read(_capture, _source, *, loop_on_eof: bool = True):
            self.assertFalse(loop_on_eof)
            index = read_state["index"]
            if index >= len(frames):
                return False, None
            read_state["index"] = index + 1
            return True, frames[index]

        with (
            mock.patch(
                "object_apriltag.cli.charuco.open_frame_source",
                return_value=fake_capture,
            ),
            mock.patch("object_apriltag.cli.charuco.read_frame", side_effect=fake_read),
            mock.patch("object_apriltag.cli.charuco.cv2.imshow") as imshow,
            mock.patch("object_apriltag.cli.charuco.cv2.waitKey") as wait_key,
        ):
            object_points, image_points, width, height = capture_from_video(
                Path("calibration.mov"),
                board_type="charuco_board",
                layout_width=10,
                layout_height=7,
                square_size=0.024,
                sample_rate_hz=sample_rate_hz,
                charuco_board=board,
                charuco_detector=detector,
            )

        expected = _expected_video_sample_count(frame_count, fps, sample_rate_hz)
        self.assertEqual(len(object_points), expected)
        self.assertEqual(len(image_points), expected)
        self.assertEqual((width, height), (1200, 1200))
        self.assertGreaterEqual(expected, 3)
        self.assertLess(expected, frame_count)
        imshow.assert_not_called()
        wait_key.assert_not_called()
        fake_capture.release.assert_called_once()


def _project_points(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    camera_matrix = np.array(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    return image_points.reshape(-1, 2).astype(np.float32)


class PlanarCalibrationViewTests(unittest.TestCase):
    def test_collinear_points_are_not_usable(self) -> None:
        line = np.array(
            [[0.0, 0.0, 0.0], [0.025, 0.0, 0.0], [0.05, 0.0, 0.0], [0.075, 0.0, 0.0]],
            dtype=np.float32,
        )
        image_points = _project_points(
            line,
            np.zeros((3, 1), dtype=np.float64),
            np.array([[0.0], [0.0], [0.5]], dtype=np.float64),
        )
        self.assertFalse(planar_calibration_view_usable(line, image_points))

    def test_solve_drops_collinear_views_instead_of_opencv_assert(self) -> None:
        grid = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.025, 0.025, 0.0],
                [0.0, 0.025, 0.0],
                [0.05, 0.0, 0.0],
                [0.05, 0.025, 0.0],
            ],
            dtype=np.float32,
        )
        line = np.array(
            [[0.0, 0.0, 0.0], [0.025, 0.0, 0.0], [0.05, 0.0, 0.0], [0.075, 0.0, 0.0]],
            dtype=np.float32,
        )
        object_points_list = []
        image_points_list = []
        for yaw in (0.1, -0.2, 0.3):
            rvec = np.array([[0.2], [yaw], [0.05]], dtype=np.float64)
            tvec = np.array([[0.0], [0.0], [0.5]], dtype=np.float64)
            object_points_list.append(grid.copy())
            image_points_list.append(_project_points(grid, rvec, tvec))
        object_points_list.append(line)
        image_points_list.append(
            _project_points(
                line,
                np.zeros((3, 1), dtype=np.float64),
                np.array([[0.0], [0.0], [0.5]], dtype=np.float64),
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "intrinsics.json"
            solve_and_write_intrinsics(
                output,
                board_type="charuco_board",
                object_points_list=object_points_list,
                image_points_list=image_points_list,
                image_width=1280,
                image_height=720,
                layout_width=8,
                layout_height=11,
                square_size=0.025,
                marker_size=0.02,
                dictionary="4x4_50",
            )
            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["captured_frames"], 3)


if __name__ == "__main__":
    unittest.main()
