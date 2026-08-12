"""Tests for board pose estimation and frame overlay helpers."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.board_model import (
    build_charuco_board_from_geometry,
    load_board_model,
    reject_mixed_board_model_args,
)
from object_apriltag.board_pose import (
    BoardPoseEstimate,
    board_point_to_camera,
    board_points_to_opencv,
    camera_point_to_board,
    charuco_corners_consistent,
    charuco_draw_arrays,
    detect_charuco_intersections,
    estimate_board_pose,
    make_charuco_detector,
    select_board_pose,
    solve_board_pose,
)
from object_apriltag.cli.charuco import draw_charuco_overlay
from object_apriltag.cli.visualize_board_frame import process_frame, validate_image_size
from object_apriltag.viz.board_frame import (
    board_extent_m,
    grid_line_points_board,
    project_board_points,
    sample_board_segment,
)


DEFAULT_BOARD_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json"
)
DEFAULT_BOARD_IMAGE_PATH = DEFAULT_BOARD_MODEL_PATH.with_name("a4_borderless.png")


def synthetic_camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


class BoardPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = load_board_model(DEFAULT_BOARD_MODEL_PATH)
        self.board, self.detector = make_charuco_detector(self.model)
        self.camera_matrix, self.dist_coeffs = synthetic_camera()

    def test_board_points_to_opencv_matches_origin(self) -> None:
        np.testing.assert_allclose(board_points_to_opencv(np.zeros(3)), np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(
            board_points_to_opencv(np.array([0.1, 0.2, 0.3])),
            np.array([0.1, 0.3, -0.2]),
            atol=1e-9,
        )

    def test_board_camera_roundtrip(self) -> None:
        pose = BoardPoseEstimate(
            rotation=np.array(
                [[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]],
                dtype=np.float64,
            ),
            origin=np.array([0.1, 0.2, 0.8], dtype=np.float64),
            reprojection_rms_px=0.0,
            detected_intersections=10,
            total_intersections=40,
        )
        board_point = np.array([0.05, -0.02, 0.11], dtype=np.float64)
        camera_point = board_point_to_camera(board_point, pose)
        recovered = camera_point_to_board(camera_point, pose)
        np.testing.assert_allclose(recovered, board_point, atol=1e-5)
        np.testing.assert_allclose(
            camera_point_to_board(board_point_to_camera(np.zeros(3), pose), pose),
            np.zeros(3),
            atol=1e-9,
        )

    def test_select_board_pose_static_retains_last_valid(self) -> None:
        first = BoardPoseEstimate(
            rotation=np.eye(3),
            origin=np.array([0.0, 0.0, 0.5]),
            reprojection_rms_px=0.1,
            detected_intersections=8,
            total_intersections=40,
        )
        second = BoardPoseEstimate(
            rotation=np.eye(3),
            origin=np.array([0.1, 0.0, 0.5]),
            reprojection_rms_px=0.1,
            detected_intersections=8,
            total_intersections=40,
        )

        usable, retained = select_board_pose(first, None, "static")
        self.assertIs(usable, first)
        self.assertIs(retained, first)

        usable, retained = select_board_pose(second, retained, "static")
        self.assertIs(usable, second)
        self.assertIs(retained, second)

        usable, retained = select_board_pose(None, retained, "static")
        self.assertIs(usable, second)
        self.assertIs(retained, second)

    def test_select_board_pose_dynamic_clears_on_dropout(self) -> None:
        current = BoardPoseEstimate(
            rotation=np.eye(3),
            origin=np.array([0.0, 0.0, 0.5]),
            reprojection_rms_px=0.1,
            detected_intersections=8,
            total_intersections=40,
        )
        retained = BoardPoseEstimate(
            rotation=np.eye(3),
            origin=np.array([0.2, 0.0, 0.5]),
            reprojection_rms_px=0.1,
            detected_intersections=8,
            total_intersections=40,
        )

        usable, kept = select_board_pose(current, retained, "dynamic")
        self.assertIs(usable, current)
        self.assertIs(kept, retained)

        usable, kept = select_board_pose(None, retained, "dynamic")
        self.assertIsNone(usable)
        self.assertIs(kept, retained)

    def test_pose_estimation_on_synthetic_projection(self) -> None:
        image = self.board.generateImage((900, 600))
        observation = detect_charuco_intersections(image, self.board, self.detector)
        self.assertIsNotNone(observation)
        assert observation is not None

        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        projected, _ = cv2.projectPoints(
            observation.object_points_opencv.reshape(-1, 1, 3).astype(np.float32),
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        synthetic_observation = type(observation)(
            object_points_opencv=observation.object_points_opencv,
            image_points=projected.reshape(-1, 2).astype(np.float64),
            charuco_corners=projected.reshape(-1, 1, 2),
            charuco_ids=observation.charuco_ids,
        )
        pose = estimate_board_pose(
            synthetic_observation,
            self.model,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertGreaterEqual(pose.detected_intersections, 4)
        self.assertLess(pose.reprojection_rms_px, 1.0)
        y_in_camera = pose.rotation @ np.array([0.0, 1.0, 0.0])
        self.assertLess(float(y_in_camera[2]), 0.0)

    def test_board_image_recovers_known_front_facing_pose(self) -> None:
        page = cv2.imread(str(DEFAULT_BOARD_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(page)
        assert page is not None

        board_width_px = round(
            self.model.layout_width * self.model.square_size / 0.0254 * 300
        )
        board_height_px = round(
            self.model.layout_height * self.model.square_size / 0.0254 * 300
        )
        x0 = (page.shape[1] - board_width_px) // 2
        y0 = (page.shape[0] - board_height_px) // 2
        board_image = page[
            y0 : y0 + board_height_px,
            x0 : x0 + board_width_px,
        ]

        camera_matrix = np.array(
            [[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        expected_origin = np.array([-0.1, -0.1375, 0.65], dtype=np.float64)
        board_width_m = self.model.layout_width * self.model.square_size
        board_height_m = self.model.layout_height * self.model.square_size
        object_corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [board_width_m, 0.0, 0.0],
                [board_width_m, board_height_m, 0.0],
                [0.0, board_height_m, 0.0],
            ],
            dtype=np.float32,
        )
        image_corners, _ = cv2.projectPoints(
            object_corners,
            np.zeros(3),
            expected_origin,
            camera_matrix,
            dist_coeffs,
        )
        source_corners = np.array(
            [
                [0.0, 0.0],
                [board_width_px - 1.0, 0.0],
                [board_width_px - 1.0, board_height_px - 1.0],
                [0.0, board_height_px - 1.0],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(
            source_corners,
            image_corners.reshape(4, 2).astype(np.float32),
        )
        camera_image = cv2.warpPerspective(
            board_image,
            homography,
            (1920, 1080),
            borderValue=255,
        )

        preview, pose = process_frame(
            cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR),
            model=self.model,
            board=self.board,
            detector=self.detector,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            grid_margin_squares=0,
            axis_length_squares=2.0,
            show_intersections=False,
            show_hud=False,
            source_label="test board image",
        )

        self.assertIsNotNone(pose)
        assert pose is not None
        np.testing.assert_allclose(pose.origin, expected_origin, atol=1.2e-2)
        expected_rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(pose.rotation, expected_rotation, atol=6e-2)
        self.assertAlmostEqual(np.linalg.det(pose.rotation), 1.0, places=6)

        axis_length = 2.0 * self.model.square_size
        expected_axis_endpoints_camera = np.array(
            [
                expected_origin + [axis_length, 0.0, 0.0],
                expected_origin + [0.0, 0.0, -axis_length],
                expected_origin + [0.0, axis_length, 0.0],
            ],
            dtype=np.float64,
        )
        endpoints, _ = cv2.projectPoints(
            expected_axis_endpoints_camera,
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            dist_coeffs,
        )
        expected_dominant_channels = (2, 1, 0)  # X red, Y green, Z blue in BGR.
        for endpoint, channel in zip(
            endpoints.reshape(-1, 2),
            expected_dominant_channels,
            strict=True,
        ):
            x, y = np.round(endpoint).astype(int)
            neighborhood = preview[y - 15 : y + 16, x - 15 : x + 16].astype(
                np.int16
            )
            dominant = neighborhood[:, :, channel]
            other = np.max(np.delete(neighborhood, channel, axis=2), axis=2)
            self.assertTrue(
                np.any((dominant > 150) & (dominant > other + 50)),
                f"axis color missing near expected endpoint {(x, y)}",
            )

    def test_estimate_board_pose_returns_none_for_three_points_without_crashing(self) -> None:
        object_points = np.zeros((3, 3), dtype=np.float64)
        image_points = np.zeros((3, 2), dtype=np.float64)
        observation = type("Obs", (), {})()
        observation.object_points_opencv = object_points
        observation.image_points = image_points
        observation.charuco_ids = np.arange(3, dtype=np.int32)
        pose = estimate_board_pose(
            observation,
            self.model,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNone(pose)

    def test_pose_dropouts_return_none(self) -> None:
        blank = np.zeros((480, 640), dtype=np.uint8)
        observation = detect_charuco_intersections(blank, self.board, self.detector)
        self.assertIsNone(observation)
        result = solve_board_pose(
            blank, self.board, self.detector, self.model, self.camera_matrix, self.dist_coeffs
        )
        self.assertIsNone(result.pose)
        self.assertEqual(result.no_pose_reason, "no ChArUco intersections detected")

    def test_solve_board_pose_reports_insufficient_intersections(self) -> None:
        image = self.board.generateImage((900, 600))
        corners, ids, _, _ = self.detector.detectBoard(image)
        ids_three = ids[:3]
        corners_three = corners[:3]
        with mock.patch.object(
            type(self.detector),
            "detectBoard",
            return_value=(corners_three, ids_three, None, None),
        ):
            result = solve_board_pose(
                image,
                self.board,
                self.detector,
                self.model,
                self.camera_matrix,
                self.dist_coeffs,
            )
        self.assertIsNone(result.pose)
        self.assertEqual(result.no_pose_reason, "need >= 4 intersections (detected 3)")
        self.assertEqual(result.detected_intersections, 3)

    def test_grid_generation_spacing(self) -> None:
        x_min, x_max, z_min, z_max = board_extent_m(self.model, grid_margin_squares=2)
        self.assertAlmostEqual(x_min, -0.05)
        self.assertAlmostEqual(x_max, 0.25)
        x_lines, z_lines = grid_line_points_board(self.model, grid_margin_squares=2)
        self.assertGreater(len(x_lines), self.model.layout_height)
        self.assertGreater(len(z_lines), self.model.layout_width)
        first_gap = float(x_lines[1][0, 2] - x_lines[0][0, 2])
        self.assertAlmostEqual(first_gap, self.model.square_size)

    def test_sample_board_segment_count(self) -> None:
        points = sample_board_segment(np.zeros(3), np.array([1.0, 0.0, 0.0]), samples=5)
        self.assertEqual(points.shape, (5, 3))

    def test_project_board_points_identity_pose(self) -> None:
        from object_apriltag.board_pose import BoardPoseEstimate

        pose = BoardPoseEstimate(
            rotation=np.eye(3),
            origin=np.array([0.0, 0.5, 2.0]),
            reprojection_rms_px=0.0,
            detected_intersections=10,
            total_intersections=40,
        )
        projected = project_board_points(
            np.array([[0.0, 0.0, 0.0], [0.025, 0.0, 0.0]]),
            pose,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertEqual(projected.shape, (2, 2))
        self.assertGreater(projected[1, 0], projected[0, 0])

    def test_validate_image_size_rejects_mismatch(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_image_size(640, 480, 1280, 720, Path("calib.json"))

    def test_reject_mixed_board_model_args(self) -> None:
        args = argparse.Namespace(
            board_model=Path("board.json"),
            board_type="charuco_board",
            layout=[6, 9],
            marker_size=0.02,
            square_size=0.025,
            dictionary="4x4_50",
        )
        with self.assertRaises(RuntimeError):
            reject_mixed_board_model_args(args)

    def test_charuco_cli_board_model_save_board(self) -> None:
        from object_apriltag.cli.charuco import main as charuco_main

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "board.png"
            with mock.patch(
                "sys.argv",
                [
                    "object-charuco",
                    "--board-model",
                    str(DEFAULT_BOARD_MODEL_PATH),
                    "--save-board",
                    str(output),
                ],
            ):
                charuco_main()
            self.assertTrue(output.exists())


class CharucoOverlayGuardTests(unittest.TestCase):
    def test_corners_consistent_rejects_mismatched_arrays(self) -> None:
        ids = np.array([[0], [1]], dtype=np.int32)
        corners = np.zeros((1, 1, 2), dtype=np.float32)
        self.assertFalse(charuco_corners_consistent(corners, ids))

    def test_draw_overlay_skips_invalid_charuco_without_crashing(self) -> None:
        model = load_board_model(DEFAULT_BOARD_MODEL_PATH)
        board, detector = make_charuco_detector(model)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = frame[:, :, 0]
        ids = np.array([[0], [1]], dtype=np.int32)
        corners = np.zeros((1, 1, 2), dtype=np.float32)

        with mock.patch.object(
            type(detector),
            "detectBoard",
            return_value=(corners, ids, None, None),
        ):
            detected = draw_charuco_overlay(frame, gray, board, detector)

        self.assertFalse(detected)

    def test_draw_overlay_handles_opencv5_flat_arrays(self) -> None:
        model = load_board_model(DEFAULT_BOARD_MODEL_PATH)
        board, detector = make_charuco_detector(model)
        img = board.generateImage((800, 800))
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        frame = np.zeros((800, 800, 3), dtype=np.uint8)
        draw_arrays = charuco_draw_arrays(corners, ids)
        self.assertIsNotNone(draw_arrays)
        corners_draw, ids_draw = draw_arrays
        cv2.aruco.drawDetectedCornersCharuco(frame, corners_draw, ids_draw)
        detected = draw_charuco_overlay(frame, gray, board, detector)
        self.assertTrue(detected)


if __name__ == "__main__":
    unittest.main()
