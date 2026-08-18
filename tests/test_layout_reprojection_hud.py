"""Tests for layout reprojection metrics and shared status HUD rendering."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.layout import (
    layout_point_to_camera,
    load_marker_model,
    object_reference_origin,
)
from object_apriltag.pose import layout_reprojection_errors, reference_marker_camera_position
from object_apriltag.viz.overlay import (
    draw_live_hud,
    draw_status_hud_panel,
    format_reference_marker_camera_line,
)
from object_apriltag.viz.plots import LiveHud

REMOTE1_MARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/marker_model.json"
)


def synthetic_camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def synthetic_object_pose() -> tuple[np.ndarray, np.ndarray]:
    rotation = np.array(
        [
            [0.98, 0.0, 0.2],
            [0.0, 1.0, 0.0],
            [-0.2, 0.0, 0.98],
        ],
        dtype=np.float64,
    )
    origin = np.array([0.05, -0.02, 0.55], dtype=np.float64)
    return rotation, origin


class LayoutReprojectionErrorTests(unittest.TestCase):
    def test_exact_layout_corners_have_zero_error(self) -> None:
        layout = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        camera_matrix, dist_coeffs = synthetic_camera()
        object_rotation, object_origin = synthetic_object_pose()
        zero_rvec = np.zeros(3, dtype=np.float64)
        zero_tvec = np.zeros(3, dtype=np.float64)

        detections: list[tuple[np.ndarray, int]] = []
        for marker_id in sorted(layout.footprints):
            footprint = layout.footprints[marker_id]
            image_corners: list[list[float]] = []
            for point_layout in footprint.corners():
                camera_point = layout_point_to_camera(
                    point_layout, object_rotation, object_origin, layout
                )
                projected, _ = cv2.projectPoints(
                    camera_point.reshape(1, 1, 3).astype(np.float64),
                    zero_rvec,
                    zero_tvec,
                    camera_matrix,
                    dist_coeffs,
                )
                image_corners.append(projected.reshape(2).tolist())
            corners = np.asarray(image_corners, dtype=np.float32).reshape(1, 4, 2)
            detections.append((corners, marker_id))

        result = layout_reprojection_errors(
            detections,
            object_rotation,
            object_origin,
            layout,
            camera_matrix,
            dist_coeffs,
        )
        self.assertIsNotNone(result)
        mean_error, max_error = result
        self.assertLess(mean_error, 1e-3)
        self.assertLess(max_error, 1e-3)

    def test_perturbed_corners_increase_layout_error(self) -> None:
        layout = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        camera_matrix, dist_coeffs = synthetic_camera()
        object_rotation, object_origin = synthetic_object_pose()
        zero_rvec = np.zeros(3, dtype=np.float64)
        zero_tvec = np.zeros(3, dtype=np.float64)

        marker_id = layout.reference_marker_id
        footprint = layout.footprints[marker_id]
        image_corners: list[list[float]] = []
        for point_layout in footprint.corners():
            camera_point = layout_point_to_camera(
                point_layout, object_rotation, object_origin, layout
            )
            projected, _ = cv2.projectPoints(
                camera_point.reshape(1, 1, 3).astype(np.float64),
                zero_rvec,
                zero_tvec,
                camera_matrix,
                dist_coeffs,
            )
            image_corners.append(projected.reshape(2).tolist())
        corners = np.asarray(image_corners, dtype=np.float32).reshape(1, 4, 2)
        corners[0, 0, 0] += 5.0
        detections = [(corners, marker_id)]

        result = layout_reprojection_errors(
            detections,
            object_rotation,
            object_origin,
            layout,
            camera_matrix,
            dist_coeffs,
        )
        self.assertIsNotNone(result)
        mean_error, max_error = result
        self.assertGreater(mean_error, 1.0)
        self.assertGreater(max_error, 4.0)

    def test_unknown_markers_and_missing_pose_return_none(self) -> None:
        layout = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        camera_matrix, dist_coeffs = synthetic_camera()
        object_rotation, object_origin = synthetic_object_pose()
        corners = np.zeros((1, 4, 2), dtype=np.float32)

        self.assertIsNone(
            layout_reprojection_errors(
                [(corners, 999)],
                object_rotation,
                object_origin,
                layout,
                camera_matrix,
                dist_coeffs,
            )
        )
        self.assertIsNone(
            layout_reprojection_errors(
                [],
                object_rotation,
                object_origin,
                layout,
                camera_matrix,
                dist_coeffs,
            )
        )


class LiveHudLayoutReprojectionTests(unittest.TestCase):
    def test_window_averages_means_and_reports_current_frame_max(self) -> None:
        hud = LiveHud(reproj_window=3)
        hud.tick(1.0, 10.0)
        hud.tick(2.0, 2.0)
        _, avg, max_error = hud.tick(3.0, 5.0)
        self.assertAlmostEqual(avg, 2.0)
        self.assertAlmostEqual(max_error, 5.0)

    def test_current_frame_max_does_not_carry_prior_peak(self) -> None:
        hud = LiveHud(reproj_window=3)
        hud.tick(1.0, 10.0)
        _, _, max_error = hud.tick(2.0, 2.0)
        self.assertAlmostEqual(max_error, 2.0)

    def test_skips_reprojection_when_values_missing(self) -> None:
        hud = LiveHud(reproj_window=3)
        hud.tick(1.0, 2.0)
        _, avg, max_error = hud.tick()
        self.assertAlmostEqual(avg, 1.0)
        self.assertIsNone(max_error)


class ReferenceMarkerCameraPositionTests(unittest.TestCase):
    def test_matches_layout_projection_of_reference_center(self) -> None:
        layout = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        object_rotation, object_origin = synthetic_object_pose()
        expected = layout_point_to_camera(
            object_reference_origin(layout),
            object_rotation,
            object_origin,
            layout,
        )
        actual = reference_marker_camera_position(
            object_rotation, object_origin, layout
        )
        np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_format_reference_marker_camera_line(self) -> None:
        self.assertEqual(
            format_reference_marker_camera_line(14, np.array([0.5, -0.03, 0.78])),
            "ref 14 cam xyz (m): 0.500, -0.030, 0.780",
        )
        self.assertEqual(
            format_reference_marker_camera_line(14, None),
            "ref 14 cam xyz (m): --",
        )


class StatusHudPanelTests(unittest.TestCase):
    def test_draw_live_hud_uses_shared_panel_style(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with (
            mock.patch("object_apriltag.viz.overlay.cv2.rectangle") as rectangle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock,
        ):
            draw_live_hud(
                frame,
                fps=30.0,
                layout_reproj_avg=1.2,
                layout_reproj_max=3.4,
                reference_marker_id=0,
                reference_marker_camera_m=np.array([0.5, -0.03, 0.78]),
            )

        rectangle_mock.assert_called_once()
        self.assertEqual(rectangle_mock.call_args.args[4], -1)
        rendered = [call.args[1] for call in text_mock.call_args_list]
        self.assertEqual(rendered[0], "FPS: 30.0")
        self.assertEqual(rendered[1], "layout reproj avg: 1.2px")
        self.assertEqual(rendered[2], "layout reproj max: 3.4px")
        self.assertEqual(rendered[3], "ref 0 cam xyz (m): 0.500, -0.030, 0.780")
        self.assertTrue(all(call.args[5] == (255, 255, 255) for call in text_mock.call_args_list))

    def test_draw_live_hud_shows_dashes_when_unavailable(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock:
            draw_live_hud(frame, fps=24.0, layout_reproj_avg=None, layout_reproj_max=None)

        rendered = [call.args[1] for call in text_mock.call_args_list]
        self.assertEqual(rendered[1], "layout reproj avg: --")
        self.assertEqual(rendered[2], "layout reproj max: --")

    def test_draw_status_hud_panel_uses_single_white_pass(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with (
            mock.patch("object_apriltag.viz.overlay.cv2.rectangle") as rectangle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock,
        ):
            draw_status_hud_panel(frame, ["line one", "line two"])

        rectangle_mock.assert_called_once()
        self.assertEqual(rectangle_mock.call_args.args[4], -1)
        self.assertEqual(len(text_mock.call_args_list), 2)
        self.assertTrue(all(call.args[5] == (255, 255, 255) for call in text_mock.call_args_list))


if __name__ == "__main__":
    unittest.main()
