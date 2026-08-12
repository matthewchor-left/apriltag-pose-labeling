"""Regression tests for OpenCV-safe projected image coordinate conversion."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag.board_model import load_board_model
from object_apriltag.board_pose import BoardPoseEstimate
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import load_marker_model
from object_apriltag.viz.board_frame import (
    axis_segments_board,
    draw_board_axes,
    draw_projected_polyline,
)
from object_apriltag.viz.overlay import (
    draw_board_coordinate_preview,
    draw_object_orientation,
    draw_object_pose,
    draw_racket_keypoints,
)
from object_apriltag.viz.projection import object_axis_image_points, opencv_image_point
from object_apriltag.viz.skeleton import ObjectModel, load_object_model

REMOTE1_MARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/marker_model.json"
)
REMOTE1_OBJECT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/object_model.json"
)
DEFAULT_BOARD_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json"
)

HUGE_FINITE = 3.0e9
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def assert_puttext_origins_int32_safe(text_mock: mock.Mock) -> None:
    for call in text_mock.call_args_list:
        org = call.args[2]
        assert INT32_MIN <= org[0] <= INT32_MAX
        assert INT32_MIN <= org[1] <= INT32_MAX


def synthetic_camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


class OpenCvImagePointTests(unittest.TestCase):
    def test_accepts_normal_in_frame_coordinates(self) -> None:
        self.assertEqual(opencv_image_point((100.4, 200.6)), (100, 201))

    def test_rejects_non_finite(self) -> None:
        self.assertIsNone(opencv_image_point((np.nan, 1.0)))
        self.assertIsNone(opencv_image_point((1.0, np.inf)))

    def test_rejects_huge_finite_coordinates(self) -> None:
        self.assertIsNone(opencv_image_point((HUGE_FINITE, 10.0)))
        self.assertIsNone(opencv_image_point((10.0, HUGE_FINITE)))

    def test_rejects_rounding_past_int32_max(self) -> None:
        self.assertIsNone(opencv_image_point((INT32_MAX + 0.6, 0.0)))


class BoardCoordinatePreviewTests(unittest.TestCase):
    def test_draws_board_point_at_its_projected_pixel(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        camera_matrix, dist_coeffs = synthetic_camera()
        board_pose = BoardPoseEstimate(
            rotation=np.eye(3, dtype=np.float64),
            origin=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            reprojection_rms_px=0.0,
            detected_intersections=10,
            total_intersections=40,
        )

        with mock.patch("object_apriltag.viz.overlay.cv2.drawMarker") as marker_mock:
            draw_board_coordinate_preview(
                frame,
                np.zeros(3, dtype=np.float64),
                board_pose,
                camera_matrix,
                dist_coeffs,
                label="tip target",
            )

        self.assertEqual(marker_mock.call_args.args[1], (320, 240))


class DrawRacketKeypointsSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.object_model = ObjectModel(
            units="meters",
            keypoint_names=("center", "back"),
            keypoints={
                "center": np.zeros(3, dtype=np.float64),
                "back": np.zeros(3, dtype=np.float64),
            },
            skeleton_edges=(("center", "back"),),
            object_points=np.zeros((2, 3), dtype=np.float32),
        )

    def test_skips_unsafe_finite_line_endpoints_without_crashing(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image_points = np.array(
            [[200.0, 200.0], [HUGE_FINITE, 200.0]],
            dtype=np.float64,
        )

        with mock.patch("object_apriltag.viz.overlay.cv2.line") as line_mock:
            draw_racket_keypoints(frame, image_points, self.object_model, draw_point_labels=False)

        line_mock.assert_not_called()

    def test_skips_unsafe_finite_circle_and_text(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image_points = np.array(
            [[HUGE_FINITE, 200.0], [200.0, 200.0]],
            dtype=np.float64,
        )

        with (
            mock.patch("object_apriltag.viz.overlay.cv2.circle") as circle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock,
        ):
            draw_racket_keypoints(frame, image_points, self.object_model, draw_point_labels=True)

        for call in circle_mock.call_args_list:
            center = call.args[1]
            self.assertLess(abs(center[0]), INT32_MAX)
            self.assertLess(abs(center[1]), INT32_MAX)
        for call in text_mock.call_args_list:
            org = call.args[2]
            self.assertGreaterEqual(org[0], INT32_MIN)
            self.assertLessEqual(org[0], INT32_MAX)
            self.assertGreaterEqual(org[1], INT32_MIN)
            self.assertLessEqual(org[1], INT32_MAX)

    def test_skips_label_when_offset_origin_overflows_int32(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        image_points = np.array(
            [[float(INT32_MAX - 4), 200.0], [200.0, 200.0]],
            dtype=np.float64,
        )

        with (
            mock.patch("object_apriltag.viz.overlay.cv2.circle") as circle_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock,
        ):
            draw_racket_keypoints(frame, image_points, self.object_model, draw_point_labels=True)

        drawn_labels = {call.args[1] for call in text_mock.call_args_list}
        self.assertNotIn("center", drawn_labels)
        assert_puttext_origins_int32_safe(text_mock)


class DrawObjectOrientationLabelSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = synthetic_camera()
        self.pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )

    def test_skips_axis_labels_when_offset_origin_overflows_int32(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        axis_points = (
            (100, 100),
            (INT32_MAX - 2, 100),
            (100, 100),
            (100, INT32_MIN + 3),
        )

        with (
            mock.patch(
                "object_apriltag.viz.overlay.object_axis_image_points",
                return_value=axis_points,
            ),
            mock.patch("object_apriltag.viz.overlay.cv2.arrowedLine"),
            mock.patch("object_apriltag.viz.overlay.cv2.putText") as text_mock,
        ):
            draw_object_orientation(
                frame,
                self.pose,
                self.camera_matrix,
                self.dist_coeffs,
                axis_length_m=0.5,
            )

        drawn_labels = {call.args[1] for call in text_mock.call_args_list}
        self.assertNotIn("X", drawn_labels)
        self.assertNotIn("Z", drawn_labels)
        assert_puttext_origins_int32_safe(text_mock)


class DrawObjectPoseSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = synthetic_camera()
        self.marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        self.object_model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)

    def test_near_camera_plane_axes_do_not_crash(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 1e-8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )

        draw_object_pose(
            frame,
            pose,
            self.camera_matrix,
            self.dist_coeffs,
            marker_size_m=0.07,
            model=self.object_model,
            marker_model=self.marker_model,
            draw_point_labels=False,
        )

    def test_preserves_keypoint_identity_when_middle_projection_invalid(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        three_keypoint_model = ObjectModel(
            units="meters",
            keypoint_names=("a", "b", "c"),
            keypoints={
                "a": np.array([0.0, 0.0, 0.0], dtype=np.float64),
                "b": np.array([0.01, 0.0, 0.0], dtype=np.float64),
                "c": np.array([0.02, 0.0, 0.0], dtype=np.float64),
            },
            skeleton_edges=(("a", "b"), ("b", "c")),
            object_points=np.zeros((3, 3), dtype=np.float32),
        )

        projected = [
            np.array([200.0, 200.0], dtype=np.float64),
            np.array([HUGE_FINITE, 200.0], dtype=np.float64),
            np.array([220.0, 200.0], dtype=np.float64),
        ]

        with (
            mock.patch("object_apriltag.viz.overlay.draw_object_origin", return_value=None),
            mock.patch(
                "object_apriltag.viz.overlay.object_axis_image_points",
                return_value=((0, 0), (0, 0), (0, 0), (0, 0)),
            ),
            mock.patch(
                "object_apriltag.viz.overlay.project_camera_point",
                side_effect=projected,
            ),
            mock.patch("object_apriltag.viz.overlay.cv2.line") as line_mock,
            mock.patch("object_apriltag.viz.overlay.cv2.circle") as circle_mock,
        ):
            draw_object_pose(
                frame,
                pose,
                self.camera_matrix,
                self.dist_coeffs,
                marker_size_m=0.07,
                model=three_keypoint_model,
                marker_model=self.marker_model,
                draw_point_labels=False,
            )

        line_mock.assert_not_called()
        self.assertEqual(circle_mock.call_count, 4)


class ObjectAxisImagePointsSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = synthetic_camera()

    def test_raises_when_projection_unsafe(self) -> None:
        with self.assertRaises(ValueError):
            object_axis_image_points(
                np.eye(3, dtype=np.float64),
                np.array([0.0, 0.0, 1e-7], dtype=np.float64),
                self.camera_matrix,
                self.dist_coeffs,
                axis_length_m=0.5,
            )


class BoardFrameProjectionSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = synthetic_camera()
        self.board_model = load_board_model(DEFAULT_BOARD_MODEL_PATH)
        self.pose = BoardPoseEstimate(
            rotation=np.eye(3, dtype=np.float64),
            origin=np.array([0.0, 0.0, 0.5], dtype=np.float64),
            reprojection_rms_px=0.0,
            detected_intersections=10,
            total_intersections=40,
        )

    def test_draw_projected_polyline_skips_unsafe_coordinates(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        unsafe_points = np.array(
            [[100.0, 100.0], [HUGE_FINITE, 100.0], [200.0, 200.0]],
            dtype=np.float64,
        )

        with mock.patch("object_apriltag.viz.board_frame.cv2.polylines") as polyline_mock:
            draw_projected_polyline(frame, unsafe_points, (255, 255, 255), thickness=1)

        polyline_mock.assert_not_called()

    def test_draw_board_axes_skips_unsafe_axis_polylines_and_labels(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        unsafe_pose = BoardPoseEstimate(
            rotation=np.eye(3, dtype=np.float64),
            origin=np.array([0.0, 0.0, 1e-8], dtype=np.float64),
            reprojection_rms_px=0.0,
            detected_intersections=10,
            total_intersections=40,
        )

        with (
            mock.patch("object_apriltag.viz.board_frame.cv2.polylines") as polyline_mock,
            mock.patch("object_apriltag.viz.board_frame.cv2.putText") as text_mock,
            mock.patch("object_apriltag.viz.board_frame.cv2.circle") as circle_mock,
        ):
            draw_board_axes(
                frame,
                unsafe_pose,
                self.board_model,
                self.camera_matrix,
                self.dist_coeffs,
                axis_length_squares=2.0,
            )

        self.assertEqual(polyline_mock.call_count, 1)
        self.assertEqual(text_mock.call_count, 1)
        self.assertEqual(text_mock.call_args.args[1], "Z")
        circle_mock.assert_not_called()

        for call in polyline_mock.call_args_list:
            pixels = call.args[1][0]
            self.assertLessEqual(int(np.max(np.abs(pixels))), INT32_MAX)

    def test_y_origin_fallback_skips_label_when_offset_overflows_int32(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        y_origin = np.array([float(INT32_MAX - 10), float(INT32_MAX - 4)], dtype=np.float64)
        y_end = np.array([float(INT32_MAX - 10), float(INT32_MAX - 3)], dtype=np.float64)
        x_segment = axis_segments_board(self.board_model, 2.0)["x"]

        def fake_project_board_polyline(
            segment_board: np.ndarray,
            *_args: object,
            **_kwargs: object,
        ) -> np.ndarray:
            if np.array_equal(segment_board, x_segment):
                return np.array([[320.0, 240.0], [400.0, 240.0]], dtype=np.float64)
            return np.stack([y_origin, y_end], axis=0)

        with (
            mock.patch(
                "object_apriltag.viz.board_frame.project_board_polyline",
                side_effect=fake_project_board_polyline,
            ),
            mock.patch("object_apriltag.viz.board_frame.cv2.polylines"),
            mock.patch("object_apriltag.viz.board_frame.cv2.circle") as circle_mock,
            mock.patch("object_apriltag.viz.board_frame.cv2.putText") as text_mock,
        ):
            draw_board_axes(
                frame,
                self.pose,
                self.board_model,
                self.camera_matrix,
                self.dist_coeffs,
                axis_length_squares=2.0,
            )

        self.assertEqual(circle_mock.call_count, 1)
        y_label_calls = [call for call in text_mock.call_args_list if call.args[1] == "Y"]
        self.assertEqual(len(y_label_calls), 1)
        self.assertEqual(y_label_calls[0].args[2], (INT32_MAX - 10, INT32_MAX - 3))
        assert_puttext_origins_int32_safe(text_mock)


if __name__ == "__main__":
    unittest.main()
