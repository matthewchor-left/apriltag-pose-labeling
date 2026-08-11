"""Tests for layout-bounds tag erasure."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from paddle_apriltag.cli.annotation_tool import (
    clip_polygon_to_rect,
    erase_with_hull,
    layout_bounds_corners,
    project_layout_bounds_hull,
)
from paddle_apriltag.layout import DEFAULT_MARKER_LAYOUT_PATH, load_marker_layout


class LayoutBoundsCornersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)

    def test_returns_eight_corners(self) -> None:
        corners = layout_bounds_corners(self.layout, padding_m=0.02)
        self.assertEqual(corners.shape, (8, 3))

    def test_padding_expands_axis_limits(self) -> None:
        tight = layout_bounds_corners(self.layout, padding_m=0.0)
        padded = layout_bounds_corners(self.layout, padding_m=0.05)
        self.assertLess(padded[:, 0].min(), tight[:, 0].min())
        self.assertGreater(padded[:, 0].max(), tight[:, 0].max())


class EraseWithHullTests(unittest.TestCase):
    def test_pastes_plate_pixels_inside_hull(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        plate = np.zeros_like(frame)
        plate[:, :] = (0, 255, 0)
        hull = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], dtype=np.float32)

        erased = erase_with_hull(frame, plate, hull)

        self.assertEqual(tuple(erased[30, 30]), (0, 255, 0))
        self.assertEqual(tuple(erased[0, 0]), (255, 255, 255))

    def test_returns_copy_when_hull_is_none(self) -> None:
        frame = np.full((40, 40, 3), 127, dtype=np.uint8)
        plate = np.zeros_like(frame)
        erased = erase_with_hull(frame, plate, None)
        np.testing.assert_array_equal(erased, frame)

    def test_requires_matching_frame_and_plate_shapes(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        plate = np.zeros((30, 30, 3), dtype=np.uint8)
        hull = np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)
        with self.assertRaises(ValueError):
            erase_with_hull(frame, plate, hull)


class ClipPolygonToRectTests(unittest.TestCase):
    def test_clips_polygon_at_left_image_edge(self) -> None:
        polygon = np.array([[-20.0, 40.0], [60.0, 40.0], [60.0, 80.0], [-20.0, 80.0]], dtype=np.float32)
        clipped = clip_polygon_to_rect(polygon, width=100, height=100)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertGreaterEqual(clipped[:, 0].min(), 0.0)
        self.assertAlmostEqual(clipped[:, 0].min(), 0.0, places=6)
        self.assertAlmostEqual(clipped[:, 1].min(), 40.0, places=6)
        self.assertAlmostEqual(clipped[:, 1].max(), 80.0, places=6)

    def test_returns_none_when_polygon_is_fully_outside(self) -> None:
        polygon = np.array([[-50.0, -50.0], [-40.0, -50.0], [-40.0, -40.0]], dtype=np.float32)
        self.assertIsNone(clip_polygon_to_rect(polygon, width=100, height=100))


class ProjectLayoutBoundsHullTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        self.camera_matrix = np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def test_projects_hull_for_pose_facing_camera(self) -> None:
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([0.0, 0.0, 0.8], dtype=np.float64)

        hull = project_layout_bounds_hull(
            rotation,
            origin,
            self.layout,
            self.camera_matrix,
            self.dist_coeffs,
            bounds_padding_m=0.02,
            image_width=640,
            image_height=480,
        )

        self.assertIsNotNone(hull)
        assert hull is not None
        self.assertGreaterEqual(len(hull), 3)

    def test_returns_none_when_all_corners_are_behind_camera(self) -> None:
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([0.0, 0.0, -0.5], dtype=np.float64)

        hull = project_layout_bounds_hull(
            rotation,
            origin,
            self.layout,
            self.camera_matrix,
            self.dist_coeffs,
            bounds_padding_m=0.02,
            image_width=640,
            image_height=480,
        )

        self.assertIsNone(hull)

    def test_off_screen_corners_are_clipped_to_image_edge(self) -> None:
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([-0.15, 0.0, 0.5], dtype=np.float64)

        hull = project_layout_bounds_hull(
            rotation,
            origin,
            self.layout,
            self.camera_matrix,
            self.dist_coeffs,
            bounds_padding_m=0.02,
            image_width=640,
            image_height=480,
        )

        self.assertIsNotNone(hull)
        assert hull is not None
        self.assertGreaterEqual(hull[:, 0].min(), 0.0)
        self.assertGreaterEqual(hull[:, 1].min(), 0.0)
        self.assertLessEqual(hull[:, 0].max(), 640.0)
        self.assertLessEqual(hull[:, 1].max(), 480.0)
        self.assertAlmostEqual(hull[:, 0].min(), 0.0, places=0)


if __name__ == "__main__":
    unittest.main()
