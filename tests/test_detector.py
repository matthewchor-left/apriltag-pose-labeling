"""Smoke tests for PaddleDetector."""

from __future__ import annotations

import unittest

import numpy as np

from paddle_apriltag import PaddleDetector, PaddlePose
from paddle_apriltag.layout import DEFAULT_MARKER_LAYOUT_PATH, load_marker_layout
from paddle_apriltag.pose import estimate_fused_pose, paddle_pose_from_marker


class PaddleDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        self.camera_matrix = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.marker_size_m = self.layout.marker_size_m

    def _synthetic_corners(self) -> np.ndarray:
        half = self.marker_size_m / 2.0
        center = np.array([640.0, 360.0])
        scale = 800.0
        return np.array(
            [
                center + scale * np.array([-half, 0.0]),
                center + scale * np.array([half, 0.0]),
                center + scale * np.array([half, self.marker_size_m]),
                center + scale * np.array([-half, self.marker_size_m]),
            ],
            dtype=np.float32,
        ).reshape(1, 4, 2)

    def test_fused_pose_from_synthetic_marker(self) -> None:
        corners = self._synthetic_corners()
        detections = [(corners, 0)]
        origin, rotation = estimate_fused_pose(
            detections,
            self.layout,
            self.marker_size_m,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNotNone(origin)
        self.assertIsNotNone(rotation)
        assert origin is not None and rotation is not None
        self.assertEqual(origin.shape, (3,))
        self.assertEqual(rotation.shape, (3, 3))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=4)

    def test_paddle_pose_dataclass(self) -> None:
        corners = self._synthetic_corners()
        rotation, origin = paddle_pose_from_marker(
            corners, 0, self.marker_size_m, self.camera_matrix, self.dist_coeffs, self.layout
        )
        pose = PaddlePose(origin=origin, rotation=rotation)
        self.assertEqual(pose.origin.shape, (3,))
        self.assertEqual(pose.rotation.shape, (3, 3))

    def test_detector_fuse_returns_none_without_markers(self) -> None:
        detector = PaddleDetector(
            self.camera_matrix,
            self.dist_coeffs,
            marker_layout=self.layout,
        )
        self.assertIsNone(detector.fuse([]))


if __name__ == "__main__":
    unittest.main()
