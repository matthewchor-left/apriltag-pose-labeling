"""Smoke tests for ObjectDetector."""

from __future__ import annotations

import unittest

import numpy as np

from object_apriltag import ObjectDetector, ObjectPose
from object_apriltag.calibration import DEFAULT_MARKER_MODEL_PATH
from object_apriltag.layout import load_marker_model
from object_apriltag.pose import estimate_global_layout_pose, object_pose_from_marker


class ObjectDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_model = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        self.camera_matrix = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.marker_size_m = self.marker_model.marker_size_m

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

    def test_fused_pose_returns_none_for_single_marker(self) -> None:
        corners = self._synthetic_corners()
        detections = [(corners, 0)]
        pose = estimate_global_layout_pose(
            detections,
            self.marker_model,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNone(pose)

    def test_object_pose_dataclass(self) -> None:
        corners = self._synthetic_corners()
        rotation, origin = object_pose_from_marker(
            corners, 0, self.marker_model, self.camera_matrix, self.dist_coeffs
        )
        pose = ObjectPose(origin=origin, rotation=rotation)
        self.assertEqual(pose.origin.shape, (3,))
        self.assertEqual(pose.rotation.shape, (3, 3))

    def test_detector_fuse_returns_none_without_markers(self) -> None:
        detector = ObjectDetector(
            self.camera_matrix,
            self.dist_coeffs,
            marker_model=self.marker_model,
        )
        self.assertIsNone(detector.fuse([]))


if __name__ == "__main__":
    unittest.main()
