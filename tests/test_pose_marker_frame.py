"""Regression tests for marker PnP frame convention and layout compatibility."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE1_MARKER_MODEL = REPO_ROOT / "config/Model/remote1/marker_model.json"
from object_apriltag.layout import (
    layout_point_to_camera,
    load_marker_model,
    marker_origin_on_object,
)
from object_apriltag.pose import (
    estimate_marker_pose,
    marker_corner_object_points,
    object_pose_from_marker,
    object_pose_from_marker_pose,
)


def _camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


class MarkerCornerObjectPointsTests(unittest.TestCase):
    def test_origin_is_bottom_edge_center(self) -> None:
        marker_size = 0.07
        points = marker_corner_object_points(marker_size)
        bottom_center = (points[3] + points[2]) / 2.0
        np.testing.assert_allclose(bottom_center, [0.0, 0.0, 0.0], atol=1e-6)

    def test_positive_y_points_toward_tag_top(self) -> None:
        marker_size = 0.048
        points = marker_corner_object_points(marker_size)
        self.assertGreater(float(points[0, 1]), float(points[3, 1]))
        self.assertGreater(float(points[1, 1]), float(points[2, 1]))
        np.testing.assert_allclose(points[0, 1], marker_size, atol=1e-6)
        np.testing.assert_allclose(points[3, 1], 0.0, atol=1e-6)

    def test_opencv_corner_order(self) -> None:
        half = 0.035
        points = marker_corner_object_points(0.07)
        np.testing.assert_allclose(points[0], [-half, 0.07, 0.0], atol=1e-6)
        np.testing.assert_allclose(points[1], [half, 0.07, 0.0], atol=1e-6)
        np.testing.assert_allclose(points[2], [half, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(points[3], [-half, 0.0, 0.0], atol=1e-6)


class MarkerPoseLayoutCompatibilityTests(unittest.TestCase):
    def test_estimate_marker_pose_reprojects_synthetic_corners(self) -> None:
        marker_size = 0.07
        camera_matrix, dist_coeffs = _camera()
        object_points = marker_corner_object_points(marker_size)
        rvec_true = np.array([0.25, -0.15, 0.08], dtype=np.float64)
        tvec_true = np.array([[0.03], [-0.02], [0.55]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs
        )
        corners = image_points.reshape(1, 4, 2).astype(np.float32)

        rvec, tvec = estimate_marker_pose(corners, marker_size, camera_matrix, dist_coeffs)
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        np.testing.assert_allclose(projected.reshape(4, 2), corners.reshape(4, 2), atol=1e-3)

    def test_ippe_returns_low_reprojection_for_front_facing_marker(self) -> None:
        marker_size = 0.07
        camera_matrix, dist_coeffs = _camera()
        object_points = marker_corner_object_points(marker_size)
        rvec_true = np.array([0.35, -0.2, 0.12], dtype=np.float64)
        tvec_true = np.array([[0.04], [-0.01], [0.48]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs
        )
        corners = image_points.reshape(1, 4, 2).astype(np.float32)

        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            corners.reshape(4, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        self.assertTrue(ok)
        self.assertGreaterEqual(len(rvecs), 1)

        rvec, tvec = estimate_marker_pose(corners, marker_size, camera_matrix, dist_coeffs)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - corners.reshape(4, 2), axis=1)))
        self.assertLess(error, 0.01)

    def test_object_pose_from_marker_matches_layout_transform_contract(self) -> None:
        layout = load_marker_model(REMOTE1_MARKER_MODEL)
        camera_matrix, dist_coeffs = _camera()
        object_points = marker_corner_object_points(layout.marker_size_m)
        rvec_true = np.array([0.3, -0.2, 0.1], dtype=np.float64)
        tvec_true = np.array([[0.05], [-0.02], [0.55]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs
        )
        corners = image_points.reshape(1, 4, 2).astype(np.float32)

        for marker_id in sorted(layout.footprints):
            rvec, tvec = estimate_marker_pose(
                corners, layout.marker_size_m, camera_matrix, dist_coeffs
            )
            object_rotation, object_origin = object_pose_from_marker_pose(
                rvec, tvec, marker_id, layout
            )
            footprint = layout.footprints[marker_id]
            marker_rotation, _ = cv2.Rodrigues(rvec)
            marker_origin = marker_origin_on_object(footprint.bottom_left, footprint.bottom_right)
            for corner_layout in footprint.corners():
                marker_point = footprint.orientation.T @ (corner_layout - marker_origin)
                expected_camera = marker_rotation @ marker_point + tvec.reshape(3)
                actual_camera = layout_point_to_camera(
                    corner_layout, object_rotation, object_origin, layout
                )
                np.testing.assert_allclose(actual_camera, expected_camera, atol=1e-5)

    def test_object_pose_from_marker_wrapper_matches_pose_path(self) -> None:
        layout = load_marker_model(REMOTE1_MARKER_MODEL)
        camera_matrix, dist_coeffs = _camera()
        object_points = marker_corner_object_points(layout.marker_size_m)
        rvec_true = np.array([0.2, -0.1, 0.05], dtype=np.float64)
        tvec_true = np.array([[0.02], [-0.01], [0.5]], dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs
        )
        corners = image_points.reshape(1, 4, 2).astype(np.float32)
        marker_id = layout.reference_marker_id

        from_pose = object_pose_from_marker_pose(
            *estimate_marker_pose(corners, layout.marker_size_m, camera_matrix, dist_coeffs),
            marker_id,
            layout,
        )
        from_corners = object_pose_from_marker(
            corners, marker_id, layout, camera_matrix, dist_coeffs
        )
        np.testing.assert_allclose(from_pose[0], from_corners[0], atol=1e-9)
        np.testing.assert_allclose(from_pose[1], from_corners[1], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
