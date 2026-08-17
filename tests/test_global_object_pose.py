"""Tests for layout-wide runtime object pose estimation."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.layout import (
    build_marker_layout,
    footprint_from_dict,
    layout_point_to_object_frame,
    load_marker_model,
)
from object_apriltag.pose import estimate_global_layout_pose

REMOTE1_MARKER_MODEL = (
    Path(__file__).resolve().parents[1]
    / "config/Model/remote1/marker_model.json"
)


def _camera() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.zeros((5, 1), dtype=np.float64),
    )


def _project_layout_detections(
    layout,
    marker_ids: list[int],
    object_rvec: np.ndarray,
    object_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    detections = []
    for marker_id in marker_ids:
        object_points = np.stack(
            [
                layout_point_to_object_frame(point, layout)
                for point in layout.footprints[marker_id].corners()
            ]
        )
        projected, _ = cv2.projectPoints(
            object_points,
            object_rvec,
            object_origin,
            camera_matrix,
            dist_coeffs,
        )
        detections.append(
            (projected.reshape(1, 4, 2).astype(np.float32), marker_id)
        )
    return detections


def _square_payload(half: float, z: float = 0.0) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, z],
        "top_right": [half, -half, z],
        "bottom_right": [half, half, z],
        "bottom_left": [-half, half, z],
    }


class GlobalObjectPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = load_marker_model(REMOTE1_MARKER_MODEL)
        self.camera_matrix, self.dist_coeffs = _camera()
        self.object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        self.object_rotation, _ = cv2.Rodrigues(self.object_rvec)
        self.object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)
        self.marker_ids = sorted(self.layout.marker_ids)[:4]

    def assert_pose_close(
        self, origin: np.ndarray | None, rotation: np.ndarray | None
    ) -> None:
        self.assertIsNotNone(origin)
        self.assertIsNotNone(rotation)
        assert origin is not None and rotation is not None
        np.testing.assert_allclose(origin, self.object_origin, atol=1e-4)
        np.testing.assert_allclose(rotation, self.object_rotation, atol=1e-4)

    def test_recovers_one_pose_from_all_layout_corners(self) -> None:
        detections = _project_layout_detections(
            self.layout,
            self.marker_ids,
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assert_pose_close(
            *(estimate_global_layout_pose(
                detections,
                self.layout,
                self.camera_matrix,
                self.dist_coeffs,
            ) or (None, None))
        )

    def test_ransac_rejects_one_corrupted_marker(self) -> None:
        detections = _project_layout_detections(
            self.layout,
            self.marker_ids,
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        corrupted_corners, corrupted_id = detections[-1]
        detections[-1] = (
            corrupted_corners
            + np.array([[[70.0, -50.0]]], dtype=np.float32),
            corrupted_id,
        )
        self.assert_pose_close(
            *(estimate_global_layout_pose(
                detections,
                self.layout,
                self.camera_matrix,
                self.dist_coeffs,
            ) or (None, None))
        )


class CoplanarLayoutPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        marker_size_m = 0.07
        half = marker_size_m / 2.0
        self.layout = build_marker_layout(
            0,
            marker_size_m,
            {
                0: footprint_from_dict(0, _square_payload(half)),
                1: footprint_from_dict(1, _square_payload(half, z=0.12)),
            },
        )
        self.camera_matrix, self.dist_coeffs = _camera()
        self.object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        self.object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)

    def test_rejects_coplanar_markers_with_multiple_ippe_solutions(self) -> None:
        coplanar_layout = build_marker_layout(
            0,
            0.07,
            {
                0: footprint_from_dict(0, _square_payload(0.035)),
                1: footprint_from_dict(1, _square_payload(0.035)),
            },
        )
        detections = _project_layout_detections(
            coplanar_layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNone(
            estimate_global_layout_pose(
                detections,
                coplanar_layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )

    def test_allows_non_coplanar_markers(self) -> None:
        detections = _project_layout_detections(
            self.layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNotNone(
            estimate_global_layout_pose(
                detections,
                self.layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )


if __name__ == "__main__":
    unittest.main()
