"""Tests for marker layout derivation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_apriltag.calibration import DEFAULT_MARKER_MODEL_PATH
from object_apriltag.layout import (
    DEFAULT_MARKER_COLOR,
    DEFAULT_MARKER_COLOR_BGR,
    REFERENCE_MARKER_COLOR,
    REFERENCE_MARKER_COLOR_BGR,
    derive_marker_to_object_transform,
    footprint_edge_lengths,
    footprint_from_dict,
    footprint_orientation,
    layout_point_to_camera,
    layout_point_to_object_frame,
    load_marker_model,
    build_marker_layout,
    marker_color,
    marker_color_bgr,
    marker_layout_to_dict,
    marker_origin_on_object,
    object_reference_origin,
    rectangle_center,
    validate_footprint_size,
)
from object_apriltag.pose import fuse_rotations


def _square_payload(half: float, z: float = 0.0) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, z],
        "top_right": [half, -half, z],
        "bottom_right": [half, half, z],
        "bottom_left": [-half, half, z],
    }


class MarkerLayoutDerivationTests(unittest.TestCase):
    def test_reference_marker_has_identity_rotation(self) -> None:
        side = 0.048
        half = side / 2
        footprint = footprint_from_dict(0, _square_payload(half))
        object_origin = rectangle_center(*footprint.corners())
        transform = derive_marker_to_object_transform(footprint, footprint.orientation, object_origin)

        np.testing.assert_allclose(transform.rotation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(transform.offset[0], 0.0, places=6)
        self.assertAlmostEqual(transform.offset[1], side / 2, places=6)
        self.assertAlmostEqual(transform.offset[2], 0.0, places=6)

    def test_requires_all_four_corners(self) -> None:
        with self.assertRaises(ValueError):
            footprint_from_dict(0, {"top_left": [-0.024, -0.024, 0.0], "bottom_right": [0.024, 0.024, 0.0]})

    def test_accepts_legacy_xy_coordinates(self) -> None:
        footprint = footprint_from_dict(0, _square_payload(0.024))
        np.testing.assert_allclose(footprint.top_left, [-0.024, -0.024, 0.0])

    def test_reference_marker_y_points_toward_tag_top_in_layout_frame(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        y_axis = layout.footprints[0].orientation[:, 1]
        np.testing.assert_allclose(y_axis, [0.0, -1.0, 0.0], atol=1e-9)

    def test_reference_marker_z_points_into_object_in_layout_frame(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        z_axis = layout.footprints[0].orientation[:, 2]
        np.testing.assert_allclose(z_axis, [0.0, 0.0, -1.0], atol=1e-9)

    def test_footprint_orientation_is_orthonormal(self) -> None:
        orientation = footprint_orientation(
            np.array([-0.024, -0.024, 0.0]),
            np.array([0.024, -0.024, 0.0]),
            np.array([-0.024, 0.024, 0.0]),
            np.array([0.024, 0.024, 0.0]),
        )
        np.testing.assert_allclose(orientation.T @ orientation, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(orientation)), 1.0, places=6)

    def test_side_face_marker_gets_non_identity_rotation(self) -> None:
        side = 0.048
        half = side / 2
        footprint = footprint_from_dict(
            2,
            {
                "top_left": [0.0, -half, half],
                "top_right": [0.0, half, half],
                "bottom_right": [0.0, half, -half],
                "bottom_left": [0.0, -half, -half],
            },
        )
        reference = footprint_from_dict(0, _square_payload(half))
        object_origin = rectangle_center(*reference.corners())
        transform = derive_marker_to_object_transform(footprint, reference.orientation, object_origin)
        self.assertGreater(np.linalg.norm(transform.rotation - np.eye(3)), 0.1)

    def test_tilted_marker_four_loads(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        footprint = layout.footprints[4]
        edges = footprint_edge_lengths(*footprint.corners())
        for value in edges:
            self.assertAlmostEqual(value, layout.marker_size_m, places=4)

    def test_left_right_markers_are_mirrored(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        left = layout.footprints[2]
        right = layout.footprints[3]
        self.assertAlmostEqual(left.top_left[0], -right.bottom_right[0], places=4)
        self.assertAlmostEqual(left.bottom_right[0], -right.top_left[0], places=4)

    def test_all_rotations_are_proper(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        for marker_id, transform in layout.transforms.items():
            det = float(np.linalg.det(transform.rotation))
            self.assertAlmostEqual(det, 1.0, places=6, msg=f"marker {marker_id}")

    def test_marker_origin_is_bottom_edge_center(self) -> None:
        footprint = footprint_from_dict(0, _square_payload(0.024))
        origin = marker_origin_on_object(footprint.bottom_left, footprint.bottom_right)
        np.testing.assert_allclose(origin, np.array([0.0, 0.024, 0.0]))

    def test_footprints_match_marker_size_in_layout(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        for footprint in layout.footprints.values():
            for value in footprint_edge_lengths(*footprint.corners()):
                self.assertAlmostEqual(value, layout.marker_size_m, places=6)

    def test_layout_point_to_object_frame_uses_reference_orientation(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        origin = object_reference_origin(layout)
        np.testing.assert_allclose(layout_point_to_object_frame(origin, layout), np.zeros(3), atol=1e-9)
        top_left = layout.footprints[0].top_left
        point_object = layout_point_to_object_frame(top_left, layout)
        self.assertAlmostEqual(point_object[0], 0.024, places=6)
        self.assertAlmostEqual(point_object[1], 0.024, places=6)
        self.assertAlmostEqual(point_object[2], 0.0, places=6)

    def test_layout_point_to_camera_matches_marker_pose_for_each_marker(self) -> None:
        import cv2

        from object_apriltag.layout import marker_origin_on_object
        from object_apriltag.pose import estimate_marker_pose, marker_corner_object_points, object_pose_from_marker_pose

        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        camera_matrix = np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        rvec_true = np.array([0.3, -0.2, 0.1], dtype=np.float64)
        tvec_true = np.array([[0.05], [-0.02], [0.55]], dtype=np.float64)
        object_points = marker_corner_object_points(layout.marker_size_m)
        image_points, _ = cv2.projectPoints(object_points, rvec_true, tvec_true, camera_matrix, dist_coeffs)
        corners = image_points.reshape(1, 4, 2).astype(np.float32)

        for marker_id in sorted(layout.footprints):
            rvec, tvec = estimate_marker_pose(corners, layout.marker_size_m, camera_matrix, dist_coeffs)
            object_rotation, object_origin = object_pose_from_marker_pose(rvec, tvec, marker_id, layout)
            footprint = layout.footprints[marker_id]
            marker_origin = marker_origin_on_object(footprint.bottom_left, footprint.bottom_right)
            for corner_layout in footprint.corners():
                marker_point = footprint.orientation.T @ (corner_layout - marker_origin)
                rotation, _ = cv2.Rodrigues(rvec)
                expected_camera = rotation @ marker_point + tvec.reshape(3)
                actual_camera = layout_point_to_camera(corner_layout, object_rotation, object_origin, layout)
                np.testing.assert_allclose(actual_camera, expected_camera, atol=1e-6)

    def test_invalid_footprint_size_raises(self) -> None:
        footprint = footprint_from_dict(
            0,
            {
                "top_left": [-0.02, -0.02, 0.0],
                "top_right": [0.03, -0.02, 0.0],
                "bottom_right": [0.03, 0.02, 0.0],
                "bottom_left": [-0.02, 0.02, 0.0],
            },
        )
        with self.assertRaises(ValueError):
            validate_footprint_size(footprint, 0.048)


class MarkerOriginsIntegrationTests(unittest.TestCase):
    def test_layout_transforms_lookup(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        offset = layout.transforms[0].offset
        rotation = layout.transforms[0].rotation
        self.assertEqual(offset.shape, (3,))
        self.assertEqual(rotation.shape, (3, 3))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)

    def test_fuse_rotations_with_layout_rotations(self) -> None:
        layout = load_marker_model(DEFAULT_MARKER_MODEL_PATH)
        rotations = [layout.transforms[marker_id].rotation for marker_id in (2, 3)]
        fused = fuse_rotations(rotations)
        self.assertIsNotNone(fused)
        assert fused is not None
        self.assertAlmostEqual(float(np.linalg.det(fused)), 1.0, places=6)

    def test_load_from_temp_json(self) -> None:
        payload = {
            "reference_marker_id": 0,
            "units": "meters",
            "marker_size_m": 0.04,
            "markers": {
                "0": _square_payload(0.02),
                "1": {
                    "top_left": [-0.02, 0.06, 0.0],
                    "top_right": [0.02, 0.06, 0.0],
                    "bottom_right": [0.02, 0.1, 0.0],
                    "bottom_left": [-0.02, 0.1, 0.0],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            layout = load_marker_model(path)
            self.assertEqual(layout.marker_ids, {0, 1})
            self.assertIn(1, layout.transforms)

    def test_build_marker_layout_defaults_anchors_to_all_markers(self) -> None:
        footprints = {
            0: footprint_from_dict(0, _square_payload(0.02)),
            1: footprint_from_dict(
                1,
                {
                    "top_left": [-0.02, 0.06, 0.0],
                    "top_right": [0.02, 0.06, 0.0],
                    "bottom_right": [0.02, 0.1, 0.0],
                    "bottom_left": [-0.02, 0.1, 0.0],
                },
            ),
        }
        layout = build_marker_layout(0, 0.04, footprints)
        self.assertEqual(layout.anchor_marker_ids, (0, 1))
        payload = marker_layout_to_dict(layout)
        self.assertEqual(payload["anchor_marker_ids"], [0, 1])

    def test_anchor_marker_ids_round_trip(self) -> None:
        payload = {
            "reference_marker_id": 0,
            "units": "meters",
            "marker_size_m": 0.04,
            "anchor_marker_ids": [0, 1],
            "markers": {
                "0": _square_payload(0.02),
                "1": {
                    "top_left": [-0.02, 0.06, 0.0],
                    "top_right": [0.02, 0.06, 0.0],
                    "bottom_right": [0.02, 0.1, 0.0],
                    "bottom_left": [-0.02, 0.1, 0.0],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            layout = load_marker_model(path)
            self.assertEqual(layout.anchor_marker_ids, (0, 1))
            round_trip = marker_layout_to_dict(layout)
            self.assertEqual(round_trip["anchor_marker_ids"], [0, 1])


class MarkerLayoutColorTests(unittest.TestCase):
    def test_reference_marker_uses_distinct_color(self) -> None:
        reference_marker_id = 14
        self.assertEqual(marker_color(reference_marker_id, reference_marker_id), REFERENCE_MARKER_COLOR)
        self.assertEqual(
            marker_color_bgr(reference_marker_id, reference_marker_id),
            REFERENCE_MARKER_COLOR_BGR,
        )

    def test_non_reference_markers_use_yellow(self) -> None:
        reference_marker_id = 14
        for marker_id in (0, 1, 19):
            self.assertEqual(marker_color(marker_id, reference_marker_id), DEFAULT_MARKER_COLOR)
            self.assertEqual(marker_color_bgr(marker_id, reference_marker_id), DEFAULT_MARKER_COLOR_BGR)


if __name__ == "__main__":
    unittest.main()
