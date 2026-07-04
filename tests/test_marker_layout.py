"""Tests for marker layout derivation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paddle_apriltag.layout import (
    DEFAULT_MARKER_LAYOUT_PATH,
    derive_marker_to_paddle_transform,
    footprint_edge_lengths,
    footprint_from_dict,
    footprint_orientation,
    layout_point_to_camera,
    layout_point_to_paddle_frame,
    load_marker_layout,
    marker_origin_on_paddle,
    paddle_reference_origin,
    rectangle_center,
    validate_footprint_size,
)
from paddle_apriltag.pose import fuse_rotations


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
        paddle_origin = rectangle_center(*footprint.corners())
        transform = derive_marker_to_paddle_transform(footprint, footprint.orientation, paddle_origin)

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
        paddle_origin = rectangle_center(*reference.corners())
        transform = derive_marker_to_paddle_transform(footprint, reference.orientation, paddle_origin)
        self.assertGreater(np.linalg.norm(transform.rotation - np.eye(3)), 0.1)

    def test_tilted_marker_four_loads(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        footprint = layout.footprints[4]
        edges = footprint_edge_lengths(*footprint.corners())
        for value in edges:
            self.assertAlmostEqual(value, layout.marker_size_m, places=4)

    def test_left_right_markers_are_mirrored(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        left = layout.footprints[2]
        right = layout.footprints[3]
        self.assertAlmostEqual(left.top_left[0], -right.bottom_right[0], places=4)
        self.assertAlmostEqual(left.bottom_right[0], -right.top_left[0], places=4)

    def test_all_rotations_are_proper(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        for marker_id, transform in layout.transforms.items():
            det = float(np.linalg.det(transform.rotation))
            self.assertAlmostEqual(det, 1.0, places=6, msg=f"marker {marker_id}")

    def test_marker_origin_is_bottom_edge_center(self) -> None:
        footprint = footprint_from_dict(0, _square_payload(0.024))
        origin = marker_origin_on_paddle(footprint.bottom_left, footprint.bottom_right)
        np.testing.assert_allclose(origin, np.array([0.0, 0.024, 0.0]))

    def test_footprints_match_marker_size_in_layout(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        for footprint in layout.footprints.values():
            for value in footprint_edge_lengths(*footprint.corners()):
                self.assertAlmostEqual(value, layout.marker_size_m, places=6)

    def test_layout_point_to_paddle_frame_uses_reference_orientation(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        origin = paddle_reference_origin(layout)
        np.testing.assert_allclose(layout_point_to_paddle_frame(origin, layout), np.zeros(3), atol=1e-9)
        top_left = layout.footprints[0].top_left
        point_paddle = layout_point_to_paddle_frame(top_left, layout)
        self.assertAlmostEqual(point_paddle[0], -0.024, places=6)
        self.assertAlmostEqual(point_paddle[1], 0.024, places=6)
        self.assertAlmostEqual(point_paddle[2], 0.0, places=6)

    def test_layout_point_to_camera_matches_marker_pose_for_reference_marker(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        footprint = layout.footprints[0]
        paddle_rotation = footprint.orientation
        paddle_origin = np.zeros(3)
        corner = footprint.top_right
        camera_point = layout_point_to_camera(corner, paddle_rotation, paddle_origin, layout)
        expected = paddle_rotation @ layout_point_to_paddle_frame(corner, layout)
        np.testing.assert_allclose(camera_point, expected, atol=1e-9)

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
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
        offset = layout.transforms[0].offset
        rotation = layout.transforms[0].rotation
        self.assertEqual(offset.shape, (3,))
        self.assertEqual(rotation.shape, (3, 3))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)

    def test_fuse_rotations_with_layout_rotations(self) -> None:
        layout = load_marker_layout(DEFAULT_MARKER_LAYOUT_PATH)
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
            layout = load_marker_layout(path)
            self.assertEqual(layout.marker_ids, {0, 1})
            self.assertIn(1, layout.transforms)


if __name__ == "__main__":
    unittest.main()
