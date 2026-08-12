"""Tests for object skeleton JSON loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_apriltag.layout import layout_point_to_camera, load_marker_model
from object_apriltag.viz.skeleton import (
    DEFAULT_OBJECT_MODEL_PATH,
    MODEL_FRAME_NAME,
    load_object_model,
    object_world_points_from_pose,
)


REMOTE1_OBJECT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/object_model.json"
)
REMOTE1_MARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/marker_model.json"
)


class ObjectModelJsonTests(unittest.TestCase):
    def test_default_model_loads_five_keypoints(self) -> None:
        model = load_object_model(DEFAULT_OBJECT_MODEL_PATH)
        self.assertEqual(len(model.keypoint_names), 5)
        self.assertEqual(model.object_points.shape, (5, 3))
        self.assertGreaterEqual(len(model.skeleton_edges), 1)

    def test_bottom_keypoint_is_origin(self) -> None:
        model = load_object_model(DEFAULT_OBJECT_MODEL_PATH)
        np.testing.assert_allclose(model.keypoints["bottom"], [0.0, 0.0, 0.0], atol=1e-9)

    def test_accepts_legacy_xy_coordinates(self) -> None:
        payload = {
            "units": "meters",
            "keypoints": {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0], "d": [7.0, 8.0]},
            "skeleton": [["a", "b"], ["c", "d"]],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            model = load_object_model(path)
            np.testing.assert_allclose(model.keypoints["a"], [1.0, 2.0, 0.0], atol=1e-9)
        finally:
            path.unlink()

    def test_unknown_skeleton_keypoint_raises(self) -> None:
        payload = {
            "keypoints": {"a": [0.0, 0.0, 0.0]},
            "skeleton": [["a", "missing"]],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_object_model(path)
        finally:
            path.unlink()

    def test_rejects_invalid_coordinate_frame(self) -> None:
        payload = json.loads(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"))
        payload["coordinate_frame"] = "camera"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_object_model(path)
            self.assertIn(MODEL_FRAME_NAME, str(ctx.exception))
        finally:
            path.unlink()

    def test_defaults_coordinate_frame_to_marker_model(self) -> None:
        payload = json.loads(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"))
        payload.pop("coordinate_frame", None)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            model = load_object_model(path)
            self.assertEqual(len(model.keypoint_names), 2)
        finally:
            path.unlink()

    def test_object_world_points_use_marker_model_frame(self) -> None:
        object_model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)
        marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        rotation = np.eye(3, dtype=np.float64)
        origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        world_points = object_world_points_from_pose(rotation, origin, object_model, marker_model)

        self.assertEqual(set(world_points), set(object_model.keypoint_names))
        for name, camera_point in world_points.items():
            expected = layout_point_to_camera(
                object_model.keypoints[name],
                rotation,
                origin,
                marker_model,
            )
            np.testing.assert_allclose(camera_point, expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
