"""Tests for paddle skeleton JSON loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paddle_apriltag.viz.skeleton import DEFAULT_PADDLE_MODEL_PATH, load_paddle_model


class PaddleModelJsonTests(unittest.TestCase):
    def test_default_model_loads_five_keypoints(self) -> None:
        model = load_paddle_model(DEFAULT_PADDLE_MODEL_PATH)
        self.assertEqual(len(model.keypoint_names), 5)
        self.assertEqual(model.object_points.shape, (5, 3))
        self.assertGreaterEqual(len(model.skeleton_edges), 1)

    def test_bottom_keypoint_is_origin(self) -> None:
        model = load_paddle_model(DEFAULT_PADDLE_MODEL_PATH)
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
            model = load_paddle_model(path)
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
                load_paddle_model(path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
