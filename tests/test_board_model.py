"""Tests for ChArUco board model loading and frame conventions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_apriltag.board_model import (
    BOARD_REFERENCE_FROM_OPENCV,
    OPENCV_FROM_BOARD_REFERENCE,
    board_model_geometry_flags_provided,
    load_board_model,
)


DEFAULT_BOARD_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json"
)


class BoardModelTests(unittest.TestCase):
    def test_default_profile_loads(self) -> None:
        model = load_board_model(DEFAULT_BOARD_MODEL_PATH)
        self.assertEqual(model.layout_height, 11)
        self.assertEqual(model.layout_width, 8)
        self.assertEqual(model.square_size, 0.025)
        self.assertEqual(model.marker_size, 0.02)
        self.assertEqual(model.total_charuco_intersections, 70)

    def test_rejects_marker_larger_than_square(self) -> None:
        payload = json.loads(DEFAULT_BOARD_MODEL_PATH.read_text(encoding="utf-8"))
        payload["marker_size"] = payload["square_size"]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_board_model(path)
        finally:
            path.unlink()

    def test_rejects_invalid_reference_frame(self) -> None:
        payload = json.loads(DEFAULT_BOARD_MODEL_PATH.read_text(encoding="utf-8"))
        payload["reference_frame"]["y_axis"] = "up"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = Path(handle.name)
        try:
            with self.assertRaises(ValueError):
                load_board_model(path)
        finally:
            path.unlink()

    def test_board_reference_frame_axis_mapping(self) -> None:
        np.testing.assert_allclose(
            BOARD_REFERENCE_FROM_OPENCV @ np.array([1.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            BOARD_REFERENCE_FROM_OPENCV @ np.array([0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            BOARD_REFERENCE_FROM_OPENCV @ np.array([0.0, 0.0, 1.0]),
            [0.0, -1.0, 0.0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            BOARD_REFERENCE_FROM_OPENCV @ OPENCV_FROM_BOARD_REFERENCE,
            np.eye(3),
            atol=1e-9,
        )
        self.assertAlmostEqual(np.linalg.det(BOARD_REFERENCE_FROM_OPENCV), 1.0)
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(np.cross(x_axis, y_axis), z_axis, atol=1e-9)

    def test_board_model_geometry_flag_detection(self) -> None:
        class Args:
            board_type = "charuco_board"
            layout = [6, 9]
            marker_size = 0.02
            square_size = 0.025
            dictionary = "4x4_50"

        self.assertEqual(board_model_geometry_flags_provided(Args()), ["--layout", "--marker-size"])

        class BoardOnlyArgs:
            board_type = "charuco_board"
            layout = None
            marker_size = None
            square_size = 0.025
            dictionary = "4x4_50"

        self.assertEqual(board_model_geometry_flags_provided(BoardOnlyArgs()), [])


if __name__ == "__main__":
    unittest.main()
