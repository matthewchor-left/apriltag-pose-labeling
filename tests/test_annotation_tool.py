"""Tests for AprilTag erasure compositing."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from paddle_apriltag.cli.annotation_tool import erase_markers


class EraseMarkersTests(unittest.TestCase):
    def test_pastes_plate_pixels_inside_marker_quad(self) -> None:
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        plate = np.zeros_like(frame)
        plate[:, :] = (0, 255, 0)

        corners = np.array(
            [[[20, 20], [60, 20], [60, 60], [20, 60]]],
            dtype=np.float32,
        )
        detections = [(corners, 0)]

        erased = erase_markers(frame, plate, detections)

        self.assertEqual(tuple(erased[30, 30]), (0, 255, 0))
        self.assertEqual(tuple(erased[0, 0]), (255, 255, 255))

    def test_returns_original_frame_when_no_detections(self) -> None:
        frame = np.full((40, 40, 3), 127, dtype=np.uint8)
        plate = np.zeros_like(frame)
        erased = erase_markers(frame, plate, [])
        np.testing.assert_array_equal(erased, frame)

    def test_requires_matching_frame_and_plate_shapes(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        plate = np.zeros((30, 30, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            erase_markers(frame, plate, [])


if __name__ == "__main__":
    unittest.main()
