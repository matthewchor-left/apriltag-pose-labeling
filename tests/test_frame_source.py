"""Tests for frame source parsing and video looping."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag.frame_source import (
    format_frame_source,
    is_camera_source,
    open_frame_source,
    parse_frame_source,
    read_frame,
)


class FrameSourceParsingTests(unittest.TestCase):
    def test_integer_string_parses_as_camera_index(self) -> None:
        self.assertEqual(parse_frame_source("0"), 0)
        self.assertTrue(is_camera_source(parse_frame_source("2")))

    def test_existing_video_path_parses_as_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".mov") as tmp:
            source = parse_frame_source(tmp.name)
            self.assertIsInstance(source, Path)
            self.assertFalse(is_camera_source(source))

    def test_missing_video_path_raises(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_frame_source("/tmp/does-not-exist-video.mov")

    def test_format_frame_source(self) -> None:
        self.assertEqual(format_frame_source(0), "camera 0")
        self.assertEqual(format_frame_source(Path("clip.mov")), "video clip.mov")


class FrameSourceReadTests(unittest.TestCase):
    def test_read_frame_loops_video_on_eof(self) -> None:
        capture = mock.MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        capture.read.side_effect = [(False, None), (True, frame)]

        ok, returned = read_frame(capture, Path("clip.mov"))

        self.assertTrue(ok)
        np.testing.assert_array_equal(returned, frame)
        capture.set.assert_called_once()


if __name__ == "__main__":
    unittest.main()
