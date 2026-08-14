"""OpenCV frame sources: live camera index or video file path."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FrameSource = int | Path


def parse_frame_source(value: str) -> FrameSource:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("frame source must not be empty.")
    try:
        return int(stripped)
    except ValueError:
        path = Path(stripped)
        if not path.is_file():
            raise argparse.ArgumentTypeError(f"video file not found: {stripped}")
        return path


def is_camera_source(source: FrameSource) -> bool:
    return isinstance(source, int)


def format_frame_source(source: FrameSource) -> str:
    if is_camera_source(source):
        return f"camera {source}"
    return f"video {source}"


def open_frame_source(
    source: FrameSource,
    *,
    width: int | None = None,
    height: int | None = None,
) -> cv2.VideoCapture:
    if is_camera_source(source):
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {source}.")
        if width is not None and height is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        return capture

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {source}.")
    return capture


def read_frame(
    capture: cv2.VideoCapture,
    source: FrameSource,
) -> tuple[bool, np.ndarray | None]:
    ok, frame = capture.read()
    if ok or is_camera_source(source):
        return ok, frame
    # ponytail: loop video files on EOF for interactive CLIs; upgrade path is
    # explicit pause/seek controls if scrubbing becomes a requirement.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture.read()
