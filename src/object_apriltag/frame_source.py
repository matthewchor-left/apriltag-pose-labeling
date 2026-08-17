"""OpenCV frame sources: live camera index or video file path."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FrameSource = int | Path


def parse_frame_source(value: str) -> FrameSource:
    """Parse a CLI frame-source string into a camera index or video path.

    Args:
        value: Non-empty string containing an integer camera index or path to a
            video file.

    Returns:
        Camera index as ``int`` when ``value`` is numeric, otherwise a ``Path``
        to an existing video file.

    Raises:
        argparse.ArgumentTypeError: If ``value`` is empty or the video file does
            not exist.
    """
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
    """Return whether a frame source refers to a live camera index.

    Args:
        source: Camera index or video file path.

    Returns:
        ``True`` when ``source`` is an ``int`` camera index.
    """
    return isinstance(source, int)


def format_frame_source(source: FrameSource) -> str:
    """Format a frame source for human-readable logging.

    Args:
        source: Camera index or video file path.

    Returns:
        String such as ``"camera 0"`` or ``"video path/to/file.mov"``.
    """
    if is_camera_source(source):
        return f"camera {source}"
    return f"video {source}"


def open_frame_source(
    source: FrameSource,
    *,
    width: int | None = None,
    height: int | None = None,
) -> cv2.VideoCapture:
    """Open a camera or video file as an OpenCV capture device.

    Args:
        source: Camera index or path to a video file.
        width: Optional requested capture width for camera sources.
        height: Optional requested capture height for camera sources.

    Returns:
        Opened ``cv2.VideoCapture`` handle.

    Raises:
        RuntimeError: If the camera or video file cannot be opened.
    """
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
    *,
    loop_on_eof: bool = True,
) -> tuple[bool, np.ndarray | None]:
    """Read the next frame from a capture device.

    Video files optionally loop to the first frame on end-of-file.

    Args:
        capture: Open ``cv2.VideoCapture`` handle.
        source: Original frame source used to open ``capture``.
        loop_on_eof: When ``True``, rewind video files after EOF and read again.

    Returns:
        Tuple ``(ok, frame)`` as returned by ``cv2.VideoCapture.read``.
    """
    ok, frame = capture.read()
    if ok or is_camera_source(source) or not loop_on_eof:
        return ok, frame
    # loop video files on EOF for interactive CLIs; upgrade path is
    # explicit pause/seek controls if scrubbing becomes a requirement.
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture.read()
