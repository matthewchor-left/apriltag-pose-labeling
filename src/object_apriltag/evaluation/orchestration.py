"""One-pass held-out video decoding and frozen AprilTag detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.evaluation.detection_consistency import (
    FrozenFrameDetections,
    FrozenVideoDetections,
    normalize_frame_detections,
)
from object_apriltag.evaluation.manifest import repo_relative_path
from object_apriltag.frame_source import open_frame_source, read_frame
from object_apriltag.pose import Detection


@dataclass(frozen=True)
class VideoNormalizationSummary:
    source_video: str
    frame_count: int
    unknown_marker_ids: int
    duplicate_marker_skips: int
    malformed_detections: int


def freeze_held_out_video_detections(
    video_path: Path,
    detector: cv2.aruco.ArucoDetector,
    *,
    calibration_width: int,
    calibration_height: int,
    expected_marker_ids: frozenset[int],
) -> tuple[FrozenVideoDetections, VideoNormalizationSummary]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Held-out video not found: {video_path}.")

    capture = open_frame_source(video_path)
    frames: list[FrozenFrameDetections] = []
    unknown_marker_ids = 0
    duplicate_marker_skips = 0
    malformed_detections = 0
    frame_index = 0
    try:
        while True:
            ok, frame = read_frame(capture, video_path, loop_on_eof=False)
            if not ok or frame is None:
                break
            height, width = frame.shape[:2]
            if width != calibration_width or height != calibration_height:
                raise ValueError(
                    f"Held-out video {video_path} frame {frame_index} is {width}x{height}; "
                    f"intrinsics require {calibration_width}x{calibration_height}. "
                    "Do not scale intrinsics."
                )
            detections, frame_unknown, frame_duplicates, frame_malformed = _detect_frame(
                detector,
                frame,
                expected_marker_ids=expected_marker_ids,
            )
            unknown_marker_ids += frame_unknown
            duplicate_marker_skips += frame_duplicates
            malformed_detections += frame_malformed
            frames.append(FrozenFrameDetections(detections=tuple(detections)))
            frame_index += 1
            del frame
    finally:
        capture.release()

    source_video = repo_relative_path(video_path)
    return (
        FrozenVideoDetections(source_video=source_video, frames=tuple(frames)),
        VideoNormalizationSummary(
            source_video=source_video,
            frame_count=len(frames),
            unknown_marker_ids=unknown_marker_ids,
            duplicate_marker_skips=duplicate_marker_skips,
            malformed_detections=malformed_detections,
        ),
    )


def _detect_frame(
    detector: cv2.aruco.ArucoDetector,
    frame: np.ndarray,
    *,
    expected_marker_ids: frozenset[int],
) -> tuple[list[Detection], int, int, int]:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [], 0, 0, 0

    raw_detections: list[Detection] = [
        (marker_corners, int(marker_id))
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True)
    ]
    corners_by_id, duplicate_skips, unknown_ids, malformed_skips = normalize_frame_detections(
        raw_detections,
        expected_marker_ids=expected_marker_ids,
    )
    detections = [
        (corners.reshape(1, 4, 2).astype(np.float32), marker_id)
        for marker_id, corners in sorted(corners_by_id.items())
    ]
    return detections, unknown_ids, duplicate_skips, malformed_skips
