"""Frame-in / pose-out paddle detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from paddle_aruco.apriltag import DEFAULT_APRILTAG_DICTIONARY, build_aruco_detector
from paddle_aruco.layout import MarkerLayout, load_marker_layout
from paddle_aruco.pose import Detection, estimate_fused_pose


@dataclass(frozen=True)
class PaddlePose:
    origin: np.ndarray
    rotation: np.ndarray


class PaddleDetector:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        marker_layout: Path | str | MarkerLayout,
        marker_size_m: float | None = None,
        dictionary: str = DEFAULT_APRILTAG_DICTIONARY,
        sensitivity: str = "relaxed",
        marker_ids: set[int] | None = None,
    ) -> None:
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._layout = (
            marker_layout
            if isinstance(marker_layout, MarkerLayout)
            else load_marker_layout(marker_layout)
        )
        self._marker_size_m = (
            self._layout.marker_size_m if marker_size_m is None else marker_size_m
        )
        self._known_ids = self._layout.marker_ids if marker_ids is None else marker_ids
        self._detector = build_aruco_detector(dictionary, sensitivity)

    @property
    def layout(self) -> MarkerLayout:
        return self._layout

    @property
    def marker_size_m(self) -> float:
        return self._marker_size_m

    def find_markers(self, frame: np.ndarray) -> list[Detection]:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return []

        detections: list[Detection] = []
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
            marker_id = int(marker_id)
            if marker_id in self._known_ids:
                detections.append((marker_corners, marker_id))
        return detections

    def fuse(self, detections: list[Detection]) -> PaddlePose | None:
        origin, rotation = estimate_fused_pose(
            detections,
            self._layout,
            self._marker_size_m,
            self._camera_matrix,
            self._dist_coeffs,
        )
        if origin is None or rotation is None:
            return None
        return PaddlePose(origin=origin, rotation=rotation)

    def detect(self, frame: np.ndarray) -> PaddlePose | None:
        return self.fuse(self.find_markers(frame))
