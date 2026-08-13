"""Frame-in / pose-out object detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.apriltag import DEFAULT_APRILTAG_DICTIONARY, build_apriltag_detector
from object_apriltag.layout import MarkerModel, load_marker_model
from object_apriltag.pose import Detection, estimate_fused_pose


@dataclass(frozen=True)
class ObjectPose:
    origin: np.ndarray
    rotation: np.ndarray


class ObjectDetector:
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        marker_model: Path | str | MarkerModel,
        marker_size_m: float | None = None,
        dictionary: str = DEFAULT_APRILTAG_DICTIONARY,
        sensitivity: str = "relaxed",
        marker_ids: set[int] | None = None,
    ) -> None:
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._marker_model = (
            marker_model
            if isinstance(marker_model, MarkerModel)
            else load_marker_model(marker_model)
        )
        if marker_size_m is not None:
            default_size = self._marker_model.marker_size_m
            sizes = self._marker_model.marker_sizes_m
            if any(size != default_size for size in sizes.values()):
                raise ValueError(
                    "marker_size_m override is only supported for uniform marker models; "
                    "use per-marker size_m in the marker model JSON for mixed layouts."
                )
            if abs(marker_size_m - default_size) > 1e-9:
                raise ValueError(
                    f"marker_size_m override {marker_size_m:.6f} m does not match the "
                    f"uniform model default {default_size:.6f} m."
                )
        self._known_ids = self._marker_model.marker_ids if marker_ids is None else marker_ids
        self._detector = build_apriltag_detector(dictionary, sensitivity)
        self._previous_pose: ObjectPose | None = None

    @property
    def marker_model(self) -> MarkerModel:
        return self._marker_model

    @property
    def marker_size_m(self) -> float:
        return self._marker_model.marker_size_m

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

    def fuse(self, detections: list[Detection]) -> ObjectPose | None:
        previous_pose = (
            (self._previous_pose.origin, self._previous_pose.rotation)
            if self._previous_pose is not None
            else None
        )
        origin, rotation = estimate_fused_pose(
            detections,
            self._marker_model,
            self._camera_matrix,
            self._dist_coeffs,
            previous_pose=previous_pose,
        )
        if origin is None or rotation is None:
            return None
        pose = ObjectPose(origin=origin, rotation=rotation)
        self._previous_pose = pose
        return pose

    def detect(self, frame: np.ndarray) -> ObjectPose | None:
        return self.fuse(self.find_markers(frame))
