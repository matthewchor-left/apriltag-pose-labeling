"""AprilTag-based table tennis object pose estimation.

Public API:
    ObjectDetector: Detect AprilTags and fuse them into an object pose.
    ObjectPose: Fused object origin and rotation in the camera frame.
"""

from object_apriltag.detector import ObjectDetector, ObjectPose

__all__ = ["ObjectDetector", "ObjectPose"]
