"""ChArUco board pose estimation in the Board Reference Frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from object_apriltag.board_model import (
    OPENCV_FROM_BOARD_REFERENCE,
    BoardModel,
    build_charuco_board,
)

CameraMotion = Literal["static", "dynamic"]


def parse_charuco_detection(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return matching (N,2) corners and (N,) ids, or None when unusable."""
    if charuco_ids is None:
        return None
    ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    if ids_flat.shape[0] == 0:
        return None
    if charuco_corners is None:
        return None
    corners_flat = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    if corners_flat.shape[0] != ids_flat.shape[0]:
        return None
    return corners_flat, ids_flat


def charuco_corners_consistent(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> bool:
    return parse_charuco_detection(charuco_corners, charuco_ids) is not None


def charuco_draw_arrays(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Arrays shaped for OpenCV drawDetectedCornersCharuco."""
    parsed = parse_charuco_detection(charuco_corners, charuco_ids)
    if parsed is None:
        return None
    corners_flat, ids_flat = parsed
    return corners_flat.reshape(-1, 1, 2), ids_flat.reshape(-1, 1)


@dataclass(frozen=True)
class CharucoObservation:
    object_points_opencv: np.ndarray
    image_points: np.ndarray
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray


@dataclass(frozen=True)
class BoardPoseEstimate:
    rotation: np.ndarray
    origin: np.ndarray
    reprojection_rms_px: float
    detected_intersections: int
    total_intersections: int


@dataclass(frozen=True)
class BoardPoseResult:
    pose: BoardPoseEstimate | None
    detected_intersections: int
    no_pose_reason: str | None
    charuco_corners: np.ndarray | None = None


def detect_charuco_intersections(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
) -> CharucoObservation | None:
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    parsed = parse_charuco_detection(charuco_corners, charuco_ids)
    if parsed is None or parsed[1].shape[0] < 4:
        return None
    corners_flat, ids_flat = parsed
    object_points, image_points = board.matchImagePoints(
        corners_flat.reshape(-1, 1, 2),
        ids_flat.reshape(-1, 1),
    )
    if object_points is None or len(object_points) < 4:
        return None
    return CharucoObservation(
        object_points_opencv=object_points.reshape(-1, 3).astype(np.float64),
        image_points=image_points.reshape(-1, 2).astype(np.float64),
        charuco_corners=corners_flat,
        charuco_ids=ids_flat,
    )


def _opencv_pose_to_board_reference(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_opencv, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    rotation = rotation_opencv @ OPENCV_FROM_BOARD_REFERENCE
    origin = np.asarray(tvec, dtype=np.float64).reshape(3)
    return rotation, origin


def _is_physically_valid_board_pose(rotation: np.ndarray) -> bool:
    # +Y points out of the printed face toward the camera: negative camera Z component.
    y_in_camera = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return float(y_in_camera[2]) < 0.0


def _reprojection_rms_px(
    object_points_opencv: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points_opencv.reshape(-1, 1, 3).astype(np.float32),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(image_points - projected, axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def estimate_board_pose(
    observation: CharucoObservation,
    model: BoardModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> BoardPoseEstimate | None:
    pose, _ = _estimate_board_pose_with_reason(
        observation, model, camera_matrix, dist_coeffs
    )
    return pose


def _estimate_board_pose_with_reason(
    observation: CharucoObservation,
    model: BoardModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[BoardPoseEstimate | None, str | None]:
    object_points = observation.object_points_opencv.astype(np.float32)
    image_points = observation.image_points.astype(np.float32)
    if len(object_points) < 4:
        return None, f"need >= 4 matched intersections (matched {len(object_points)})"
    if len(object_points) != len(image_points):
        return None, "object/image point count mismatch"

    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or rvecs is None or tvecs is None or len(rvecs) == 0:
        return None, "solvePnP failed"

    best_rvec: np.ndarray | None = None
    best_tvec: np.ndarray | None = None
    best_rms = float("inf")
    had_physically_valid = False
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rotation, _ = _opencv_pose_to_board_reference(rvec, tvec)
        if not _is_physically_valid_board_pose(rotation):
            continue
        had_physically_valid = True
        rms = _reprojection_rms_px(
            observation.object_points_opencv,
            observation.image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        if rms < best_rms:
            best_rms = rms
            best_rvec = rvec
            best_tvec = tvec

    if best_rvec is None or best_tvec is None:
        if not had_physically_valid:
            return None, "no valid board pose (board facing away or ambiguous)"
        return None, "solvePnP failed"

    rotation, origin = _opencv_pose_to_board_reference(best_rvec, best_tvec)
    return (
        BoardPoseEstimate(
            rotation=rotation,
            origin=origin,
            reprojection_rms_px=best_rms,
            detected_intersections=len(observation.charuco_ids),
            total_intersections=model.total_charuco_intersections,
        ),
        None,
    )


def solve_board_pose(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    model: BoardModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> BoardPoseResult:
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or np.asarray(charuco_ids).size == 0:
        return BoardPoseResult(
            pose=None,
            detected_intersections=0,
            no_pose_reason="no ChArUco intersections detected",
        )

    parsed = parse_charuco_detection(charuco_corners, charuco_ids)
    if parsed is None:
        return BoardPoseResult(
            pose=None,
            detected_intersections=0,
            no_pose_reason="ChArUco corner/ID arrays inconsistent",
        )

    corners_flat, ids_flat = parsed
    detected = ids_flat.shape[0]
    if detected < 4:
        return BoardPoseResult(
            pose=None,
            detected_intersections=detected,
            no_pose_reason=f"need >= 4 intersections (detected {detected})",
            charuco_corners=corners_flat,
        )

    object_points, image_points = board.matchImagePoints(
        corners_flat.reshape(-1, 1, 2),
        ids_flat.reshape(-1, 1),
    )
    if object_points is None or len(object_points) < 4:
        matched = 0 if object_points is None else len(object_points)
        return BoardPoseResult(
            pose=None,
            detected_intersections=detected,
            no_pose_reason=f"need >= 4 matched intersections (matched {matched})",
            charuco_corners=corners_flat,
        )

    observation = CharucoObservation(
        object_points_opencv=object_points.reshape(-1, 3).astype(np.float64),
        image_points=image_points.reshape(-1, 2).astype(np.float64),
        charuco_corners=corners_flat,
        charuco_ids=ids_flat,
    )
    pose, reason = _estimate_board_pose_with_reason(
        observation, model, camera_matrix, dist_coeffs
    )
    return BoardPoseResult(
        pose=pose,
        detected_intersections=detected,
        no_pose_reason=reason,
        charuco_corners=corners_flat,
    )


def board_point_to_camera(
    point_board: np.ndarray,
    pose: BoardPoseEstimate,
) -> np.ndarray:
    point = np.asarray(point_board, dtype=np.float64).reshape(3)
    return pose.rotation @ point + pose.origin


def camera_point_to_board(
    point_camera: np.ndarray,
    pose: BoardPoseEstimate,
) -> np.ndarray:
    point = np.asarray(point_camera, dtype=np.float64).reshape(3)
    return pose.rotation.T @ (point - pose.origin)


def select_board_pose(
    current: BoardPoseEstimate | None,
    retained: BoardPoseEstimate | None,
    camera_motion: CameraMotion,
) -> tuple[BoardPoseEstimate | None, BoardPoseEstimate | None]:
    """Return the usable pose and updated retained pose for static mode."""
    if camera_motion == "dynamic":
        return current, retained
    if current is not None:
        return current, current
    return retained, retained


def board_points_to_opencv(
    points_board: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_board, dtype=np.float64).reshape(-1, 3)
    return ((OPENCV_FROM_BOARD_REFERENCE @ points.T).T).reshape(points_board.shape)


def make_charuco_detector(model: BoardModel) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector]:
    board = build_charuco_board(model)
    return board, cv2.aruco.CharucoDetector(board)
