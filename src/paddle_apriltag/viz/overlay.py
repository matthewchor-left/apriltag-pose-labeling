"""OpenCV overlays for paddle detection."""

from __future__ import annotations

import cv2
import numpy as np

from paddle_apriltag.detector import PaddlePose
from paddle_apriltag.layout import CORNER_LABELS, MarkerLayout, layout_point_to_camera, marker_color_bgr
from paddle_apriltag.viz.projection import (
    paddle_axis_image_points,
    project_camera_point,
)
from paddle_apriltag.viz.skeleton import PaddleModel

KEYPOINT_COLORS_BGR = {
    "top": (128, 0, 128),
    "left": (255, 255, 0),
    "bottom": (0, 255, 0),
    "handle": (203, 192, 255),
    "right": (165, 42, 42),
}


def draw_racket_keypoints(
    frame: np.ndarray,
    image_points: np.ndarray,
    model: PaddleModel,
) -> None:
    points_by_name = {name: image_points[index] for index, name in enumerate(model.keypoint_names)}

    for start_name, end_name in model.skeleton_edges:
        start = points_by_name[start_name]
        end = points_by_name[end_name]
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            continue
        cv2.line(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    for index, name in enumerate(model.keypoint_names):
        x, y = image_points[index]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        point = (int(round(x)), int(round(y)))
        color = KEYPOINT_COLORS_BGR.get(name, (200, 200, 200))
        cv2.circle(frame, point, 7, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, point, 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame, name, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, name, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_paddle_origin(
    frame: np.ndarray,
    paddle_origin_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    radius: int = 9,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "paddle origin",
) -> tuple[int, int] | None:
    point = project_camera_point(paddle_origin_camera, camera_matrix, dist_coeffs)
    if not np.all(np.isfinite(point)):
        return None

    x, y = int(round(point[0])), int(round(point[1]))
    height, width = frame.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None

    cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (x, y), radius, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return x, y


def draw_paddle_orientation(
    frame: np.ndarray,
    pose: PaddlePose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> None:
    """Draw fused paddle X/Y/Z axes in the camera image (OpenCV camera frame)."""
    try:
        origin_xy, x_end, y_end, z_end = paddle_axis_image_points(
            pose.rotation,
            pose.origin,
            camera_matrix,
            dist_coeffs,
            axis_length_m,
        )
    except (RuntimeError, ValueError):
        return

    cv2.circle(frame, origin_xy, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, origin_xy, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    axes = (
        (x_end, (0, 0, 255), "X"),
        (y_end, (0, 255, 0), "Y"),
        (z_end, (255, 0, 0), "Z"),
    )
    for tip, color, label in axes:
        cv2.arrowedLine(frame, origin_xy, tip, color, 2, tipLength=0.2, line_type=cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (tip[0] + 4, tip[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_paddle_pose(
    frame: np.ndarray,
    pose: PaddlePose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size_m: float,
    model: PaddleModel,
) -> None:
    draw_paddle_origin(frame, pose.origin, camera_matrix, dist_coeffs, label="paddle origin")
    try:
        origin_xy, x_end, y_end, z_end = paddle_axis_image_points(
            pose.rotation, pose.origin, camera_matrix, dist_coeffs, marker_size_m * 0.5
        )
        cv2.arrowedLine(frame, origin_xy, x_end, (0, 0, 255), 2, tipLength=0.25)
        cv2.arrowedLine(frame, origin_xy, y_end, (0, 255, 0), 2, tipLength=0.25)
        cv2.arrowedLine(frame, origin_xy, z_end, (255, 0, 0), 2, tipLength=0.25)
    except (RuntimeError, ValueError):
        return

    image_points = []
    for point in model.object_points:
        camera_point = pose.rotation @ point + pose.origin
        projected = project_camera_point(camera_point, camera_matrix, dist_coeffs)
        if np.all(np.isfinite(projected)):
            image_points.append(projected)
    if image_points:
        draw_racket_keypoints(frame, np.asarray(image_points, dtype=np.float32), model)


def draw_marker_layout_footprints(
    frame: np.ndarray,
    pose: PaddlePose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    layout: MarkerLayout,
) -> None:
    height, width = frame.shape[:2]
    for marker_id in sorted(layout.footprints):
        footprint = layout.footprints[marker_id]
        color = marker_color_bgr(marker_id)
        corners_paddle = footprint.corners()
        image_corners: list[tuple[int, int]] = []
        for point_layout in corners_paddle:
            camera_point = layout_point_to_camera(point_layout, pose.rotation, pose.origin, layout)
            projected = project_camera_point(camera_point, camera_matrix, dist_coeffs)
            if not np.all(np.isfinite(projected)):
                continue
            x, y = int(round(projected[0])), int(round(projected[1]))
            if 0 <= x < width and 0 <= y < height:
                image_corners.append((x, y))

        if len(image_corners) == 4:
            cv2.polylines(
                frame,
                [np.asarray(image_corners, dtype=np.int32)],
                isClosed=True,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        for corner_name, point_layout in footprint.corners_by_name().items():
            label = CORNER_LABELS[corner_name]
            radius = 6 if label in {"tl", "br"} else 5
            camera_point = layout_point_to_camera(point_layout, pose.rotation, pose.origin, layout)
            projected = project_camera_point(camera_point, camera_matrix, dist_coeffs)
            if not np.all(np.isfinite(projected)):
                continue
            x, y = int(round(projected[0])), int(round(projected[1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame, f"{marker_id}:{label}", (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )


def draw_marker_annotations(
    frame: np.ndarray,
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    layout: MarkerLayout,
    *,
    draw: bool = True,
) -> bool:
    del marker_size_m, camera_matrix, dist_coeffs, layout
    if not draw:
        return True

    pts = corners.reshape(4, 2).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    anchor = tuple(int(round(v)) for v in pts[0])
    cv2.putText(
        frame,
        f"id={marker_id}",
        (anchor[0] + 8, anchor[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return True


def draw_live_hud(
    frame: np.ndarray,
    fps: float,
    avg_reproj_error: float | None,
    origin: tuple[int, int] = (10, 30),
    line_spacing: int = 30,
) -> None:
    lines = [f"FPS: {fps:.1f}"]
    lines.append(f"avg reproj: {avg_reproj_error:.1f}px" if avg_reproj_error is not None else "avg reproj: --")
    for index, line in enumerate(lines):
        x, y = origin[0], origin[1] + index * line_spacing
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
