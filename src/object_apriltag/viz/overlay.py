"""OpenCV overlays for object detection."""

from __future__ import annotations

import cv2
import numpy as np

from object_apriltag.board_pose import BoardPoseEstimate, board_point_to_camera, camera_point_to_board
from object_apriltag.detector import ObjectPose
from object_apriltag.eraser import EraserModel, eraser_offset_to_model_point
from object_apriltag.layout import CORNER_NAMES, CORNER_LABELS, MarkerLayout, layout_point_to_camera, marker_color_bgr
from object_apriltag.viz.projection import (
    object_axis_image_points,
    opencv_image_point,
    project_camera_point,
)
from object_apriltag.viz.skeleton import ObjectModel

BOARD_LABEL_COLOR_BGR = (255, 255, 255)


def format_board_coordinate_mm(point_board: np.ndarray) -> str:
    point = np.asarray(point_board, dtype=np.float64).reshape(3) * 1000.0
    return f"({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}) mm"


def format_board_coordinate_hud_row(identity: str, point_board: np.ndarray) -> str:
    return f"{identity}: {format_board_coordinate_mm(point_board)}"


def draw_board_coordinate_preview(
    frame: np.ndarray,
    point_board: np.ndarray,
    board_pose: BoardPoseEstimate,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    label: str = "preview",
) -> None:
    point_camera = board_point_to_camera(point_board, board_pose)
    if point_camera[2] <= 0.0:
        return
    point = opencv_image_point(project_camera_point(point_camera, camera_matrix, dist_coeffs))
    if point is None:
        return

    color = (255, 0, 255)
    cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    label_origin = opencv_image_point((point[0] + 10, point[1] - 10))
    if label_origin is not None:
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_board_coordinates_hud(
    frame: np.ndarray,
    entries: list[tuple[str, np.ndarray]],
    *,
    title: str = "Board coordinates",
    margin: int = 10,
    padding: int = 8,
    line_height: int = 18,
    font_scale: float = 0.45,
    color: tuple[int, int, int] = BOARD_LABEL_COLOR_BGR,
) -> None:
    if not entries:
        return

    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    entry_lines = [format_board_coordinate_hud_row(identity, point_board) for identity, point_board in entries]
    lines = [title, *entry_lines]
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_width = max(size[0] for size in text_sizes) + 2 * padding
    available_height = height - 2 * margin
    max_content_lines = max(1, (available_height - 2 * padding) // line_height)
    truncated = len(lines) > max_content_lines
    if truncated:
        omitted = len(lines) - max_content_lines + 1
        lines = lines[: max_content_lines - 1] + [f"... +{omitted} more"]

    panel_height = len(lines) * line_height + 2 * padding
    x0 = max(margin, width - panel_width - margin)
    y0 = margin
    x1 = min(width - margin, x0 + panel_width)
    y1 = min(height - margin, y0 + panel_height)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)

    for index, line in enumerate(lines):
        text_y = y0 + padding + (index + 1) * line_height - 4
        cv2.putText(
            frame,
            line,
            (x0 + padding, text_y),
            font,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (x0 + padding, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_board_coordinate_labels_for_points(
    frame: np.ndarray,
    board_pose: BoardPoseEstimate,
    labeled_camera_points: list[tuple[str, np.ndarray]],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    colors: list[tuple[int, int, int]] | None = None,
) -> None:
    del camera_matrix, dist_coeffs, colors
    entries = [
        (identity, camera_point_to_board(camera_point, board_pose))
        for identity, camera_point in labeled_camera_points
    ]
    draw_board_coordinates_hud(frame, entries)


def draw_object_model_board_coordinate_labels(
    frame: np.ndarray,
    pose: ObjectPose,
    board_pose: BoardPoseEstimate,
    marker_model: MarkerLayout,
    object_model: ObjectModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    labeled_points: list[tuple[str, np.ndarray]] = []
    colors: list[tuple[int, int, int]] = []
    for name in object_model.keypoint_names:
        labeled_points.append(
            (
                name,
                layout_point_to_camera(
                    object_model.keypoints[name],
                    pose.rotation,
                    pose.origin,
                    marker_model,
                ),
            )
        )
        colors.append(KEYPOINT_COLORS_BGR.get(name, (200, 200, 200)))
    _draw_board_coordinate_labels_for_points(
        frame, board_pose, labeled_points, camera_matrix, dist_coeffs, colors=colors
    )


def draw_marker_model_board_coordinate_labels(
    frame: np.ndarray,
    pose: ObjectPose,
    board_pose: BoardPoseEstimate,
    marker_model: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    labeled_points: list[tuple[str, np.ndarray]] = []
    colors: list[tuple[int, int, int]] = []
    for marker_id in sorted(marker_model.footprints):
        footprint = marker_model.footprints[marker_id]
        color = marker_color_bgr(marker_id)
        for corner_name, point_layout in footprint.corners_by_name().items():
            labeled_points.append(
                (
                    f"{marker_id}:{CORNER_LABELS[corner_name]}",
                    layout_point_to_camera(point_layout, pose.rotation, pose.origin, marker_model),
                )
            )
            colors.append(color)
    _draw_board_coordinate_labels_for_points(
        frame, board_pose, labeled_points, camera_matrix, dist_coeffs, colors=colors
    )


def draw_eraser_model_board_coordinate_labels(
    frame: np.ndarray,
    pose: ObjectPose,
    board_pose: BoardPoseEstimate,
    eraser_model: EraserModel,
    marker_model: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    labeled_points: list[tuple[str, np.ndarray]] = []
    eraser_color = (0, 255, 255)
    for plane_index, plane in enumerate(eraser_model.planes):
        plane_ref = plane.plane_id if plane.plane_id is not None else str(plane_index)
        for corner_name, offset in zip(CORNER_NAMES, plane.corners(), strict=True):
            model_point = eraser_offset_to_model_point(offset, marker_model)
            labeled_points.append(
                (
                    f"{plane_ref}:{CORNER_LABELS[corner_name]}",
                    layout_point_to_camera(model_point, pose.rotation, pose.origin, marker_model),
                )
            )
    _draw_board_coordinate_labels_for_points(
        frame,
        board_pose,
        labeled_points,
        camera_matrix,
        dist_coeffs,
        colors=[eraser_color] * len(labeled_points),
    )


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
    model: ObjectModel,
    *,
    draw_point_labels: bool = True,
) -> None:
    points_by_name = {name: image_points[index] for index, name in enumerate(model.keypoint_names)}

    for start_name, end_name in model.skeleton_edges:
        start_pt = opencv_image_point(points_by_name[start_name])
        end_pt = opencv_image_point(points_by_name[end_name])
        if start_pt is None or end_pt is None:
            continue
        cv2.line(
            frame,
            start_pt,
            end_pt,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    for index, name in enumerate(model.keypoint_names):
        point = opencv_image_point(image_points[index])
        if point is None:
            continue
        color = KEYPOINT_COLORS_BGR.get(name, (200, 200, 200))
        cv2.circle(frame, point, 7, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, point, 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        if draw_point_labels:
            label_origin = opencv_image_point((point[0] + 8, point[1] - 8))
            if label_origin is not None:
                cv2.putText(frame, name, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, name, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_object_origin(
    frame: np.ndarray,
    object_origin_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    radius: int = 9,
    color: tuple[int, int, int] = (0, 255, 0),
    label: str = "object origin",
) -> tuple[int, int] | None:
    point = project_camera_point(object_origin_camera, camera_matrix, dist_coeffs)
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


def draw_object_orientation(
    frame: np.ndarray,
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
) -> None:
    """Draw fused object X/Y/Z axes in the camera image (OpenCV camera frame)."""
    try:
        origin_xy, x_end, y_end, z_end = object_axis_image_points(
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
        label_origin = opencv_image_point((tip[0] + 4, tip[1] - 4))
        if label_origin is not None:
            cv2.putText(
                frame,
                label,
                label_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_object_pose(
    frame: np.ndarray,
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_size_m: float,
    model: ObjectModel,
    marker_model: MarkerLayout,
    *,
    draw_point_labels: bool = True,
) -> None:
    draw_object_origin(frame, pose.origin, camera_matrix, dist_coeffs, label="object origin")
    try:
        origin_xy, x_end, y_end, z_end = object_axis_image_points(
            pose.rotation, pose.origin, camera_matrix, dist_coeffs, marker_size_m * 0.5
        )
        cv2.arrowedLine(frame, origin_xy, x_end, (0, 0, 255), 2, tipLength=0.25)
        cv2.arrowedLine(frame, origin_xy, y_end, (0, 255, 0), 2, tipLength=0.25)
        cv2.arrowedLine(frame, origin_xy, z_end, (255, 0, 0), 2, tipLength=0.25)
    except (RuntimeError, ValueError):
        return

    image_points = np.full((len(model.keypoint_names), 2), np.nan, dtype=np.float64)
    for index, name in enumerate(model.keypoint_names):
        camera_point = layout_point_to_camera(
            model.keypoints[name],
            pose.rotation,
            pose.origin,
            marker_model,
        )
        image_points[index] = project_camera_point(camera_point, camera_matrix, dist_coeffs)
    if np.any(np.isfinite(image_points)):
        draw_racket_keypoints(
            frame,
            image_points,
            model,
            draw_point_labels=draw_point_labels,
        )


def draw_marker_model_footprints(
    frame: np.ndarray,
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_model: MarkerLayout,
    *,
    draw_point_labels: bool = True,
) -> None:
    height, width = frame.shape[:2]
    for marker_id in sorted(marker_model.footprints):
        footprint = marker_model.footprints[marker_id]
        color = marker_color_bgr(marker_id)
        corners_layout = footprint.corners()
        image_corners: list[tuple[int, int]] = []
        for point_layout in corners_layout:
            camera_point = layout_point_to_camera(point_layout, pose.rotation, pose.origin, marker_model)
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
            camera_point = layout_point_to_camera(point_layout, pose.rotation, pose.origin, marker_model)
            projected = project_camera_point(camera_point, camera_matrix, dist_coeffs)
            if not np.all(np.isfinite(projected)):
                continue
            x, y = int(round(projected[0])), int(round(projected[1]))
            if not (0 <= x < width and 0 <= y < height):
                continue
            cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            if draw_point_labels:
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


def draw_eraser_planes(frame: np.ndarray, polygons: list[np.ndarray]) -> None:
    for polygon in polygons:
        points = np.round(polygon).astype(np.int32)
        cv2.polylines(
            frame,
            [points],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )


def draw_object_model_edit_hud(
    frame: np.ndarray,
    *,
    dirty: bool,
    status_message: str,
    origin: tuple[int, int] = (10, 90),
    line_spacing: int = 22,
) -> None:
    lines = [
        "edit: e add/update  s save  q quit  x discard+quit",
        "modified" if dirty else "saved",
    ]
    if status_message:
        lines.append(status_message)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    color = (0, 255, 255) if dirty else (200, 255, 200)
    for index, line in enumerate(lines):
        x, y = origin[0], origin[1] + index * line_spacing
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


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
