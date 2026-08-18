"""OpenCV overlays for object detection."""

from __future__ import annotations

import cv2
import numpy as np

from object_apriltag.detector import ObjectPose
from object_apriltag.layout import (
    CORNER_NAMES,
    CORNER_LABELS,
    MarkerLayout,
    layout_point_to_camera,
    marker_color_bgr,
    object_reference_orientation,
    object_reference_origin,
)
from object_apriltag.viz.projection import (
    object_axis_image_points,
    opencv_image_point,
    project_camera_point,
)
from object_apriltag.viz.skeleton import ObjectModel

AXIS_COLORS_LABELS = (
    ((0, 0, 255), "X"),
    ((0, 255, 0), "Y"),
    ((255, 0, 0), "Z"),
)


def _layout_point_image_xy(
    point_layout: np.ndarray,
    pose: ObjectPose,
    marker_model: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[int, int] | None:
    """Project one layout-frame point to an OpenCV-safe image coordinate.

    Args:
        point_layout: Point in marker_model/layout coordinates.
        pose: Fused object pose in the camera frame.
        marker_model: Marker layout defining the object frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        Integer pixel coordinate, or ``None`` when projection is unusable.
    """
    camera_point = layout_point_to_camera(
        point_layout, pose.rotation, pose.origin, marker_model
    )
    projected = project_camera_point(camera_point, camera_matrix, dist_coeffs)
    return opencv_image_point(projected)


def _draw_axis_triad(
    frame: np.ndarray,
    origin_xy: tuple[int, int],
    axis_tips: tuple[tuple[tuple[int, int], tuple[int, int, int], str], ...],
) -> None:
    """Draw a labeled RGB axis triad on a BGR frame.

    Args:
        frame: BGR image to draw on.
        origin_xy: Axis origin in pixel coordinates.
        axis_tips: Tuple of ``(tip_xy, color_bgr, label)`` for each axis.
    """
    cv2.circle(frame, origin_xy, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, origin_xy, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    for tip, color, label in axis_tips:
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
    """Draw object skeleton edges and keypoints on a BGR frame.

    Args:
        frame: BGR image to draw on.
        image_points: Keypoint image coordinates in ``model.keypoint_names`` order.
        model: Object skeleton model.
        draw_point_labels: Whether to draw text labels beside keypoints.
    """
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
    """Draw the fused object origin on a BGR frame.

    Args:
        frame: BGR image to draw on.
        object_origin_camera: Object origin in camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        radius: Circle radius in pixels.
        color: BGR fill color.
        label: Text label beside the origin.

    Returns:
        Integer pixel coordinate when drawn, or ``None`` when off-screen or invalid.
    """
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
    """Draw fused object X/Y/Z axes in the camera image.

    Args:
        frame: BGR image to draw on.
        pose: Fused object pose in the camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        axis_length_m: Axis length in meters.
    """
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

    _draw_axis_triad(
        frame,
        origin_xy,
        (
            (x_end, AXIS_COLORS_LABELS[0][0], "X"),
            (y_end, AXIS_COLORS_LABELS[1][0], "Y"),
            (z_end, AXIS_COLORS_LABELS[2][0], "Z"),
        ),
    )


def draw_reference_marker_orientation(
    frame: np.ndarray,
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_model: MarkerLayout,
) -> None:
    """Draw reference-marker X/Y/Z axes from the calibrated layout footprint.

    Args:
        frame: BGR image to draw on.
        pose: Fused object pose in the camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        marker_model: Marker layout with reference marker geometry.
    """
    origin_layout = object_reference_origin(marker_model)
    orientation = object_reference_orientation(marker_model)
    axis_length = marker_model.marker_size_for(marker_model.reference_marker_id) * 0.5

    origin_xy = _layout_point_image_xy(
        origin_layout, pose, marker_model, camera_matrix, dist_coeffs
    )
    if origin_xy is None:
        return

    axis_tips: list[tuple[tuple[int, int], tuple[int, int, int], str]] = []
    for axis_index, (color, label) in enumerate(AXIS_COLORS_LABELS):
        tip_layout = origin_layout + orientation[:, axis_index] * axis_length
        tip_xy = _layout_point_image_xy(
            tip_layout, pose, marker_model, camera_matrix, dist_coeffs
        )
        if tip_xy is None:
            return
        axis_tips.append((tip_xy, color, label))

    _draw_axis_triad(frame, origin_xy, tuple(axis_tips))


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
    """Draw fused object origin, axes, and skeleton keypoints on a BGR frame.

    Args:
        frame: BGR image to draw on.
        pose: Fused object pose in the camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        marker_size_m: Unused; retained for API compatibility.
        model: Object skeleton model.
        marker_model: Marker layout defining the object frame.
        draw_point_labels: Whether to draw text labels beside keypoints.
    """
    del marker_size_m
    axis_length_m = marker_model.marker_size_for(marker_model.reference_marker_id) * 0.5
    draw_object_origin(frame, pose.origin, camera_matrix, dist_coeffs, label="object origin")
    try:
        origin_xy, x_end, y_end, z_end = object_axis_image_points(
            pose.rotation, pose.origin, camera_matrix, dist_coeffs, axis_length_m
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
    """Draw projected marker footprints and reference-marker axes on a BGR frame.

    Args:
        frame: BGR image to draw on.
        pose: Fused object pose in the camera frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        marker_model: Marker layout with footprint geometry.
        draw_point_labels: Whether to draw marker and corner labels.
    """
    height, width = frame.shape[:2]
    for marker_id in sorted(marker_model.footprints):
        footprint = marker_model.footprints[marker_id]
        color = marker_color_bgr(marker_id, marker_model.reference_marker_id)
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

    draw_reference_marker_orientation(
        frame, pose, camera_matrix, dist_coeffs, marker_model
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
    """Draw a detected marker outline and ID label on a BGR frame.

    Args:
        frame: BGR image to draw on.
        corners: Detected marker corners.
        marker_id: AprilTag marker ID.
        marker_size_m: Unused; retained for API compatibility.
        camera_matrix: Unused; retained for API compatibility.
        dist_coeffs: Unused; retained for API compatibility.
        layout: Marker layout used to choose marker color.
        draw: When false, skip drawing and return success.

    Returns:
        ``True`` when annotations were drawn or skipped intentionally.
    """
    del marker_size_m, camera_matrix, dist_coeffs
    if not draw:
        return True

    color = marker_color_bgr(marker_id, layout.reference_marker_id)
    pts = corners.reshape(4, 2).astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    anchor = tuple(int(round(v)) for v in pts[0])
    cv2.putText(
        frame,
        f"id={marker_id}",
        (anchor[0] + 8, anchor[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )
    return True


def draw_eraser_planes(frame: np.ndarray, polygons: list[np.ndarray]) -> None:
    """Draw eraser-plane polygons on a BGR frame.

    Args:
        frame: BGR image to draw on.
        polygons: Polygon vertex arrays in image coordinates.
    """
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


STATUS_HUD_FONT_SCALE = 0.55
STATUS_HUD_THICKNESS = 1
STATUS_HUD_LINE_SPACING = 22
STATUS_HUD_ORIGIN_X = 10
STATUS_HUD_FIRST_LINE_Y = 24


def draw_status_hud_panel(frame: np.ndarray, lines: list[str]) -> None:
    """Draw a semi-opaque status HUD panel in the top-left corner.

    Args:
        frame: BGR image to draw on.
        lines: Text lines to render inside the panel.
    """
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = STATUS_HUD_FONT_SCALE
    thickness = STATUS_HUD_THICKNESS
    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    panel_right = min(frame.shape[1] - 1, text_width + 20)
    panel_bottom = min(frame.shape[0] - 1, len(lines) * STATUS_HUD_LINE_SPACING + 10)
    cv2.rectangle(frame, (0, 0), (panel_right, panel_bottom), (0, 0, 0), -1)

    y = STATUS_HUD_FIRST_LINE_Y
    for line in lines:
        cv2.putText(
            frame,
            line,
            (STATUS_HUD_ORIGIN_X, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += STATUS_HUD_LINE_SPACING


def format_reference_marker_camera_line(
    reference_marker_id: int,
    camera_point_m: np.ndarray | None,
) -> str:
    """Format a HUD line with the reference marker camera-frame position.

    Args:
        reference_marker_id: Reference marker ID.
        camera_point_m: Reference marker position in camera frame, or ``None``.

    Returns:
        Single-line HUD text with coordinates or placeholders.
    """
    if camera_point_m is None:
        return f"ref {reference_marker_id} cam xyz (m): --"
    point = np.asarray(camera_point_m, dtype=np.float64).reshape(3)
    return (
        f"ref {reference_marker_id} cam xyz (m): "
        f"{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}"
    )


def draw_live_hud(
    frame: np.ndarray,
    fps: float,
    layout_reproj_avg: float | None,
    layout_reproj_max: float | None,
    *,
    reference_marker_id: int | None = None,
    reference_marker_camera_m: np.ndarray | None = None,
) -> None:
    """Draw live FPS, layout reprojection, and optional reference-marker HUD lines.

    Args:
        frame: BGR image to draw on.
        fps: Smoothed frames-per-second estimate.
        layout_reproj_avg: Rolling mean layout reprojection error in pixels.
        layout_reproj_max: Current-frame max layout reprojection error in pixels.
        reference_marker_id: Optional reference marker ID for the extra HUD line.
        reference_marker_camera_m: Optional reference marker camera position in meters.
    """
    lines = [f"FPS: {fps:.1f}"]
    lines.append(
        f"layout reproj avg: {layout_reproj_avg:.1f}px"
        if layout_reproj_avg is not None
        else "layout reproj avg: --"
    )
    lines.append(
        f"layout reproj max: {layout_reproj_max:.1f}px"
        if layout_reproj_max is not None
        else "layout reproj max: --"
    )
    if reference_marker_id is not None:
        lines.append(
            format_reference_marker_camera_line(
                reference_marker_id, reference_marker_camera_m
            )
        )
    draw_status_hud_panel(frame, lines)
