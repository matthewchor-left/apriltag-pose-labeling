"""Board Reference Frame overlay rendering."""

from __future__ import annotations

import cv2
import numpy as np

from object_apriltag.board_model import BoardModel
from object_apriltag.board_pose import BoardPoseEstimate, board_point_to_camera

COLOR_X = (0, 0, 255)
COLOR_Y = (0, 255, 0)
COLOR_Z = (255, 0, 0)
COLOR_GRID = (160, 160, 160)
COLOR_INTERSECTION = (255, 255, 0)
COLOR_HUD = (255, 255, 255)
Y_AXIS_MIN_SCREEN_PX = 18.0


def board_extent_m(model: BoardModel, grid_margin_squares: int) -> tuple[float, float, float, float]:
    margin = grid_margin_squares * model.square_size
    width_m = model.layout_width * model.square_size
    height_m = model.layout_height * model.square_size
    return (-margin, width_m + margin, -margin, height_m + margin)


def grid_line_points_board(
    model: BoardModel,
    grid_margin_squares: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_min, x_max, z_min, z_max = board_extent_m(model, grid_margin_squares)
    square = model.square_size
    x_lines: list[np.ndarray] = []
    z_lines: list[np.ndarray] = []

    z = z_min
    while z <= z_max + 1e-9:
        x_lines.append(
            np.array([[x_min, 0.0, z], [x_max, 0.0, z]], dtype=np.float64)
        )
        z += square

    x = x_min
    while x <= x_max + 1e-9:
        z_lines.append(
            np.array([[x, 0.0, z_min], [x, 0.0, z_max]], dtype=np.float64)
        )
        x += square

    return x_lines, z_lines


def sample_board_segment(
    start_board: np.ndarray,
    end_board: np.ndarray,
    samples: int,
) -> np.ndarray:
    start = np.asarray(start_board, dtype=np.float64).reshape(3)
    end = np.asarray(end_board, dtype=np.float64).reshape(3)
    alphas = np.linspace(0.0, 1.0, max(2, samples), dtype=np.float64)
    return start[None, :] + alphas[:, None] * (end - start)[None, :]


def project_board_points(
    points_board: np.ndarray,
    pose: BoardPoseEstimate,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    points_camera = np.stack(
        [board_point_to_camera(point, pose) for point in points_board.reshape(-1, 3)],
        axis=0,
    )
    projected, _ = cv2.projectPoints(
        points_camera.reshape(-1, 1, 3).astype(np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def project_board_polyline(
    segment_board: np.ndarray,
    pose: BoardPoseEstimate,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    samples: int = 32,
) -> np.ndarray:
    points = sample_board_segment(segment_board[0], segment_board[1], samples)
    return project_board_points(points, pose, camera_matrix, dist_coeffs)


def draw_projected_polyline(
    frame: np.ndarray,
    image_points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    pixels = np.round(image_points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pixels], False, color, thickness, cv2.LINE_AA)


def axis_segments_board(model: BoardModel, axis_length_squares: float) -> dict[str, np.ndarray]:
    length = axis_length_squares * model.square_size
    origin = np.zeros(3, dtype=np.float64)
    return {
        "x": np.array([origin, [length, 0.0, 0.0]], dtype=np.float64),
        "y": np.array([origin, [0.0, length, 0.0]], dtype=np.float64),
        "z": np.array([origin, [0.0, 0.0, length]], dtype=np.float64),
    }


def draw_board_axes(
    frame: np.ndarray,
    pose: BoardPoseEstimate,
    model: BoardModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_squares: float,
) -> None:
    segments = axis_segments_board(model, axis_length_squares)
    colors = {"x": COLOR_X, "y": COLOR_Y, "z": COLOR_Z}
    labels = {"x": "X", "y": "Y", "z": "Z"}

    for axis_name, segment in segments.items():
        projected = project_board_polyline(segment, pose, camera_matrix, dist_coeffs)
        draw_projected_polyline(frame, projected, colors[axis_name], thickness=2)
        end = tuple(int(round(v)) for v in projected[-1])
        cv2.putText(
            frame,
            labels[axis_name],
            end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[axis_name],
            1,
            cv2.LINE_AA,
        )

    y_segment = segments["y"]
    y_projected = project_board_polyline(y_segment, pose, camera_matrix, dist_coeffs)
    origin = y_projected[0]
    y_end = y_projected[-1]
    if float(np.linalg.norm(y_end - origin)) < Y_AXIS_MIN_SCREEN_PX:
        origin_px = (int(round(origin[0])), int(round(origin[1])))
        cv2.circle(frame, origin_px, 10, COLOR_Y, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Y",
            (origin_px[0] + 12, origin_px[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_Y,
            1,
            cv2.LINE_AA,
        )


def draw_board_grid(
    frame: np.ndarray,
    pose: BoardPoseEstimate,
    model: BoardModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    grid_margin_squares: int,
) -> None:
    x_lines, z_lines = grid_line_points_board(model, grid_margin_squares)
    for segment in x_lines + z_lines:
        projected = project_board_polyline(segment, pose, camera_matrix, dist_coeffs)
        draw_projected_polyline(frame, projected, COLOR_GRID, thickness=1)


def draw_charuco_intersections(
    frame: np.ndarray,
    charuco_corners: np.ndarray,
) -> None:
    for corner in charuco_corners.reshape(-1, 2):
        center = (int(round(float(corner[0]))), int(round(float(corner[1]))))
        cv2.circle(frame, center, 4, COLOR_INTERSECTION, -1, cv2.LINE_AA)


def draw_board_frame_hud(
    frame: np.ndarray,
    *,
    pose: BoardPoseEstimate | None,
    detected_intersections: int,
    total_intersections: int,
    source_label: str,
    image_width: int,
    image_height: int,
    show_hud: bool,
    no_pose_reason: str | None = None,
) -> None:
    if not show_hud:
        return

    if pose is None:
        reason = no_pose_reason or "unknown"
        lines = [
            "board pose: NO POSE",
            f"reason: {reason}",
            f"ChArUco intersections: {detected_intersections} / {total_intersections}",
            f"source: {source_label}",
            f"resolution: {image_width}x{image_height}",
        ]
    else:
        lines = [
            "board pose: VALID",
            (
                "ChArUco intersections: "
                f"{pose.detected_intersections} / {pose.total_intersections}"
            ),
            f"pose reprojection RMS: {pose.reprojection_rms_px:.2f} px",
            f"source: {source_label}",
            f"resolution: {image_width}x{image_height}",
        ]

    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_HUD,
            1,
            cv2.LINE_AA,
        )
        y += 22


def render_board_frame_grid_axes(
    frame: np.ndarray,
    *,
    model: BoardModel,
    pose: BoardPoseEstimate,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    grid_margin_squares: int = 2,
    axis_length_squares: float = 2.0,
) -> None:
    draw_board_grid(
        frame,
        pose,
        model,
        camera_matrix,
        dist_coeffs,
        grid_margin_squares,
    )
    draw_board_axes(
        frame,
        pose,
        model,
        camera_matrix,
        dist_coeffs,
        axis_length_squares,
    )


def render_board_frame_overlay(
    frame: np.ndarray,
    *,
    model: BoardModel,
    pose: BoardPoseEstimate | None,
    charuco_corners: np.ndarray | None,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    grid_margin_squares: int,
    axis_length_squares: float,
    show_intersections: bool,
    show_hud: bool,
    source_label: str,
    detected_intersections: int | None = None,
    no_pose_reason: str | None = None,
) -> None:
    image_height, image_width = frame.shape[:2]
    if detected_intersections is None:
        detected_intersections = 0
        if charuco_corners is not None:
            detected_intersections = len(charuco_corners.reshape(-1, 2))
    if pose is not None:
        draw_board_grid(
            frame,
            pose,
            model,
            camera_matrix,
            dist_coeffs,
            grid_margin_squares,
        )
        draw_board_axes(
            frame,
            pose,
            model,
            camera_matrix,
            dist_coeffs,
            axis_length_squares,
        )
    if show_intersections and charuco_corners is not None:
        draw_charuco_intersections(frame, charuco_corners)
    draw_board_frame_hud(
        frame,
        pose=pose,
        detected_intersections=detected_intersections,
        total_intersections=model.total_charuco_intersections,
        source_label=source_label,
        image_width=image_width,
        image_height=image_height,
        show_hud=show_hud,
        no_pose_reason=no_pose_reason,
    )
