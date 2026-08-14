"""Live camera calibration with a ChArUco or checkerboard."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from object_apriltag.board_model import (
    ARUCO_DICTIONARIES,
    build_charuco_board_from_geometry,
    load_board_model,
    reject_mixed_board_model_args,
)
from object_apriltag.board_pose import (
    charuco_corners_consistent,
    charuco_draw_arrays,
    parse_charuco_detection,
)
from object_apriltag.frame_source import format_frame_source, open_frame_source, parse_frame_source, read_frame


METERS_PER_INCH = 0.0254
A4_WIDTH_M = 0.210
A4_HEIGHT_M = 0.297
DEFAULT_PRINT_DPI = 300.0


def meters_to_pixels(length_m: float, dpi: float) -> int:
    return max(1, int(round(length_m * dpi / METERS_PER_INCH)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from live ChArUco or checkerboard detections.",
        epilog=(
            "Controls:\n"
            "  Space  capture a frame when the board is fully detected\n"
            "  q      finish calibration and write --output\n"
            "\n"
            "Layout is height × width (rows × columns):\n"
            "  charuco_board  chess square count (e.g. 7×10 → --layout 7 10)\n"
            "  checkerboard   inner corner count (e.g. 6×9 for a 7×10 square board)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--board-type",
        choices=("charuco_board", "checkerboard"),
        default="charuco_board",
        help="Calibration target (default: charuco_board).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        help="ArUco marker edge length in meters (required for charuco_board).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="Chess square edge length in meters (default: 0.025).",
    )
    parser.add_argument(
        "--layout",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="Board layout as height × width (rows × columns).",
    )
    parser.add_argument(
        "--source",
        type=parse_frame_source,
        help="Frame source: camera device index (e.g. 0) or path to a video file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Intrinsics JSON output path (e.g. config/Camera/<profile>/intrinsics.json).",
    )
    parser.add_argument(
        "--dictionary",
        choices=tuple(ARUCO_DICTIONARIES),
        default="4x4_50",
        help="ArUco dictionary for charuco_board (default: 4x4_50).",
    )
    parser.add_argument(
        "--save-board",
        type=Path,
        help=(
            "Save a true-scale ChArUco board on an A4 page (PNG with DPI metadata) and exit. "
            "Print at 100%% scale so each square matches --square-size."
        ),
    )
    parser.add_argument(
        "--print-dpi",
        type=float,
        default=DEFAULT_PRINT_DPI,
        help="Print resolution for --save-board (default: 300). Used for pixel size and PNG DPI metadata.",
    )
    parser.add_argument(
        "--board-model",
        type=Path,
        help="ChArUco board model JSON path (mutually exclusive with individual geometry flags).",
    )
    return parser.parse_args()


def checkerboard_object_points(layout_width: int, layout_height: int, square_size: float) -> np.ndarray:
    grid = np.mgrid[0:layout_width, 0:layout_height].T.reshape(-1, 2)
    points = np.zeros((layout_width * layout_height, 3), dtype=np.float32)
    points[:, :2] = grid.astype(np.float32)
    points *= float(square_size)
    return points


# Backward-compatible alias for tests and callers using geometry parameters.
build_charuco_board = build_charuco_board_from_geometry


def save_png_with_dpi(image: np.ndarray, path: Path, dpi: float) -> None:
    if image.ndim == 2:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(path, dpi=(dpi, dpi))


def save_charuco_board_a4(
    board: cv2.aruco.CharucoBoard,
    path: Path,
    *,
    layout_width: int,
    layout_height: int,
    square_size: float,
    dpi: float,
) -> None:
    if dpi <= 0.0:
        raise ValueError("--print-dpi must be positive.")

    board_width_m = layout_width * square_size
    board_height_m = layout_height * square_size
    board_width_px = meters_to_pixels(board_width_m, dpi)
    board_height_px = meters_to_pixels(board_height_m, dpi)
    page_width_px = meters_to_pixels(A4_WIDTH_M, dpi)
    page_height_px = meters_to_pixels(A4_HEIGHT_M, dpi)

    board_image = board.generateImage(
        (board_width_px, board_height_px),
        marginSize=0,
        borderBits=1,
    )

    board_width_mm = board_width_m * 1000.0
    board_height_mm = board_height_m * 1000.0
    if board_width_mm > A4_WIDTH_M * 1000.0 or board_height_mm > A4_HEIGHT_M * 1000.0:
        warnings.warn(
            f"Board size {board_width_mm:.1f}×{board_height_mm:.1f} mm exceeds A4 "
            f"({A4_WIDTH_M * 1000:.0f}×{A4_HEIGHT_M * 1000:.0f} mm); pattern will be clipped on a true-scale A4 print.",
            stacklevel=2,
        )

    offset_x = (page_width_px - board_width_px) // 2
    offset_y = (page_height_px - board_height_px) // 2
    page = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)

    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    src_x1 = min(board_width_px, page_width_px - offset_x)
    src_y1 = min(board_height_px, page_height_px - offset_y)
    dst_x0 = max(0, offset_x)
    dst_y0 = max(0, offset_y)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        page[dst_y0:dst_y1, dst_x0:dst_x1] = board_image[src_y0:src_y1, src_x0:src_x1]

    save_png_with_dpi(page, path, dpi)
    print(f"Saved board image: {path}")
    print(f"A4 page: {page_width_px}×{page_height_px} px at {dpi:.0f} DPI")
    print(
        f"Board on page: {board_width_mm:.1f}×{board_height_mm:.1f} mm "
        f"({board_width_px}×{board_height_px} px, square {square_size * 1000:.1f} mm)"
    )
    print("Print at 100% scale (not 'fit to page') so squares match --square-size.")


def detect_charuco(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
) -> tuple[np.ndarray, np.ndarray] | None:
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if not charuco_corners_consistent(charuco_corners, charuco_ids):
        return None
    corners_flat, ids_flat = parse_charuco_detection(charuco_corners, charuco_ids)
    assert corners_flat is not None and ids_flat is not None
    object_points, image_points = board.matchImagePoints(
        corners_flat.reshape(-1, 1, 2),
        ids_flat.reshape(-1, 1),
    )
    if object_points is None or len(object_points) < 4:
        return None
    return object_points.reshape(-1, 3), image_points.reshape(-1, 2)


def detect_checkerboard(
    gray: np.ndarray,
    layout_width: int,
    layout_height: int,
    square_size: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    pattern_size = (layout_width, layout_height)
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    if not found:
        return None
    object_points = checkerboard_object_points(layout_width, layout_height, square_size)
    image_points = corners.reshape(-1, 2)
    return object_points, image_points


def mean_reprojection_error(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
) -> float:
    total_error = 0.0
    total_points = 0
    for object_points, image_points, rvec, tvec in zip(
        object_points_list, image_points_list, rvecs, tvecs, strict=True
    ):
        projected, _ = cv2.projectPoints(
            object_points.reshape(-1, 1, 3),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(image_points - projected, axis=1)
        total_error += float(np.sum(errors))
        total_points += len(errors)
    if total_points == 0:
        return 0.0
    return total_error / total_points


def write_intrinsics_json(
    path: Path,
    *,
    calibration_source: str,
    image_width: int,
    image_height: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    mean_reprojection_error_px: float,
    layout_width: int,
    layout_height: int,
    square_size: float,
    captured_frames: int,
    marker_size: float | None = None,
    dictionary: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "calibration_source": calibration_source,
        "image_size": [image_width, image_height],
        "camera_matrix": camera_matrix.reshape(3, 3).tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "mean_reprojection_error_px": mean_reprojection_error_px,
        "squares_x": layout_width,
        "squares_y": layout_height,
        "inner_corners_x": layout_width,
        "inner_corners_y": layout_height,
        "square_size": square_size,
        "captured_frames": captured_frames,
    }
    if marker_size is not None:
        payload["marker_size"] = marker_size
    if dictionary is not None:
        payload["dictionary"] = dictionary
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def draw_charuco_overlay(
    frame: np.ndarray,
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
) -> bool:
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    draw_arrays = charuco_draw_arrays(charuco_corners, charuco_ids)
    if draw_arrays is not None:
        corners_draw, ids_draw = draw_arrays
        cv2.aruco.drawDetectedCornersCharuco(frame, corners_draw, ids_draw)
    return draw_arrays is not None and draw_arrays[1].shape[0] >= 4


def draw_checkerboard_overlay(
    frame: np.ndarray,
    gray: np.ndarray,
    layout_width: int,
    layout_height: int,
) -> bool:
    pattern_size = (layout_width, layout_height)
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    if found:
        cv2.drawChessboardCorners(frame, pattern_size, corners, found)
    return bool(found)


def main() -> None:
    args = parse_args()
    reject_mixed_board_model_args(args)

    board_model = None
    if args.board_model is not None:
        board_model = load_board_model(args.board_model)
        layout_height = board_model.layout_height
        layout_width = board_model.layout_width
        square_size = board_model.square_size
        marker_size = board_model.marker_size
        dictionary = board_model.dictionary
        board_type = board_model.board_type
    else:
        if args.layout is None:
            raise RuntimeError("--layout is required when --board-model is not provided.")
        layout_height, layout_width = args.layout
        square_size = args.square_size
        marker_size = args.marker_size
        dictionary = args.dictionary
        board_type = args.board_type

    if args.save_board is None:
        if args.source is None:
            raise RuntimeError("--source is required for live calibration.")
        if args.output is None:
            raise RuntimeError("--output is required for live calibration.")

    if board_type == "charuco_board" and marker_size is None:
        raise RuntimeError("--marker-size is required when --board-type is charuco_board.")

    charuco_board: cv2.aruco.CharucoBoard | None = None
    charuco_detector: cv2.aruco.CharucoDetector | None = None
    if board_type == "charuco_board":
        charuco_board = build_charuco_board_from_geometry(
            layout_width,
            layout_height,
            square_size,
            marker_size,
            dictionary,
        )
        charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
        if args.save_board is not None:
            save_charuco_board_a4(
                charuco_board,
                args.save_board,
                layout_width=layout_width,
                layout_height=layout_height,
                square_size=square_size,
                dpi=args.print_dpi,
            )
            return

    capture = open_frame_source(args.source)
    source_label = format_frame_source(args.source)

    print(f"Board type: {board_type}")
    print(f"Layout: {layout_height} rows × {layout_width} columns")
    print(f"Square size: {square_size:.4f} m")
    if board_type == "charuco_board":
        print(f"Marker size: {marker_size:.4f} m")
        print(f"Dictionary: {dictionary}")
    print(source_label)
    print("Space = capture frame, q = calibrate and save")

    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    image_width = 0
    image_height = 0

    while True:
        ok, frame = read_frame(capture, args.source)
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a frame from {source_label}.")

        image_height, image_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preview = frame.copy()

        if board_type == "charuco_board":
            detected = draw_charuco_overlay(
                preview, gray, charuco_board, charuco_detector
            )
        else:
            detected = draw_checkerboard_overlay(preview, gray, layout_width, layout_height)

        status = "board detected" if detected else "searching for board"
        cv2.putText(
            preview,
            f"{status} | captured: {len(object_points_list)} | Space capture | q save",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Camera calibration", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" ") and detected:
            if board_type == "charuco_board":
                detection = detect_charuco(gray, charuco_board, charuco_detector)
            else:
                detection = detect_checkerboard(gray, layout_width, layout_height, square_size)
            if detection is None:
                continue
            object_points, image_points = detection
            object_points_list.append(object_points.astype(np.float32))
            image_points_list.append(image_points.astype(np.float32))
            print(f"Captured frame {len(object_points_list)} ({len(image_points)} points)")

    capture.release()
    cv2.destroyAllWindows()

    if len(object_points_list) < 3:
        raise RuntimeError("Need at least 3 captured frames with a detected board.")

    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list,
        image_points_list,
        (image_width, image_height),
        None,
        None,
    )
    reproj_error = mean_reprojection_error(
        object_points_list,
        image_points_list,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
    )

    calibration_source = "charuco" if board_type == "charuco_board" else "checkerboard"
    write_intrinsics_json(
        args.output,
        calibration_source=calibration_source,
        image_width=image_width,
        image_height=image_height,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        mean_reprojection_error_px=reproj_error,
        layout_width=layout_width,
        layout_height=layout_height,
        square_size=square_size,
        captured_frames=len(object_points_list),
        marker_size=marker_size if board_type == "charuco_board" else None,
        dictionary=dictionary if board_type == "charuco_board" else None,
    )

    print(f"Saved calibration: {args.output}")
    print(f"Captured frames: {len(object_points_list)}")
    print(f"Mean reprojection error: {reproj_error:.3f} px")
    print(f"Image size: {image_width}x{image_height}")


if __name__ == "__main__":
    main()
