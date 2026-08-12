"""Visualize the Board Reference Frame from a camera or still image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.board_model import load_board_model
from object_apriltag.board_pose import (
    detect_charuco_intersections,
    solve_board_pose,
    make_charuco_detector,
)
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.viz.board_frame import render_board_frame_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the Board Reference Frame from ChArUco intersections. "
            "Estimates a Board Pose Estimate in the camera frame and overlays axes and grid."
        )
    )
    parser.add_argument("--calibration", type=Path, required=True, help="Camera intrinsics JSON path.")
    parser.add_argument("--board-model", type=Path, required=True, help="ChArUco board model JSON path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", type=int, help="Camera device index.")
    source.add_argument("--image", type=Path, help="Still image path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Save rendered output. Still image: written automatically. Live camera: press S.",
    )
    parser.add_argument(
        "--grid-margin",
        type=int,
        default=2,
        help="Grid extension beyond the board in square counts (default: 2).",
    )
    parser.add_argument(
        "--axis-length-squares",
        type=float,
        default=2.0,
        help="Axis length in square counts (default: 2).",
    )
    parser.add_argument(
        "--show-intersections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw detected ChArUco intersections (default: on).",
    )
    parser.add_argument(
        "--hud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw diagnostics HUD (default: on).",
    )
    return parser.parse_args()


def validate_image_size(
    frame_width: int,
    frame_height: int,
    calibration_width: int,
    calibration_height: int,
    calibration_path: Path,
) -> None:
    if frame_width != calibration_width or frame_height != calibration_height:
        raise RuntimeError(
            "Frame size "
            f"{frame_width}x{frame_height} does not match calibration image size "
            f"{calibration_width}x{calibration_height} from {calibration_path}. "
            "Do not scale intrinsics."
        )


def process_frame(
    frame: np.ndarray,
    *,
    model,
    board,
    detector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    grid_margin_squares: int,
    axis_length_squares: float,
    show_intersections: bool,
    show_hud: bool,
    source_label: str,
) -> tuple[np.ndarray, object | None]:
    preview = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = solve_board_pose(
        gray, board, detector, model, camera_matrix, dist_coeffs
    )
    render_board_frame_overlay(
        preview,
        model=model,
        pose=result.pose,
        charuco_corners=result.charuco_corners,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        grid_margin_squares=grid_margin_squares,
        axis_length_squares=axis_length_squares,
        show_intersections=show_intersections,
        show_hud=show_hud,
        source_label=source_label,
        detected_intersections=result.detected_intersections,
        no_pose_reason=result.no_pose_reason,
    )
    return preview, result.pose


def run_still_image(args: argparse.Namespace) -> None:
    camera_matrix, dist_coeffs, image_width, image_height, _ = load_intrinsics(args.calibration)
    calibration_width, calibration_height = require_calibration_image_size(
        image_width, image_height, args.calibration
    )
    model = load_board_model(args.board_model)
    board, detector = make_charuco_detector(model)

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {args.image}")
    frame_height, frame_width = frame.shape[:2]
    validate_image_size(frame_width, frame_height, calibration_width, calibration_height, args.calibration)

    preview, _ = process_frame(
        frame,
        model=model,
        board=board,
        detector=detector,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        grid_margin_squares=args.grid_margin,
        axis_length_squares=args.axis_length_squares,
        show_intersections=args.show_intersections,
        show_hud=args.hud,
        source_label=args.image.name,
    )
    cv2.imshow("Board Reference Frame", preview)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), preview)
        print(f"Saved overlay: {args.output}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live_camera(args: argparse.Namespace) -> None:
    camera_matrix, dist_coeffs, image_width, image_height, _ = load_intrinsics(args.calibration)
    calibration_width, calibration_height = require_calibration_image_size(
        image_width, image_height, args.calibration
    )
    model = load_board_model(args.board_model)
    board, detector = make_charuco_detector(model)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}.")

    print(f"Camera {args.camera}")
    print("q = quit")
    if args.output is not None:
        print("S = save current rendered frame")

    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read a frame from camera {args.camera}.")
        frame_height, frame_width = frame.shape[:2]
        validate_image_size(frame_width, frame_height, calibration_width, calibration_height, args.calibration)

        preview, _ = process_frame(
            frame,
            model=model,
            board=board,
            detector=detector,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            grid_margin_squares=args.grid_margin,
            axis_length_squares=args.axis_length_squares,
            show_intersections=args.show_intersections,
            show_hud=args.hud,
            source_label=f"camera {args.camera}",
        )
        cv2.imshow("Board Reference Frame", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), preview)
            print(f"Saved overlay: {args.output}")

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.grid_margin < 0:
        raise RuntimeError("--grid-margin must be non-negative.")
    if args.axis_length_squares <= 0.0:
        raise RuntimeError("--axis-length-squares must be positive.")
    if args.image is not None:
        run_still_image(args)
    else:
        run_live_camera(args)


if __name__ == "__main__":
    main()
