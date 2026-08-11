"""Live AprilTag paddle detection CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from paddle_apriltag.apriltag import DEFAULT_APRILTAG_DICTIONARY
from paddle_apriltag.calibration import DEFAULT_CALIBRATION_PATH, load_intrinsics
from paddle_apriltag.detector import PaddleDetector
from paddle_apriltag.layout import DEFAULT_MARKER_LAYOUT_PATH
from paddle_apriltag.pose import mean_reprojection_error
from paddle_apriltag.viz import (
    DEFAULT_AXIS_LIMITS,
    LiveHud,
    draw_live_hud,
    draw_marker_annotations,
    draw_marker_layout_footprints,
    draw_paddle_orientation,
    draw_paddle_pose,
    load_paddle_model,
    make_side_by_side,
    paddle_world_points_from_pose,
    render_pose_plots,
)


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return capture


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect AprilTag markers and estimate fused paddle pose.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--detection-sensitivity", choices=("default", "relaxed", "aggressive"), default="relaxed")
    parser.add_argument("--dictionary", default=DEFAULT_APRILTAG_DICTIONARY)
    parser.add_argument("--marker-size", type=float, default=None)
    parser.add_argument("--plot-width", type=int, default=540)
    parser.add_argument("--marker-id", type=int, default=None)
    parser.add_argument("--marker-layout", type=Path, default=DEFAULT_MARKER_LAYOUT_PATH)
    parser.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--overlay-layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Project marker_layout.json sticker footprints onto the camera preview",
    )
    parser.add_argument(
        "--overlay-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Project paddle_model.json skeleton onto the camera preview",
    )
    parser.add_argument("--axis-length", type=float, default=0.08, help="Paddle axis length to draw, in meters")
    args = parser.parse_args()

    if args.overlay_model and args.overlay_layout:
        raise RuntimeError("Use only one of --overlay-model or --overlay-layout.")
    if not args.preview and not args.plot_graph:
        raise RuntimeError("Enable at least one of --preview or --plot-graph.")
    if not args.calibration.exists():
        raise RuntimeError(
            f"Calibration file not found: {args.calibration}\n"
            "Run `uv run paddle calibrate-camera` from the repo root first "
            "(use --output calibration/camera_intrinsics.json)."
        )
    if not args.marker_layout.exists():
        raise RuntimeError(f"Marker layout file not found: {args.marker_layout}")

    marker_ids = {args.marker_id} if args.marker_id is not None else None
    camera_matrix, dist_coeffs, _, _, calibration_source = load_intrinsics(args.calibration)
    detector = PaddleDetector(
        camera_matrix,
        dist_coeffs,
        marker_layout=args.marker_layout,
        marker_size_m=args.marker_size,
        dictionary=args.dictionary,
        sensitivity=args.detection_sensitivity,
        marker_ids=marker_ids,
    )
    layout = detector.layout
    marker_size_m = detector.marker_size_m
    if args.marker_size is not None and abs(marker_size_m - layout.marker_size_m) > 1e-4:
        raise RuntimeError(f"--marker-size {marker_size_m} does not match layout marker_size_m {layout.marker_size_m}.")

    from paddle_apriltag.calibration import DEFAULT_PADDLE_MODEL_PATH

    paddle_model = load_paddle_model(DEFAULT_PADDLE_MODEL_PATH) if args.overlay_model or args.plot_graph else None

    print(f"Using marker layout: {args.marker_layout} ({len(layout.marker_ids)} markers)")
    print(f"Marker size: {marker_size_m:.4f} m")
    print(f"Using camera calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    cap = open_camera(args.camera, args.width, args.height)
    print(f"Camera {args.camera}: {args.width}x{args.height}")
    print("Press q to quit.")

    hud = LiveHud()
    plot_figsize = (args.plot_width / 50.0, args.plot_width / 100.0)

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read a frame from camera {args.camera}.")

        detections = detector.find_markers(frame)
        pose = detector.fuse(detections)
        preview_frame = frame.copy() if args.visualize else frame

        if args.visualize:
            for corners, marker_id in detections:
                draw_marker_annotations(
                    preview_frame,
                    corners,
                    marker_id,
                    marker_size_m,
                    camera_matrix,
                    dist_coeffs,
                    layout,
                    draw=True,
                )
            if pose is not None:
                if args.overlay_model:
                    draw_paddle_pose(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        marker_size_m,
                        paddle_model,
                    )
                elif args.overlay_layout:
                    draw_marker_layout_footprints(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        layout,
                    )
                else:
                    draw_paddle_orientation(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        args.axis_length,
                    )

        reproj_error = mean_reprojection_error(detections, marker_size_m, camera_matrix, dist_coeffs)
        fps, avg_reproj_error = hud.tick(reproj_error)
        if args.visualize:
            draw_live_hud(preview_frame, fps, avg_reproj_error)

        world_points = (
            paddle_world_points_from_pose(pose.rotation, pose.origin, paddle_model)
            if pose is not None and paddle_model is not None
            else {}
        )
        if args.preview and args.plot_graph:
            plot_bgr = render_pose_plots(world_points, paddle_model, DEFAULT_AXIS_LIMITS, figsize=plot_figsize)
            display_frame = make_side_by_side(preview_frame, plot_bgr, preview_frame.shape[0])
        elif args.preview:
            display_frame = preview_frame
        else:
            display_frame = render_pose_plots(world_points, paddle_model, DEFAULT_AXIS_LIMITS, figsize=plot_figsize)

        cv2.imshow("Paddle AprilTag Detector", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
