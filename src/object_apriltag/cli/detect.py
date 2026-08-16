"""Live AprilTag object detection CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.board_model import load_board_model
from object_apriltag.board_pose import make_charuco_detector, select_board_pose, solve_board_pose
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.detector import ObjectDetector
from object_apriltag.frame_source import (
    format_frame_source,
    open_frame_source,
    parse_frame_source,
    read_frame,
)
from object_apriltag.object_model_edit import (
    ObjectModelEditSession,
    load_object_model_document,
    object_model_for_render,
)
from object_apriltag.eraser import load_eraser_model, project_eraser_planes
from object_apriltag.pose import layout_reprojection_errors, reference_marker_camera_position
from object_apriltag.viz import (
    DEFAULT_AXIS_LIMITS,
    LiveHud,
    draw_eraser_planes,
    draw_live_hud,
    draw_marker_annotations,
    draw_marker_model_footprints,
    draw_object_orientation,
    draw_object_pose,
    load_object_model,
    make_side_by_side,
    object_world_points_from_pose,
    render_pose_plots,
)
from object_apriltag.viz.board_frame import render_board_frame_grid_axes
from object_apriltag.viz.overlay import (
    draw_board_coordinate_preview,
    draw_eraser_model_board_coordinate_labels,
    draw_marker_model_board_coordinate_labels,
    draw_object_model_board_coordinate_labels,
    draw_object_model_edit_hud,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect AprilTag markers and estimate fused object pose.",
        epilog=(
            "Terms:\n"
            "  detection outline  AprilTag corner box and ID drawn on the camera frame.\n"
            "  pose projection    3D object geometry drawn on the camera frame from fused pose.\n"
            "  skeleton chart     Separate matplotlib 3D plot (--plot-graph).\n"
            "\n"
            "Display:\n"
            "  --preview          Open the live OpenCV window.\n"
            "  --visualize        Draw detection outlines, HUD, and pose projection on the camera frame.\n"
            "  --plot-graph       Add the skeleton chart (--object-model required).\n"
            "\n"
            "Pose projection (--visualize on; enable one style, or neither for axis arrows):\n"
            "  --overlay-marker-model   Marker sticker footprint quads (--marker-model).\n"
            "  --overlay-object-model   Object skeleton keypoints and edges (--object-model).\n"
            "  --overlay-eraser-model   Eraser plane quads (--eraser-model).\n"
            "  (neither)                RGB object axis arrows (--axis-length).\n"
            "\n"
            "CAD (--cad-model; registration loaded from sibling cad_registration.json, or fitted\n"
            "     from matching GLB landmarks and --object-model keypoint_sources when absent):\n"
            "  --overlay-cad-model      Semi-transparent GLB on camera frame (--visualize on).\n"
            "  --side2side-cad-model    Side-by-side preview: camera | opaque CAD pane (--preview).\n"
            "  With --plot-graph: camera | CAD | skeleton chart.\n"
            "\n"
            "Object Model keypoint editing (requires --preview, --board-frame, "
            "--overlay-object-model):\n"
            "  e    Terminal prompt: keypoint-id x_mm y_mm z_mm (Board Coordinates).\n"
            "  s    Save keypoints to --object-model.\n"
            "  q    Quit when saved; refuses while modified.\n"
            "  x    Discard unsaved edits and quit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=parse_frame_source,
        required=True,
        help="Frame source: camera device index (e.g. 0) or path to a video file.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Camera intrinsics JSON. Image width and height are read from this file.",
    )
    parser.add_argument(
        "--detection-sensitivity",
        choices=("default", "relaxed", "aggressive"),
        required=True,
        help="AprilTag detector preset: default (OpenCV defaults), relaxed, or aggressive.",
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        help="AprilTag dictionary name (e.g. 36h11, 25h9).",
    )
    parser.add_argument(
        "--plot-width",
        type=int,
        default=540,
        help="Width in pixels of the matplotlib skeleton plot when --plot-graph is enabled.",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        help="Track only this marker ID. Omit to fuse all markers listed in --marker-model.",
    )
    parser.add_argument(
        "--marker-model",
        type=Path,
        required=True,
        help="Marker model JSON (sticker footprint positions and marker_size_m).",
    )
    parser.add_argument(
        "--object-model",
        type=Path,
        help="Object skeleton JSON. Required for --overlay-object-model or --plot-graph.",
    )
    parser.add_argument(
        "--eraser-model",
        type=Path,
        help="Eraser planes JSON. Required for --overlay-eraser-model.",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the live OpenCV display window. --no-preview: skeleton chart only (--plot-graph).",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Draw on the camera frame: detection outlines (box and ID per AprilTag), "
            "FPS/reprojection HUD, and pose projection when tracking succeeds. "
            "--no-visualize: raw camera frame (HUD still drawn if --preview)."
        ),
    )
    parser.add_argument(
        "--plot-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show a skeleton chart: matplotlib 3D plot of --object-model keypoints. "
            "With --preview: side-by-side with the camera frame. Without: chart-only window."
        ),
    )
    parser.add_argument(
        "--overlay-marker-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pose projection: marker sticker footprint quads from --marker-model. "
            "Mutually exclusive with other overlay styles."
        ),
    )
    parser.add_argument(
        "--overlay-object-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pose projection: object skeleton keypoints and bone lines from --object-model. "
            "Mutually exclusive with other overlay styles."
        ),
    )
    parser.add_argument(
        "--overlay-eraser-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pose projection: eraser plane quads from --eraser-model. "
            "Mutually exclusive with other overlay styles."
        ),
    )
    parser.add_argument(
        "--overlay-cad-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "CAD overlay: render --cad-model GLB on the camera frame when fused pose exists. "
            "Additive with pose projection styles."
        ),
    )
    parser.add_argument(
        "--side2side-cad-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Side-by-side preview: camera frame on the left, opaque CAD-only rendering on the "
            "right (black when pose is unavailable). Requires --preview and --cad-model."
        ),
    )
    parser.add_argument(
        "--cad-model",
        type=Path,
        help=(
            "GLB CAD model path. Required when --overlay-cad-model or --side2side-cad-model "
            "is enabled. A missing sibling cad_registration.json is generated in memory from "
            "--object-model and --marker-model."
        ),
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.08,
        help=(
            "Pose projection: length (meters) of RGB object axis arrows when no overlay style is enabled."
        ),
    )
    parser.add_argument(
        "--board-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlay the Board Reference Frame grid and axes, and annotate model points in board coordinates.",
    )
    parser.add_argument(
        "--board-model",
        type=Path,
        help="ChArUco board model JSON. Required when --board-frame is enabled.",
    )
    parser.add_argument(
        "--camera-motion",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "Board pose retention when tracking drops out: static keeps the latest valid estimate; "
            "dynamic clears labels and grid until the board is visible again."
        ),
    )
    args = parser.parse_args()

    overlay_styles = (
        args.overlay_marker_model,
        args.overlay_object_model,
        args.overlay_eraser_model,
    )
    if sum(overlay_styles) > 1:
        raise RuntimeError(
            "Only one pose projection overlay: --overlay-marker-model, --overlay-object-model, "
            "or --overlay-eraser-model."
        )
    if not args.preview and not args.plot_graph:
        raise RuntimeError("Enable at least one of --preview or --plot-graph.")
    if (args.overlay_object_model or args.plot_graph) and args.object_model is None:
        raise RuntimeError("--object-model is required when --overlay-object-model or --plot-graph is enabled.")
    if args.overlay_eraser_model and args.eraser_model is None:
        raise RuntimeError("--eraser-model is required when --overlay-eraser-model is enabled.")
    if args.board_frame and args.board_model is None:
        raise RuntimeError("--board-model is required when --board-frame is enabled.")
    needs_cad = args.overlay_cad_model or args.side2side_cad_model
    if args.side2side_cad_model and not args.preview:
        raise RuntimeError("--side2side-cad-model requires --preview.")
    if needs_cad and args.cad_model is None:
        raise RuntimeError(
            "--cad-model is required when --overlay-cad-model or --side2side-cad-model is enabled."
        )
    cad_registration_path = (
        args.cad_model.with_name("cad_registration.json") if args.cad_model is not None else None
    )
    if needs_cad and not args.cad_model.exists():
        raise RuntimeError(f"CAD model file not found: {args.cad_model}")
    if needs_cad and not cad_registration_path.exists() and args.object_model is None:
        raise RuntimeError(
            f"CAD registration file not found: {cad_registration_path}. "
            "Pass --object-model to generate it in memory from matching named landmarks."
        )
    if not args.calibration.exists():
        raise RuntimeError(
            f"Calibration file not found: {args.calibration}\n"
            "Run `uv run object-charuco` first "
            "(use --output config/Camera/<profile>/intrinsics.json)."
        )
    if not args.marker_model.exists():
        raise RuntimeError(f"Marker model file not found: {args.marker_model}")
    if args.eraser_model is not None and not args.eraser_model.exists():
        raise RuntimeError(f"Eraser model file not found: {args.eraser_model}")
    if args.object_model is not None and not args.object_model.exists():
        raise RuntimeError(f"Object model file not found: {args.object_model}")
    if args.board_frame and not args.board_model.exists():
        raise RuntimeError(f"Board model file not found: {args.board_model}")

    marker_ids = {args.marker_id} if args.marker_id is not None else None
    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    detector = ObjectDetector(
        camera_matrix,
        dist_coeffs,
        marker_model=args.marker_model,
        dictionary=args.dictionary,
        sensitivity=args.detection_sensitivity,
        marker_ids=marker_ids,
    )
    marker_model = detector.marker_model
    marker_size_m = detector.marker_size_m

    needs_object_model = args.overlay_object_model or args.plot_graph
    edit_enabled = bool(
        args.preview and args.board_frame and args.overlay_object_model and args.object_model is not None
    )
    edit_session = ObjectModelEditSession.from_path(args.object_model) if edit_enabled else None
    object_model = (
        None
        if edit_session is not None
        else load_object_model(args.object_model)
        if needs_object_model
        else None
    )
    eraser_model = (
        load_eraser_model(args.eraser_model)
        if args.overlay_eraser_model
        else None
    )
    board_model = load_board_model(args.board_model) if args.board_frame else None
    board_charuco = make_charuco_detector(board_model) if board_model is not None else None
    board_retained_pose = None
    cad_model = None
    cad_registration = None
    draw_cad_model_overlay = None
    render_cad_model_view = None
    if needs_cad:
        from object_apriltag.cad import (
            load_cad_landmarks,
            load_cad_model,
            load_cad_registration,
        )

        cad_model = load_cad_model(args.cad_model)
        if cad_registration_path.exists():
            cad_registration = load_cad_registration(cad_registration_path)
        else:
            from object_apriltag.evaluation.cad_geometry import fit_cad_registration

            _, object_model_document = load_object_model_document(args.object_model)
            try:
                cad_registration = fit_cad_registration(
                    load_cad_landmarks(args.cad_model),
                    object_model_document,
                    marker_model,
                )
            except ValueError as error:
                raise RuntimeError(
                    f"Cannot generate CAD registration from {args.cad_model} and "
                    f"{args.object_model}: {error}"
                ) from error
        if args.overlay_cad_model:
            from object_apriltag.viz.cad_overlay import draw_cad_model_overlay as _draw_cad_overlay

            draw_cad_model_overlay = _draw_cad_overlay
        if args.side2side_cad_model:
            from object_apriltag.viz.cad_overlay import render_cad_model_view as _render_cad_view

            render_cad_model_view = _render_cad_view

    print(f"Using marker model: {args.marker_model} ({len(marker_model.marker_ids)} markers)")
    print(f"Marker size: {marker_size_m:.4f} m")
    if eraser_model is not None:
        print(f"Using eraser model: {args.eraser_model} ({len(eraser_model.planes)} planes)")
    if board_model is not None:
        print(f"Using board model: {args.board_model}")
        print(f"Board camera motion: {args.camera_motion}")
    if cad_model is not None:
        print(f"Using CAD model: {args.cad_model}")
        if cad_registration_path.exists():
            print(f"Using CAD registration: {cad_registration_path}")
        else:
            print(
                "Generated CAD registration in memory from "
                f"{args.object_model} and {args.marker_model}"
            )
    print(f"Using camera calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    cap = open_frame_source(args.source, width=width, height=height)
    source_label = format_frame_source(args.source)
    if isinstance(args.source, int):
        print(f"{source_label}: target {width}x{height}")
    else:
        print(source_label)
    if edit_enabled:
        print(f"Object model editing enabled: {args.object_model}")
        print("Keys: e edit, s save, q quit (when saved), x discard+quit.")
    else:
        print("Press q to quit.")

    hud = LiveHud()
    plot_figsize = (args.plot_width / 50.0, args.plot_width / 100.0)

    while True:
        ok, frame = read_frame(cap, args.source)
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a frame from {source_label}.")

        detections = detector.find_markers(frame)
        pose = detector.fuse(detections)
        current_object_model = object_model_for_render(object_model, edit_session)
        preview_frame = frame.copy() if args.visualize or edit_session is not None else frame

        board_pose = None
        if args.board_frame and board_model is not None and board_charuco is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            board, board_detector = board_charuco
            board_result = solve_board_pose(
                gray,
                board,
                board_detector,
                board_model,
                camera_matrix,
                dist_coeffs,
            )
            board_pose, board_retained_pose = select_board_pose(
                board_result.pose,
                board_retained_pose,
                args.camera_motion,
            )

        if args.visualize:
            if board_pose is not None and board_model is not None:
                render_board_frame_grid_axes(
                    preview_frame,
                    model=board_model,
                    pose=board_pose,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
            for corners, marker_id in detections:
                draw_marker_annotations(
                    preview_frame,
                    corners,
                    marker_id,
                    marker_model.marker_size_for(marker_id),
                    camera_matrix,
                    dist_coeffs,
                    marker_model,
                    draw=True,
                )
            if pose is not None:
                if args.overlay_object_model:
                    draw_object_pose(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        marker_size_m,
                        current_object_model,
                        marker_model,
                    )
                elif args.overlay_marker_model:
                    draw_marker_model_footprints(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        marker_model,
                    )
                elif args.overlay_eraser_model and eraser_model is not None:
                    polygons = project_eraser_planes(
                        eraser_model,
                        pose.rotation,
                        pose.origin,
                        marker_model,
                        camera_matrix,
                        dist_coeffs,
                        image_width=width,
                        image_height=height,
                    )
                    draw_eraser_planes(preview_frame, polygons)
                else:
                    draw_object_orientation(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        args.axis_length,
                    )
                if draw_cad_model_overlay is not None and cad_model is not None and cad_registration is not None:
                    draw_cad_model_overlay(
                        preview_frame,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        marker_model,
                        cad_model,
                        cad_registration,
                    )
                if board_pose is not None:
                    if args.overlay_object_model and current_object_model is not None:
                        draw_object_model_board_coordinate_labels(
                            preview_frame,
                            pose,
                            board_pose,
                            marker_model,
                            current_object_model,
                            camera_matrix,
                            dist_coeffs,
                        )
                    elif args.overlay_marker_model:
                        draw_marker_model_board_coordinate_labels(
                            preview_frame,
                            pose,
                            board_pose,
                            marker_model,
                            camera_matrix,
                            dist_coeffs,
                        )
                    elif args.overlay_eraser_model and eraser_model is not None:
                        draw_eraser_model_board_coordinate_labels(
                            preview_frame,
                            pose,
                            board_pose,
                            eraser_model,
                            marker_model,
                            camera_matrix,
                            dist_coeffs,
                        )

        if (
            edit_session is not None
            and edit_session.preview_board_point_m is not None
            and board_pose is not None
        ):
            draw_board_coordinate_preview(
                preview_frame,
                edit_session.preview_board_point_m,
                board_pose,
                camera_matrix,
                dist_coeffs,
                label=f"{edit_session.preview_keypoint_id} target",
            )

        layout_reproj = (
            layout_reprojection_errors(
                detections,
                pose.rotation,
                pose.origin,
                marker_model,
                camera_matrix,
                dist_coeffs,
            )
            if pose is not None
            else None
        )
        if layout_reproj is not None:
            fps, avg_reproj_error, max_reproj_error = hud.tick(*layout_reproj)
        else:
            fps, avg_reproj_error, max_reproj_error = hud.tick()
        reference_camera_m = (
            reference_marker_camera_position(
                pose.rotation, pose.origin, marker_model
            )
            if pose is not None
            else None
        )
        if args.visualize:
            draw_live_hud(
                preview_frame,
                fps,
                avg_reproj_error,
                max_reproj_error,
                reference_marker_id=marker_model.reference_marker_id,
                reference_marker_camera_m=reference_camera_m,
            )
        if edit_session is not None:
            draw_object_model_edit_hud(
                preview_frame,
                dirty=edit_session.dirty,
                status_message=edit_session.status_message,
            )

        world_points = (
            object_world_points_from_pose(pose.rotation, pose.origin, current_object_model, marker_model)
            if pose is not None and current_object_model is not None
            else {}
        )
        target_height = preview_frame.shape[0]
        display_frame = preview_frame
        if args.side2side_cad_model and render_cad_model_view is not None:
            if pose is not None and cad_model is not None and cad_registration is not None:
                cad_view_bgr = render_cad_model_view(
                    preview_frame.shape[:2],
                    pose,
                    camera_matrix,
                    dist_coeffs,
                    marker_model,
                    cad_model,
                    cad_registration,
                )
            else:
                height, width = preview_frame.shape[:2]
                cad_view_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            display_frame = make_side_by_side(display_frame, cad_view_bgr, target_height)
        if args.plot_graph:
            plot_bgr = render_pose_plots(
                world_points, current_object_model, DEFAULT_AXIS_LIMITS, figsize=plot_figsize
            )
            if args.preview:
                display_frame = make_side_by_side(display_frame, plot_bgr, target_height)
            else:
                display_frame = plot_bgr

        cv2.imshow("Object AprilTag Detector", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            if edit_session is not None and edit_session.dirty:
                print("Unsaved object model edits; press s to save or x to discard and quit.")
                edit_session.status_message = "unsaved: s save or x discard+quit"
            else:
                break
        elif key == ord("x") and edit_session is not None:
            edit_session.discard()
            break
        elif key == ord("s") and edit_session is not None:
            if edit_session.save():
                print(f"Saved object model: {args.object_model}")
            else:
                print(edit_session.status_message)
        elif key == ord("e") and edit_session is not None:
            try:
                line = input("keypoint-id x_mm y_mm z_mm: ")
            except EOFError:
                line = ""
            if edit_session.apply_keypoint_edit(line, pose, board_pose, marker_model):
                print(edit_session.status_message)
            else:
                print(edit_session.status_message)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
