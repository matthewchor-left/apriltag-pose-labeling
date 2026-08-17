"""Live AprilTag eraser: paste a background plate over projected eraser planes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.detector import ObjectDetector, ObjectPose
from object_apriltag.eraser import EraserModel, load_eraser_model, project_eraser_planes
from object_apriltag.frame_source import (
    format_frame_source,
    open_frame_source,
    parse_frame_source,
    read_frame,
)
from object_apriltag.layout import MarkerModel
from object_apriltag.viz.overlay import draw_object_pose
from object_apriltag.viz.skeleton import load_object_model


def erase_with_mask(
    frame: np.ndarray,
    plate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Replace masked pixels in ``frame`` with values from ``plate``.

    Args:
        frame: Input BGR image.
        plate: Background plate with the same shape as ``frame``.
        mask: Single-channel mask; pixels with value > 0 are replaced.

    Returns:
        Copy of ``frame`` with masked pixels taken from ``plate``.

    Raises:
        ValueError: ``plate`` shape does not match ``frame``.
    """
    if plate.shape != frame.shape:
        raise ValueError("Background plate must match the frame shape.")
    output = frame.copy()
    output[mask > 0] = plate[mask > 0]
    return output


def build_eraser_mask(
    frame_shape: tuple[int, int],
    polygons: list[np.ndarray],
) -> np.ndarray:
    """Rasterize projected eraser polygons into a binary mask.

    Args:
        frame_shape: ``(height, width)`` of the target image.
        polygons: Projected eraser quads as ``(N, 2)`` float image points.

    Returns:
        ``uint8`` mask with 255 inside filled polygons and 0 elsewhere.
    """
    mask = np.zeros(frame_shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 255)
    return mask


def erase_with_planes(
    frame: np.ndarray,
    plate: np.ndarray,
    polygons: list[np.ndarray],
) -> np.ndarray:
    """Erase projected eraser regions by compositing ``plate`` through a polygon mask.

    Args:
        frame: Input BGR image.
        plate: Background plate captured before the object entered the scene.
        polygons: Projected eraser quads as ``(N, 2)`` float image points.

    Returns:
        Copy of ``frame`` with eraser regions replaced by ``plate`` pixels, or
        an unchanged copy when ``polygons`` is empty.
    """
    if not polygons:
        return frame.copy()
    mask = build_eraser_mask(frame.shape[:2], polygons)
    return erase_with_mask(frame, plate, mask)


def draw_status_hud(frame: np.ndarray, *, plate_captured: bool, plane_count: int) -> None:
    """Draw capture status and keyboard hints on ``frame`` in place.

    Args:
        frame: Preview image to annotate.
        plate_captured: Whether a background plate has been captured.
        plane_count: Number of eraser planes in the loaded eraser model.
    """
    status = "plate: captured (erasing)" if plate_captured else "plate: none (press C to capture)"
    cv2.putText(
        frame,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"eraser planes: {plane_count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "C capture plate | q quit",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def erase_eraser_planes(
    frame: np.ndarray,
    plate: np.ndarray,
    pose: ObjectPose,
    eraser_model: EraserModel,
    marker_model: MarkerModel,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Project eraser planes from ``pose`` and composite ``plate`` over those regions.

    Args:
        frame: Input BGR image.
        plate: Background plate for masked replacement.
        pose: Fused object pose in marker-model coordinates.
        eraser_model: Eraser plane definitions relative to the object frame.
        marker_model: Marker layout used for pose projection.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        Tuple of the erased image copy and the projected eraser polygons.
    """
    height, width = frame.shape[:2]
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
    return erase_with_planes(frame, plate, polygons), polygons


def main() -> None:
    """Run the live eraser CLI: capture a plate, then mask projected eraser planes.

    Raises:
        RuntimeError: Missing required model files, invalid flag combinations, or frame read failure.
    """
    parser = argparse.ArgumentParser(
        description="Erase AprilTag marker regions using projected eraser planes.",
        epilog=(
            "Workflow: press C to capture a background plate (empty scene), then move the object "
            "into view. Detected tag regions are replaced with the plate pixels each frame."
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
        "--marker-model",
        type=Path,
        required=True,
        help="Marker model JSON (sticker footprints) used for pose fusion.",
    )
    parser.add_argument(
        "--eraser-model",
        type=Path,
        required=True,
        help="Eraser planes JSON: quads to mask, offsets from reference marker center.",
    )
    parser.add_argument(
        "--object-model",
        type=Path,
        help="Object skeleton JSON. Required for --overlay-object-model.",
    )
    parser.add_argument(
        "--overlay-object-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw object skeleton keypoints and bone lines from --object-model on the preview frame.",
    )
    args = parser.parse_args()

    if args.overlay_object_model and args.object_model is None:
        raise RuntimeError("--object-model is required when --overlay-object-model is enabled.")

    if not args.calibration.exists():
        raise RuntimeError(f"Calibration file not found: {args.calibration}")
    if not args.marker_model.exists():
        raise RuntimeError(f"Marker model file not found: {args.marker_model}")
    if not args.eraser_model.exists():
        raise RuntimeError(f"Eraser model file not found: {args.eraser_model}")
    if args.object_model is not None and not args.object_model.exists():
        raise RuntimeError(f"Object model file not found: {args.object_model}")

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    eraser_model = load_eraser_model(args.eraser_model)
    detector = ObjectDetector(
        camera_matrix,
        dist_coeffs,
        marker_model=args.marker_model,
        dictionary=args.dictionary,
        sensitivity=args.detection_sensitivity,
    )
    marker_model = detector.marker_model
    marker_size_m = detector.marker_size_m
    object_model = load_object_model(args.object_model) if args.overlay_object_model else None

    print(f"Using marker model: {args.marker_model} ({len(marker_model.marker_ids)} markers)")
    print(f"Using eraser model: {args.eraser_model} ({len(eraser_model.planes)} planes)")
    if object_model is not None:
        print(f"Using object model: {args.object_model} ({len(object_model.keypoint_names)} keypoints)")
    print(f"Eraser origin: {eraser_model.origin}")
    print(f"Marker size: {marker_size_m:.4f} m")
    print(f"Using camera calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    cap = open_frame_source(args.source, width=width, height=height)
    source_label = format_frame_source(args.source)
    if isinstance(args.source, int):
        print(f"{source_label}: target {width}x{height}")
    else:
        print(source_label)
    print("Press C to capture the background plate, q to quit.")

    background_plate: np.ndarray | None = None

    while True:
        ok, frame = read_frame(cap, args.source)
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read a frame from {source_label}.")

        detections = detector.find_markers(frame)
        preview = frame.copy()
        pose = detector.fuse(detections)

        if background_plate is not None and pose is not None:
            preview, _ = erase_eraser_planes(
                preview,
                background_plate,
                pose,
                eraser_model,
                marker_model,
                camera_matrix,
                dist_coeffs,
            )
        if args.overlay_object_model and pose is not None and object_model is not None:
            draw_object_pose(
                preview,
                pose,
                camera_matrix,
                dist_coeffs,
                marker_size_m,
                object_model,
                marker_model,
            )

        draw_status_hud(
            preview,
            plate_captured=background_plate is not None,
            plane_count=len(eraser_model.planes),
        )

        cv2.imshow("AprilTag Eraser", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in (ord("c"), ord("C")):
            background_plate = frame.copy()
            print("Captured background plate.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
