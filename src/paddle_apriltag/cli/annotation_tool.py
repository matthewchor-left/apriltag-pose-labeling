"""Live AprilTag eraser: paste a captured background plate over detected markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from paddle_apriltag.apriltag import DEFAULT_APRILTAG_DICTIONARY
from paddle_apriltag.calibration import DEFAULT_CALIBRATION_PATH, load_intrinsics
from paddle_apriltag.detector import PaddleDetector
from paddle_apriltag.layout import DEFAULT_MARKER_LAYOUT_PATH
from paddle_apriltag.pose import Detection
from paddle_apriltag.viz import draw_marker_annotations


def erase_markers(
    frame: np.ndarray,
    plate: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    if plate.shape != frame.shape:
        raise ValueError("Background plate must match the frame shape.")

    output = frame.copy()
    for corners, _marker_id in detections:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = corners.reshape(4, 2).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        output[mask > 0] = plate[mask > 0]
    return output


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return capture


def draw_status_hud(frame: np.ndarray, *, plate_captured: bool) -> None:
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
        "C capture plate | q quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Erase AprilTag markers using a captured background plate.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--detection-sensitivity", choices=("default", "relaxed", "aggressive"), default="relaxed")
    parser.add_argument("--dictionary", default=DEFAULT_APRILTAG_DICTIONARY)
    parser.add_argument("--marker-size", type=float, default=None)
    parser.add_argument("--marker-layout", type=Path, default=DEFAULT_MARKER_LAYOUT_PATH)
    args = parser.parse_args()

    if not args.calibration.exists():
        raise RuntimeError(f"Calibration file not found: {args.calibration}")
    if not args.marker_layout.exists():
        raise RuntimeError(f"Marker layout file not found: {args.marker_layout}")

    camera_matrix, dist_coeffs, _, _, calibration_source = load_intrinsics(args.calibration)
    detector = PaddleDetector(
        camera_matrix,
        dist_coeffs,
        marker_layout=args.marker_layout,
        marker_size_m=args.marker_size,
        dictionary=args.dictionary,
        sensitivity=args.detection_sensitivity,
    )
    layout = detector.layout
    marker_size_m = detector.marker_size_m

    print(f"Using marker layout: {args.marker_layout} ({len(layout.marker_ids)} markers)")
    print(f"Marker size: {marker_size_m:.4f} m")
    print(f"Using camera calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    cap = open_camera(args.camera, args.width, args.height)
    print(f"Camera {args.camera}: {args.width}x{args.height}")
    print("Press C to capture the background plate, q to quit.")

    background_plate: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read a frame from camera {args.camera}.")

        detections = detector.find_markers(frame)
        preview = frame.copy()
        if background_plate is not None:
            preview = erase_markers(preview, background_plate, detections)
            for corners, marker_id in detections:
                draw_marker_annotations(
                    preview,
                    corners,
                    marker_id,
                    marker_size_m,
                    camera_matrix,
                    dist_coeffs,
                    layout,
                    draw=True,
                )
        draw_status_hud(preview, plate_captured=background_plate is not None)

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
