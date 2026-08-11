"""Live AprilTag eraser: paste a background plate over projected eraser planes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from paddle_apriltag.apriltag import DEFAULT_APRILTAG_DICTIONARY
from paddle_apriltag.calibration import DEFAULT_CALIBRATION_PATH, load_intrinsics
from paddle_apriltag.detector import PaddleDetector, PaddlePose
from paddle_apriltag.eraser import DEFAULT_ERASER_MODEL_PATH, EraserModel, load_eraser_model
from paddle_apriltag.layout import (
    DEFAULT_MARKER_LAYOUT_PATH,
    MarkerLayout,
    layout_point_to_camera,
    paddle_reference_origin,
)
from paddle_apriltag.viz.projection import project_camera_point


def clip_polygon_to_rect(polygon: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Clip a polygon to the image rectangle [0, width] x [0, height]."""
    if len(polygon) < 3:
        return None

    def _clip(points: np.ndarray, inside, intersect) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)
        output: list[np.ndarray] = []
        previous = points[-1]
        for current in points:
            current_inside = inside(current)
            previous_inside = inside(previous)
            if current_inside:
                if previous_inside:
                    output.append(current)
                else:
                    output.append(intersect(previous, current))
                    output.append(current)
            elif previous_inside:
                output.append(intersect(previous, current))
            previous = current
        if not output:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(output, dtype=np.float64)

    def _intersect_x(edge_x: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta_x = b[0] - a[0]
        t = 0.0 if delta_x == 0.0 else (edge_x - a[0]) / delta_x
        return np.array([edge_x, a[1] + t * (b[1] - a[1])], dtype=np.float64)

    def _intersect_y(edge_y: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta_y = b[1] - a[1]
        t = 0.0 if delta_y == 0.0 else (edge_y - a[1]) / delta_y
        return np.array([a[0] + t * (b[0] - a[0]), edge_y], dtype=np.float64)

    points = polygon.astype(np.float64)
    for edge_value, inside, intersect in (
        (0.0, lambda p: p[0] >= 0.0, lambda a, b: _intersect_x(0.0, a, b)),
        (float(width), lambda p: p[0] <= float(width), lambda a, b: _intersect_x(float(width), a, b)),
        (0.0, lambda p: p[1] >= 0.0, lambda a, b: _intersect_y(0.0, a, b)),
        (float(height), lambda p: p[1] <= float(height), lambda a, b: _intersect_y(float(height), a, b)),
    ):
        del edge_value
        points = _clip(points, inside, intersect)
        if len(points) < 3:
            return None
    return points


def eraser_offset_to_layout_point(offset: np.ndarray, layout: MarkerLayout) -> np.ndarray:
    return paddle_reference_origin(layout) + offset


def project_eraser_plane(
    plane_corners: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    paddle_rotation: np.ndarray,
    paddle_origin: np.ndarray,
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray | None:
    image_points: list[np.ndarray] = []
    for offset in plane_corners:
        layout_point = eraser_offset_to_layout_point(offset, layout)
        camera_point = layout_point_to_camera(layout_point, paddle_rotation, paddle_origin, layout)
        if camera_point[2] <= 0.0:
            continue
        projected = project_camera_point(camera_point, camera_matrix, dist_coeffs)
        if not (np.isfinite(projected[0]) and np.isfinite(projected[1])):
            continue
        image_points.append(projected)

    if len(image_points) < 3:
        return None

    polygon = np.asarray(image_points, dtype=np.float64)
    return clip_polygon_to_rect(polygon, image_width, image_height)


def project_eraser_planes(
    eraser_model: EraserModel,
    paddle_rotation: np.ndarray,
    paddle_origin: np.ndarray,
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for plane in eraser_model.planes:
        polygon = project_eraser_plane(
            plane.corners(),
            paddle_rotation,
            paddle_origin,
            layout,
            camera_matrix,
            dist_coeffs,
            image_width=image_width,
            image_height=image_height,
        )
        if polygon is not None:
            polygons.append(polygon)
    return polygons


def erase_with_mask(
    frame: np.ndarray,
    plate: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if plate.shape != frame.shape:
        raise ValueError("Background plate must match the frame shape.")
    output = frame.copy()
    output[mask > 0] = plate[mask > 0]
    return output


def build_eraser_mask(
    frame_shape: tuple[int, int],
    polygons: list[np.ndarray],
) -> np.ndarray:
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
    if not polygons:
        return frame.copy()
    mask = build_eraser_mask(frame.shape[:2], polygons)
    return erase_with_mask(frame, plate, mask)


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


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return capture


def draw_status_hud(frame: np.ndarray, *, plate_captured: bool, plane_count: int) -> None:
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
    pose: PaddlePose,
    eraser_model: EraserModel,
    layout: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = frame.shape[:2]
    polygons = project_eraser_planes(
        eraser_model,
        pose.rotation,
        pose.origin,
        layout,
        camera_matrix,
        dist_coeffs,
        image_width=width,
        image_height=height,
    )
    return erase_with_planes(frame, plate, polygons), polygons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Erase AprilTag marker regions using projected eraser planes.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--detection-sensitivity", choices=("default", "relaxed", "aggressive"), default="relaxed")
    parser.add_argument("--dictionary", default=DEFAULT_APRILTAG_DICTIONARY)
    parser.add_argument("--marker-size", type=float, default=None)
    parser.add_argument("--marker-layout", type=Path, default=DEFAULT_MARKER_LAYOUT_PATH)
    parser.add_argument("--eraser-model", type=Path, default=DEFAULT_ERASER_MODEL_PATH)
    args = parser.parse_args()

    if not args.calibration.exists():
        raise RuntimeError(f"Calibration file not found: {args.calibration}")
    if not args.marker_layout.exists():
        raise RuntimeError(f"Marker layout file not found: {args.marker_layout}")
    if not args.eraser_model.exists():
        raise RuntimeError(f"Eraser model file not found: {args.eraser_model}")

    camera_matrix, dist_coeffs, _, _, calibration_source = load_intrinsics(args.calibration)
    eraser_model = load_eraser_model(args.eraser_model)
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
    print(f"Using eraser model: {args.eraser_model} ({len(eraser_model.planes)} planes)")
    print(f"Eraser origin: {eraser_model.origin}")
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
        polygons: list[np.ndarray] = []
        if background_plate is not None:
            pose = detector.fuse(detections)
            if pose is not None:
                preview, polygons = erase_eraser_planes(
                    preview,
                    background_plate,
                    pose,
                    eraser_model,
                    layout,
                    camera_matrix,
                    dist_coeffs,
                )
            draw_eraser_planes(preview, polygons)

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
