"""Live marker layout calibration CLI."""

from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.apriltag import build_apriltag_detector
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.layout import marker_color_bgr, save_marker_model
from object_apriltag.marker_layout_calibration import (
    CalibrationQualityReport,
    CalibrationResult,
    CalibrationSettings,
    FrameObservation,
    calibrate_marker_layout,
)

DEFAULT_SAMPLE_RATE_HZ = 2.0
DEFAULT_MIN_PAIR_INLIERS = 20
DEFAULT_REPROJECTION_RMS_GATE_PX = 2.0
DEFAULT_PAIR_TRANSLATION_RMS_GATE_RATIO = 0.10
DEFAULT_PAIR_ROTATION_RMS_GATE_DEG = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate marker sticker layout from live co-visible AprilTag detections.",
        epilog=(
            "Controls:\n"
            "  S  solve from captured samples and write --output when quality gates pass\n"
            "  Q  quit without writing\n"
            "\n"
            "Sampling:\n"
            "  Frames are recorded at --sample-rate-hz when at least two expected marker IDs "
            "are visible. Move the object so every expected ID co-appears with others often "
            "enough to connect through the reference marker.\n"
            "\n"
            "Scale:\n"
            "  Metric layout depends on --marker-size and calibrated camera intrinsics. "
            "Wrong marker size or scaled intrinsics will bias the solved geometry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--camera", type=int, required=True, help="Camera device index.")
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Camera intrinsics JSON. Frame size must match this file exactly.",
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        help="AprilTag dictionary name (e.g. 36h11, 25h9).",
    )
    parser.add_argument(
        "--detection-sensitivity",
        choices=("default", "relaxed", "aggressive"),
        required=True,
        help="AprilTag detector preset.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        required=True,
        help="Physical AprilTag edge length in meters (uniform for all expected markers).",
    )
    parser.add_argument(
        "--marker-ids",
        type=int,
        nargs="+",
        required=True,
        metavar="ID",
        help="Expected unique AprilTag marker IDs on the object.",
    )
    parser.add_argument(
        "--reference-marker-id",
        type=int,
        required=True,
        help="Reference marker ID; must appear in --marker-ids.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Marker model JSON output path (written only after a successful solve).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing --output file.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=f"Co-visibility sample rate in Hz (default: {DEFAULT_SAMPLE_RATE_HZ:g}).",
    )
    parser.add_argument(
        "--min-pair-inliers",
        type=int,
        default=DEFAULT_MIN_PAIR_INLIERS,
        help=f"Minimum co-visible frames required per marker pair (default: {DEFAULT_MIN_PAIR_INLIERS}).",
    )
    parser.add_argument(
        "--reprojection-rms-gate-px",
        type=float,
        default=DEFAULT_REPROJECTION_RMS_GATE_PX,
        help=f"Accepted global reprojection RMS gate in pixels (default: {DEFAULT_REPROJECTION_RMS_GATE_PX:g}).",
    )
    parser.add_argument(
        "--pair-translation-rms-gate-ratio",
        type=float,
        default=DEFAULT_PAIR_TRANSLATION_RMS_GATE_RATIO,
        help=(
            "Accepted pair translation RMS gate as a fraction of --marker-size "
            f"(default: {DEFAULT_PAIR_TRANSLATION_RMS_GATE_RATIO:g})."
        ),
    )
    parser.add_argument(
        "--pair-rotation-rms-gate-deg",
        type=float,
        default=DEFAULT_PAIR_ROTATION_RMS_GATE_DEG,
        help=f"Accepted pair rotation RMS gate in degrees (default: {DEFAULT_PAIR_ROTATION_RMS_GATE_DEG:g}).",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], CalibrationSettings]:
    if not args.calibration.exists():
        raise RuntimeError(
            f"Calibration file not found: {args.calibration}\n"
            "Run `uv run object-charuco` first."
        )
    if args.output.exists() and not args.force:
        raise RuntimeError(
            f"Output already exists: {args.output}. Pass --force to overwrite."
        )

    expected_ids = sorted({int(marker_id) for marker_id in args.marker_ids})
    if len(expected_ids) != len(args.marker_ids):
        raise RuntimeError("--marker-ids must contain unique values.")
    if args.reference_marker_id not in expected_ids:
        raise RuntimeError("--reference-marker-id must appear in --marker-ids.")
    if args.marker_size <= 0.0:
        raise RuntimeError("--marker-size must be positive.")
    if args.sample_rate_hz <= 0.0:
        raise RuntimeError("--sample-rate-hz must be positive.")
    if args.min_pair_inliers <= 0:
        raise RuntimeError("--min-pair-inliers must be positive.")
    for name, value in (
        ("--reprojection-rms-gate-px", args.reprojection_rms_gate_px),
        ("--pair-translation-rms-gate-ratio", args.pair_translation_rms_gate_ratio),
        ("--pair-rotation-rms-gate-deg", args.pair_rotation_rms_gate_deg),
    ):
        if value <= 0.0:
            raise RuntimeError(f"{name} must be positive.")

    settings = CalibrationSettings(
        min_inliers_per_edge=args.min_pair_inliers,
        reprojection_rms_gate_px=args.reprojection_rms_gate_px,
        pair_translation_rms_gate_ratio=args.pair_translation_rms_gate_ratio,
        pair_rotation_rms_gate_deg=args.pair_rotation_rms_gate_deg,
    )
    return expected_ids, settings


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return capture


def require_frame_size(
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


def detect_expected_markers(
    detector: cv2.aruco.ArucoDetector,
    frame: np.ndarray,
    expected_ids: set[int],
) -> dict[int, np.ndarray]:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}

    visible: dict[int, np.ndarray] = {}
    for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
        marker_id = int(marker_id)
        if marker_id in expected_ids:
            visible[marker_id] = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
    return visible


def pair_covisibility_counts(
    observations: list[FrameObservation],
    expected_ids: list[int],
) -> dict[tuple[int, int], int]:
    counts = {pair: 0 for pair in combinations(expected_ids, 2)}
    for observation in observations:
        present = set(observation.markers)
        for pair in combinations(sorted(present), 2):
            if pair in counts:
                counts[pair] += 1
    return counts


def connected_marker_ids(
    pair_counts: dict[tuple[int, int], int],
    reference_marker_id: int,
    min_inliers: int,
) -> set[int]:
    graph: dict[int, set[int]] = {}
    for (marker_a, marker_b), count in pair_counts.items():
        if count < min_inliers:
            continue
        graph.setdefault(marker_a, set()).add(marker_b)
        graph.setdefault(marker_b, set()).add(marker_a)

    connected = {reference_marker_id}
    queue = [reference_marker_id]
    while queue:
        current = queue.pop()
        for neighbor in graph.get(current, set()):
            if neighbor not in connected:
                connected.add(neighbor)
                queue.append(neighbor)
    return connected


def draw_detection_outlines(frame: np.ndarray, visible: dict[int, np.ndarray]) -> None:
    for marker_id, corners in visible.items():
        points = corners.reshape(4, 2).astype(np.int32)
        color = marker_color_bgr(marker_id)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
        anchor = tuple(int(round(value)) for value in points[0])
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


def format_solve_frame_counts(quality: CalibrationQualityReport) -> str:
    return (
        f"frames input/accepted/rejected: "
        f"{quality.input_frame_count}/{quality.accepted_frame_count}/{quality.rejected_frame_count}"
    )


def draw_calibration_hud(
    frame: np.ndarray,
    *,
    expected_ids: list[int],
    visible_ids: list[int],
    sample_count: int,
    pair_counts: dict[tuple[int, int], int],
    connected_ids: set[int],
    reference_marker_id: int,
    min_pair_inliers: int,
    status_line: str | None,
    last_solve_quality: CalibrationQualityReport | None,
) -> None:
    qualified_pairs = [
        f"({marker_a},{marker_b})={count}"
        for (marker_a, marker_b), count in sorted(pair_counts.items())
        if count > 0
    ]
    pair_summary = ", ".join(qualified_pairs[:6])
    if len(qualified_pairs) > 6:
        pair_summary += ", ..."

    ready_pairs = sum(1 for count in pair_counts.values() if count >= min_pair_inliers)
    lines = [
        f"expected: {expected_ids}",
        f"visible: {visible_ids}",
        f"samples: {sample_count}",
        f"pairs ready: {ready_pairs}/{len(pair_counts)} (>= {min_pair_inliers})",
        f"connected from ref {reference_marker_id}: {len(connected_ids)}/{len(expected_ids)}",
        "S=solve  Q=quit",
    ]
    if pair_summary:
        lines.insert(4, f"pair counts: {pair_summary}")
    if last_solve_quality is not None:
        lines.insert(4, format_solve_frame_counts(last_solve_quality))
    if status_line:
        lines.append(status_line)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    panel_right = min(frame.shape[1] - 1, text_width + 20)
    panel_bottom = min(frame.shape[0] - 1, len(lines) * 22 + 10)
    cv2.rectangle(frame, (0, 0), (panel_right, panel_bottom), (0, 0, 0), -1)

    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += 22


def print_refusal(result: CalibrationResult) -> None:
    print(f"Calibration refused: {result.failure_reason}")
    quality = result.quality
    if quality is None:
        return
    print(f"  {format_solve_frame_counts(quality)}")
    print(f"  inlier corners: {quality.inlier_corner_count}")
    print(f"  reprojection RMS: {quality.reprojection_rms_px:.3f} px")
    print(
        "  connected markers:",
        sorted(quality.connected_marker_ids),
        "missing:",
        sorted(quality.missing_expected_ids),
    )
    for edge in quality.edges:
        print(
            f"  pair ({edge.marker_a}, {edge.marker_b}): "
            f"inliers={edge.inlier_count} "
            f"trans_rms={edge.translation_rms_m:.4f} m "
            f"rot_rms={edge.rotation_rms_deg:.2f} deg"
        )


def print_success(result: CalibrationResult, output: Path) -> None:
    print(f"Saved marker model: {output}")
    quality = result.quality
    if quality is None:
        return
    print(f"  {format_solve_frame_counts(quality)}")
    print(f"  reprojection RMS: {quality.reprojection_rms_px:.3f} px")


def run_capture(args: argparse.Namespace) -> bool:
    """Capture live samples; return True when a model was saved."""
    expected_ids, settings = validate_args(args)
    expected_id_set = set(expected_ids)

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    detector = build_apriltag_detector(args.dictionary, args.detection_sensitivity)

    print(f"Expected marker IDs: {expected_ids}")
    print(f"Reference marker ID: {args.reference_marker_id}")
    print(f"Marker size: {args.marker_size:.4f} m")
    print(f"Sample rate: {args.sample_rate_hz:g} Hz")
    print(f"Using calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    capture = open_camera(args.camera, width, height)
    try:
        print(f"Camera {args.camera}: target {width}x{height}")
        print("Press S to solve, Q to quit.")

        observations: list[FrameObservation] = []
        sample_interval = 1.0 / args.sample_rate_hz
        next_sample_time = time.monotonic()
        status_line: str | None = None
        last_solve_quality: CalibrationQualityReport | None = None

        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read a frame from camera {args.camera}.")
            require_frame_size(frame.shape[1], frame.shape[0], width, height, args.calibration)

            visible = detect_expected_markers(detector, frame, expected_id_set)
            preview = frame.copy()
            draw_detection_outlines(preview, visible)

            now = time.monotonic()
            if now >= next_sample_time and len(visible) >= 2:
                observations.append(
                    FrameObservation(
                        frame_id=len(observations),
                        markers={
                            marker_id: corners.copy() for marker_id, corners in visible.items()
                        },
                    )
                )
                next_sample_time = now + sample_interval

            pair_counts = pair_covisibility_counts(observations, expected_ids)
            connected_ids = connected_marker_ids(
                pair_counts,
                args.reference_marker_id,
                args.min_pair_inliers,
            )
            draw_calibration_hud(
                preview,
                expected_ids=expected_ids,
                visible_ids=sorted(visible),
                sample_count=len(observations),
                pair_counts=pair_counts,
                connected_ids=connected_ids,
                reference_marker_id=args.reference_marker_id,
                min_pair_inliers=args.min_pair_inliers,
                status_line=status_line,
                last_solve_quality=last_solve_quality,
            )

            cv2.imshow("Marker layout calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Calibration cancelled.")
                return False
            if key in (ord("s"), ord("S")):
                result = calibrate_marker_layout(
                    observations,
                    camera_matrix,
                    dist_coeffs,
                    expected_marker_ids=expected_ids,
                    reference_marker_id=args.reference_marker_id,
                    marker_size_m=args.marker_size,
                    settings=settings,
                )
                if result.failure_reason is not None or result.layout is None:
                    print_refusal(result)
                    last_solve_quality = result.quality
                    status_line = f"refused: {result.failure_reason}"
                    continue

                save_marker_model(args.output, result.layout)
                print_success(result, args.output)
                return True
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    try:
        run_capture(parse_args())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
