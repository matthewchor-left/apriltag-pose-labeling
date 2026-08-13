"""Live marker layout calibration CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.apriltag import build_apriltag_detector
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.layout import marker_color_bgr, save_marker_model
from object_apriltag.cli.calibration_diagnostics import (
    format_quality_diagnostics_lines,
    save_calibration_diagnostics,
)
from object_apriltag.cli.live_pair_readiness_worker import (
    LivePairReadinessView,
    LivePairReadinessWorker,
)
from object_apriltag.viz.overlay import draw_status_hud_panel
from object_apriltag.marker_layout_calibration import (
    AssignmentRejectionSummary,
    CalibrationQualityReport,
    CalibrationResult,
    CalibrationSettings,
    FrameObservation,
    PairReadinessEdge,
    calibrate_marker_layout,
    compute_live_pair_readiness,
    parse_anchor_marker_ids,
    parse_marker_id_spec,
    resolve_marker_sizes_for_calibration,
)

DEFAULT_ASSIGNMENT_REJECTION_SUMMARY_LINES = 3

DEFAULT_MIN_PAIR_INLIERS = 20
DEFAULT_REPROJECTION_RMS_GATE_PX = 2.0
DEFAULT_PAIR_TRANSLATION_RMS_GATE_RATIO = 0.10
DEFAULT_PAIR_ROTATION_RMS_GATE_DEG = 5.0
DEFAULT_PAIR_READINESS_HUD_LINES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate marker sticker layout from live co-visible AprilTag detections.",
        epilog=(
            "Controls:\n"
            "  C  capture the current frame when at least two expected markers are visible\n"
            "  S  solve from captured samples and write --output when quality gates pass\n"
            "  Q  quit without writing\n"
            "\n"
            "Capture:\n"
            "  Inspect the live image, then press C only for sharp, geometrically diverse "
            "views. Every expected ID must co-appear with others often enough to connect "
            "through the reference marker.\n"
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
        help="Default physical AprilTag edge length in meters.",
    )
    parser.add_argument(
        "--marker-size-for",
        type=str,
        action="append",
        nargs="+",
        default=None,
        metavar="ID_OR_RANGE:SIZE",
        help=(
            "Per-marker physical edge length override (repeatable), e.g. "
            "--marker-size-for 4:0.03 --marker-size-for 10-12:0.025 or "
            "--marker-size-for 4:0.03 10-12:0.025. "
            "Override IDs must be a subset of --marker-ids."
        ),
    )
    parser.add_argument(
        "--marker-ids",
        type=str,
        nargs="+",
        required=True,
        metavar="ID",
        help="Expected unique AprilTag marker IDs (e.g. 0 1 2 3-10 11).",
    )
    parser.add_argument(
        "--reference-marker-id",
        type=int,
        required=True,
        help="Reference marker ID; must appear in --marker-ids.",
    )
    parser.add_argument(
        "--anchor-marker-ids",
        type=str,
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Optional anchor-core marker IDs for bootstrap assignment (supports ranges, "
            "e.g. 0 1 4-7). Must include --reference-marker-id and be a subset of "
            "--marker-ids. Omitted uses full-set exhaustive IPPE assignment."
        ),
    )
    parser.add_argument(
        "--anchor-stop-after-expansion",
        action="store_true",
        help=(
            "Anchor-core only: after hierarchical expansion, write marker poses directly "
            "without full IPPE reassignment, bundle adjustment, or quality gates. "
            "Requires --anchor-marker-ids."
        ),
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
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Optional JSON path for full post-solve calibration diagnostics.",
    )
    return parser.parse_args()


def flatten_marker_size_override_tokens(
    tokens: list[str] | list[list[str]] | None,
) -> list[str] | None:
    if tokens is None:
        return None
    flattened: list[str] = []
    for token in tokens:
        if isinstance(token, list):
            flattened.extend(token)
        else:
            flattened.append(token)
    return flattened


def validate_args(
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, float], CalibrationSettings, tuple[int, ...] | None, bool]:
    if not args.calibration.exists():
        raise RuntimeError(
            f"Calibration file not found: {args.calibration}\n"
            "Run `uv run object-charuco` first."
        )
    if args.output.exists() and not args.force:
        raise RuntimeError(
            f"Output already exists: {args.output}. Pass --force to overwrite."
        )

    expected_ids, marker_ids_failure = parse_marker_id_spec(args.marker_ids)
    if marker_ids_failure is not None:
        raise RuntimeError(f"--marker-ids {marker_ids_failure}")
    if args.reference_marker_id not in expected_ids:
        raise RuntimeError("--reference-marker-id must appear in --marker-ids.")
    anchor_list: list[int] | None
    if args.anchor_marker_ids is None:
        anchor_list = None
    else:
        anchor_list, anchor_parse_failure = parse_marker_id_spec(args.anchor_marker_ids)
        if anchor_parse_failure is not None:
            raise RuntimeError(f"--anchor-marker-ids {anchor_parse_failure}")
    anchor_ids, anchor_failure = parse_anchor_marker_ids(
        anchor_list,
        expected_ids,
        args.reference_marker_id,
    )
    if anchor_failure is not None:
        raise RuntimeError(anchor_failure)
    if args.anchor_stop_after_expansion and anchor_ids is None:
        raise RuntimeError("--anchor-stop-after-expansion requires --anchor-marker-ids.")
    if args.marker_size <= 0.0:
        raise RuntimeError("--marker-size must be positive.")
    marker_sizes_m, sizes_failure = resolve_marker_sizes_for_calibration(
        expected_ids,
        args.marker_size,
        flatten_marker_size_override_tokens(args.marker_size_for),
    )
    if sizes_failure is not None:
        raise RuntimeError(sizes_failure)
    assert marker_sizes_m is not None
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
    return expected_ids, marker_sizes_m, settings, anchor_ids, args.anchor_stop_after_expansion


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


def draw_detection_outlines(
    frame: np.ndarray,
    visible: dict[int, np.ndarray],
    reference_marker_id: int,
) -> None:
    for marker_id, corners in visible.items():
        points = corners.reshape(4, 2).astype(np.int32)
        color = marker_color_bgr(marker_id, reference_marker_id)
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


def format_assignment_rejection_cause(cause_count) -> str:
    pair_text = ""
    if cause_count.marker_pair is not None:
        pair_text = f" pair=({cause_count.marker_pair[0]},{cause_count.marker_pair[1]})"
    return f"{cause_count.reason}{pair_text} x{cause_count.count}"


def format_assignment_rejection_summary(
    summary: AssignmentRejectionSummary,
    *,
    max_lines: int = DEFAULT_ASSIGNMENT_REJECTION_SUMMARY_LINES,
) -> list[str]:
    if summary.total_rejected == 0:
        return []
    return [
        format_assignment_rejection_cause(cause)
        for cause in summary.top_causes[:max_lines]
    ]


def format_pair_readiness_edge(edge: PairReadinessEdge) -> str:
    return (
        f"({edge.marker_a},{edge.marker_b}) {edge.status} "
        f"raw={edge.raw_covisible_frames}"
    )


def format_marker_connectivity_line(
    connected_ids: set[int],
    missing_ids: frozenset[int],
    *,
    max_listed: int = 12,
) -> str:
    def compact(marker_ids: list[int]) -> str:
        if len(marker_ids) <= max_listed:
            return str(marker_ids)
        head = marker_ids[:max_listed]
        return f"{head} +{len(marker_ids) - max_listed}"

    connected = compact(sorted(connected_ids))
    missing = compact(sorted(missing_ids))
    return f"markers connected {connected} missing {missing}"


def format_readiness_snapshot_line(
    *,
    current_sample_count: int,
    represented_sample_count: int,
    is_computing: bool,
) -> str:
    suffix = " (computing...)" if is_computing else ""
    if represented_sample_count == current_sample_count and not is_computing:
        return f"readiness@{current_sample_count} samples"
    return f"readiness@{represented_sample_count}/{current_sample_count} samples{suffix}"


def build_pair_readiness_hud_lines(
    *,
    expected_ids: list[int],
    visible_ids: list[int],
    current_sample_count: int,
    readiness_view: LivePairReadinessView,
    reference_marker_id: int,
    max_pair_lines: int = DEFAULT_PAIR_READINESS_HUD_LINES,
) -> list[str]:
    diagnostics = readiness_view.diagnostics
    status_counts = {"pass": 0, "weak": 0, "fail": 0}
    for edge in diagnostics.pairs:
        status_counts[edge.status] = status_counts.get(edge.status, 0) + 1

    connected_ids = set(diagnostics.connected_marker_ids)
    lines = [
        f"expected: {expected_ids}",
        f"visible: {visible_ids}",
        f"samples: {current_sample_count}",
        format_readiness_snapshot_line(
            current_sample_count=current_sample_count,
            represented_sample_count=readiness_view.represented_sample_count,
            is_computing=readiness_view.is_computing,
        ),
        format_marker_connectivity_line(connected_ids, diagnostics.missing_marker_ids),
        (
            f"graph: {len(connected_ids)}/{len(expected_ids)} connected from ref {reference_marker_id}"
        ),
        (
            f"pairs: {status_counts['pass']} pass, {status_counts['weak']} weak, "
            f"{status_counts['fail']} fail ({len(diagnostics.pairs)} observed)"
        ),
        "S=solve  Q=quit",
    ]
    if diagnostics.failure_reason:
        lines.insert(7, f"readiness error: {diagnostics.failure_reason}")

    pair_lines = [format_pair_readiness_edge(edge) for edge in diagnostics.pairs]
    if len(pair_lines) > max_pair_lines:
        pair_lines = pair_lines[:max_pair_lines]
        pair_lines.append("...")
    lines[7:7] = pair_lines
    return lines


def build_pair_readiness_hud_lines_from_diagnostics(
    *,
    expected_ids: list[int],
    visible_ids: list[int],
    current_sample_count: int,
    readiness_view: LivePairReadinessView,
    reference_marker_id: int,
    max_pair_lines: int = DEFAULT_PAIR_READINESS_HUD_LINES,
) -> list[str]:
    return build_pair_readiness_hud_lines(
        expected_ids=expected_ids,
        visible_ids=visible_ids,
        current_sample_count=current_sample_count,
        readiness_view=readiness_view,
        reference_marker_id=reference_marker_id,
        max_pair_lines=max_pair_lines,
    )


def draw_calibration_hud(
    frame: np.ndarray,
    *,
    hud_lines: list[str],
    last_solve_quality: CalibrationQualityReport | None,
) -> None:
    lines = list(hud_lines)
    if last_solve_quality is not None:
        lines.insert(5, format_solve_frame_counts(last_solve_quality))
        if last_solve_quality.assignment_rejections is not None:
            rejection_lines = format_assignment_rejection_summary(
                last_solve_quality.assignment_rejections,
                max_lines=2,
            )
            if rejection_lines:
                lines[6:6] = [
                    f"assignment rejections: {last_solve_quality.assignment_rejections.total_rejected} total",
                    *rejection_lines,
                ]

    draw_status_hud_panel(frame, lines)


def print_refusal(result: CalibrationResult) -> None:
    print(f"Calibration refused: {result.failure_reason}")
    quality = result.quality
    if quality is None:
        return
    print(f"  {format_solve_frame_counts(quality)}")
    for line in format_quality_diagnostics_lines(quality):
        print(f"  {line}")


def print_success(result: CalibrationResult, output: Path) -> None:
    print(f"Saved marker model: {output}")
    quality = result.quality
    if quality is None:
        return
    print(f"  {format_solve_frame_counts(quality)}")
    for line in format_quality_diagnostics_lines(quality):
        print(f"  {line}")


def write_calibration_diagnostics_if_requested(
    diagnostics_output: Path | None,
    result: CalibrationResult,
) -> None:
    if diagnostics_output is None or result.quality is None:
        return
    path = save_calibration_diagnostics(diagnostics_output, result)
    print(f"Wrote calibration diagnostics: {path}")


def run_capture(args: argparse.Namespace) -> bool:
    """Capture live samples; return True when a model was saved."""
    expected_ids, marker_sizes_m, settings, anchor_ids, anchor_stop_after_expansion = validate_args(args)
    expected_id_set = set(expected_ids)

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    detector = build_apriltag_detector(args.dictionary, args.detection_sensitivity)

    print(f"Expected marker IDs: {expected_ids}")
    print(f"Reference marker ID: {args.reference_marker_id}")
    print(f"Default marker size: {args.marker_size:.4f} m")
    overrides = {
        marker_id: size
        for marker_id, size in sorted(marker_sizes_m.items())
        if size != args.marker_size
    }
    if overrides:
        print(f"Marker size overrides: {overrides}")
    else:
        print("Marker size overrides: none (uniform)")
    print("Capture mode: manual (press C)")
    print(f"Using calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    capture = open_camera(args.camera, width, height)
    readiness_worker = LivePairReadinessWorker(
        compute_fn=compute_live_pair_readiness,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        expected_marker_ids=expected_ids,
        reference_marker_id=args.reference_marker_id,
        settings=settings,
    )
    try:
        print(f"Camera {args.camera}: target {width}x{height}")
        print("Press C to capture, S to solve, Q to quit.")

        observations: list[FrameObservation] = []
        status_line: str | None = None
        last_solve_quality: CalibrationQualityReport | None = None

        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read a frame from camera {args.camera}.")
            require_frame_size(frame.shape[1], frame.shape[0], width, height, args.calibration)

            visible = detect_expected_markers(detector, frame, expected_id_set)
            preview = frame.copy()
            draw_detection_outlines(preview, visible, args.reference_marker_id)

            readiness_view = readiness_worker.poll(len(observations))
            hud_lines = build_pair_readiness_hud_lines_from_diagnostics(
                expected_ids=expected_ids,
                visible_ids=sorted(visible),
                current_sample_count=len(observations),
                readiness_view=readiness_view,
                reference_marker_id=args.reference_marker_id,
            )
            if status_line:
                hud_lines.append(status_line)
            draw_calibration_hud(
                preview,
                hud_lines=hud_lines,
                last_solve_quality=last_solve_quality,
            )

            cv2.imshow("Marker layout calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Calibration cancelled.")
                return False
            if key in (ord("c"), ord("C")):
                if len(visible) < 2:
                    status_line = "capture skipped: need at least 2 expected markers"
                    continue
                observations.append(
                    FrameObservation(
                        frame_id=len(observations),
                        markers={
                            marker_id: corners.copy()
                            for marker_id, corners in visible.items()
                        },
                    )
                )
                readiness_worker.submit(observations)
                status_line = (
                    f"captured sample {len(observations)}: "
                    f"markers {sorted(visible)}"
                )
                continue
            if key in (ord("s"), ord("S")):
                result = calibrate_marker_layout(
                    observations,
                    camera_matrix,
                    dist_coeffs,
                    expected_marker_ids=expected_ids,
                    reference_marker_id=args.reference_marker_id,
                    marker_size_m=args.marker_size,
                    settings=settings,
                    anchor_marker_ids=anchor_ids,
                    anchor_stop_after_expansion=anchor_stop_after_expansion,
                    marker_sizes_m=marker_sizes_m,
                )
                if result.failure_reason is not None or result.layout is None:
                    print_refusal(result)
                    try:
                        write_calibration_diagnostics_if_requested(args.diagnostics_output, result)
                    except RuntimeError as error:
                        print(error, file=sys.stderr)
                    last_solve_quality = result.quality
                    status_line = f"refused: {result.failure_reason}"
                    continue

                save_marker_model(args.output, result.layout)
                print_success(result, args.output)
                try:
                    write_calibration_diagnostics_if_requested(args.diagnostics_output, result)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                return True
    finally:
        readiness_worker.shutdown(join_timeout=0.0)
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
