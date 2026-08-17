"""Live marker layout calibration CLI."""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_apriltag.apriltag import build_apriltag_detector
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.frame_source import (
    format_frame_source,
    open_frame_source,
    parse_frame_source,
    read_frame,
)
from object_apriltag.layout import (
    CORNER_NAMES,
    MarkerLayout,
    footprint_corner_with_padding,
    footprint_from_dict,
    marker_color_bgr,
    save_marker_model,
)
from object_apriltag.pose import estimate_marker_pose, marker_corner_object_points
from object_apriltag.viz.projection import opencv_image_point, project_camera_point
from object_apriltag.object_model_edit import (
    apply_keypoint_sources_from_layout,
    build_object_model_document_from_layout,
    load_object_model_document,
    missing_source_marker_ids,
    parse_keypoint_sources,
    save_object_model_document,
    save_object_model_keypoints,
)
from object_apriltag.cli.calibration_diagnostics import (
    format_omitted_marker_lines,
    format_quality_diagnostics_lines,
    save_calibration_diagnostics,
)
from object_apriltag.cli.live_pair_readiness_worker import (
    LivePairReadinessView,
    LivePairReadinessWorker,
)
from object_apriltag.viz.overlay import draw_status_hud_panel
from object_apriltag.marker_layout_calibration.recipe import (
    BenchmarkExecution,
    InteractiveExecution,
    load_calibration_recipe,
)
from object_apriltag.marker_layout_calibration import (
    AssignmentRejectionSummary,
    CalibrationQualityReport,
    CalibrationResult,
    CalibrationSettings,
    CalibrationSolveDiagnostics,
    FrameObservation,
    PairReadinessEdge,
    calibrate_marker_layout,
    compute_live_pair_readiness,
    parse_anchor_marker_ids,
    parse_marker_id_spec,
    resolve_marker_sizes_for_calibration,
)

DEFAULT_ASSIGNMENT_REJECTION_SUMMARY_LINES = 3

DEFAULT_SAMPLE_RATE_HZ = 10.0
DEFAULT_MIN_PAIR_INLIERS = 20
DEFAULT_REPROJECTION_RMS_GATE_PX = 2.0
DEFAULT_PAIR_TRANSLATION_RMS_GATE_RATIO = 0.10
DEFAULT_PAIR_ROTATION_RMS_GATE_DEG = 5.0
DEFAULT_PAIR_READINESS_HUD_LINES = 8

BENCHMARK_FRAME_SELECTION_UNIFORM = "uniform"
BENCHMARK_FRAME_SELECTION_SHARPEST = "sharpest"
DEFAULT_BENCHMARK_FRAME_SELECTION = BENCHMARK_FRAME_SELECTION_UNIFORM

LEGACY_CALIBRATION_FLAGS = frozenset(
    {
        "--source",
        "--calibration",
        "--dictionary",
        "--detection-sensitivity",
        "--marker-size",
        "--marker-size-for",
        "--marker-ids",
        "--reference-marker-id",
        "--anchor-marker-ids",
        "--anchor-stop-after-expansion",
        "--best-effort",
        "--partial-output",
        "--output",
        "--auto",
        "--sample-rate-hz",
        "--min-pair-inliers",
        "--reprojection-rms-gate-px",
        "--pair-translation-rms-gate-ratio",
        "--pair-rotation-rms-gate-deg",
        "--object-model",
        "--overlay-object-model",
        "--no-overlay-object-model",
        "--diagnostics-output",
        "--benchmark",
        "--benchmark-frame-selection",
    }
)


def _argv_uses_config(argv: list[str]) -> bool:
    """Return whether ``argv`` requests config mode."""
    for token in argv:
        if token == "--config" or token.startswith("--config="):
            return True
    return False


def _legacy_flags_in_argv(argv: list[str]) -> list[str]:
    """Collect legacy calibration flag tokens present in ``argv``."""
    legacy: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        if name == "--config" and "=" not in token:
            skip_next = True
            continue
        if name in LEGACY_CALIBRATION_FLAGS:
            legacy.append(token)
    return legacy


def _assert_config_mode_argv(argv: list[str]) -> None:
    """Reject legacy calibration flags mixed with ``--config``."""
    legacy = _legacy_flags_in_argv(argv)
    if legacy:
        joined = " ".join(legacy)
        raise RuntimeError(f"--config cannot be mixed with legacy calibration flags: {joined}")


def namespace_from_recipe(config_path: Path, *, force: bool) -> argparse.Namespace:
    """Build a CLI namespace from a parsed Calibration Recipe."""
    try:
        recipe = load_calibration_recipe(config_path)
    except (OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    execution = recipe.execution
    benchmark = isinstance(execution, BenchmarkExecution)
    if benchmark:
        sample_rate_hz = execution.sample_rate_hz
        benchmark_frame_selection = execution.frame_selection
        auto = False
        overlay_object_model = False
    else:
        assert isinstance(execution, InteractiveExecution)
        auto = execution.capture == "auto"
        sample_rate_hz = (
            execution.sample_rate_hz
            if execution.sample_rate_hz is not None
            else DEFAULT_SAMPLE_RATE_HZ
        )
        benchmark_frame_selection = DEFAULT_BENCHMARK_FRAME_SELECTION
        overlay_object_model = execution.preview == "keypoint_sources"

    return argparse.Namespace(
        from_config=True,
        recipe=recipe,
        config_path=config_path.resolve(),
        source=recipe.source,
        calibration=recipe.intrinsics_path,
        dictionary=recipe.dictionary,
        detection_sensitivity=recipe.sensitivity,
        marker_size=recipe.default_marker_size_m,
        marker_size_for=None,
        marker_ids=[str(marker_id) for marker_id in recipe.expected_marker_ids],
        reference_marker_id=recipe.reference_marker_id,
        anchor_marker_ids=(
            None
            if recipe.anchor_marker_ids is None
            else [str(marker_id) for marker_id in recipe.anchor_marker_ids]
        ),
        anchor_stop_after_expansion=recipe.anchor_stop_after_expansion,
        best_effort=recipe.policy == "best_effort",
        partial_output=recipe.partial_output,
        output=recipe.paths.marker_model_path,
        force=force,
        auto=auto,
        sample_rate_hz=sample_rate_hz,
        min_pair_inliers=recipe.settings.min_inliers_per_edge,
        reprojection_rms_gate_px=recipe.settings.reprojection_rms_gate_px,
        pair_translation_rms_gate_ratio=recipe.settings.pair_translation_rms_gate_ratio,
        pair_rotation_rms_gate_deg=recipe.settings.pair_rotation_rms_gate_deg,
        object_model=recipe.paths.object_model_path,
        overlay_object_model=overlay_object_model,
        diagnostics_output=recipe.paths.diagnostics_path,
        benchmark=benchmark,
        benchmark_frame_selection=benchmark_frame_selection,
    )


def _parse_config_mode_args() -> argparse.Namespace:
    """Parse config-mode CLI arguments."""
    _assert_config_mode_argv(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="Calibrate marker and object models from a Calibration Recipe config.json.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Calibration Workspace config.json recipe.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing marker_model.json and object_model.json outputs.",
    )
    args = parser.parse_args()
    return namespace_from_recipe(args.config, force=args.force)


def _is_benchmark_mode(args: argparse.Namespace) -> bool:
    """Return whether ``--benchmark`` was set on the parsed CLI namespace.

    Args:
        args: Parsed CLI namespace.

    Returns:
        True when benchmark mode is enabled.
    """
    return getattr(args, "benchmark", False) is True


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for config mode or legacy live/benchmark calibration.

    Returns:
        Parsed argument namespace with calibration flags and frame-source options.
    """
    if _argv_uses_config(sys.argv[1:]):
        return _parse_config_mode_args()
    return _parse_legacy_args()


def _parse_legacy_args() -> argparse.Namespace:
    """Parse legacy flag-based CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Calibrate marker sticker layout from live co-visible AprilTag detections.",
        epilog=(
            "Controls:\n"
            "  C  capture the current frame when at least two expected markers are visible\n"
            "  S  solve from captured samples and write --output when quality gates pass\n"
            "  Q  quit without writing\n"
            "\n"
            "Capture:\n"
            "  Default is manual capture (press C). Pass --auto to record frames at "
            "--sample-rate-hz when at least two expected markers are visible; C still "
            "captures an extra frame in --auto mode. Inspect the live image and prefer "
            "sharp, geometrically diverse views. Every expected ID must co-appear with "
            "others often enough to connect through the reference marker.\n"
            "\n"
            "Scale:\n"
            "  Metric layout depends on --marker-size and calibrated camera intrinsics. "
            "Wrong marker size or scaled intrinsics will bias the solved geometry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Recommended: run from a Calibration Workspace config.json. "
            "May only be combined with --force."
        ),
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
        default=None,
        help=(
            "Reference marker ID; must appear in --marker-ids. "
            "Omitted selects automatically from pair connectivity and keypoint sources."
        ),
    )
    parser.add_argument(
        "--anchor-marker-ids",
        type=str,
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Optional anchor-core marker IDs for bootstrap assignment (supports ranges, "
            "e.g. 0 1 4-7). When --reference-marker-id is set, it must appear here. "
            "Subset of --marker-ids. Omitted uses full-set exhaustive IPPE assignment."
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
        "--best-effort",
        action="store_true",
        help=(
            "Write a provisional marker model when optimization completes but strict "
            "reprojection, translation, or rotation quality gates fail."
        ),
    )
    parser.add_argument(
        "--partial-output",
        action="store_true",
        help=(
            "With --best-effort, write a reference-connected partial marker model when "
            "some requested markers remain unobservable or disconnected after weak-edge "
            "recovery."
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
        "--auto",
        action="store_true",
        help="Enable periodic automatic capture at --sample-rate-hz (default: manual C only).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=(
            "Automatic capture rate in Hz when --auto is set "
            f"(default: {DEFAULT_SAMPLE_RATE_HZ:g}; ignored in manual mode)."
        ),
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
        "--object-model",
        type=Path,
        default=None,
        help=(
            "Optional object model JSON to update after a successful marker-model save. "
            "Requires non-empty keypoint_sources mapping keypoint names to marker corners "
            "(optional padding_mm per source)."
        ),
    )
    parser.add_argument(
        "--overlay-object-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preview object-model keypoint_sources on the live view. Requires --object-model. "
            "Draws each mapped corner (and padding_mm offset) on visible source tags."
        ),
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=None,
        help="Optional JSON path for full post-solve calibration diagnostics.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Headless single-pass benchmark over a video file. Requires "
            "--diagnostics-output; decodes every frame but runs AprilTag detection only "
            "on frames selected by --benchmark-frame-selection (default: uniform scheduled "
            "samples at --sample-rate-hz), then solves once at EOF."
        ),
    )
    parser.add_argument(
        "--benchmark-frame-selection",
        choices=(
            BENCHMARK_FRAME_SELECTION_UNIFORM,
            BENCHMARK_FRAME_SELECTION_SHARPEST,
        ),
        default=DEFAULT_BENCHMARK_FRAME_SELECTION,
        help=(
            "Benchmark-only: which decoded frames run AprilTag detection. "
            f"'{BENCHMARK_FRAME_SELECTION_UNIFORM}' (default) detects on scheduled "
            "video-time samples at --sample-rate-hz. "
            f"'{BENCHMARK_FRAME_SELECTION_SHARPEST}' still decodes every frame to score "
            "relative sharpness, groups frames into half-open windows of "
            "1/--sample-rate-hz seconds, and detects only the sharpest frame per window "
            "(earliest frame on ties); the final partial window is flushed at EOF. "
            "Selecting sharpest requires --benchmark."
        ),
    )
    return parser.parse_args()


def flatten_marker_size_override_tokens(
    tokens: list[str] | list[list[str]] | None,
) -> list[str] | None:
    """Flatten nested ``--marker-size-for`` nargs groups into one token list.

    Args:
        tokens: Raw ``action="append"`` token groups from argparse, or ``None``.

    Returns:
        Flattened token list, or ``None`` when ``tokens`` is ``None``.
    """
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
) -> tuple[list[int], dict[int, float], CalibrationSettings, tuple[int, ...] | None, bool, bool, bool]:
    """Validate CLI inputs and derive calibration settings.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Tuple of ``(expected_ids, marker_sizes_m, settings, anchor_ids,
        anchor_stop_after_expansion, best_effort, partial_output)``.

    Raises:
        RuntimeError: Invalid flag combinations, missing files, or out-of-range values.
    """
    if getattr(args, "from_config", False) is True:
        return _validate_config_args(args)

    object_model_sources: dict[str, tuple[int, str, float]] | None = None
    if not args.calibration.exists():
        raise RuntimeError(
            f"Calibration file not found: {args.calibration}\n"
            "Run `uv run object-charuco` first."
        )
    if args.output.exists() and not args.force:
        raise RuntimeError(
            f"Output already exists: {args.output}. Pass --force to overwrite."
        )
    if isinstance(args.object_model, Path):
        if not args.object_model.exists():
            raise RuntimeError(f"Object model file not found: {args.object_model}")
        try:
            _, document = load_object_model_document(args.object_model)
            object_model_sources = parse_keypoint_sources(document)
        except (ValueError, OSError) as error:
            raise RuntimeError(str(error)) from error
    if args.overlay_object_model is True and not isinstance(args.object_model, Path):
        raise RuntimeError("--object-model is required when --overlay-object-model is enabled.")
    if _is_benchmark_mode(args):
        if args.diagnostics_output is None:
            raise RuntimeError("--diagnostics-output is required with --benchmark.")
        if isinstance(getattr(args, "source", None), int):
            raise RuntimeError("--benchmark requires a video file --source, not a camera index.")
        if args.overlay_object_model is True:
            raise RuntimeError(
                "--overlay-object-model is preview-only and cannot be used with --benchmark."
            )

    expected_ids, marker_ids_failure = parse_marker_id_spec(args.marker_ids)
    if marker_ids_failure is not None:
        raise RuntimeError(f"--marker-ids {marker_ids_failure}")
    if object_model_sources is not None:
        missing_source_ids = sorted(
            {marker_id for marker_id, _, _ in object_model_sources.values()}
            - set(expected_ids)
        )
        if missing_source_ids:
            raise RuntimeError(
                "Object-model keypoint source marker IDs must appear in --marker-ids: "
                f"{missing_source_ids}."
            )
    if args.reference_marker_id is not None and args.reference_marker_id not in expected_ids:
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
    if args.best_effort and args.anchor_stop_after_expansion:
        raise RuntimeError(
            "--best-effort cannot be used with --anchor-stop-after-expansion."
        )
    if args.partial_output and not args.best_effort:
        raise RuntimeError("--partial-output requires --best-effort.")
    args.keypoint_sources = object_model_sources or {}
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
    if not math.isfinite(args.sample_rate_hz) or args.sample_rate_hz <= 0.0:
        raise RuntimeError("--sample-rate-hz must be finite and positive.")
    frame_selection = getattr(
        args, "benchmark_frame_selection", DEFAULT_BENCHMARK_FRAME_SELECTION
    )
    if frame_selection == BENCHMARK_FRAME_SELECTION_SHARPEST and not _is_benchmark_mode(args):
        raise RuntimeError("--benchmark-frame-selection requires --benchmark.")
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
    return expected_ids, marker_sizes_m, settings, anchor_ids, args.anchor_stop_after_expansion, args.best_effort, args.partial_output


def _validate_config_args(
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, float], CalibrationSettings, tuple[int, ...] | None, bool, bool, bool]:
    """Validate config-mode namespace values derived from a Calibration Recipe."""
    recipe = args.recipe
    if not recipe.intrinsics_path.is_file():
        raise RuntimeError(
            f"Calibration file not found: {recipe.intrinsics_path}\n"
            "Run `uv run object-charuco` first."
        )
    existing_outputs = [
        path
        for path in (
            recipe.paths.marker_model_path,
            recipe.paths.object_model_path,
            recipe.paths.diagnostics_path,
        )
        if path.exists()
    ]
    if existing_outputs and not args.force:
        joined = ", ".join(str(path) for path in existing_outputs)
        raise RuntimeError(
            f"Output already exists: {joined}. Pass --force to overwrite."
        )
    if _is_benchmark_mode(args):
        if isinstance(args.source, int):
            raise RuntimeError("benchmark execution requires a video file source, not a camera index.")
        if args.overlay_object_model is True:
            raise RuntimeError(
                "execution.preview 'keypoint_sources' cannot be used with benchmark execution."
            )
    return (
        list(recipe.expected_marker_ids),
        dict(recipe.marker_sizes_m),
        recipe.settings,
        recipe.anchor_marker_ids,
        recipe.anchor_stop_after_expansion,
        recipe.policy == "best_effort",
        recipe.partial_output,
    )


def _sync_selected_reference_marker_id(
    args: argparse.Namespace,
    result: CalibrationResult,
) -> None:
    """Copy auto-selected reference marker ID back onto CLI args for reporting."""
    if args.reference_marker_id is None and result.layout is not None:
        args.reference_marker_id = result.layout.reference_marker_id


def require_frame_size(
    frame_width: int,
    frame_height: int,
    calibration_width: int,
    calibration_height: int,
    calibration_path: Path,
) -> None:
    """Reject frames whose resolution differs from the intrinsics calibration image size.

    Args:
        frame_width: Decoded frame width in pixels.
        frame_height: Decoded frame height in pixels.
        calibration_width: Width from the intrinsics calibration file.
        calibration_height: Height from the intrinsics calibration file.
        calibration_path: Path shown in the mismatch error message.

    Raises:
        RuntimeError: Frame dimensions do not match calibrated image size.
    """
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
    """Detect AprilTags in ``frame`` and return corners only for ``expected_ids``.

    Args:
        detector: OpenCV ArUco/AprilTag detector.
        frame: BGR or grayscale image.
        expected_ids: Marker IDs to retain from detection output.

    Returns:
        Map from marker ID to ``(4, 2)`` float64 corner array in image pixels.
    """
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
    reference_marker_id: int | None,
) -> None:
    """Draw marker outlines and ID labels on ``frame`` in place.

    Args:
        frame: Image to annotate (modified in place).
        visible: Detected marker corners keyed by marker ID.
        reference_marker_id: ID used for reference-marker coloring, or ``None``.
    """
    ref_id = reference_marker_id if reference_marker_id is not None else -1
    for marker_id, corners in visible.items():
        points = corners.reshape(4, 2).astype(np.int32)
        color = marker_color_bgr(marker_id, ref_id)
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


KEYPOINT_SOURCE_OVERLAY_COLORS_BGR = {
    "top": (128, 0, 128),
    "bottom": (0, 255, 0),
    "left": (255, 255, 0),
    "right": (165, 42, 42),
}


def marker_frame_footprint(marker_id: int, marker_size_m: float):
    """Build a marker-corner footprint in the marker frame from physical edge length.

    Args:
        marker_id: AprilTag marker ID for the footprint record.
        marker_size_m: Physical marker edge length in meters.

    Returns:
        Marker footprint with corners named per ``CORNER_NAMES``.
    """
    object_points = marker_corner_object_points(marker_size_m)
    payload = {
        corner_name: object_points[index].astype(np.float64).tolist()
        for index, corner_name in enumerate(CORNER_NAMES)
    }
    return footprint_from_dict(marker_id, payload)


def project_keypoint_source_on_marker(
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float,
    corner_name: str,
    padding_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project a keypoint source corner (and optional padding) into image coordinates.

    Args:
        corners: Detected marker corners ``(4, 2)``.
        marker_id: Source marker ID.
        marker_size_m: Physical edge length for pose estimation.
        corner_name: Footprint corner name (e.g. ``"top"``).
        padding_m: Outward padding along the corner normal in meters.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Distortion coefficients.

    Returns:
        ``(raw_corner_px, padded_corner_px)`` image points, or ``None`` when pose
        estimation fails or either point projects behind the camera.
    """
    try:
        rvec, tvec = estimate_marker_pose(corners, marker_size_m, camera_matrix, dist_coeffs)
    except RuntimeError:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    footprint = marker_frame_footprint(marker_id, marker_size_m)
    corner_point = footprint.corners_by_name()[corner_name]
    target_point = footprint_corner_with_padding(footprint, corner_name, padding_m)

    raw_camera = rotation @ corner_point + translation
    target_camera = rotation @ target_point + translation
    if raw_camera[2] <= 0.0 or target_camera[2] <= 0.0:
        return None

    raw_image = project_camera_point(raw_camera, camera_matrix, dist_coeffs)
    target_image = project_camera_point(target_camera, camera_matrix, dist_coeffs)
    if not (np.all(np.isfinite(raw_image)) and np.all(np.isfinite(target_image))):
        return None
    return raw_image, target_image


def draw_keypoint_source_overlays(
    frame: np.ndarray,
    visible: dict[int, np.ndarray],
    keypoint_sources: dict[str, tuple[int, str, float]],
    marker_sizes_m: dict[int, float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    """Preview object-model keypoint sources on visible markers.

    Args:
        frame: Live preview image (modified in place).
        visible: Detected marker corners keyed by marker ID.
        keypoint_sources: Keypoint name to ``(marker_id, corner_name, padding_m)`` map.
        marker_sizes_m: Per-marker physical edge lengths.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Distortion coefficients.
    """
    for keypoint_name, (marker_id, corner_name, padding_m) in keypoint_sources.items():
        corners = visible.get(marker_id)
        if corners is None:
            continue
        projected = project_keypoint_source_on_marker(
            corners,
            marker_id,
            marker_sizes_m[marker_id],
            corner_name,
            padding_m,
            camera_matrix,
            dist_coeffs,
        )
        if projected is None:
            continue

        color = KEYPOINT_SOURCE_OVERLAY_COLORS_BGR.get(keypoint_name, (200, 200, 200))
        raw_point = opencv_image_point(projected[0])
        target_point = opencv_image_point(projected[1])
        if raw_point is None or target_point is None:
            continue

        cv2.circle(frame, raw_point, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, raw_point, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        if padding_m > 0.0 and raw_point != target_point:
            cv2.line(frame, raw_point, target_point, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, target_point, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, target_point, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            label_point = target_point
        else:
            label_point = raw_point
        cv2.putText(
            frame,
            keypoint_name,
            (label_point[0] + 8, label_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            keypoint_name,
            (label_point[0] + 8, label_point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def format_solve_frame_counts(quality: CalibrationQualityReport) -> str:
    """Format input/accepted/rejected frame counts from a solve quality report.

    Args:
        quality: Post-solve calibration quality report.

    Returns:
        Human-readable frame count summary line.
    """
    return (
        f"frames input/accepted/rejected: "
        f"{quality.input_frame_count}/{quality.accepted_frame_count}/{quality.rejected_frame_count}"
    )


def format_assignment_rejection_cause(cause_count) -> str:
    """Short HUD line for one grouped assignment-rejection cause and count.

    Args:
        cause_count: Grouped rejection statistics for one cause.

    Returns:
        Compact HUD line with reason, optional pair, and count.
    """
    pair_text = ""
    if cause_count.marker_pair is not None:
        pair_text = f" pair=({cause_count.marker_pair[0]},{cause_count.marker_pair[1]})"
    return f"{cause_count.reason}{pair_text} x{cause_count.count}"


def format_assignment_rejection_summary(
    summary: AssignmentRejectionSummary,
    *,
    max_lines: int = DEFAULT_ASSIGNMENT_REJECTION_SUMMARY_LINES,
) -> list[str]:
    """Return top assignment-rejection causes for the HUD.

    Args:
        summary: Assignment-rejection rollup from a quality report.
        max_lines: Maximum number of cause lines to include.

    Returns:
        Up to ``max_lines`` formatted cause lines; empty when nothing was rejected.
    """
    if summary.total_rejected == 0:
        return []
    return [
        format_assignment_rejection_cause(cause)
        for cause in summary.top_causes[:max_lines]
    ]


def format_pair_readiness_edge(edge: PairReadinessEdge) -> str:
    """Format one live pair-readiness edge for the capture HUD.

    Args:
        edge: Pair-readiness edge from the background worker snapshot.

    Returns:
        Compact HUD line with pair IDs, status, and raw co-visible frame count.
    """
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
    """Summarize connected vs missing markers, truncating long ID lists.

    Args:
        connected_ids: Marker IDs connected to the reference in the readiness graph.
        missing_ids: Expected marker IDs still missing from the graph.
        max_listed: Maximum IDs to show before abbreviating with ``+N``.

    Returns:
        Single-line connected/missing summary.
    """
    def compact(marker_ids: list[int]) -> str:
        """Render a marker ID list, abbreviating when longer than ``max_listed``.

        Args:
            marker_ids: Sorted marker IDs to display.

        Returns:
            Bracketed ID list, or head IDs plus ``+N`` when truncated.
        """
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
    """Label which captured samples the latest readiness snapshot represents.

    Args:
        current_sample_count: Number of captured observations so far.
        represented_sample_count: Sample count the readiness snapshot was computed from.
        is_computing: Whether the readiness worker is still computing.

    Returns:
        Readiness snapshot label for the HUD, with optional ``(computing...)`` suffix.
    """
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
    reference_marker_id: int | None,
    max_pair_lines: int = DEFAULT_PAIR_READINESS_HUD_LINES,
) -> list[str]:
    """Build live pair-readiness HUD lines from the background worker snapshot.

    Inserts per-pair status lines before the control hint; may append ``...`` when
    ``max_pair_lines`` truncates the pair list.

    Args:
        expected_ids: Full list of expected marker IDs.
        visible_ids: Marker IDs detected in the current frame.
        current_sample_count: Number of captured observations.
        readiness_view: Latest pair-readiness snapshot from the worker.
        reference_marker_id: Reference marker for connectivity reporting.
        max_pair_lines: Maximum per-pair lines before truncation.

    Returns:
        Ordered HUD text lines for the live capture overlay.
    """
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
            "graph: "
            f"{len(connected_ids)}/{len(expected_ids)} connected from ref "
            f"{reference_marker_id if reference_marker_id is not None else 'auto'}"
        ),
        (
            f"pairs: {status_counts['pass']} pass, {status_counts['weak']} weak, "
            f"{status_counts['fail']} fail ({len(diagnostics.pairs)} observed)"
        ),
        "C=capture  S=solve  Q=quit",
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
    """Alias for :func:`build_pair_readiness_hud_lines` used by the capture loop.

    Args:
        expected_ids: Full list of expected marker IDs.
        visible_ids: Marker IDs detected in the current frame.
        current_sample_count: Number of captured observations.
        readiness_view: Latest pair-readiness snapshot from the worker.
        reference_marker_id: Reference marker for connectivity reporting.
        max_pair_lines: Maximum per-pair lines before truncation.

    Returns:
        Ordered HUD text lines for the live capture overlay.
    """
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
    """Draw the calibration HUD, optionally appending the last solve's frame stats.

    Args:
        frame: Preview image (HUD drawn in place).
        hud_lines: Base HUD lines from pair-readiness and capture status.
        last_solve_quality: Optional quality report from the most recent solve attempt.
    """
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
    """Print refusal reason and quality diagnostics to stdout.

    Args:
        result: Calibration result with ``layout`` unset or a failure reason.
    """
    print(f"Calibration refused: {result.failure_reason}")
    quality = result.quality
    if quality is None:
        return
    print(f"  {format_solve_frame_counts(quality)}")
    for line in format_omitted_marker_lines(result.omitted_markers):
        print(f"  {line}")
    for line in format_quality_diagnostics_lines(quality):
        print(f"  {line}")


def print_success(result: CalibrationResult, output: Path) -> None:
    """Print outcome-specific warnings, save path, and quality diagnostics.

    Args:
        result: Successful or best-effort calibration result.
        output: Marker model path that was written.
    """
    if result.outcome == "partial":
        print("WARNING: Saved partial marker model (some requested markers were omitted).")
        for line in format_omitted_marker_lines(result.omitted_markers):
            print(f"  {line}")
    elif result.outcome == "provisional":
        if result.failed_refinement_stage is not None:
            print(
                "WARNING: Saved provisional marker model from optimization checkpoint "
                f"'{result.selected_checkpoint_stage}' after '{result.failed_refinement_stage}' failed."
            )
        else:
            print("WARNING: Saved provisional marker model (strict quality gates failed).")
        for gate in result.failed_quality_gates:
            print(f"  failed gate: {gate}")
    print(f"Saved marker model: {output}")
    quality = result.quality
    if quality is None:
        return
    print(f"  {format_solve_frame_counts(quality)}")
    if result.outcome != "partial":
        for line in format_omitted_marker_lines(result.omitted_markers):
            print(f"  {line}")
    for line in format_quality_diagnostics_lines(quality):
        print(f"  {line}")


def write_calibration_diagnostics_if_requested(
    diagnostics_output: Path | None,
    result: CalibrationResult,
    *,
    benchmark: Mapping[str, Any] | None = None,
) -> None:
    """Write ``--diagnostics-output`` JSON when requested and quality is available.

    Prints the output path on success; diagnostics write errors go to stderr.

    Args:
        diagnostics_output: Optional diagnostics JSON path from CLI flags.
        result: Calibration result containing the quality report.
        benchmark: Optional benchmark timing metadata for diagnostics export.

    Notes:
        No-op when ``diagnostics_output`` is ``None`` or ``result.quality`` is missing.
    """
    if diagnostics_output is None or result.quality is None:
        return
    if benchmark is None:
        path = save_calibration_diagnostics(diagnostics_output, result)
    else:
        path = save_calibration_diagnostics(diagnostics_output, result, benchmark=benchmark)
    print(f"Wrote calibration diagnostics: {path}")


def update_object_model_from_layout(object_model_path: Path, layout: MarkerLayout) -> None:
    """Apply solved marker layout keypoints to an object-model JSON file in place.

    Args:
        object_model_path: Object-model JSON to update.
        layout: Solved marker layout with keypoint source positions.
    """
    model, document = load_object_model_document(object_model_path)
    updated = apply_keypoint_sources_from_layout(model, document, layout)
    save_object_model_keypoints(object_model_path, updated, document)
    print(f"Updated object model: {object_model_path}")


def format_capture_mode(auto: bool, sample_rate_hz: float) -> str:
    """Describe manual vs automatic capture for HUD and startup logs.

    Args:
        auto: Whether automatic periodic capture is enabled.
        sample_rate_hz: Automatic capture rate when ``auto`` is true.

    Returns:
        Short capture-mode description string.
    """
    if auto:
        return f"automatic at {sample_rate_hz:g} Hz"
    return "manual (press C)"


def append_capture_observation(
    observations: list[FrameObservation],
    visible: dict[int, np.ndarray],
    *,
    readiness_worker: LivePairReadinessWorker | None = None,
) -> FrameObservation:
    """Append a captured frame observation and notify the readiness worker.

    Corners are copied so later in-place frame edits cannot mutate stored samples.

    Args:
        observations: Growing list of captured frame observations.
        visible: Detected marker corners for this capture.
        readiness_worker: Optional worker to recompute pair readiness.

    Returns:
        The newly appended ``FrameObservation``.
    """
    observation = FrameObservation(
        frame_id=len(observations),
        markers={marker_id: corners.copy() for marker_id, corners in visible.items()},
    )
    observations.append(observation)
    if readiness_worker is not None:
        readiness_worker.submit(observations)
    return observation


def keypoint_sources_for_solve(args: argparse.Namespace) -> dict[str, tuple[int, str, float]]:
    """Return object-model keypoint sources used for automatic reference selection."""
    if getattr(args, "from_config", False) is True:
        return dict(args.recipe.keypoint_sources)
    return dict(getattr(args, "keypoint_sources", {}) or {})


def solve_calibration_from_observations(
    observations: list[FrameObservation],
    args: argparse.Namespace,
    *,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    expected_ids: list[int],
    marker_sizes_m: dict[int, float],
    settings: CalibrationSettings,
    anchor_ids: tuple[int, ...] | None,
    anchor_stop_after_expansion: bool,
    best_effort: bool,
    partial_output: bool,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
) -> CalibrationResult:
    """Run marker-layout calibration on captured observations via library solver.

    Args:
        observations: Captured frame observations to solve from.
        args: Parsed CLI namespace (reference ID, marker size, flags).
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Distortion coefficients.
        expected_ids: Expected marker IDs for the layout.
        marker_sizes_m: Per-marker physical edge lengths.
        settings: Quality gates and pair-inlier thresholds.
        anchor_ids: Optional anchor-core marker subset.
        anchor_stop_after_expansion: Stop after hierarchical expansion without full BA.
        best_effort: Allow provisional output when strict gates fail.
        partial_output: Allow reference-connected partial layout output.
        solve_diagnostics: Optional container for per-stage solve timings.

    Returns:
        ``CalibrationResult`` from ``calibrate_marker_layout``.
    """
    return calibrate_marker_layout(
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
        best_effort=best_effort,
        partial_output=partial_output,
        solve_diagnostics=solve_diagnostics,
        keypoint_sources=keypoint_sources_for_solve(args),
    )


def apply_calibration_result(
    args: argparse.Namespace,
    result: CalibrationResult,
    *,
    benchmark: Mapping[str, Any] | None = None,
) -> bool:
    """Write outputs for a calibration result.

    Args:
        args: Parsed CLI namespace with output paths and flags.
        result: Calibration solve result.
        benchmark: Optional benchmark metadata for diagnostics export.

    Returns:
        True when the marker model was saved; False on refusal.

    Raises:
        RuntimeError: Marker model saved but object-model update failed.
    """
    if result.layout is None:
        print_refusal(result)
        try:
            write_calibration_diagnostics_if_requested(
                args.diagnostics_output,
                result,
                benchmark=benchmark,
            )
        except RuntimeError as error:
            print(error, file=sys.stderr)
        return False

    if getattr(args, "from_config", False) is True:
        return _publish_config_mode_results(args, result, benchmark=benchmark)

    save_marker_model(args.output, result.layout)
    print_success(result, args.output)
    if isinstance(args.object_model, Path):
        try:
            update_object_model_from_layout(args.object_model, result.layout)
        except (ValueError, OSError) as error:
            raise RuntimeError(
                f"Marker model saved to {args.output}, but object model "
                f"update failed: {error}"
            ) from error
    try:
        write_calibration_diagnostics_if_requested(
            args.diagnostics_output,
            result,
            benchmark=benchmark,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
    return True


def _publish_config_mode_results(
    args: argparse.Namespace,
    result: CalibrationResult,
    *,
    benchmark: Mapping[str, Any] | None = None,
) -> bool:
    """Publish paired model outputs for config mode after validation."""
    recipe = args.recipe
    missing_markers = missing_source_marker_ids(result.layout, recipe.keypoint_sources)
    if missing_markers:
        print(
            "Calibration solved but publication refused: keypoint-source markers "
            f"missing from layout: {list(missing_markers)}"
        )
        for line in format_omitted_marker_lines(result.omitted_markers):
            print(f"  {line}")
        try:
            write_calibration_diagnostics_if_requested(
                args.diagnostics_output,
                result,
                benchmark=benchmark,
            )
        except RuntimeError as error:
            print(error, file=sys.stderr)
        return False

    try:
        object_document = build_object_model_document_from_layout(
            result.layout,
            recipe.keypoint_sources,
            recipe.skeleton,
        )
    except ValueError as error:
        print(f"Calibration solved but publication refused: {error}")
        try:
            write_calibration_diagnostics_if_requested(
                args.diagnostics_output,
                result,
                benchmark=benchmark,
            )
        except RuntimeError as diagnostics_error:
            print(diagnostics_error, file=sys.stderr)
        return False

    save_marker_model(recipe.paths.marker_model_path, result.layout)
    save_object_model_document(recipe.paths.object_model_path, object_document)
    print_success(result, recipe.paths.marker_model_path)
    print(f"Saved object model: {recipe.paths.object_model_path}")
    try:
        write_calibration_diagnostics_if_requested(
            args.diagnostics_output,
            result,
            benchmark=benchmark,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
    return True


def _frame_sharpness_score(frame: np.ndarray) -> float:
    """Score frame sharpness via downsampled grayscale Laplacian variance.

    Args:
        frame: BGR or grayscale image frame.

    Returns:
        Relative sharpness score (higher is sharper).
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(small, cv2.CV_64F)
    return float(laplacian.var())


def _benchmark_window_index(video_time: float, sample_interval: float) -> int:
    """Map video time to a sample window index.

    Uses a tiny epsilon so frames exactly on a window boundary belong to the
    following window, matching uniform sampling semantics.

    Args:
        video_time: Elapsed video time in seconds.
        sample_interval: Window width in seconds (``1 / sample_rate_hz``).

    Returns:
        Zero-based window index for ``video_time``.
    """
    return int(math.floor(video_time / sample_interval + 1e-12))


def _benchmark_detect_on_frame(
    detector: cv2.aruco.ArucoDetector,
    frame: np.ndarray,
    expected_id_set: set[int],
) -> tuple[dict[int, np.ndarray], int]:
    """Detect expected markers on one frame and time the detection pass.

    Args:
        detector: OpenCV ArUco/AprilTag detector.
        frame: Image frame to detect on.
        expected_id_set: Marker IDs to retain.

    Returns:
        Tuple of detected corners map and detection elapsed time in nanoseconds.
    """
    detection_start_ns = time.perf_counter_ns()
    visible = detect_expected_markers(detector, frame, expected_id_set)
    return visible, time.perf_counter_ns() - detection_start_ns


def _record_benchmark_detection(
    visible: dict[int, np.ndarray],
    observations: list[FrameObservation],
    *,
    detector_invocations: int,
    frames_with_expected_markers: int,
    covisible_frames: int,
    detected_markers: int,
) -> tuple[int, int, int, int]:
    """Update benchmark counters and append an observation when >=2 markers co-visible.

    Args:
        visible: Detected marker corners for this detection invocation.
        observations: Observation list to append to when co-visible.
        detector_invocations: Running detector call count.
        frames_with_expected_markers: Frames with at least one expected marker.
        covisible_frames: Frames with two or more co-visible markers.
        detected_markers: Total marker detections across invocations.

    Returns:
        Updated ``(detector_invocations, frames_with_expected_markers,
        covisible_frames, detected_markers)`` counters.
    """
    detector_invocations += 1
    detected_markers += len(visible)
    if visible:
        frames_with_expected_markers += 1
    if len(visible) >= 2:
        covisible_frames += 1
        append_capture_observation(observations, visible)
    return (
        detector_invocations,
        frames_with_expected_markers,
        covisible_frames,
        detected_markers,
    )


def _timing_seconds_from_ns(duration_ns: int) -> float:
    """Convert a non-negative nanosecond duration to seconds for benchmark reporting.

    Args:
        duration_ns: Elapsed time in nanoseconds.

    Returns:
        Duration in seconds; non-finite input yields ``0.0``.
    """
    seconds = max(0.0, duration_ns / 1_000_000_000)
    if not math.isfinite(seconds):
        return 0.0
    return seconds


def _benchmark_environment() -> dict[str, str]:
    """Collect runtime library versions embedded in benchmark diagnostics JSON.

    Returns:
        Map of library and platform version strings.
    """
    import scipy

    return {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }


def _build_benchmark_payload(
    *,
    source_path: Path,
    reported_fps: float,
    reported_frame_count: int,
    image_size: tuple[int, int],
    decoded_frames: int,
    detector_invocations: int,
    frames_skipped_before_detection: int,
    frames_with_expected_markers: int,
    covisible_frames: int,
    sampled_observations: int,
    detected_markers: int,
    open_source_ns: int,
    decode_ns: int,
    detection_ns: int,
    ingest_total_ns: int,
    calibration_solve_ns: int,
    total_through_solve_ns: int,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
    frame_selection: str = DEFAULT_BENCHMARK_FRAME_SELECTION,
    sharpness_scoring_ns: int | None = None,
) -> dict[str, Any]:
    """Assemble benchmark timing, throughput, and environment metadata for diagnostics export.

    Args:
        source_path: Video file path used for the benchmark.
        reported_fps: FPS from video metadata.
        reported_frame_count: Frame count from video metadata.
        image_size: ``(width, height)`` of decoded frames.
        decoded_frames: Total frames decoded from the video.
        detector_invocations: Number of detection passes executed.
        frames_skipped_before_detection: Frames decoded but not sent to the detector.
        frames_with_expected_markers: Frames where at least one expected marker was found.
        covisible_frames: Frames with two or more co-visible expected markers.
        sampled_observations: Observations appended for calibration.
        detected_markers: Total marker detections across invocations.
        open_source_ns: Time to open the video source.
        decode_ns: Total frame decode time.
        detection_ns: Total AprilTag detection time.
        ingest_total_ns: End-to-end ingest time through last decoded frame.
        calibration_solve_ns: Marker-layout solve time.
        total_through_solve_ns: Total benchmark time through solve completion.
        solve_diagnostics: Optional per-stage solve timing container.
        frame_selection: ``uniform`` or ``sharpest`` frame-selection mode.
        sharpness_scoring_ns: Total sharpness scoring time for ``sharpest`` mode.

    Returns:
        Benchmark metadata dict for ``--diagnostics-output`` JSON export.
    """
    ingest_seconds = _timing_seconds_from_ns(ingest_total_ns)
    detection_seconds = _timing_seconds_from_ns(detection_ns)
    pipeline_fps = decoded_frames / ingest_seconds if ingest_seconds > 0.0 else 0.0
    detection_fps = (
        detector_invocations / detection_seconds if detection_seconds > 0.0 else 0.0
    )
    if not math.isfinite(pipeline_fps):
        pipeline_fps = 0.0
    if not math.isfinite(detection_fps):
        detection_fps = 0.0
    timing_seconds: dict[str, Any] = {
        "open_source": _timing_seconds_from_ns(open_source_ns),
        "decode": _timing_seconds_from_ns(decode_ns),
        "detection": _timing_seconds_from_ns(detection_ns),
        "ingest_total": ingest_seconds,
        "calibration_solve": _timing_seconds_from_ns(calibration_solve_ns),
        "total_through_solve": _timing_seconds_from_ns(total_through_solve_ns),
    }
    if solve_diagnostics is not None:
        timing_seconds["solve_stages"] = dict(solve_diagnostics.solve_stages_seconds)
    if sharpness_scoring_ns is not None:
        timing_seconds["sharpness_scoring"] = _timing_seconds_from_ns(sharpness_scoring_ns)
    payload: dict[str, Any] = {
        "frame_selection": frame_selection,
        "source": {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "reported_fps": reported_fps,
            "reported_frame_count": reported_frame_count,
            "image_size": [image_size[0], image_size[1]],
        },
        "counts": {
            "decoded_frames": decoded_frames,
            "detector_invocations": detector_invocations,
            "frames_skipped_before_detection": frames_skipped_before_detection,
            "frames_with_expected_markers": frames_with_expected_markers,
            "covisible_frames": covisible_frames,
            "sampled_observations": sampled_observations,
            "detected_markers": detected_markers,
        },
        "timing_seconds": timing_seconds,
        "throughput": {
            "pipeline_frames_per_second": pipeline_fps,
            "detector_invocations_per_second": detection_fps,
        },
        "environment": _benchmark_environment(),
    }
    if solve_diagnostics is not None:
        payload["optimizer_runs"] = list(solve_diagnostics.optimizer_runs)
    return payload


def run_benchmark(args: argparse.Namespace) -> bool:
    """Benchmark video ingest and calibration on a headless video pass.

    Args:
        args: Parsed CLI namespace; requires video ``--source`` and ``--benchmark``.

    Returns:
        True when a marker model was saved.

    Raises:
        RuntimeError: Invalid benchmark configuration or video metadata.
    """
    if not isinstance(args.source, Path):
        raise RuntimeError("--benchmark requires a video file --source, not a camera index.")

    total_start_ns = time.perf_counter_ns()
    (
        expected_ids,
        marker_sizes_m,
        settings,
        anchor_ids,
        anchor_stop_after_expansion,
        best_effort,
        partial_output,
    ) = validate_args(args)
    expected_id_set = set(expected_ids)

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    detector = build_apriltag_detector(args.dictionary, args.detection_sensitivity)

    frame_selection = getattr(
        args, "benchmark_frame_selection", DEFAULT_BENCHMARK_FRAME_SELECTION
    )

    print(f"Expected marker IDs: {expected_ids}")
    print(f"Reference marker ID: {args.reference_marker_id if args.reference_marker_id is not None else 'auto'}")
    if frame_selection == BENCHMARK_FRAME_SELECTION_SHARPEST:
        print(
            "Benchmark capture: sharpest frame per "
            f"{1.0 / args.sample_rate_hz:g}s window "
            f"({args.sample_rate_hz:g} Hz; every frame decoded for scoring)"
        )
    else:
        print(f"Benchmark capture: automatic at {args.sample_rate_hz:g} Hz (video time)")
    print(f"Using calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")
    print(format_frame_source(args.source))

    open_start_ns = time.perf_counter_ns()
    capture = open_frame_source(args.source, width=width, height=height)
    open_source_ns = time.perf_counter_ns() - open_start_ns

    try:
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(reported_fps) or reported_fps <= 0.0:
            raise RuntimeError(
                f"Video reported FPS must be finite and positive; got {reported_fps!r}."
            )
        raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_frame_count = (
            int(raw_frame_count)
            if math.isfinite(raw_frame_count) and raw_frame_count >= 0.0
            else 0
        )

        observations: list[FrameObservation] = []
        sample_interval = 1.0 / args.sample_rate_hz
        frame_index = 0
        decoded_frames = 0
        detector_invocations = 0
        frames_skipped_before_detection = 0
        frames_with_expected_markers = 0
        covisible_frames = 0
        detected_markers = 0
        decode_ns = 0
        detection_ns = 0
        sharpness_scoring_ns = 0

        if frame_selection == BENCHMARK_FRAME_SELECTION_SHARPEST:
            current_window: int | None = None
            best_frame: np.ndarray | None = None
            best_sharpness = float("-inf")
            window_frame_count = 0

            def flush_sharpest_window() -> None:
                """Detect on the sharpest frame in the closed window and reset window state.

                Updates benchmark detection counters and appends an observation when at
                least two expected markers are co-visible. Resets per-window best-frame
                tracking.
                """
                nonlocal best_frame, best_sharpness, window_frame_count
                nonlocal detector_invocations, frames_skipped_before_detection
                nonlocal frames_with_expected_markers, covisible_frames, detected_markers
                nonlocal detection_ns
                if best_frame is None:
                    return
                visible, elapsed_ns = _benchmark_detect_on_frame(
                    detector, best_frame, expected_id_set
                )
                detection_ns += elapsed_ns
                (
                    detector_invocations,
                    frames_with_expected_markers,
                    covisible_frames,
                    detected_markers,
                ) = _record_benchmark_detection(
                    visible,
                    observations,
                    detector_invocations=detector_invocations,
                    frames_with_expected_markers=frames_with_expected_markers,
                    covisible_frames=covisible_frames,
                    detected_markers=detected_markers,
                )
                frames_skipped_before_detection += window_frame_count - 1
                best_frame = None
                best_sharpness = float("-inf")
                window_frame_count = 0

            while True:
                decode_start_ns = time.perf_counter_ns()
                ok, frame = read_frame(capture, args.source, loop_on_eof=False)
                decode_ns += time.perf_counter_ns() - decode_start_ns
                if not ok or frame is None:
                    break

                decoded_frames += 1
                require_frame_size(
                    frame.shape[1], frame.shape[0], width, height, args.calibration
                )

                video_time = frame_index / reported_fps
                window_index = _benchmark_window_index(video_time, sample_interval)
                if current_window is None:
                    current_window = window_index
                elif window_index != current_window:
                    flush_sharpest_window()
                    current_window = window_index

                score_start_ns = time.perf_counter_ns()
                sharpness = _frame_sharpness_score(frame)
                sharpness_scoring_ns += time.perf_counter_ns() - score_start_ns

                window_frame_count += 1
                # Earliest frame wins ties: update only on strictly higher sharpness.
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = frame.copy()

                frame_index += 1

            flush_sharpest_window()
        else:
            next_sample_time = 0.0
            while True:
                decode_start_ns = time.perf_counter_ns()
                ok, frame = read_frame(capture, args.source, loop_on_eof=False)
                decode_ns += time.perf_counter_ns() - decode_start_ns
                if not ok or frame is None:
                    break

                decoded_frames += 1
                require_frame_size(
                    frame.shape[1], frame.shape[0], width, height, args.calibration
                )

                video_time = frame_index / reported_fps
                if video_time >= next_sample_time:
                    visible, elapsed_ns = _benchmark_detect_on_frame(
                        detector, frame, expected_id_set
                    )
                    detection_ns += elapsed_ns
                    (
                        detector_invocations,
                        frames_with_expected_markers,
                        covisible_frames,
                        detected_markers,
                    ) = _record_benchmark_detection(
                        visible,
                        observations,
                        detector_invocations=detector_invocations,
                        frames_with_expected_markers=frames_with_expected_markers,
                        covisible_frames=covisible_frames,
                        detected_markers=detected_markers,
                    )
                    next_sample_time = video_time + sample_interval
                else:
                    frames_skipped_before_detection += 1

                frame_index += 1

        ingest_total_ns = time.perf_counter_ns() - total_start_ns

        solve_start_ns = time.perf_counter_ns()
        solve_diagnostics = CalibrationSolveDiagnostics()
        result = solve_calibration_from_observations(
            observations,
            args,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            expected_ids=expected_ids,
            marker_sizes_m=marker_sizes_m,
            settings=settings,
            anchor_ids=anchor_ids,
            anchor_stop_after_expansion=anchor_stop_after_expansion,
            best_effort=best_effort,
            partial_output=partial_output,
            solve_diagnostics=solve_diagnostics,
        )
        _sync_selected_reference_marker_id(args, result)
        calibration_solve_ns = time.perf_counter_ns() - solve_start_ns
        total_through_solve_ns = time.perf_counter_ns() - total_start_ns

        benchmark_payload = _build_benchmark_payload(
            source_path=args.source,
            reported_fps=reported_fps,
            reported_frame_count=reported_frame_count,
            image_size=(width, height),
            decoded_frames=decoded_frames,
            detector_invocations=detector_invocations,
            frames_skipped_before_detection=frames_skipped_before_detection,
            frames_with_expected_markers=frames_with_expected_markers,
            covisible_frames=covisible_frames,
            sampled_observations=len(observations),
            detected_markers=detected_markers,
            open_source_ns=open_source_ns,
            decode_ns=decode_ns,
            detection_ns=detection_ns,
            ingest_total_ns=ingest_total_ns,
            calibration_solve_ns=calibration_solve_ns,
            total_through_solve_ns=total_through_solve_ns,
            solve_diagnostics=solve_diagnostics,
            frame_selection=frame_selection,
            sharpness_scoring_ns=(
                sharpness_scoring_ns
                if frame_selection == BENCHMARK_FRAME_SELECTION_SHARPEST
                else None
            ),
        )
        return apply_calibration_result(args, result, benchmark=benchmark_payload)
    finally:
        capture.release()


def run_capture(args: argparse.Namespace) -> bool:
    """Capture live samples from camera or looping video; solve on ``S``.

    Args:
        args: Parsed CLI namespace for capture and solve flags.

    Returns:
        True when a marker model was saved before quit.

    Raises:
        RuntimeError: Frame read failure from the configured source.
    """
    (
        expected_ids,
        marker_sizes_m,
        settings,
        anchor_ids,
        anchor_stop_after_expansion,
        best_effort,
        partial_output,
    ) = validate_args(args)
    expected_id_set = set(expected_ids)

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    detector = build_apriltag_detector(args.dictionary, args.detection_sensitivity)

    print(f"Expected marker IDs: {expected_ids}")
    print(f"Reference marker ID: {args.reference_marker_id if args.reference_marker_id is not None else 'auto'}")
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
    print(f"Capture mode: {format_capture_mode(args.auto, args.sample_rate_hz)}")
    print(f"Using calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    keypoint_sources: dict[str, tuple[int, str, float]] | None = None
    if args.overlay_object_model is True:
        if getattr(args, "from_config", False) is True:
            keypoint_sources = dict(args.recipe.keypoint_sources)
        else:
            _, object_model_document = load_object_model_document(args.object_model)
            keypoint_sources = parse_keypoint_sources(object_model_document)

    capture = open_frame_source(args.source, width=width, height=height)
    source_label = format_frame_source(args.source)
    readiness_worker = LivePairReadinessWorker(
        compute_fn=compute_live_pair_readiness,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        expected_marker_ids=expected_ids,
        reference_marker_id=args.reference_marker_id,
        settings=settings,
    )
    try:
        if isinstance(args.source, int):
            print(f"{source_label}: target {width}x{height}")
        else:
            print(source_label)
        print("Press C to capture, S to solve, Q to quit.")

        observations: list[FrameObservation] = []
        sample_interval = 1.0 / args.sample_rate_hz if args.auto else None
        next_sample_time = time.monotonic()
        status_line: str | None = None
        last_solve_quality: CalibrationQualityReport | None = None

        while True:
            ok, frame = read_frame(capture, args.source)
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read a frame from {source_label}.")
            require_frame_size(frame.shape[1], frame.shape[0], width, height, args.calibration)

            visible = detect_expected_markers(detector, frame, expected_id_set)
            preview = frame.copy()
            draw_detection_outlines(preview, visible, args.reference_marker_id)
            if keypoint_sources is not None:
                draw_keypoint_source_overlays(
                    preview,
                    visible,
                    keypoint_sources,
                    marker_sizes_m,
                    camera_matrix,
                    dist_coeffs,
                )

            now = time.monotonic()
            manual_capture = False
            auto_due = False
            readiness_view = readiness_worker.poll(len(observations))
            hud_lines = build_pair_readiness_hud_lines_from_diagnostics(
                expected_ids=expected_ids,
                visible_ids=sorted(visible),
                current_sample_count=len(observations),
                readiness_view=readiness_view,
                reference_marker_id=args.reference_marker_id,
            )
            hud_lines.insert(0, f"capture: {format_capture_mode(args.auto, args.sample_rate_hz)}")
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
                manual_capture = True
            if args.auto and sample_interval is not None and now >= next_sample_time:
                auto_due = True
            if len(visible) >= 2 and (manual_capture or auto_due):
                append_capture_observation(
                    observations,
                    visible,
                    readiness_worker=readiness_worker,
                )
                if auto_due:
                    next_sample_time = now + sample_interval
                status_line = (
                    f"captured sample {len(observations)}: "
                    f"markers {sorted(visible)}"
                )
                continue
            if manual_capture:
                status_line = "capture skipped: need at least 2 expected markers"
                continue
            if key in (ord("s"), ord("S")):
                result = solve_calibration_from_observations(
                    observations,
                    args,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    expected_ids=expected_ids,
                    marker_sizes_m=marker_sizes_m,
                    settings=settings,
                    anchor_ids=anchor_ids,
                    anchor_stop_after_expansion=anchor_stop_after_expansion,
                    best_effort=best_effort,
                    partial_output=partial_output,
                )
                _sync_selected_reference_marker_id(args, result)
                if apply_calibration_result(args, result):
                    return True
                last_solve_quality = result.quality
                status_line = f"refused: {result.failure_reason}"
                continue
    finally:
        readiness_worker.shutdown(join_timeout=0.0)
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    """CLI entry point: run headless benchmark or interactive capture mode.

    Raises:
        SystemExit: Exit code 1 after printing ``RuntimeError`` to stderr.
    """
    try:
        args = parse_args()
        if getattr(args, "from_config", False) is not True:
            warnings.warn(
                "Legacy calibration CLI flags are deprecated; use "
                "`--config <workspace>/config.json`.",
                FutureWarning,
                stacklevel=1,
            )
        if _is_benchmark_mode(args):
            run_benchmark(args)
        else:
            run_capture(args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
