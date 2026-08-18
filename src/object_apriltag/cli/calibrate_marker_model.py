"""Config-driven marker layout calibration CLI."""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_apriltag.apriltag import build_apriltag_detector
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.frame_source import format_frame_source, open_frame_source, read_frame
from object_apriltag.layout import save_marker_model
from object_apriltag.object_model_edit import (
    build_object_model_document_from_layout,
    keypoint_sources_for_layout,
    missing_source_marker_ids,
    save_object_model_document,
    skeleton_for_keypoint_names,
)
from object_apriltag.cli.calibration_diagnostics import (
    format_omitted_marker_lines,
    format_quality_diagnostics_lines,
    save_calibration_diagnostics,
)
from object_apriltag.marker_layout_calibration.recipe import (
    BENCHMARK_FRAME_SELECTION_SHARPEST,
    BenchmarkExecution,
    CalibrationRecipe,
    load_calibration_recipe,
)
from object_apriltag.marker_layout_calibration import (
    AssignmentRejectionSummary,
    CalibrationQualityReport,
    CalibrationResult,
    CalibrationSolveDiagnostics,
    FrameObservation,
    calibrate_marker_layout,
)

DEFAULT_ASSIGNMENT_REJECTION_SUMMARY_LINES = 3


def parse_args() -> tuple[Path, bool]:
    """Parse ``--config`` and optional ``--force``.

    Returns:
        Resolved config path and overwrite flag.
    """
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
        help="Overwrite existing marker_model.json, object_model.json, and diagnostics.json outputs.",
    )
    args = parser.parse_args()
    return args.config, args.force


def validate_recipe(recipe: CalibrationRecipe, *, force: bool = False) -> None:
    """Validate recipe inputs and refuse when outputs exist without ``--force``.

    Args:
        recipe: Parsed calibration recipe.
        force: Whether to allow overwriting existing workspace outputs.

    Raises:
        RuntimeError: Missing intrinsics, camera source, or existing outputs without force.
    """
    if not recipe.intrinsics_path.is_file():
        raise RuntimeError(
            f"Calibration file not found: {recipe.intrinsics_path}\n"
            "Run `uv run object-charuco` first."
        )
    if isinstance(recipe.source, int):
        raise RuntimeError("benchmark execution requires a video file source, not a camera index.")
    if not isinstance(recipe.execution, BenchmarkExecution):
        raise RuntimeError("only benchmark execution is supported.")

    existing_outputs = [
        path
        for path in (
            recipe.paths.marker_model_path,
            recipe.paths.object_model_path,
            recipe.paths.diagnostics_path,
        )
        if path.exists()
    ]
    if existing_outputs and not force:
        joined = ", ".join(str(path) for path in existing_outputs)
        raise RuntimeError(
            f"Output already exists: {joined}. Pass --force to overwrite."
        )


def require_frame_size(
    frame_width: int,
    frame_height: int,
    calibration_width: int,
    calibration_height: int,
    calibration_path: Path,
) -> None:
    """Reject frames whose resolution differs from the intrinsics calibration image size."""
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
    """Detect AprilTags in ``frame`` and return corners only for ``expected_ids``."""
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


def format_solve_frame_counts(quality: CalibrationQualityReport) -> str:
    """Format input/accepted/rejected frame counts from a solve quality report."""
    return (
        f"frames input/accepted/rejected: "
        f"{quality.input_frame_count}/{quality.accepted_frame_count}/{quality.rejected_frame_count}"
    )


def format_assignment_rejection_cause(cause_count) -> str:
    """Short line for one grouped assignment-rejection cause and count."""
    pair_text = ""
    if cause_count.marker_pair is not None:
        pair_text = f" pair=({cause_count.marker_pair[0]},{cause_count.marker_pair[1]})"
    return f"{cause_count.reason}{pair_text} x{cause_count.count}"


def format_assignment_rejection_summary(
    summary: AssignmentRejectionSummary,
    *,
    max_lines: int = DEFAULT_ASSIGNMENT_REJECTION_SUMMARY_LINES,
) -> list[str]:
    """Return top assignment-rejection causes for console output."""
    if summary.total_rejected == 0:
        return []
    return [
        format_assignment_rejection_cause(cause)
        for cause in summary.top_causes[:max_lines]
    ]


def print_refusal(result: CalibrationResult) -> None:
    """Print refusal reason and quality diagnostics to stdout."""
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
    """Print outcome-specific warnings, save path, and quality diagnostics."""
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
    """Write diagnostics JSON when a path is configured and quality is available."""
    if diagnostics_output is None or result.quality is None:
        return
    if benchmark is None:
        path = save_calibration_diagnostics(diagnostics_output, result)
    else:
        path = save_calibration_diagnostics(diagnostics_output, result, benchmark=benchmark)
    print(f"Wrote calibration diagnostics: {path}")


def append_capture_observation(
    observations: list[FrameObservation],
    visible: dict[int, np.ndarray],
) -> FrameObservation:
    """Append a captured frame observation with copied marker corners."""
    observation = FrameObservation(
        frame_id=len(observations),
        markers={marker_id: corners.copy() for marker_id, corners in visible.items()},
    )
    observations.append(observation)
    return observation


def solve_calibration_from_observations(
    observations: list[FrameObservation],
    recipe: CalibrationRecipe,
    *,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    solve_diagnostics: CalibrationSolveDiagnostics | None = None,
) -> CalibrationResult:
    """Run marker-layout calibration on captured observations."""
    return calibrate_marker_layout(
        observations,
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=list(recipe.expected_marker_ids),
        reference_marker_id=None,
        marker_size_m=recipe.default_marker_size_m,
        settings=recipe.settings,
        marker_sizes_m=dict(recipe.marker_sizes_m),
        solve_diagnostics=solve_diagnostics,
        keypoint_sources=dict(recipe.keypoint_sources),
    )


def apply_calibration_result(
    recipe: CalibrationRecipe,
    result: CalibrationResult,
    *,
    benchmark: Mapping[str, Any] | None = None,
) -> bool:
    """Write paired workspace outputs for a calibration result.

    Returns:
        True when marker and object models were saved; False on refusal.
    """
    if result.layout is None:
        print_refusal(result)
        try:
            write_calibration_diagnostics_if_requested(
                recipe.paths.diagnostics_path,
                result,
                benchmark=benchmark,
            )
        except RuntimeError as error:
            print(error, file=sys.stderr)
        return False

    sources = dict(recipe.keypoint_sources)
    skeleton = recipe.skeleton
    missing_markers = missing_source_marker_ids(result.layout, sources)
    if missing_markers:
        allow_keypoint_omission = (
            result.outcome == "partial" and result.partial_output
        )
        if not allow_keypoint_omission:
            print(
                "Calibration solved but publication refused: keypoint-source markers "
                f"missing from layout: {list(missing_markers)}"
            )
            for line in format_omitted_marker_lines(result.omitted_markers):
                print(f"  {line}")
            try:
                write_calibration_diagnostics_if_requested(
                    recipe.paths.diagnostics_path,
                    result,
                    benchmark=benchmark,
                )
            except RuntimeError as error:
                print(error, file=sys.stderr)
            return False
        sources = keypoint_sources_for_layout(result.layout, sources)
        if not sources:
            print(
                "Calibration solved but publication refused: no keypoint-source markers "
                "remain in the partial layout."
            )
            for line in format_omitted_marker_lines(result.omitted_markers):
                print(f"  {line}")
            try:
                write_calibration_diagnostics_if_requested(
                    recipe.paths.diagnostics_path,
                    result,
                    benchmark=benchmark,
                )
            except RuntimeError as error:
                print(error, file=sys.stderr)
            return False
        skeleton = skeleton_for_keypoint_names(skeleton, frozenset(sources.keys()))

    try:
        object_document = build_object_model_document_from_layout(
            result.layout,
            sources,
            skeleton,
        )
    except ValueError as error:
        print(f"Calibration solved but publication refused: {error}")
        try:
            write_calibration_diagnostics_if_requested(
                recipe.paths.diagnostics_path,
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
            recipe.paths.diagnostics_path,
            result,
            benchmark=benchmark,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
    return True


def _frame_sharpness_score(frame: np.ndarray) -> float:
    """Score frame sharpness via downsampled grayscale Laplacian variance."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    laplacian = cv2.Laplacian(small, cv2.CV_64F)
    return float(laplacian.var())


def _benchmark_window_index(video_time: float, sample_interval: float) -> int:
    """Map video time to a half-open sample window index."""
    return int(math.floor(video_time / sample_interval + 1e-12))


def _benchmark_detect_on_frame(
    detector: cv2.aruco.ArucoDetector,
    frame: np.ndarray,
    expected_id_set: set[int],
) -> tuple[dict[int, np.ndarray], int]:
    """Detect expected markers on one frame and time the detection pass."""
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
    """Update benchmark counters and append an observation when >=2 markers co-visible."""
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
    """Convert a non-negative nanosecond duration to seconds."""
    seconds = max(0.0, duration_ns / 1_000_000_000)
    if not math.isfinite(seconds):
        return 0.0
    return seconds


def _benchmark_environment() -> dict[str, str]:
    """Collect runtime library versions for benchmark diagnostics JSON."""
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
    frame_selection: str = BENCHMARK_FRAME_SELECTION_SHARPEST,
    sharpness_scoring_ns: int | None = None,
) -> dict[str, Any]:
    """Assemble benchmark timing, throughput, and environment metadata."""
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


def run_benchmark(recipe: CalibrationRecipe) -> bool:
    """Benchmark video ingest and calibration on a headless sharpest-frame pass."""
    execution = recipe.execution
    if not isinstance(execution, BenchmarkExecution):
        raise RuntimeError("only benchmark execution is supported.")
    source = recipe.source
    if not isinstance(source, Path):
        raise RuntimeError("benchmark execution requires a video file source, not a camera index.")

    total_start_ns = time.perf_counter_ns()
    expected_ids = list(recipe.expected_marker_ids)
    expected_id_set = set(expected_ids)

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        recipe.intrinsics_path
    )
    width, height = require_calibration_image_size(
        image_width, image_height, recipe.intrinsics_path
    )
    detector = build_apriltag_detector(recipe.dictionary, recipe.sensitivity)

    print(f"Expected marker IDs: {expected_ids}")
    print("Reference marker ID: auto")
    print(
        "Benchmark capture: sharpest frame per "
        f"{1.0 / execution.sample_rate_hz:g}s window "
        f"({execution.sample_rate_hz:g} Hz; every frame decoded for scoring)"
    )
    print(f"Using calibration: {recipe.intrinsics_path}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")
    print(format_frame_source(source))

    open_start_ns = time.perf_counter_ns()
    capture = open_frame_source(source, width=width, height=height)
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
        sample_interval = 1.0 / execution.sample_rate_hz
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

        current_window: int | None = None
        best_frame: np.ndarray | None = None
        best_sharpness = float("-inf")
        window_frame_count = 0

        def flush_sharpest_window() -> None:
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
            ok, frame = read_frame(capture, source, loop_on_eof=False)
            decode_ns += time.perf_counter_ns() - decode_start_ns
            if not ok or frame is None:
                break

            decoded_frames += 1
            require_frame_size(
                frame.shape[1], frame.shape[0], width, height, recipe.intrinsics_path
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
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame.copy()

            frame_index += 1

        flush_sharpest_window()
        ingest_total_ns = time.perf_counter_ns() - total_start_ns

        solve_start_ns = time.perf_counter_ns()
        solve_diagnostics = CalibrationSolveDiagnostics()
        result = solve_calibration_from_observations(
            observations,
            recipe,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            solve_diagnostics=solve_diagnostics,
        )
        calibration_solve_ns = time.perf_counter_ns() - solve_start_ns
        total_through_solve_ns = time.perf_counter_ns() - total_start_ns

        benchmark_payload = _build_benchmark_payload(
            source_path=source,
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
            frame_selection=BENCHMARK_FRAME_SELECTION_SHARPEST,
            sharpness_scoring_ns=sharpness_scoring_ns,
        )
        return apply_calibration_result(recipe, result, benchmark=benchmark_payload)
    finally:
        capture.release()


def main() -> None:
    """CLI entry point: load recipe, validate, and run benchmark calibration."""
    try:
        config_path, force = parse_args()
        recipe = load_calibration_recipe(config_path)
        validate_recipe(recipe, force=force)
        run_benchmark(recipe)
    except (RuntimeError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
