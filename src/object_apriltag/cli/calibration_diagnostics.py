"""Formatting and JSON export for marker calibration diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

import numpy as np

from object_apriltag.marker_layout_calibration import (
    AssignmentRejectionCauseStats,
    AssignmentRejectionSummary,
    CalibrationQualityReport,
    CalibrationResult,
    DroppedPairEdge,
    EdgeDiagnostics,
    FrameAssignmentRejectionRecord,
    MeasurementDistribution,
    OmittedMarkerDiagnostic,
    RestoredPairEdge,
)

CALIBRATION_DIAGNOSTICS_VERSION = 9


def format_reprojection_rms_px(value: float) -> str:
    """Format global reprojection RMS for console/HUD display.

    Args:
        value: Global reprojection RMS in pixels.

    Returns:
        Fixed-precision pixel string, or ``N/A`` when non-finite.
    """
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.3f} px"


def _format_optional_float(value: float | None, *, precision: int = 3, suffix: str = "") -> str:
    """Format a nullable float with fixed precision.

    Args:
        value: Float to format, or ``None`` when absent.
        precision: Decimal places in the formatted string.
        suffix: Optional unit suffix appended to the number.

    Returns:
        Formatted string, or ``N/A`` when missing or non-finite.
    """
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _format_measurement_distribution(
    distribution: MeasurementDistribution | None,
    *,
    label: str,
    precision: int = 3,
) -> str | None:
    """Render min/median/p95/max for a measurement distribution.

    Args:
        distribution: Optional measurement distribution to format.
        label: Prefix label for the min/med/p95/max line.
        precision: Decimal places for each statistic.

    Returns:
        Formatted distribution line, or ``None`` when ``distribution`` is absent.
    """
    if distribution is None:
        return None
    return (
        f"{label} min/med/p95/max="
        f"{_format_optional_float(distribution.min, precision=precision)}/"
        f"{_format_optional_float(distribution.median, precision=precision)}/"
        f"{_format_optional_float(distribution.p95, precision=precision)}/"
        f"{_format_optional_float(distribution.max, precision=precision)}"
    )


def format_assignment_rejection_cause_detail(cause: AssignmentRejectionCauseStats) -> str:
    """One-line human summary of a grouped frame-assignment rejection cause.

    Args:
        cause: Grouped rejection statistics for one reason and optional marker pair.

    Returns:
        Single-line summary with counts, gates, and sample frame IDs.
    """
    pair_text = ""
    if cause.marker_pair is not None:
        pair_text = f" pair=({cause.marker_pair[0]},{cause.marker_pair[1]})"
    parts = [f"assignment {cause.reason}{pair_text} x{cause.count}"]
    for segment in (
        _format_measurement_distribution(cause.translation_error_m, label="tr_err_m"),
        _format_measurement_distribution(cause.rotation_error_deg, label="rot_err_deg", precision=1),
        f"tr_gate_m={_format_optional_float(cause.translation_gate_m)}",
        f"rot_gate_deg={_format_optional_float(cause.rotation_gate_deg, precision=1)}",
        _format_measurement_distribution(cause.translation_error_ratio, label="tr_ratio"),
        _format_measurement_distribution(cause.rotation_error_ratio, label="rot_ratio"),
        f"sample_frames=[{', '.join(str(frame_id) for frame_id in cause.sample_frame_ids)}]",
    ):
        if segment:
            parts.append(segment)
    return " ".join(parts)


def format_dropped_pair_edge(edge: DroppedPairEdge) -> str:
    """Summarize a marker pair removed during graph pruning or weak-edge filtering.

    Args:
        edge: Dropped pair edge diagnostic from the quality report.

    Returns:
        Single-line summary with stage, support counts, and RMS metrics.
    """
    return " ".join(
        [
            f"dropped pair ({edge.marker_a},{edge.marker_b})",
            f"stage={edge.stage}",
            f"reason={edge.reason}",
            f"support={edge.supported_count}/{edge.required_count}",
            f"observed={edge.observed_count}",
            f"tr_rms={_format_optional_float(edge.translation_rms_m)}m",
            f"rot_rms={_format_optional_float(edge.rotation_rms_deg, precision=1)}deg",
            f"tr_gate={_format_optional_float(edge.translation_gate_m)}m",
            f"rot_gate={_format_optional_float(edge.rotation_gate_deg, precision=1)}deg",
        ]
    )


def format_restored_pair_edge(edge: RestoredPairEdge) -> str:
    """Summarize a previously dropped pair edge that weak-edge recovery reinstated.

    Args:
        edge: Restored pair edge diagnostic from the quality report.

    Returns:
        Single-line summary with original drop reason and support metrics.
    """
    return " ".join(
        [
            f"restored pair ({edge.marker_a},{edge.marker_b})",
            f"stage={edge.stage}",
            f"original_stage={edge.original_stage}",
            f"reason={edge.original_reason}",
            f"support={edge.supported_count}/{edge.required_count}",
            f"observed={edge.observed_count}",
            f"support_fraction={_format_optional_float(edge.support_fraction, precision=3)}",
            f"tr_rms={_format_optional_float(edge.translation_rms_m)}m",
            f"rot_rms={_format_optional_float(edge.rotation_rms_deg, precision=1)}deg",
            f"tr_gate={_format_optional_float(edge.translation_gate_m)}m",
            f"rot_gate={_format_optional_float(edge.rotation_gate_deg, precision=1)}deg",
        ]
    )


def format_omitted_marker_diagnostic(record: OmittedMarkerDiagnostic) -> str:
    """Format one omitted-marker reason for console output.

    Args:
        record: Omitted-marker diagnostic entry.

    Returns:
        Single-line ``omitted marker`` summary.
    """
    return f"omitted marker {record.marker_id}: {record.reason}"


def format_omitted_marker_lines(
    omitted_markers: tuple[OmittedMarkerDiagnostic, ...] | None,
) -> list[str]:
    """Return formatted omitted-marker lines.

    Args:
        omitted_markers: Omitted-marker diagnostics tuple, or legacy ``None`` value.

    Returns:
        Formatted lines; empty when ``omitted_markers`` is absent or not a tuple.
    """
    if omitted_markers is None or not isinstance(omitted_markers, tuple):
        return []
    return [format_omitted_marker_diagnostic(record) for record in omitted_markers]


def format_quality_diagnostics_lines(quality: CalibrationQualityReport) -> list[str]:
    """Assemble post-solve quality, pair-edge, and assignment diagnostics for printing.

    Args:
        quality: Post-solve calibration quality report.

    Returns:
        Ordered console lines for reprojection, pair edges, and rejections.
    """
    lines = [
        f"reprojection RMS: {format_reprojection_rms_px(quality.reprojection_rms_px)}",
        f"inlier corners: {quality.inlier_corner_count}",
        (
            "connected markers: "
            f"{sorted(quality.connected_marker_ids)} "
            f"missing: {sorted(quality.missing_expected_ids)}"
        ),
    ]
    for edge in quality.edges:
        lines.append(
            "pair "
            f"({edge.marker_a}, {edge.marker_b}): "
            f"inliers={edge.inlier_count} "
            f"trans_rms={_format_optional_float(edge.translation_rms_m, precision=4)} m "
            f"rot_rms={_format_optional_float(edge.rotation_rms_deg, precision=2)} deg"
        )
    if quality.assignment_rejections is not None and quality.assignment_rejections.total_rejected > 0:
        lines.append(
            f"assignment rejections: {quality.assignment_rejections.total_rejected} total"
        )
        for cause in quality.assignment_rejections.by_cause:
            lines.append(format_assignment_rejection_cause_detail(cause))
    if isinstance(quality.dropped_pair_edges, tuple):
        for edge in quality.dropped_pair_edges:
            lines.append(format_dropped_pair_edge(edge))
    if isinstance(quality.restored_pair_edges, tuple):
        for edge in quality.restored_pair_edges:
            lines.append(format_restored_pair_edge(edge))
    return lines


def _json_safe_float(value: float | None) -> float | None:
    """Return a finite float for JSON export.

    Args:
        value: Scalar to coerce, or ``None`` when absent.

    Returns:
        Finite float, or ``None`` for missing or non-finite values.
    """
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _json_safe_benchmark_value(value: Any) -> Any:
    """Recursively coerce benchmark metadata to JSON-serializable scalars and containers.

    Args:
        value: Benchmark metadata value or nested structure.

    Returns:
        JSON-safe scalar, list, or dict; unknown types are returned unchanged.
    """
    if isinstance(value, Mapping):
        return {key: _json_safe_benchmark_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_benchmark_value(item) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _json_safe_float(float(value))
    return value


def _normalize_benchmark_payload(
    benchmark: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Copy optional benchmark metadata through :func:`_json_safe_benchmark_value`.

    Args:
        benchmark: Optional benchmark metadata mapping from the CLI benchmark run.

    Returns:
        Sanitized benchmark dict, or ``None`` when ``benchmark`` is absent.
    """
    if benchmark is None:
        return None
    return _json_safe_benchmark_value(dict(benchmark))


def _measurement_distribution_to_dict(
    distribution: MeasurementDistribution | None,
) -> dict[str, float | None] | None:
    """Serialize a measurement distribution for the diagnostics JSON document.

    Args:
        distribution: Optional measurement distribution from rejection diagnostics.

    Returns:
        Dict with min/median/p95/max keys, or ``None`` when absent.
    """
    if distribution is None:
        return None
    return {
        "min": _json_safe_float(distribution.min),
        "median": _json_safe_float(distribution.median),
        "p95": _json_safe_float(distribution.p95),
        "max": _json_safe_float(distribution.max),
    }


def _marker_pair_to_list(pair: tuple[int, int] | None) -> list[int] | None:
    """Convert an optional marker-pair tuple to a JSON-friendly two-element list.

    Args:
        pair: Optional ``(marker_a, marker_b)`` tuple.

    Returns:
        Two-element marker ID list, or ``None`` when ``pair`` is absent.
    """
    if pair is None:
        return None
    return [pair[0], pair[1]]


def _assignment_rejection_cause_to_dict(
    cause: AssignmentRejectionCauseStats,
) -> dict[str, Any]:
    """Serialize grouped assignment-rejection statistics for JSON export.

    Args:
        cause: Grouped rejection statistics for one reason and optional marker pair.

    Returns:
        JSON-serializable dict with counts, gates, and error distributions.
    """
    return {
        "reason": cause.reason,
        "marker_pair": _marker_pair_to_list(cause.marker_pair),
        "count": cause.count,
        "sample_frame_ids": list(cause.sample_frame_ids),
        "translation_error_m": _measurement_distribution_to_dict(cause.translation_error_m),
        "rotation_error_deg": _measurement_distribution_to_dict(cause.rotation_error_deg),
        "translation_gate_m": _json_safe_float(cause.translation_gate_m),
        "rotation_gate_deg": _json_safe_float(cause.rotation_gate_deg),
        "translation_error_ratio": _measurement_distribution_to_dict(cause.translation_error_ratio),
        "rotation_error_ratio": _measurement_distribution_to_dict(cause.rotation_error_ratio),
    }


def _assignment_rejection_summary_to_dict(
    summary: AssignmentRejectionSummary | None,
) -> dict[str, Any] | None:
    """Serialize the assignment-rejection rollup attached to a quality report.

    Args:
        summary: Optional assignment-rejection summary from the quality report.

    Returns:
        JSON-serializable rollup dict, or ``None`` when absent.
    """
    if summary is None:
        return None
    return {
        "total_rejected": summary.total_rejected,
        "by_reason": [[reason, count] for reason, count in summary.by_reason],
        "by_pair": [[list(pair), count] for pair, count in summary.by_pair],
        "top_causes": [
            {
                "reason": cause.reason,
                "marker_pair": _marker_pair_to_list(cause.marker_pair),
                "count": cause.count,
            }
            for cause in summary.top_causes
        ],
        "by_cause": [_assignment_rejection_cause_to_dict(cause) for cause in summary.by_cause],
    }


def _assignment_rejection_record_to_dict(
    record: FrameAssignmentRejectionRecord,
) -> dict[str, Any]:
    """Serialize one per-frame assignment rejection for JSON export.

    Args:
        record: Per-frame assignment rejection record.

    Returns:
        JSON-serializable dict with frame metadata and gate/error values.
    """
    return {
        "frame_index": record.frame_index,
        "frame_id": record.frame_id,
        "visible_marker_ids": list(record.visible_marker_ids),
        "reason": record.reason,
        "marker_pair": _marker_pair_to_list(record.marker_pair),
        "translation_error_m": _json_safe_float(record.translation_error_m),
        "rotation_error_deg": _json_safe_float(record.rotation_error_deg),
        "translation_gate_m": _json_safe_float(record.translation_gate_m),
        "rotation_gate_deg": _json_safe_float(record.rotation_gate_deg),
    }


def _edge_diagnostics_to_dict(edge: EdgeDiagnostics) -> dict[str, Any]:
    """Serialize pair-edge inlier and RMS metrics for JSON export.

    Args:
        edge: Pair-edge diagnostic from the quality report.

    Returns:
        JSON-serializable dict with inlier count and RMS metrics.
    """
    return {
        "marker_a": edge.marker_a,
        "marker_b": edge.marker_b,
        "inlier_count": edge.inlier_count,
        "translation_rms_m": _json_safe_float(edge.translation_rms_m),
        "rotation_rms_deg": _json_safe_float(edge.rotation_rms_deg),
    }


def _dropped_pair_edge_to_dict(edge: DroppedPairEdge) -> dict[str, Any]:
    """Serialize a dropped pair edge, including stage, support counts, and gates.

    Args:
        edge: Dropped pair edge diagnostic.

    Returns:
        JSON-serializable dict with stage, support counts, and RMS metrics.
    """
    return {
        "marker_a": edge.marker_a,
        "marker_b": edge.marker_b,
        "stage": edge.stage,
        "reason": edge.reason,
        "observed_count": edge.observed_count,
        "supported_count": edge.supported_count,
        "required_count": edge.required_count,
        "translation_rms_m": _json_safe_float(edge.translation_rms_m),
        "rotation_rms_deg": _json_safe_float(edge.rotation_rms_deg),
        "translation_gate_m": _json_safe_float(edge.translation_gate_m),
        "rotation_gate_deg": _json_safe_float(edge.rotation_gate_deg),
    }


def _restored_pair_edge_to_dict(edge: RestoredPairEdge) -> dict[str, Any]:
    """Serialize a restored pair edge, preserving the original drop reason and stage.

    Args:
        edge: Restored pair edge diagnostic.

    Returns:
        JSON-serializable dict with original drop metadata and support metrics.
    """
    return {
        "marker_a": edge.marker_a,
        "marker_b": edge.marker_b,
        "stage": edge.stage,
        "original_stage": edge.original_stage,
        "original_reason": edge.original_reason,
        "observed_count": edge.observed_count,
        "supported_count": edge.supported_count,
        "required_count": edge.required_count,
        "support_fraction": _json_safe_float(edge.support_fraction),
        "translation_rms_m": _json_safe_float(edge.translation_rms_m),
        "rotation_rms_deg": _json_safe_float(edge.rotation_rms_deg),
        "translation_gate_m": _json_safe_float(edge.translation_gate_m),
        "rotation_gate_deg": _json_safe_float(edge.rotation_gate_deg),
    }


def _quality_report_to_dict(quality: CalibrationQualityReport) -> dict[str, Any]:
    """Serialize core quality metrics for the diagnostics JSON document.

    Extended diagnostics (rejections, dropped edges) are serialized
    in sibling top-level keys of the full document.

    Args:
        quality: Post-solve calibration quality report.

    Returns:
        JSON-serializable dict with reprojection, edges, and frame counts.
    """
    per_marker = {
        str(marker_id): _json_safe_float(value)
        for marker_id, value in sorted(quality.per_marker_reprojection_rms_px.items())
    }
    return {
        "reprojection_rms_px": _json_safe_float(quality.reprojection_rms_px),
        "per_marker_reprojection_rms_px": per_marker,
        "edges": [_edge_diagnostics_to_dict(edge) for edge in quality.edges],
        "pair_translation_rms_max_m": _json_safe_float(quality.pair_translation_rms_max_m),
        "pair_rotation_rms_max_deg": _json_safe_float(quality.pair_rotation_rms_max_deg),
        "frame_count": quality.frame_count,
        "observation_count": quality.observation_count,
        "inlier_corner_count": quality.inlier_corner_count,
        "input_frame_count": quality.input_frame_count,
        "rejected_frame_count": quality.rejected_frame_count,
        "accepted_frame_count": quality.accepted_frame_count,
        "connected_marker_ids": sorted(quality.connected_marker_ids),
        "missing_expected_ids": sorted(quality.missing_expected_ids),
        "unused_expected_ids": sorted(quality.unused_expected_ids),
    }


def _serialize_assignment_rejection_records(
    records: tuple[FrameAssignmentRejectionRecord, ...] | None,
) -> list[dict[str, Any]] | None:
    """Serialize per-frame rejection records.

    Args:
        records: Optional per-frame rejection record tuple.

    Returns:
        List of serialized records, or ``None`` when diagnostics were not collected.
    """
    if records is None:
        return None
    return [_assignment_rejection_record_to_dict(record) for record in records]


def _serialize_dropped_pair_edges(
    edges: tuple[DroppedPairEdge, ...] | None,
) -> list[dict[str, Any]] | None:
    """Serialize dropped-pair edge diagnostics.

    Args:
        edges: Optional dropped-pair edge tuple from the quality report.

    Returns:
        List of serialized edges, or ``None`` when absent.
    """
    if edges is None:
        return None
    return [_dropped_pair_edge_to_dict(edge) for edge in edges]


def _serialize_restored_pair_edges(
    edges: tuple[RestoredPairEdge, ...] | None,
) -> list[dict[str, Any]] | None:
    """Serialize restored-pair edge diagnostics.

    Args:
        edges: Optional restored-pair edge tuple from the quality report.

    Returns:
        List of serialized edges, or ``None`` when absent.
    """
    if edges is None:
        return None
    return [_restored_pair_edge_to_dict(edge) for edge in edges]


def _omitted_marker_to_dict(record: OmittedMarkerDiagnostic) -> dict[str, Any]:
    """Serialize one omitted-marker diagnostic entry.

    Args:
        record: Omitted-marker diagnostic entry.

    Returns:
        JSON-serializable dict with marker ID and omission reason.
    """
    return {
        "marker_id": record.marker_id,
        "reason": record.reason,
    }


def build_calibration_diagnostics_document(
    quality: CalibrationQualityReport,
    *,
    succeeded: bool,
    failure_reason: str | None,
    calibration_policy: str = "best_effort",
    outcome: str = "refused",
    failed_quality_gates: tuple[str, ...] | list[str] = (),
    selected_checkpoint_stage: str | None = None,
    failed_refinement_stage: str | None = None,
    omitted_markers: tuple[OmittedMarkerDiagnostic, ...] = (),
    partial_output: bool = False,
    benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the versioned calibration diagnostics document for ``--diagnostics-output``.

    Combines solve outcome metadata with the quality report and optional benchmark
    payload.

    Args:
        quality: Post-solve calibration quality report.
        succeeded: Whether a marker layout was produced without failure reason.
        failure_reason: Refusal or partial-outcome reason when solve did not fully succeed.
        calibration_policy: Strict vs best-effort policy label.
        outcome: Solve outcome label (e.g. ``strict``, ``partial``, ``provisional``).
        failed_quality_gates: Quality gates that failed on best-effort output.
        selected_checkpoint_stage: Optimization checkpoint used for provisional output.
        failed_refinement_stage: Refinement stage that failed before checkpoint export.
        omitted_markers: Markers omitted from the saved layout.
        partial_output: Whether a reference-connected partial layout was written.
        benchmark: Optional benchmark timing metadata from a headless video run.

    Returns:
        Versioned diagnostics document dict ready for JSON serialization.

    Notes:
        Pure data assembly; no filesystem or console side effects.
    """
    return {
        "version": CALIBRATION_DIAGNOSTICS_VERSION,
        "benchmark": _normalize_benchmark_payload(benchmark),
        "succeeded": succeeded,
        "failure_reason": failure_reason,
        "calibration_policy": calibration_policy,
        "outcome": outcome,
        "partial_output": partial_output,
        "failed_quality_gates": list(failed_quality_gates),
        "selected_checkpoint_stage": selected_checkpoint_stage,
        "failed_refinement_stage": failed_refinement_stage,
        "omitted_markers": [_omitted_marker_to_dict(record) for record in omitted_markers],
        "quality": _quality_report_to_dict(quality),
        "assignment_rejections": _assignment_rejection_summary_to_dict(
            quality.assignment_rejections
        ),
        "assignment_rejection_records": _serialize_assignment_rejection_records(
            quality.assignment_rejection_records
        ),
        "dropped_pair_edges": _serialize_dropped_pair_edges(quality.dropped_pair_edges),
        "restored_pair_edges": _serialize_restored_pair_edges(quality.restored_pair_edges),
    }


def serialize_calibration_diagnostics_document(document: dict[str, Any]) -> str:
    """Serialize a diagnostics document to indented JSON with a trailing newline.

    Args:
        document: Versioned calibration diagnostics document dict.

    Returns:
        Indented JSON text with trailing newline.

    Notes:
        Uses ``allow_nan=False``; non-finite floats must be sanitized beforehand.
    """
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def save_calibration_diagnostics(
    path: str | Path,
    result: CalibrationResult,
    *,
    serialize_fn: Callable[[dict[str, Any]], str] | None = None,
    benchmark: Mapping[str, Any] | None = None,
) -> Path:
    """Write diagnostics JSON atomically from a :class:`CalibrationResult`.

    Args:
        path: Output JSON path; parent directories are created as needed.
        result: Calibration result containing the quality report.
        serialize_fn: Optional serializer; defaults to
            :func:`serialize_calibration_diagnostics_document`.
        benchmark: Optional benchmark timing metadata to embed in the document.

    Returns:
        Resolved output path after the atomic write completes.

    Raises:
        RuntimeError: ``result.quality`` is missing, or serialization/write fails.

    Notes:
        Writes via a temp file and ``os.replace`` so readers never see a partial file.
    """
    if result.quality is None:
        raise RuntimeError("Cannot write calibration diagnostics without a quality report.")
    path = Path(path)
    succeeded = result.layout is not None and result.failure_reason is None
    document = build_calibration_diagnostics_document(
        result.quality,
        succeeded=succeeded,
        failure_reason=result.failure_reason,
        calibration_policy=result.calibration_policy,
        outcome=result.outcome or "refused",
        failed_quality_gates=result.failed_quality_gates,
        selected_checkpoint_stage=result.selected_checkpoint_stage,
        failed_refinement_stage=result.failed_refinement_stage,
        omitted_markers=result.omitted_markers,
        partial_output=result.partial_output,
        benchmark=benchmark,
    )
    serialize = serialize_fn or serialize_calibration_diagnostics_document
    try:
        text = serialize(document)
    except Exception as error:
        raise RuntimeError(f"Failed to write calibration diagnostics to {path}: {error}") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception as error:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Failed to write calibration diagnostics to {path}: {error}") from error
    return path
