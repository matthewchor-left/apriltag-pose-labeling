"""Formatting and JSON export for marker calibration diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
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
)

CALIBRATION_DIAGNOSTICS_VERSION = 1


def format_reprojection_rms_px(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.3f} px"


def _format_optional_float(value: float | None, *, precision: int = 3, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{precision}f}{suffix}"


def _format_measurement_distribution(
    distribution: MeasurementDistribution | None,
    *,
    label: str,
    precision: int = 3,
) -> str | None:
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


def format_quality_diagnostics_lines(quality: CalibrationQualityReport) -> list[str]:
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
    if quality.dropped_pair_edges is not None:
        for edge in quality.dropped_pair_edges:
            lines.append(format_dropped_pair_edge(edge))
    return lines


def _json_safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _measurement_distribution_to_dict(
    distribution: MeasurementDistribution | None,
) -> dict[str, float | None] | None:
    if distribution is None:
        return None
    return {
        "min": _json_safe_float(distribution.min),
        "median": _json_safe_float(distribution.median),
        "p95": _json_safe_float(distribution.p95),
        "max": _json_safe_float(distribution.max),
    }


def _marker_pair_to_list(pair: tuple[int, int] | None) -> list[int] | None:
    if pair is None:
        return None
    return [pair[0], pair[1]]


def _assignment_rejection_cause_to_dict(
    cause: AssignmentRejectionCauseStats,
) -> dict[str, Any]:
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
    return {
        "marker_a": edge.marker_a,
        "marker_b": edge.marker_b,
        "inlier_count": edge.inlier_count,
        "translation_rms_m": _json_safe_float(edge.translation_rms_m),
        "rotation_rms_deg": _json_safe_float(edge.rotation_rms_deg),
    }


def _dropped_pair_edge_to_dict(edge: DroppedPairEdge) -> dict[str, Any]:
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


def _quality_report_to_dict(quality: CalibrationQualityReport) -> dict[str, Any]:
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
    if records is None:
        return None
    return [_assignment_rejection_record_to_dict(record) for record in records]


def _serialize_dropped_pair_edges(
    edges: tuple[DroppedPairEdge, ...] | None,
) -> list[dict[str, Any]] | None:
    if edges is None:
        return None
    return [_dropped_pair_edge_to_dict(edge) for edge in edges]


def build_calibration_diagnostics_document(
    quality: CalibrationQualityReport,
    *,
    succeeded: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "version": CALIBRATION_DIAGNOSTICS_VERSION,
        "succeeded": succeeded,
        "failure_reason": failure_reason,
        "quality": _quality_report_to_dict(quality),
        "assignment_rejections": _assignment_rejection_summary_to_dict(
            quality.assignment_rejections
        ),
        "assignment_rejection_records": _serialize_assignment_rejection_records(
            quality.assignment_rejection_records
        ),
        "dropped_pair_edges": _serialize_dropped_pair_edges(quality.dropped_pair_edges),
    }


def serialize_calibration_diagnostics_document(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def save_calibration_diagnostics(
    path: str | Path,
    result: CalibrationResult,
    *,
    serialize_fn: Callable[[dict[str, Any]], str] | None = None,
) -> Path:
    if result.quality is None:
        raise RuntimeError("Cannot write calibration diagnostics without a quality report.")
    path = Path(path)
    succeeded = result.failure_reason is None and result.layout is not None
    document = build_calibration_diagnostics_document(
        result.quality,
        succeeded=succeeded,
        failure_reason=result.failure_reason,
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
