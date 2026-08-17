"""Versioned marker-model evaluation report, serialization, and console output."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from object_apriltag.evaluation.manifest import EvaluationManifest, repo_relative_path
from object_apriltag.evaluation.orchestration import VideoNormalizationSummary
from object_apriltag.evaluation.types import (
    CadGeometryEvaluation,
    DetectionConsistencyCandidateResult,
    DistanceCadDisagreementReport,
    LeaveOneMarkerCadPrediction,
    LeaveOneMarkerCadPredictionFold,
    MetricSummaryMm,
    MetricSummaryPx,
    PerMarkerDetectionSummary,
    RigidCadFit,
    RigidRotationValidation,
    SourceVideoDetectionSummary,
    VisibleMarkerCountStratum,
)

MARKER_MODEL_EVALUATION_REPORT_VERSION = 1


@dataclass(frozen=True)
class CandidateEvaluationResult:
    """Per-candidate CAD geometry and detection-consistency results.

    Attributes:
        name: Candidate name from the evaluation manifest.
        capture_session: Capture session label for grouping.
        solver_variant: Solver variant label for grouping.
        marker_model_path: Path to the candidate marker layout JSON.
        calibration_source_video: Optional calibration video provenance.
        cad_geometry: CAD geometry evaluation metrics.
        detection: Detection-consistency evaluation metrics.
    """

    name: str
    capture_session: str
    solver_variant: str
    marker_model_path: Path
    calibration_source_video: Path | None
    cad_geometry: CadGeometryEvaluation
    detection: DetectionConsistencyCandidateResult


@dataclass(frozen=True)
class CandidateRankingEntry:
    """One entry in a candidate ranking table.

    Attributes:
        candidate_name: Ranked candidate name.
        metric_value: Metric value used for ordering (lower is better).
    """

    candidate_name: str
    metric_value: float


@dataclass(frozen=True)
class SameSessionSolverGroup:
    """Candidates from the same capture session for solver comparison.

    Attributes:
        capture_session: Shared capture session label.
        candidate_names: Candidate names in the group.
        solver_variants: Distinct solver variants present in the group.
    """

    capture_session: str
    candidate_names: tuple[str, ...]
    solver_variants: tuple[str, ...]


@dataclass(frozen=True)
class CrossSessionSameVariantGroup:
    """Candidates sharing a solver variant across capture sessions.

    Attributes:
        solver_variant: Shared solver variant label.
        capture_sessions: Distinct capture sessions in the group.
        candidate_names: Candidate names in the group.
    """

    solver_variant: str
    capture_sessions: tuple[str, ...]
    candidate_names: tuple[str, ...]


@dataclass(frozen=True)
class CandidateGroupingReport:
    """Repeatability grouping metadata for manifest candidates.

    Attributes:
        same_session_solver_groups: Groups with multiple candidates per session.
        cross_session_same_variant_groups: Groups with one variant across sessions.
        notes: Human-readable notes when grouping is insufficient.
    """

    same_session_solver_groups: tuple[SameSessionSolverGroup, ...]
    cross_session_same_variant_groups: tuple[CrossSessionSameVariantGroup, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class MarkerModelEvaluationReport:
    """Complete marker-model evaluation report.

    Attributes:
        version: Report schema version.
        manifest_path: Absolute path to the source evaluation manifest.
        inputs: Serialized manifest inputs with file hashes.
        correspondence: Landmark and marker ID correspondence metadata.
        held_out_declaration: User-provided held-out video declaration.
        normalization: Per-video detection normalization diagnostics.
        candidates: Per-candidate evaluation results.
        rankings: Separate CAD and detection metric rankings.
        grouping: Repeatability grouping metadata.
    """

    version: int
    manifest_path: Path
    inputs: dict[str, Any]
    correspondence: dict[str, Any]
    held_out_declaration: dict[str, Any]
    normalization: dict[str, Any]
    candidates: tuple[CandidateEvaluationResult, ...]
    rankings: dict[str, tuple[CandidateRankingEntry, ...]]
    grouping: CandidateGroupingReport


def build_marker_model_evaluation_report(
    *,
    manifest: EvaluationManifest,
    landmark_names: tuple[str, ...],
    expected_marker_ids: tuple[int, ...],
    calibration_width: int,
    calibration_height: int,
    calibration_source: str | None,
    candidate_results: tuple[CandidateEvaluationResult, ...],
    normalization_summaries: tuple[VideoNormalizationSummary, ...],
) -> MarkerModelEvaluationReport:
    """Assemble a structured marker-model evaluation report.

    Args:
        manifest: Source evaluation manifest.
        landmark_names: Ordered landmark names from object-model correspondence.
        expected_marker_ids: Marker IDs shared by all candidate layouts.
        calibration_width: Intrinsics calibration image width.
        calibration_height: Intrinsics calibration image height.
        calibration_source: Optional intrinsics calibration provenance string.
        candidate_results: Per-candidate CAD and detection metrics.
        normalization_summaries: Per-video detection normalization diagnostics.

    Returns:
        Structured report with rankings and grouping metadata.
    """
    inputs = _build_inputs_section(
        manifest=manifest,
        calibration_width=calibration_width,
        calibration_height=calibration_height,
        calibration_source=calibration_source,
    )
    correspondence = {
        "landmark_names": list(landmark_names),
        "expected_marker_ids": list(expected_marker_ids),
        "cad_landmark_count": len(landmark_names),
        "object_model_landmark_count": len(landmark_names),
    }
    held_out_declaration = {
        "user_provided": True,
        "verifiable_by_tool": False,
        "videos": [
            {
                "path": _repo_relative_path(video.path),
                "held_out": video.held_out,
            }
            for video in manifest.held_out_videos
        ],
    }
    normalization = {
        "per_video": [_video_normalization_to_dict(summary) for summary in normalization_summaries],
        "totals": _aggregate_normalization_totals(normalization_summaries),
    }
    rankings = _build_rankings(candidate_results)
    grouping = _build_grouping(manifest, candidate_results)
    return MarkerModelEvaluationReport(
        version=MARKER_MODEL_EVALUATION_REPORT_VERSION,
        manifest_path=manifest.manifest_path,
        inputs=inputs,
        correspondence=correspondence,
        held_out_declaration=held_out_declaration,
        normalization=normalization,
        candidates=candidate_results,
        rankings=rankings,
        grouping=grouping,
    )


def build_marker_model_evaluation_document(report: MarkerModelEvaluationReport) -> dict[str, Any]:
    """Serialize a marker-model evaluation report to a JSON-ready dict.

    Args:
        report: Structured evaluation report.

    Returns:
        JSON-serializable document with interpretation notes.
    """
    return {
        "version": report.version,
        "manifest_path": _repo_relative_path(report.manifest_path),
        "inputs": report.inputs,
        "correspondence": report.correspondence,
        "held_out_declaration": report.held_out_declaration,
        "normalization": report.normalization,
        "candidates": [_candidate_result_to_dict(candidate) for candidate in report.candidates],
        "rankings": {
            metric_name: [_ranking_entry_to_dict(entry) for entry in entries]
            for metric_name, entries in report.rankings.items()
        },
        "grouping": _grouping_to_dict(report.grouping),
        "interpretation": {
            "cad_disagreement": (
                "CAD disagreement combines nominal CAD geometry, physical installation, "
                "export/padding choices, and vision-calibration effects. It is not "
                "calibration error because physical installation is unsurveyed."
            ),
            "detection_consistency": (
                "Detection consistency scores held-out marker corner prediction from "
                "other visible markers. It does not claim absolute pose accuracy."
            ),
            "held_out_declaration": (
                "Held-out status is declared in the manifest and preserved here; the tool "
                "does not independently verify that videos were excluded from calibration."
            ),
            "no_overall_winner": (
                "Geometry and detection rankings are reported separately and may disagree."
            ),
        },
    }


def serialize_marker_model_evaluation_document(document: dict[str, Any]) -> str:
    """Serialize an evaluation document dict to indented JSON text.

    Args:
        document: JSON-ready evaluation document.

    Returns:
        Indented JSON string with trailing newline.
    """
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def save_marker_model_evaluation_report(
    path: str | Path,
    report: MarkerModelEvaluationReport,
) -> Path:
    """Atomically write a marker-model evaluation report JSON file.

    Args:
        path: Destination file path.
        report: Structured evaluation report.

    Returns:
        Resolved destination path.

    Raises:
        RuntimeError: If the report cannot be written.
    """
    path = Path(path)
    document = build_marker_model_evaluation_document(report)
    text = serialize_marker_model_evaluation_document(document)
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
        raise RuntimeError(f"Failed to write marker model evaluation report to {path}: {error}") from error
    return path


def format_marker_model_evaluation_console_summary(report: MarkerModelEvaluationReport) -> str:
    """Format a human-readable console summary of evaluation rankings.

    Args:
        report: Structured evaluation report.

    Returns:
        Multi-line text summary with CAD and detection rankings.
    """
    lines = [
        f"Marker model evaluation report v{report.version}",
        f"Manifest: {_repo_relative_path(report.manifest_path)}",
        f"Candidates: {len(report.candidates)} "
        f"({sum(1 for candidate in report.candidates if candidate.detection.compatible)} compatible)",
        "",
        "CAD disagreement ranking (leave-one-marker-out RMSE mm, lower is better):",
    ]
    cad_ranking = report.rankings.get("cad_leave_one_marker_out_rmse_mm", ())
    if cad_ranking:
        for rank, entry in enumerate(cad_ranking, start=1):
            lines.append(f"  {rank}. {entry.candidate_name}: {entry.metric_value:.3f} mm")
    else:
        lines.append("  (no compatible candidates)")

    lines.extend(["", "Detection consistency ranking (held-out P95 px, lower is better):"])
    detection_ranking = report.rankings.get("detection_held_out_p95_px", ())
    if detection_ranking:
        for rank, entry in enumerate(detection_ranking, start=1):
            lines.append(f"  {rank}. {entry.candidate_name}: {entry.metric_value:.3f} px")
    else:
        lines.append("  (no compatible candidates)")

    if report.grouping.notes:
        lines.extend(["", "Grouping notes:"])
        lines.extend(f"  - {note}" for note in report.grouping.notes)
    lines.append("")
    return "\n".join(lines)


def _build_inputs_section(
    *,
    manifest: EvaluationManifest,
    calibration_width: int,
    calibration_height: int,
    calibration_source: str | None,
) -> dict[str, Any]:
    """Build the report ``inputs`` section with file hashes.

    Args:
        manifest: Source evaluation manifest.
        calibration_width: Intrinsics calibration image width.
        calibration_height: Intrinsics calibration image height.
        calibration_source: Optional intrinsics calibration provenance string.

    Returns:
        Serialized inputs section.
    """
    return {
        "cad_model": _file_input(manifest.cad_model),
        "object_model": _file_input(manifest.object_model),
        "intrinsics": {
            **_file_input(manifest.intrinsics),
            "image_width": calibration_width,
            "image_height": calibration_height,
            "calibration_source": calibration_source,
        },
        "detector": {
            "dictionary": manifest.detector.dictionary,
            "sensitivity": manifest.detector.sensitivity,
        },
        "held_out_videos": [_file_input(video.path) for video in manifest.held_out_videos],
        "candidates": [
            {
                "name": candidate.name,
                "marker_model": _file_input(candidate.marker_model_path),
                "capture_session": candidate.capture_session,
                "solver_variant": candidate.solver_variant,
                "calibration_source": {
                    "video": _file_input(candidate.calibration_source.video)
                    if candidate.calibration_source.video is not None
                    else None,
                },
            }
            for candidate in manifest.candidates
        ],
    }


def _build_rankings(
    candidate_results: tuple[CandidateEvaluationResult, ...],
) -> dict[str, tuple[CandidateRankingEntry, ...]]:
    """Build separate CAD and detection metric rankings for compatible candidates.

    Args:
        candidate_results: Per-candidate evaluation results.

    Returns:
        Rankings keyed by metric name.
    """
    compatible = [candidate for candidate in candidate_results if candidate.detection.compatible]
    cad_ranked = sorted(
        compatible,
        key=lambda candidate: candidate.cad_geometry.leave_one_marker_out.all_excluded_summary_mm.rmse_mm,
    )
    detection_ranked = sorted(
        compatible,
        key=lambda candidate: candidate.detection.summary_px.p95_px,
    )
    return {
        "cad_leave_one_marker_out_rmse_mm": tuple(
            CandidateRankingEntry(
                candidate_name=candidate.name,
                metric_value=candidate.cad_geometry.leave_one_marker_out.all_excluded_summary_mm.rmse_mm,
            )
            for candidate in cad_ranked
        ),
        "detection_held_out_p95_px": tuple(
            CandidateRankingEntry(
                candidate_name=candidate.name,
                metric_value=candidate.detection.summary_px.p95_px,
            )
            for candidate in detection_ranked
        ),
    }


def _build_grouping(
    manifest: EvaluationManifest,
    candidate_results: tuple[CandidateEvaluationResult, ...],
) -> CandidateGroupingReport:
    """Derive repeatability grouping metadata from manifest candidate labels.

    Args:
        manifest: Source evaluation manifest.
        candidate_results: Per-candidate evaluation results (unused; kept for API symmetry).

    Returns:
        Grouping report with same-session and cross-session groups.
    """
    by_session: dict[str, list[str]] = defaultdict(list)
    by_variant: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for candidate in manifest.candidates:
        by_session[candidate.capture_session].append(candidate.name)
        by_variant[candidate.solver_variant].append((candidate.capture_session, candidate.name))

    same_session_groups: list[SameSessionSolverGroup] = []
    for capture_session, names in sorted(by_session.items()):
        if len(names) < 2:
            continue
        solver_variants = tuple(
            sorted(
                {
                    candidate.solver_variant
                    for candidate in manifest.candidates
                    if candidate.capture_session == capture_session
                }
            )
        )
        same_session_groups.append(
            SameSessionSolverGroup(
                capture_session=capture_session,
                candidate_names=tuple(names),
                solver_variants=solver_variants,
            )
        )

    cross_session_groups: list[CrossSessionSameVariantGroup] = []
    for solver_variant, entries in sorted(by_variant.items()):
        sessions = {capture_session for capture_session, _name in entries}
        if len(sessions) < 2:
            continue
        cross_session_groups.append(
            CrossSessionSameVariantGroup(
                solver_variant=solver_variant,
                capture_sessions=tuple(sorted(sessions)),
                candidate_names=tuple(name for _session, name in sorted(entries)),
            )
        )

    notes: list[str] = []
    if len(manifest.candidates) < 2:
        notes.append("insufficient_candidates_for_repeatability_grouping")
    if not same_session_groups:
        notes.append("insufficient_candidates_for_same_session_solver_comparison")
    if not cross_session_groups:
        notes.append("insufficient_candidates_for_cross_session_same_variant_comparison")

    return CandidateGroupingReport(
        same_session_solver_groups=tuple(same_session_groups),
        cross_session_same_variant_groups=tuple(cross_session_groups),
        notes=tuple(notes),
    )


def _aggregate_normalization_totals(
    summaries: tuple[VideoNormalizationSummary, ...],
) -> dict[str, int]:
    """Sum per-video normalization counters across all held-out videos.

    Args:
        summaries: Per-video normalization diagnostics.

    Returns:
        Aggregate counter dict.
    """
    return {
        "unknown_marker_ids": sum(summary.unknown_marker_ids for summary in summaries),
        "duplicate_marker_skips": sum(summary.duplicate_marker_skips for summary in summaries),
        "malformed_detections": sum(summary.malformed_detections for summary in summaries),
        "frame_count": sum(summary.frame_count for summary in summaries),
    }


def _file_input(path: Path | None) -> dict[str, Any]:
    """Serialize one input file path with SHA-256 hash.

    Args:
        path: Input file path, or ``None``.

    Returns:
        Dict with repo-relative path and hash, or null path/hash when missing.
    """
    if path is None:
        return {"path": None, "sha256": None}
    resolved = path.resolve()
    return {
        "path": _repo_relative_path(resolved),
        "sha256": _sha256_file(resolved),
    }


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(path: Path) -> str:
    """Convert a path to repo-relative POSIX form.

    Args:
        path: Absolute or relative path.

    Returns:
        Repo-relative POSIX path string.
    """
    return repo_relative_path(path)


def _candidate_result_to_dict(candidate: CandidateEvaluationResult) -> dict[str, Any]:
    """Serialize one candidate evaluation result.

    Args:
        candidate: Per-candidate evaluation result.

    Returns:
        JSON-ready candidate result dict.
    """
    return {
        "name": candidate.name,
        "capture_session": candidate.capture_session,
        "solver_variant": candidate.solver_variant,
        "marker_model": _file_input(candidate.marker_model_path),
        "calibration_source": {
            "video": _file_input(candidate.calibration_source_video),
        },
        "cad_geometry": _cad_geometry_to_dict(candidate.cad_geometry),
        "detection_consistency": _detection_candidate_to_dict(candidate.detection),
    }


def _cad_geometry_to_dict(evaluation: CadGeometryEvaluation) -> dict[str, Any]:
    """Serialize CAD geometry evaluation metrics.

    Args:
        evaluation: CAD geometry evaluation result.

    Returns:
        JSON-ready CAD geometry dict.
    """
    return {
        "landmark_names": list(evaluation.landmark_names),
        "cad_landmarks_m": evaluation.cad_landmarks_m,
        "marker_derived_landmarks_m": evaluation.marker_derived_landmarks_m,
        "rigid_fit": _rigid_fit_to_dict(evaluation.rigid_fit),
        "pair_distance_disagreement": _distance_report_to_dict(evaluation.pair_distance_disagreement),
        "skeleton_edge_disagreement": _distance_report_to_dict(evaluation.skeleton_edge_disagreement),
        "leave_one_marker_out": _leave_one_marker_cad_to_dict(evaluation.leave_one_marker_out),
    }


def _rigid_fit_to_dict(fit: RigidCadFit) -> dict[str, Any]:
    """Serialize rigid CAD fit metrics.

    Args:
        fit: Rigid CAD alignment result.

    Returns:
        JSON-ready rigid fit dict.
    """
    return {
        "rotation": fit.rotation,
        "translation_m": fit.translation_m,
        "rotation_validation": _rotation_validation_to_dict(fit.rotation_validation),
        "per_landmark": [
            {
                "landmark_name": entry.landmark_name,
                "cad_disagreement_mm": entry.cad_disagreement_mm,
                "error_mm": list(entry.error_mm),
            }
            for entry in fit.per_landmark
        ],
        "summary_mm": _metric_summary_mm_to_dict(fit.summary_mm),
    }


def _rotation_validation_to_dict(validation: RigidRotationValidation) -> dict[str, Any]:
    """Serialize rotation validation checks.

    Args:
        validation: Rotation matrix validation record.

    Returns:
        JSON-ready validation dict.
    """
    return {
        "determinant": validation.determinant,
        "orthonormality_frobenius_error": validation.orthonormality_frobenius_error,
        "is_proper_rotation": validation.is_proper_rotation,
    }


def _distance_report_to_dict(report: DistanceCadDisagreementReport) -> dict[str, Any]:
    """Serialize a distance disagreement report.

    Args:
        report: Pairwise distance disagreement report.

    Returns:
        JSON-ready distance report dict.
    """
    return {
        "distances": [
            {
                "start_landmark": entry.start_landmark,
                "end_landmark": entry.end_landmark,
                "cad_distance_mm": entry.cad_distance_mm,
                "marker_derived_distance_mm": entry.marker_derived_distance_mm,
                "cad_disagreement_mm": entry.cad_disagreement_mm,
            }
            for entry in report.distances
        ],
        "summary_mm": _metric_summary_mm_to_dict(report.summary_mm),
    }


def _leave_one_marker_cad_to_dict(prediction: LeaveOneMarkerCadPrediction) -> dict[str, Any]:
    """Serialize leave-one-marker-out CAD prediction metrics.

    Args:
        prediction: Leave-one-marker-out CAD prediction result.

    Returns:
        JSON-ready prediction dict.
    """
    return {
        "folds": [_leave_one_marker_cad_fold_to_dict(fold) for fold in prediction.folds],
        "eligible_fold_count": prediction.eligible_fold_count,
        "refused_fold_count": prediction.refused_fold_count,
        "all_excluded_summary_mm": _metric_summary_mm_to_dict(prediction.all_excluded_summary_mm),
    }


def _leave_one_marker_cad_fold_to_dict(fold: LeaveOneMarkerCadPredictionFold) -> dict[str, Any]:
    """Serialize one leave-one-marker-out CAD prediction fold.

    Args:
        fold: Single CAD prediction fold.

    Returns:
        JSON-ready fold dict.
    """
    return {
        "held_out_marker_id": fold.held_out_marker_id,
        "eligible": fold.eligible,
        "refusal_reason": fold.refusal_reason,
        "excluded_landmark_names": list(fold.excluded_landmark_names),
        "retained_landmark_count": fold.retained_landmark_count,
        "per_landmark_cad_disagreement_mm": dict(fold.per_landmark_cad_disagreement_mm),
        "summary_mm": _metric_summary_mm_to_dict(fold.summary_mm)
        if fold.summary_mm is not None
        else None,
    }


def _detection_candidate_to_dict(result: DetectionConsistencyCandidateResult) -> dict[str, Any]:
    """Serialize detection-consistency results for one candidate.

    Args:
        result: Per-candidate detection-consistency result.

    Returns:
        JSON-ready detection result dict.
    """
    return {
        "candidate_name": result.candidate_name,
        "compatible": result.compatible,
        "incompatibility_reason": result.incompatibility_reason,
        "summary_px": _metric_summary_px_to_dict(result.summary_px),
        "eligible_fold_count": result.eligible_fold_count,
        "possible_fold_count": result.possible_fold_count,
        "solve_failure_count": result.solve_failure_count,
        "ineligible_fold_count": result.ineligible_fold_count,
        "per_marker": [_per_marker_detection_to_dict(entry) for entry in result.per_marker],
        "visible_marker_count_strata": [
            _visible_marker_count_stratum_to_dict(entry)
            for entry in result.visible_marker_count_strata
        ],
        "per_source_video": [_source_video_detection_to_dict(entry) for entry in result.per_source_video],
    }


def _per_marker_detection_to_dict(entry: PerMarkerDetectionSummary) -> dict[str, Any]:
    """Serialize per-marker detection summary.

    Args:
        entry: Per-marker detection summary.

    Returns:
        JSON-ready per-marker dict.
    """
    return {
        "marker_id": entry.marker_id,
        "summary_px": _metric_summary_px_to_dict(entry.summary_px),
        "eligible_fold_count": entry.eligible_fold_count,
        "possible_fold_count": entry.possible_fold_count,
        "solve_failure_count": entry.solve_failure_count,
        "ineligible_fold_count": entry.ineligible_fold_count,
    }


def _visible_marker_count_stratum_to_dict(entry: VisibleMarkerCountStratum) -> dict[str, Any]:
    """Serialize visible-marker-count stratum summary.

    Args:
        entry: Stratum-level detection summary.

    Returns:
        JSON-ready stratum dict.
    """
    return {
        "visible_marker_count": entry.visible_marker_count,
        "summary_px": _metric_summary_px_to_dict(entry.summary_px),
        "eligible_fold_count": entry.eligible_fold_count,
        "possible_fold_count": entry.possible_fold_count,
        "solve_failure_count": entry.solve_failure_count,
        "ineligible_fold_count": entry.ineligible_fold_count,
    }


def _source_video_detection_to_dict(entry: SourceVideoDetectionSummary) -> dict[str, Any]:
    """Serialize per-video detection summary.

    Args:
        entry: Per-video detection summary.

    Returns:
        JSON-ready per-video dict.
    """
    return {
        "source_video": entry.source_video,
        "summary_px": _metric_summary_px_to_dict(entry.summary_px),
        "eligible_fold_count": entry.eligible_fold_count,
        "possible_fold_count": entry.possible_fold_count,
        "solve_failure_count": entry.solve_failure_count,
        "ineligible_fold_count": entry.ineligible_fold_count,
    }


def _metric_summary_mm_to_dict(summary: MetricSummaryMm | None) -> dict[str, Any] | None:
    """Serialize a millimeter metric summary.

    Args:
        summary: Metric summary, or ``None``.

    Returns:
        JSON-ready summary dict, or ``None``.
    """
    if summary is None:
        return None
    return {
        "count": summary.count,
        "min_mm": summary.min_mm,
        "median_mm": summary.median_mm,
        "rmse_mm": summary.rmse_mm,
        "p95_mm": summary.p95_mm,
        "max_mm": summary.max_mm,
    }


def _metric_summary_px_to_dict(summary: MetricSummaryPx) -> dict[str, Any]:
    """Serialize a pixel metric summary.

    Args:
        summary: Pixel metric summary.

    Returns:
        JSON-ready summary dict.
    """
    return {
        "count": summary.count,
        "min_px": summary.min_px,
        "median_px": summary.median_px,
        "rmse_px": summary.rmse_px,
        "p95_px": summary.p95_px,
        "max_px": summary.max_px,
    }


def _ranking_entry_to_dict(entry: CandidateRankingEntry) -> dict[str, Any]:
    """Serialize one ranking table entry.

    Args:
        entry: Candidate ranking entry.

    Returns:
        JSON-ready ranking entry dict.
    """
    return {
        "candidate_name": entry.candidate_name,
        "metric_value": entry.metric_value,
    }


def _grouping_to_dict(grouping: CandidateGroupingReport) -> dict[str, Any]:
    """Serialize candidate grouping metadata.

    Args:
        grouping: Repeatability grouping report.

    Returns:
        JSON-ready grouping dict.
    """
    return {
        "same_session_solver_groups": [
            {
                "capture_session": group.capture_session,
                "candidate_names": list(group.candidate_names),
                "solver_variants": list(group.solver_variants),
            }
            for group in grouping.same_session_solver_groups
        ],
        "cross_session_same_variant_groups": [
            {
                "solver_variant": group.solver_variant,
                "capture_sessions": list(group.capture_sessions),
                "candidate_names": list(group.candidate_names),
            }
            for group in grouping.cross_session_same_variant_groups
        ],
        "notes": list(grouping.notes),
    }


def _video_normalization_to_dict(summary: VideoNormalizationSummary) -> dict[str, Any]:
    """Serialize per-video normalization diagnostics.

    Args:
        summary: Per-video normalization summary.

    Returns:
        JSON-ready normalization dict.
    """
    return {
        "source_video": summary.source_video,
        "frame_count": summary.frame_count,
        "unknown_marker_ids": summary.unknown_marker_ids,
        "duplicate_marker_skips": summary.duplicate_marker_skips,
        "malformed_detections": summary.malformed_detections,
    }
