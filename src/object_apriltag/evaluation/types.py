"""Typed CAD geometry evaluation results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummaryMm:
    count: int
    min_mm: float
    median_mm: float
    rmse_mm: float
    p95_mm: float
    max_mm: float


@dataclass(frozen=True)
class RigidRotationValidation:
    determinant: float
    orthonormality_frobenius_error: float
    is_proper_rotation: bool


@dataclass(frozen=True)
class LandmarkCadDisagreement:
    landmark_name: str
    cad_disagreement_mm: float
    error_mm: tuple[float, float, float]


@dataclass(frozen=True)
class RigidCadFit:
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation_m: tuple[float, float, float]
    rotation_validation: RigidRotationValidation
    per_landmark: tuple[LandmarkCadDisagreement, ...]
    summary_mm: MetricSummaryMm


@dataclass(frozen=True)
class DistanceCadDisagreement:
    start_landmark: str
    end_landmark: str
    cad_distance_mm: float
    marker_derived_distance_mm: float
    cad_disagreement_mm: float


@dataclass(frozen=True)
class DistanceCadDisagreementReport:
    distances: tuple[DistanceCadDisagreement, ...]
    summary_mm: MetricSummaryMm


@dataclass(frozen=True)
class LeaveOneMarkerCadPredictionFold:
    held_out_marker_id: int
    eligible: bool
    refusal_reason: str | None
    excluded_landmark_names: tuple[str, ...]
    retained_landmark_count: int
    per_landmark_cad_disagreement_mm: dict[str, float]
    summary_mm: MetricSummaryMm | None


@dataclass(frozen=True)
class LeaveOneMarkerCadPrediction:
    folds: tuple[LeaveOneMarkerCadPredictionFold, ...]
    eligible_fold_count: int
    refused_fold_count: int
    all_excluded_summary_mm: MetricSummaryMm


@dataclass(frozen=True)
class CadGeometryEvaluation:
    landmark_names: tuple[str, ...]
    cad_landmarks_m: dict[str, tuple[float, float, float]]
    marker_derived_landmarks_m: dict[str, tuple[float, float, float]]
    rigid_fit: RigidCadFit
    pair_distance_disagreement: DistanceCadDisagreementReport
    skeleton_edge_disagreement: DistanceCadDisagreementReport
    leave_one_marker_out: LeaveOneMarkerCadPrediction


@dataclass(frozen=True)
class MetricSummaryPx:
    count: int
    min_px: float
    median_px: float
    rmse_px: float
    p95_px: float
    max_px: float


@dataclass(frozen=True)
class LeaveOneMarkerDetectionFold:
    source_video: str
    frame_index: int
    held_out_marker_id: int
    visible_marker_count: int
    eligible: bool
    refusal_reason: str | None
    solve_failed: bool
    corner_errors_px: tuple[float, float, float, float] | None
    summary_px: MetricSummaryPx | None


@dataclass(frozen=True)
class PerMarkerDetectionSummary:
    marker_id: int
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int


@dataclass(frozen=True)
class VisibleMarkerCountStratum:
    visible_marker_count: int
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int


@dataclass(frozen=True)
class SourceVideoDetectionSummary:
    source_video: str
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int


@dataclass(frozen=True)
class DetectionConsistencyCandidateResult:
    candidate_name: str
    compatible: bool
    incompatibility_reason: str | None
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int
    per_marker: tuple[PerMarkerDetectionSummary, ...]
    visible_marker_count_strata: tuple[VisibleMarkerCountStratum, ...]
    per_source_video: tuple[SourceVideoDetectionSummary, ...]


@dataclass(frozen=True)
class DetectionConsistencyEvaluation:
    expected_marker_ids: tuple[int, ...]
    candidates: tuple[DetectionConsistencyCandidateResult, ...]
