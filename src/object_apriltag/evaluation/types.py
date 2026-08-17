"""Typed CAD geometry evaluation results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummaryMm:
    """Millimeter-scale error distribution summary.

    Attributes:
        count: Number of samples summarized.
        min_mm: Minimum error in millimeters.
        median_mm: Median error in millimeters.
        rmse_mm: Root-mean-square error in millimeters.
        p95_mm: 95th-percentile error in millimeters.
        max_mm: Maximum error in millimeters.
    """

    count: int
    min_mm: float
    median_mm: float
    rmse_mm: float
    p95_mm: float
    max_mm: float


@dataclass(frozen=True)
class RigidRotationValidation:
    """Sanity checks for a fitted 3x3 rotation matrix.

    Attributes:
        determinant: Matrix determinant (should be +1 for a proper rotation).
        orthonormality_frobenius_error: Frobenius norm of ``R.T @ R - I``.
        is_proper_rotation: Whether determinant is positive and orthonormality is tight.
    """

    determinant: float
    orthonormality_frobenius_error: float
    is_proper_rotation: bool


@dataclass(frozen=True)
class LandmarkCadDisagreement:
    """Per-landmark CAD-to-marker disagreement after rigid alignment.

    Attributes:
        landmark_name: Named landmark from ``keypoint_sources``.
        cad_disagreement_mm: Euclidean error norm in millimeters.
        error_mm: Component-wise error vector in millimeters.
    """

    landmark_name: str
    cad_disagreement_mm: float
    error_mm: tuple[float, float, float]


@dataclass(frozen=True)
class RigidCadFit:
    """Kabsch rigid fit mapping CAD landmarks onto marker-derived landmarks.

    Attributes:
        rotation: 3x3 rotation matrix as nested tuples.
        translation_m: Translation vector in meters.
        rotation_validation: Rotation matrix sanity checks.
        per_landmark: Per-landmark disagreement after alignment.
        summary_mm: Aggregate disagreement statistics in millimeters.
    """

    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation_m: tuple[float, float, float]
    rotation_validation: RigidRotationValidation
    per_landmark: tuple[LandmarkCadDisagreement, ...]
    summary_mm: MetricSummaryMm


@dataclass(frozen=True)
class DistanceCadDisagreement:
    """Pairwise distance disagreement between CAD and marker-derived geometry.

    Attributes:
        start_landmark: Start landmark name.
        end_landmark: End landmark name.
        cad_distance_mm: Edge length from CAD landmarks in millimeters.
        marker_derived_distance_mm: Edge length from marker-derived landmarks in millimeters.
        cad_disagreement_mm: Absolute distance difference in millimeters.
    """

    start_landmark: str
    end_landmark: str
    cad_distance_mm: float
    marker_derived_distance_mm: float
    cad_disagreement_mm: float


@dataclass(frozen=True)
class DistanceCadDisagreementReport:
    """Collection of pairwise distance disagreements with aggregate summary.

    Attributes:
        distances: Per-edge distance disagreement records.
        summary_mm: Aggregate disagreement statistics in millimeters.
    """

    distances: tuple[DistanceCadDisagreement, ...]
    summary_mm: MetricSummaryMm


@dataclass(frozen=True)
class LeaveOneMarkerCadPredictionFold:
    """Leave-one-marker-out CAD prediction fold for one held-out marker.

    Attributes:
        held_out_marker_id: Marker ID excluded from the fit.
        eligible: Whether the fold met minimum retained-landmark requirements.
        refusal_reason: Human-readable refusal when ``eligible`` is false.
        excluded_landmark_names: Landmark names tied to the held-out marker.
        retained_landmark_count: Number of landmarks used for the fit.
        per_landmark_cad_disagreement_mm: Held-out landmark errors in millimeters.
        summary_mm: Fold-level error summary, or ``None`` when ineligible.
    """

    held_out_marker_id: int
    eligible: bool
    refusal_reason: str | None
    excluded_landmark_names: tuple[str, ...]
    retained_landmark_count: int
    per_landmark_cad_disagreement_mm: dict[str, float]
    summary_mm: MetricSummaryMm | None


@dataclass(frozen=True)
class LeaveOneMarkerCadPrediction:
    """Leave-one-marker-out CAD geometry prediction across all markers.

    Attributes:
        folds: Per-marker fold results.
        eligible_fold_count: Number of folds that completed successfully.
        refused_fold_count: Number of folds refused for degeneracy or insufficient landmarks.
        all_excluded_summary_mm: Aggregate error over all held-out landmarks.
    """

    folds: tuple[LeaveOneMarkerCadPredictionFold, ...]
    eligible_fold_count: int
    refused_fold_count: int
    all_excluded_summary_mm: MetricSummaryMm


@dataclass(frozen=True)
class CadGeometryEvaluation:
    """Full CAD-vs-marker geometry evaluation for one candidate layout.

    Attributes:
        landmark_names: Ordered landmark names from ``keypoint_sources``.
        cad_landmarks_m: CAD landmark positions in meters.
        marker_derived_landmarks_m: Marker-layout-derived positions in meters.
        rigid_fit: Global rigid alignment metrics.
        pair_distance_disagreement: All-pairs distance disagreement report.
        skeleton_edge_disagreement: Skeleton-edge distance disagreement report.
        leave_one_marker_out: Leave-one-marker-out prediction metrics.
    """

    landmark_names: tuple[str, ...]
    cad_landmarks_m: dict[str, tuple[float, float, float]]
    marker_derived_landmarks_m: dict[str, tuple[float, float, float]]
    rigid_fit: RigidCadFit
    pair_distance_disagreement: DistanceCadDisagreementReport
    skeleton_edge_disagreement: DistanceCadDisagreementReport
    leave_one_marker_out: LeaveOneMarkerCadPrediction


@dataclass(frozen=True)
class MetricSummaryPx:
    """Pixel-scale error distribution summary.

    Attributes:
        count: Number of samples summarized.
        min_px: Minimum error in pixels.
        median_px: Median error in pixels.
        rmse_px: Root-mean-square error in pixels.
        p95_px: 95th-percentile error in pixels.
        max_px: Maximum error in pixels.
    """

    count: int
    min_px: float
    median_px: float
    rmse_px: float
    p95_px: float
    max_px: float


@dataclass(frozen=True)
class LeaveOneMarkerDetectionFold:
    """Single held-out marker detection-consistency fold.

    Attributes:
        source_video: Repo-relative path of the source video.
        frame_index: Zero-based frame index within the video.
        held_out_marker_id: Marker ID excluded from pose solve.
        visible_marker_count: Total expected markers visible in the frame.
        eligible: Whether the fold had enough training markers to attempt solve.
        refusal_reason: Human-readable refusal when ``eligible`` is false.
        solve_failed: Whether pose solve or projection failed on an eligible fold.
        corner_errors_px: Per-corner reprojection errors, or ``None`` on failure.
        summary_px: Fold-level error summary, or ``None`` on failure.
    """

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
    """Detection-consistency summary aggregated for one marker ID.

    Attributes:
        marker_id: Marker ID, or ``-1`` for non-marker groupings.
        summary_px: Aggregate corner error statistics in pixels.
        eligible_fold_count: Folds with enough training markers to attempt solve.
        possible_fold_count: Total folds where the marker was visible.
        solve_failure_count: Eligible folds where pose solve or projection failed.
        ineligible_fold_count: Folds refused for insufficient training markers.
    """

    marker_id: int
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int


@dataclass(frozen=True)
class VisibleMarkerCountStratum:
    """Detection-consistency summary for one visible-marker-count stratum.

    Attributes:
        visible_marker_count: Number of expected markers visible in the frame.
        summary_px: Aggregate corner error statistics in pixels.
        eligible_fold_count: Folds with enough training markers to attempt solve.
        possible_fold_count: Total folds in this stratum.
        solve_failure_count: Eligible folds where pose solve or projection failed.
        ineligible_fold_count: Folds refused for insufficient training markers.
    """

    visible_marker_count: int
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int


@dataclass(frozen=True)
class SourceVideoDetectionSummary:
    """Detection-consistency summary aggregated for one held-out video.

    Attributes:
        source_video: Repo-relative path of the source video.
        summary_px: Aggregate corner error statistics in pixels.
        eligible_fold_count: Folds with enough training markers to attempt solve.
        possible_fold_count: Total folds in this video.
        solve_failure_count: Eligible folds where pose solve or projection failed.
        ineligible_fold_count: Folds refused for insufficient training markers.
    """

    source_video: str
    summary_px: MetricSummaryPx
    eligible_fold_count: int
    possible_fold_count: int
    solve_failure_count: int
    ineligible_fold_count: int


@dataclass(frozen=True)
class DetectionConsistencyCandidateResult:
    """Detection-consistency evaluation result for one candidate layout.

    Attributes:
        candidate_name: Candidate name from the evaluation manifest.
        compatible: Whether layout marker IDs match the expected set.
        incompatibility_reason: Refusal reason when ``compatible`` is false.
        summary_px: Aggregate corner error statistics in pixels.
        eligible_fold_count: Folds with enough training markers to attempt solve.
        possible_fold_count: Total folds evaluated.
        solve_failure_count: Eligible folds where pose solve or projection failed.
        ineligible_fold_count: Folds refused for insufficient training markers.
        per_marker: Per-marker breakdowns.
        visible_marker_count_strata: Breakdowns by visible marker count.
        per_source_video: Per-video breakdowns.
    """

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
    """Detection-consistency evaluation across all candidate layouts.

    Attributes:
        expected_marker_ids: Marker IDs required by the evaluation manifest.
        candidates: Per-candidate detection-consistency results.
    """

    expected_marker_ids: tuple[int, ...]
    candidates: tuple[DetectionConsistencyCandidateResult, ...]
