"""Public calibration contracts and diagnostic types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from object_apriltag.layout import MarkerLayout
    from object_apriltag.marker_layout_calibration.solve_primitives import MarkerCandidate

MarkerPair = tuple[int, int]

DiscreteAssignmentMethod = Literal["pair_consensus", "rotation_consistent"]


@dataclass(frozen=True)
class FrameObservation:
    """One camera sample with expected marker corners in OpenCV order."""

    frame_id: str | int
    markers: dict[int, np.ndarray]


@dataclass(frozen=True)
class CalibrationSettings:
    """Thresholds and iteration limits for multi-view marker layout calibration."""

    min_inliers_per_edge: int = 20
    reprojection_rms_gate_px: float = 2.0
    pair_translation_rms_gate_ratio: float = 0.10
    pair_rotation_rms_gate_deg: float = 5.0
    huber_delta_px: float = 1.0
    corner_outlier_px: float = 3.0
    max_ba_iterations: int = 50
    discrete_method: DiscreteAssignmentMethod = "pair_consensus"


@dataclass(frozen=True)
class EdgeDiagnostics:
    """Per-edge pair-consensus quality metrics for one marker pair."""

    marker_a: int
    marker_b: int
    inlier_count: int
    translation_rms_m: float
    rotation_rms_deg: float


@dataclass(frozen=True)
class CalibrationQualityReport:
    """Aggregate diagnostic snapshot produced during or after calibration."""

    reprojection_rms_px: float
    per_marker_reprojection_rms_px: dict[int, float]
    edges: tuple[EdgeDiagnostics, ...]
    pair_translation_rms_max_m: float
    pair_rotation_rms_max_deg: float
    frame_count: int
    observation_count: int
    inlier_corner_count: int
    input_frame_count: int
    rejected_frame_count: int
    accepted_frame_count: int
    connected_marker_ids: frozenset[int]
    missing_expected_ids: frozenset[int]
    unused_expected_ids: frozenset[int]
    assignment_rejections: AssignmentRejectionSummary | None = None
    assignment_rejection_records: tuple[FrameAssignmentRejectionRecord, ...] | None = None
    fallback_assignment_records: tuple["FrameFallbackAssignmentRecord", ...] | None = None
    dropped_pair_edges: tuple[DroppedPairEdge, ...] | None = None
    restored_pair_edges: tuple[RestoredPairEdge, ...] | None = None
    anchor_core: "AnchorCoreDiagnostics | None" = None


@dataclass(frozen=True)
class MarkerExpansionRecord:
    """Outcome of one anchor-core expansion attempt for a candidate marker."""

    marker_id: int
    status: str
    support_frames: int = 0
    reason: str | None = None
    stage: str = "expand"


@dataclass(frozen=True)
class AnchorCoreBootstrapDiagnostics:
    """Summary of the anchor-core bootstrap phase before hierarchical expansion."""

    status: str
    frames_considered: int
    frames_accepted: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class AnchorCoreDiagnostics:
    """End-to-end anchor-core solve trace from bootstrap through expansion."""

    mode: str
    configured_anchor_ids: tuple[int, ...]
    bootstrap: AnchorCoreBootstrapDiagnostics
    expansion: tuple[MarkerExpansionRecord, ...]
    final_solved_ids: frozenset[int]
    unresolved_ids: frozenset[int]
    stopped_after_expansion: bool = False


@dataclass(frozen=True)
class OmittedMarkerDiagnostic:
    """Records why a requested marker ID was omitted from the calibration output."""

    marker_id: int
    reason: str


@dataclass(frozen=True)
class QualityGateFailure:
    """Categorized quality-gate failure.

    ``strict`` gates may be waived under best-effort policy.
    """

    category: Literal["strict", "connectivity", "data"]
    message: str


@dataclass(frozen=True)
class CalibrationResult:
    """Outcome of a calibration attempt, including layout, quality, and refusal metadata."""

    layout: MarkerLayout | None
    quality: CalibrationQualityReport | None
    failure_reason: str | None
    outcome: Literal["accepted", "provisional", "partial", "refused"] | None = None
    calibration_policy: Literal["strict", "best_effort"] = "strict"
    failed_quality_gates: tuple[str, ...] = ()
    selected_checkpoint_stage: str | None = None
    failed_refinement_stage: str | None = None
    omitted_markers: tuple[OmittedMarkerDiagnostic, ...] = ()
    partial_output: bool = False

    def __post_init__(self) -> None:
        """Infer ``outcome`` from layout, failure, and omission fields when not preset."""
        if self.outcome is not None:
            return
        if self.layout is not None and self.failure_reason is None:
            resolved: Literal["accepted", "provisional", "partial", "refused"] = (
                "partial"
                if self.omitted_markers
                else (
                    "provisional"
                    if self.failed_quality_gates or self.failed_refinement_stage
                    else "accepted"
                )
            )
        else:
            resolved = "refused"
        object.__setattr__(self, "outcome", resolved)


@dataclass(frozen=True)
class PairReadinessEdge:
    """Live pair-readiness status for one marker pair before full calibration."""

    marker_a: int
    marker_b: int
    raw_covisible_frames: int
    robust_inlier_count: int
    translation_rms_m: float | None
    rotation_rms_deg: float | None
    status: str


@dataclass(frozen=True)
class LivePairReadinessDiagnostics:
    """Snapshot of marker-pair readiness and connectivity from a live observation sample."""

    pairs: tuple[PairReadinessEdge, ...]
    connected_marker_ids: frozenset[int]
    missing_marker_ids: frozenset[int]
    sample_count: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class FrameAssignmentRejection:
    """Why IPPE assignment failed for one frame."""

    reason: str
    marker_pair: MarkerPair | None = None
    translation_error_m: float | None = None
    rotation_error_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None


@dataclass(frozen=True)
class FrameAssignmentResult:
    """Per-frame IPPE assignment outcome or rejection detail."""

    assignment: dict[int, "MarkerCandidate"] | None
    rejection: FrameAssignmentRejection | None


@dataclass(frozen=True)
class AssignmentRejectionCauseCount:
    """Occurrence count for one assignment-rejection cause."""

    reason: str
    marker_pair: MarkerPair | None
    count: int


@dataclass(frozen=True)
class MeasurementDistribution:
    """Summary statistics for a scalar measurement sample."""

    min: float | None
    median: float | None
    p95: float | None
    max: float | None


@dataclass(frozen=True)
class FrameAssignmentRejectionRecord:
    """One rejected frame preserved for assignment diagnostics."""

    frame_index: int
    frame_id: str | int
    visible_marker_ids: tuple[int, ...]
    reason: str
    marker_pair: MarkerPair | None = None
    translation_error_m: float | None = None
    rotation_error_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None


@dataclass(frozen=True)
class FrameFallbackAssignmentRecord:
    """One fallback IPPE assignment chosen after primary assignment failed."""

    frame_index: int
    frame_id: str | int
    visible_marker_ids: tuple[int, ...]
    disagreement_cost: float
    marker_pair: MarkerPair | None = None
    translation_error_m: float | None = None
    rotation_error_deg: float | None = None


@dataclass(frozen=True)
class FrameFallbackAssignment:
    """Compact fallback-assignment summary for solve-time bookkeeping."""

    frame_index: int
    disagreement_cost: float
    marker_pair: MarkerPair | None
    translation_error_m: float
    rotation_error_deg: float


@dataclass(frozen=True)
class AssignmentRejectionCauseStats:
    """Aggregated assignment-rejection statistics for one cause."""

    reason: str
    marker_pair: MarkerPair | None
    count: int
    sample_frame_ids: tuple[str | int, ...]
    translation_error_m: MeasurementDistribution | None
    rotation_error_deg: MeasurementDistribution | None
    translation_gate_m: float | None
    rotation_gate_deg: float | None
    translation_error_ratio: MeasurementDistribution | None
    rotation_error_ratio: MeasurementDistribution | None


@dataclass(frozen=True)
class AssignmentRejectionSummary:
    """Roll-up of frame assignment rejections across a calibration run."""

    total_rejected: int
    by_reason: tuple[tuple[str, int], ...]
    by_pair: tuple[tuple[MarkerPair, int], ...]
    top_causes: tuple[AssignmentRejectionCauseCount, ...]
    by_cause: tuple[AssignmentRejectionCauseStats, ...] = ()


@dataclass(frozen=True)
class DroppedPairEdge:
    """Pair edge removed from consensus during pruning or support filtering."""

    marker_a: int
    marker_b: int
    stage: str
    reason: str
    observed_count: int
    supported_count: int
    required_count: int
    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None

    @property
    def marker_pair(self) -> MarkerPair:
        """Return the pair key used to index pair-consensus maps.

        Returns:
            ``(marker_a, marker_b)``.
        """
        return (self.marker_a, self.marker_b)


@dataclass(frozen=True)
class RestoredPairEdge:
    """Pair edge reinstated into consensus after an earlier drop."""

    marker_a: int
    marker_b: int
    stage: str
    original_stage: str
    original_reason: str
    observed_count: int
    supported_count: int
    required_count: int
    support_fraction: float
    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None

    @property
    def marker_pair(self) -> MarkerPair:
        """Return the pair key used to index pair-consensus maps.

        Returns:
            ``(marker_a, marker_b)``.
        """
        return (self.marker_a, self.marker_b)
