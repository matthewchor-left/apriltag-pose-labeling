"""Multi-view marker layout calibration from co-visible AprilTag corners.

This package exposes the calibration pipeline, input parsers, diagnostic types,
and assignment helpers used to recover marker footprints from multi-frame
corner observations.
"""

from __future__ import annotations

from object_apriltag.marker_layout_calibration.assignment import (
    FrameFallbackAssignmentResult,
    build_assignment_rejection_records,
    build_fallback_assignment_records,
    resolve_frame_ippe_assignment,
    resolve_frame_ippe_fallback_assignment,
    summarize_assignment_rejection_records,
    summarize_assignment_rejections,
)
from object_apriltag.marker_layout_calibration.discrete_graph import compute_live_pair_readiness
from object_apriltag.marker_layout_calibration.input import (
    parse_anchor_marker_ids,
    parse_marker_id_spec,
    parse_marker_size_override_spec,
    resolve_marker_sizes_for_calibration,
    uniform_marker_sizes,
)
from object_apriltag.marker_layout_calibration.pipeline import calibrate_marker_layout
from object_apriltag.marker_layout_calibration.solve_primitives import CalibrationSolveDiagnostics
from object_apriltag.marker_layout_calibration.types import (
    AnchorCoreBootstrapDiagnostics,
    AnchorCoreDiagnostics,
    AssignmentRejectionCauseCount,
    AssignmentRejectionCauseStats,
    AssignmentRejectionSummary,
    CalibrationQualityReport,
    CalibrationResult,
    CalibrationSettings,
    DroppedPairEdge,
    EdgeDiagnostics,
    FrameAssignmentRejection,
    FrameAssignmentRejectionRecord,
    FrameAssignmentResult,
    FrameFallbackAssignmentRecord,
    FrameObservation,
    LivePairReadinessDiagnostics,
    MarkerExpansionRecord,
    MeasurementDistribution,
    OmittedMarkerDiagnostic,
    PairReadinessEdge,
    QualityGateFailure,
    RestoredPairEdge,
)

__all__ = [
    "AnchorCoreBootstrapDiagnostics",
    "AnchorCoreDiagnostics",
    "AssignmentRejectionCauseCount",
    "AssignmentRejectionCauseStats",
    "AssignmentRejectionSummary",
    "CalibrationQualityReport",
    "CalibrationResult",
    "CalibrationSettings",
    "CalibrationSolveDiagnostics",
    "DroppedPairEdge",
    "EdgeDiagnostics",
    "FrameAssignmentRejection",
    "FrameAssignmentRejectionRecord",
    "FrameAssignmentResult",
    "FrameFallbackAssignmentRecord",
    "FrameFallbackAssignmentResult",
    "FrameObservation",
    "LivePairReadinessDiagnostics",
    "MarkerExpansionRecord",
    "MeasurementDistribution",
    "OmittedMarkerDiagnostic",
    "PairReadinessEdge",
    "QualityGateFailure",
    "RestoredPairEdge",
    "build_assignment_rejection_records",
    "build_fallback_assignment_records",
    "calibrate_marker_layout",
    "compute_live_pair_readiness",
    "parse_anchor_marker_ids",
    "parse_marker_id_spec",
    "parse_marker_size_override_spec",
    "resolve_frame_ippe_assignment",
    "resolve_frame_ippe_fallback_assignment",
    "resolve_marker_sizes_for_calibration",
    "summarize_assignment_rejection_records",
    "summarize_assignment_rejections",
    "uniform_marker_sizes",
]


def _self_check() -> None:
    """Run pipeline self-checks for import-time validation.

    Raises:
        AssertionError: When any pipeline invariant check fails.
    """
    from object_apriltag.marker_layout_calibration.pipeline import _self_check as pipeline_self_check

    pipeline_self_check()


def _input_boundary_self_check() -> None:
    """Run input-parser boundary self-checks.

    Raises:
        AssertionError: When any input validation invariant check fails.
    """
    from object_apriltag.marker_layout_calibration.pipeline import (
        _input_boundary_self_check as pipeline_input_boundary_self_check,
    )

    pipeline_input_boundary_self_check()
