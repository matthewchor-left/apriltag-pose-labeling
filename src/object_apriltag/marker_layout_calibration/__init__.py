"""Multi-view marker layout calibration from co-visible AprilTag corners.

This package exposes the calibration pipeline, input parsers, diagnostic types,
and assignment helpers used to recover marker footprints from multi-frame
corner observations.
"""

from __future__ import annotations

from object_apriltag.marker_layout_calibration.input import (
    parse_marker_id_spec,
    parse_marker_size_override_spec,
    resolve_marker_sizes_for_calibration,
    uniform_marker_sizes,
)
from object_apriltag.marker_layout_calibration.pipeline import calibrate_marker_layout
from object_apriltag.marker_layout_calibration.rotation_consistent_assignment import (
    build_assignment_rejection_records,
    summarize_assignment_rejection_records,
    summarize_assignment_rejections,
)
from object_apriltag.marker_layout_calibration.solve_primitives import CalibrationSolveDiagnostics
from object_apriltag.marker_layout_calibration.types import (
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
    FrameObservation,
    MeasurementDistribution,
    OmittedMarkerDiagnostic,
    QualityGateFailure,
    RestoredPairEdge,
)

__all__ = [
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
    "FrameObservation",
    "MeasurementDistribution",
    "OmittedMarkerDiagnostic",
    "QualityGateFailure",
    "RestoredPairEdge",
    "build_assignment_rejection_records",
    "calibrate_marker_layout",
    "parse_marker_id_spec",
    "parse_marker_size_override_spec",
    "resolve_marker_sizes_for_calibration",
    "summarize_assignment_rejection_records",
    "summarize_assignment_rejections",
    "uniform_marker_sizes",
]


def _self_check() -> None:
    """Run pipeline self-checks for import-time validation."""
    from object_apriltag.marker_layout_calibration.pipeline import _self_check as pipeline_self_check

    pipeline_self_check()


def _input_boundary_self_check() -> None:
    """Run input-parser boundary self-checks."""
    from object_apriltag.marker_layout_calibration.pipeline import (
        _input_boundary_self_check as pipeline_input_boundary_self_check,
    )

    pipeline_input_boundary_self_check()
