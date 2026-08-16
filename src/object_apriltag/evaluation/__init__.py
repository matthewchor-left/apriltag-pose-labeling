"""Evaluation helpers for CAD geometry and detection consistency."""

from object_apriltag.evaluation.cad_geometry import (
    derive_marker_derived_landmarks,
    evaluate_cad_geometry,
    fit_cad_registration,
    metric_summary_mm,
)
from object_apriltag.evaluation.detection_consistency import (
    DetectionCandidate,
    FrozenFrameDetections,
    FrozenVideoDetections,
    evaluate_detection_consistency,
    metric_summary_px,
    normalize_frame_detections,
)
from object_apriltag.evaluation.kabsch import kabsch_rigid_transform, validate_rigid_rotation
from object_apriltag.evaluation.manifest import (
    EVALUATION_MANIFEST_VERSION,
    EvaluationManifest,
    load_evaluation_manifest,
    resolve_manifest_path,
)
from object_apriltag.evaluation.orchestration import freeze_held_out_video_detections
from object_apriltag.evaluation.report import (
    MARKER_MODEL_EVALUATION_REPORT_VERSION,
    MarkerModelEvaluationReport,
    build_marker_model_evaluation_document,
    format_marker_model_evaluation_console_summary,
    save_marker_model_evaluation_report,
)
from object_apriltag.evaluation.runner import (
    evaluate_marker_models,
    evaluate_marker_models_from_manifest,
)
from object_apriltag.evaluation.types import (
    CadGeometryEvaluation,
    DetectionConsistencyEvaluation,
    MetricSummaryMm,
    MetricSummaryPx,
)

__all__ = [
    "CadGeometryEvaluation",
    "DetectionCandidate",
    "DetectionConsistencyEvaluation",
    "EVALUATION_MANIFEST_VERSION",
    "EvaluationManifest",
    "FrozenFrameDetections",
    "FrozenVideoDetections",
    "MARKER_MODEL_EVALUATION_REPORT_VERSION",
    "MarkerModelEvaluationReport",
    "MetricSummaryMm",
    "MetricSummaryPx",
    "build_marker_model_evaluation_document",
    "derive_marker_derived_landmarks",
    "evaluate_cad_geometry",
    "evaluate_detection_consistency",
    "evaluate_marker_models",
    "evaluate_marker_models_from_manifest",
    "fit_cad_registration",
    "format_marker_model_evaluation_console_summary",
    "freeze_held_out_video_detections",
    "kabsch_rigid_transform",
    "load_evaluation_manifest",
    "metric_summary_mm",
    "metric_summary_px",
    "normalize_frame_detections",
    "resolve_manifest_path",
    "save_marker_model_evaluation_report",
    "validate_rigid_rotation",
]
