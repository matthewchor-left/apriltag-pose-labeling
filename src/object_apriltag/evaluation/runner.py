"""Marker-model evaluation orchestration."""

from __future__ import annotations

from pathlib import Path

from object_apriltag.apriltag import build_apriltag_detector
from object_apriltag.cad import load_cad_landmarks
from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.evaluation.cad_geometry import evaluate_cad_geometry
from object_apriltag.evaluation.detection_consistency import (
    DetectionCandidate,
    evaluate_detection_consistency,
)
from object_apriltag.evaluation.manifest import (
    EvaluationManifest,
    load_candidate_layouts,
    load_evaluation_manifest,
    validate_object_model_correspondence,
)
from object_apriltag.evaluation.orchestration import freeze_held_out_video_detections
from object_apriltag.evaluation.report import (
    CandidateEvaluationResult,
    MarkerModelEvaluationReport,
    build_marker_model_evaluation_report,
)
from object_apriltag.object_model_edit import load_object_model_document


def evaluate_marker_models_from_manifest(
    manifest_path: str | Path,
) -> MarkerModelEvaluationReport:
    """Run marker-model evaluation from a manifest file path.

    Args:
        manifest_path: Path to the evaluation manifest JSON file.

    Returns:
        Complete marker-model evaluation report.
    """
    manifest = load_evaluation_manifest(manifest_path)
    return evaluate_marker_models(manifest)


def evaluate_marker_models(manifest: EvaluationManifest) -> MarkerModelEvaluationReport:
    """Run CAD geometry and detection-consistency evaluation for all candidates.

    Args:
        manifest: Parsed and validated evaluation manifest.

    Returns:
        Complete marker-model evaluation report with per-candidate metrics,
        rankings, and grouping.

    Raises:
        FileNotFoundError: If manifest input files or held-out videos are missing.
    """
    _validate_manifest_inputs(manifest)

    _, object_model_document = load_object_model_document(manifest.object_model)
    layouts, expected_marker_ids = load_candidate_layouts(manifest)
    landmark_names, expected_marker_id_tuple = validate_object_model_correspondence(
        object_model_document,
        expected_marker_ids,
    )
    cad_landmarks = load_cad_landmarks(manifest.cad_model, required_names=landmark_names)

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        manifest.intrinsics
    )
    calibration_width, calibration_height = require_calibration_image_size(
        image_width,
        image_height,
        manifest.intrinsics,
    )

    detector = build_apriltag_detector(
        dictionary=manifest.detector.dictionary,
        sensitivity=manifest.detector.sensitivity,
    )

    frozen_videos = []
    normalization_summaries = []
    for held_out_video in manifest.held_out_videos:
        frozen, summary = freeze_held_out_video_detections(
            held_out_video.path,
            detector,
            calibration_width=calibration_width,
            calibration_height=calibration_height,
            expected_marker_ids=expected_marker_ids,
        )
        frozen_videos.append(frozen)
        normalization_summaries.append(summary)

    detection_candidates = tuple(
        DetectionCandidate(name=name, layout=layout) for name, layout in layouts.items()
    )
    detection_evaluation = evaluate_detection_consistency(
        expected_marker_ids=expected_marker_ids,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        videos=tuple(frozen_videos),
        candidates=detection_candidates,
    )
    detection_by_name = {
        candidate.candidate_name: candidate for candidate in detection_evaluation.candidates
    }

    candidate_results: list[CandidateEvaluationResult] = []
    for candidate in manifest.candidates:
        layout = layouts[candidate.name]
        cad_geometry = evaluate_cad_geometry(
            cad_landmarks,
            object_model_document,
            layout,
        )
        candidate_results.append(
            CandidateEvaluationResult(
                name=candidate.name,
                capture_session=candidate.capture_session,
                solver_variant=candidate.solver_variant,
                marker_model_path=candidate.marker_model_path,
                calibration_source_video=candidate.calibration_source.video,
                cad_geometry=cad_geometry,
                detection=detection_by_name[candidate.name],
            )
        )

    return build_marker_model_evaluation_report(
        manifest=manifest,
        landmark_names=landmark_names,
        expected_marker_ids=expected_marker_id_tuple,
        calibration_width=calibration_width,
        calibration_height=calibration_height,
        calibration_source=calibration_source,
        candidate_results=tuple(candidate_results),
        normalization_summaries=tuple(normalization_summaries),
    )


def _validate_manifest_inputs(manifest: EvaluationManifest) -> None:
    """Verify that manifest input files exist on disk.

    Args:
        manifest: Evaluation manifest to validate.

    Raises:
        FileNotFoundError: If a required input path does not exist.
    """
    for path, label in (
        (manifest.cad_model, "cad_model"),
        (manifest.object_model, "object_model"),
        (manifest.intrinsics, "intrinsics"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest {label} not found: {path}.")
    for held_out_video in manifest.held_out_videos:
        if not held_out_video.path.is_file():
            raise FileNotFoundError(f"Held-out video not found: {held_out_video.path}.")
    for candidate in manifest.candidates:
        if candidate.calibration_source.video is not None and not candidate.calibration_source.video.is_file():
            raise FileNotFoundError(
                f"Candidate {candidate.name!r} calibration_source.video not found: "
                f"{candidate.calibration_source.video}."
            )
