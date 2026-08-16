"""Tests for manifest-driven marker-model evaluation (Stage 3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.evaluation.manifest import (
    EVALUATION_MANIFEST_VERSION,
    load_candidate_layouts,
    load_evaluation_manifest,
    resolve_manifest_path,
)
from object_apriltag.evaluation.orchestration import freeze_held_out_video_detections
from object_apriltag.evaluation.report import (
    MARKER_MODEL_EVALUATION_REPORT_VERSION,
    CandidateEvaluationResult,
    MarkerModelEvaluationReport,
    build_marker_model_evaluation_document,
    build_marker_model_evaluation_report,
    format_marker_model_evaluation_console_summary,
    save_marker_model_evaluation_report,
)
from object_apriltag.evaluation.runner import evaluate_marker_models
from object_apriltag.evaluation.types import MetricSummaryMm, MetricSummaryPx
from object_apriltag.cli.evaluate_marker_model import main as evaluate_marker_model_main
from object_apriltag.layout import build_marker_layout, footprint_from_dict, save_marker_model

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_MANIFEST = REPO_ROOT / "config/evaluation/playground_static_4_tag/manifest.json"


def _square_footprint(marker_id: int, top_left: list[float], marker_size: float = 0.04) -> object:
    x, y, z = top_left
    return footprint_from_dict(
        marker_id,
        {
            "top_left": [x, y, z],
            "top_right": [x + marker_size, y, z],
            "bottom_right": [x + marker_size, y + marker_size, z],
            "bottom_left": [x, y + marker_size, z],
        },
    )


def _write_marker_model(path: Path, marker_ids: list[int]) -> None:
    footprints = {
        marker_id: _square_footprint(marker_id, [index * 0.1, 0.0, 0.0])
        for index, marker_id in enumerate(marker_ids)
    }
    layout = build_marker_layout(marker_ids[0], 0.04, footprints)
    save_marker_model(path, layout)


def _write_intrinsics(path: Path, *, width: int, height: int) -> None:
    path.write_text(
        json.dumps(
            {
                "calibration_source": "test",
                "image_size": [width, height],
                "camera_matrix": [[900.0, 0.0, width / 2], [0.0, 900.0, height / 2], [0.0, 0.0, 1.0]],
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )


def _write_object_model(path: Path, marker_ids: list[int]) -> None:
    names = [f"kp_{marker_id}" for marker_id in marker_ids[:4]]
    path.write_text(
        json.dumps(
            {
                "units": "meters",
                "coordinate_frame": "marker_model",
                "keypoint_sources": {
                    name: {"marker_id": marker_id, "corner": "top_left"}
                    for name, marker_id in zip(names, marker_ids[:4], strict=True)
                },
                "keypoints": {name: [0.0, 0.0, 0.0] for name in names},
                "skeleton": [[names[0], names[1]]] if len(names) >= 2 else [],
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    *,
    workspace: Path,
    marker_ids: list[int],
    candidates: list[dict[str, object]] | None = None,
    held_out: bool = True,
) -> None:
    marker_model = workspace / "marker_a.json"
    _write_marker_model(marker_model, marker_ids)
    if candidates is None:
        candidates = [
            {
                "name": "candidate_a",
                "marker_model": str(marker_model),
                "capture_session": "session_a",
                "solver_variant": "default",
            }
        ]
    else:
        candidates = [
            {
                **candidate,
                "marker_model": str(workspace / str(candidate["marker_model"])),
            }
            for candidate in candidates
        ]
    payload = {
        "manifest_version": EVALUATION_MANIFEST_VERSION,
        "cad_model": str(workspace / "cad.glb"),
        "object_model": str(workspace / "object_model.json"),
        "intrinsics": str(workspace / "intrinsic.json"),
        "detector": {"dictionary": "36h11", "sensitivity": "relaxed"},
        "held_out_videos": [{"path": str(workspace / "held_out.mov"), "held_out": held_out}],
        "candidates": candidates,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_synthetic_video(path: Path, *, width: int, height: int, frame_count: int = 2) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open synthetic video writer.")
    try:
        for _ in range(frame_count):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    finally:
        writer.release()


class MarkerModelEvaluationManifestTests(unittest.TestCase):
    def test_resolve_manifest_path_uses_repo_root(self) -> None:
        manifest_path = REPO_ROOT / "config/evaluation/playground_static_4_tag/manifest.json"
        resolved = resolve_manifest_path(
            "config/Model/playground_static_4_tag/marker_model.json",
            manifest_path=manifest_path,
        )
        self.assertEqual(
            resolved,
            (REPO_ROOT / "config/Model/playground_static_4_tag/marker_model.json").resolve(),
        )

    def test_example_manifest_loads_and_requires_held_out(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        self.assertEqual(manifest.version, EVALUATION_MANIFEST_VERSION)
        self.assertTrue(all(video.held_out for video in manifest.held_out_videos))
        self.assertEqual(len(manifest.candidates), 1)

    def test_manifest_rejects_missing_held_out_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.json"
            _write_manifest(manifest_path, workspace=root, marker_ids=[0, 1, 2, 3], held_out=False)
            with self.assertRaisesRegex(ValueError, "held_out: true"):
                load_evaluation_manifest(manifest_path)

    def test_candidate_marker_ids_must_match(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        with tempfile.TemporaryDirectory() as tmp_dir:
            other = Path(tmp_dir) / "other.json"
            _write_marker_model(other, [0, 1, 2])
            broken = manifest
            object.__setattr__(
                broken,
                "candidates",
                (
                    *manifest.candidates,
                    type(manifest.candidates[0])(
                        name="mismatch",
                        marker_model_path=other,
                        capture_session="x",
                        solver_variant="default",
                        calibration_source=manifest.candidates[0].calibration_source,
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "marker IDs"):
                load_candidate_layouts(broken)


class MarkerModelEvaluationReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._marker_model_path = Path(self._temp_dir.name) / "marker.json"
        _write_marker_model(self._marker_model_path, [0, 1, 2, 3])

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def _minimal_candidate_result(self, name: str, rmse_mm: float, p95_px: float) -> CandidateEvaluationResult:
        from object_apriltag.evaluation.types import (
            CadGeometryEvaluation,
            DetectionConsistencyCandidateResult,
            DistanceCadDisagreementReport,
            LeaveOneMarkerCadPrediction,
            RigidCadFit,
            RigidRotationValidation,
        )

        summary_mm = MetricSummaryMm(
            count=1,
            min_mm=rmse_mm,
            median_mm=rmse_mm,
            rmse_mm=rmse_mm,
            p95_mm=rmse_mm,
            max_mm=rmse_mm,
        )
        summary_px = MetricSummaryPx(
            count=1,
            min_px=p95_px,
            median_px=p95_px,
            rmse_px=p95_px,
            p95_px=p95_px,
            max_px=p95_px,
        )
        rigid_fit = RigidCadFit(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation_m=(0.0, 0.0, 0.0),
            rotation_validation=RigidRotationValidation(
                determinant=1.0,
                orthonormality_frobenius_error=0.0,
                is_proper_rotation=True,
            ),
            per_landmark=(),
            summary_mm=summary_mm,
        )
        empty_distance = DistanceCadDisagreementReport(distances=(), summary_mm=summary_mm)
        cad_geometry = CadGeometryEvaluation(
            landmark_names=("a",),
            cad_landmarks_m={"a": (0.0, 0.0, 0.0)},
            marker_derived_landmarks_m={"a": (0.0, 0.0, 0.0)},
            rigid_fit=rigid_fit,
            pair_distance_disagreement=empty_distance,
            skeleton_edge_disagreement=empty_distance,
            leave_one_marker_out=LeaveOneMarkerCadPrediction(
                folds=(),
                eligible_fold_count=0,
                refused_fold_count=0,
                all_excluded_summary_mm=summary_mm,
            ),
        )
        detection = DetectionConsistencyCandidateResult(
            candidate_name=name,
            compatible=True,
            incompatibility_reason=None,
            summary_px=summary_px,
            eligible_fold_count=1,
            possible_fold_count=1,
            solve_failure_count=0,
            ineligible_fold_count=0,
            per_marker=(),
            visible_marker_count_strata=(),
            per_source_video=(),
        )
        return CandidateEvaluationResult(
            name=name,
            capture_session="session",
            solver_variant="default",
            marker_model_path=self._marker_model_path,
            calibration_source_video=None,
            cad_geometry=cad_geometry,
            detection=detection,
        )

    def test_separate_rankings_without_overall_winner(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        report = build_marker_model_evaluation_report(
            manifest=manifest,
            landmark_names=("a",),
            expected_marker_ids=(0, 1),
            calibration_width=640,
            calibration_height=480,
            calibration_source="test",
            candidate_results=(
                self._minimal_candidate_result("better_cad", rmse_mm=1.0, p95_px=9.0),
                self._minimal_candidate_result("better_detection", rmse_mm=9.0, p95_px=1.0),
            ),
            normalization_summaries=(),
        )
        self.assertEqual(
            [entry.candidate_name for entry in report.rankings["cad_leave_one_marker_out_rmse_mm"]],
            ["better_cad", "better_detection"],
        )
        self.assertEqual(
            [entry.candidate_name for entry in report.rankings["detection_held_out_p95_px"]],
            ["better_detection", "better_cad"],
        )
        document = build_marker_model_evaluation_document(report)
        self.assertNotIn("overall", document["rankings"])
        self.assertIn("no_overall_winner", document["interpretation"])

    def test_grouping_notes_when_insufficient_candidates(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        report = build_marker_model_evaluation_report(
            manifest=manifest,
            landmark_names=("a",),
            expected_marker_ids=(0,),
            calibration_width=640,
            calibration_height=480,
            calibration_source="test",
            candidate_results=(self._minimal_candidate_result("solo", 1.0, 1.0),),
            normalization_summaries=(),
        )
        self.assertIn("insufficient_candidates_for_repeatability_grouping", report.grouping.notes)

    def test_console_summary_derived_from_report(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        report = build_marker_model_evaluation_report(
            manifest=manifest,
            landmark_names=("a",),
            expected_marker_ids=(0,),
            calibration_width=640,
            calibration_height=480,
            calibration_source="test",
            candidate_results=(self._minimal_candidate_result("solo", 2.5, 3.5),),
            normalization_summaries=(),
        )
        summary = format_marker_model_evaluation_console_summary(report)
        self.assertIn("CAD disagreement ranking", summary)
        self.assertIn("Detection consistency ranking", summary)
        self.assertIn("solo: 2.500 mm", summary)
        self.assertIn("solo: 3.500 px", summary)

    def test_grouping_reports_cross_session_same_variant_when_present(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        object.__setattr__(
            manifest,
            "candidates",
            (
                type(manifest.candidates[0])(
                    name="candidate_a",
                    marker_model_path=self._marker_model_path,
                    capture_session="session_a",
                    solver_variant="default",
                    calibration_source=manifest.candidates[0].calibration_source,
                ),
                type(manifest.candidates[0])(
                    name="candidate_b",
                    marker_model_path=self._marker_model_path,
                    capture_session="session_b",
                    solver_variant="default",
                    calibration_source=manifest.candidates[0].calibration_source,
                ),
            ),
        )
        report = build_marker_model_evaluation_report(
            manifest=manifest,
            landmark_names=("a",),
            expected_marker_ids=(0,),
            calibration_width=640,
            calibration_height=480,
            calibration_source="test",
            candidate_results=(
                self._minimal_candidate_result("candidate_a", 1.0, 1.0),
                self._minimal_candidate_result("candidate_b", 2.0, 2.0),
            ),
            normalization_summaries=(),
        )
        self.assertEqual(len(report.grouping.cross_session_same_variant_groups), 1)
        self.assertNotIn(
            "insufficient_candidates_for_cross_session_same_variant_comparison",
            report.grouping.notes,
        )

    def test_serialization_is_deterministic(self) -> None:
        if not EXAMPLE_MANIFEST.is_file():
            self.skipTest("example manifest is not available")
        manifest = load_evaluation_manifest(EXAMPLE_MANIFEST)
        report = build_marker_model_evaluation_report(
            manifest=manifest,
            landmark_names=("a",),
            expected_marker_ids=(0,),
            calibration_width=640,
            calibration_height=480,
            calibration_source="test",
            candidate_results=(self._minimal_candidate_result("solo", 1.0, 1.0),),
            normalization_summaries=(),
        )
        first = json.dumps(build_marker_model_evaluation_document(report), sort_keys=True)
        second = json.dumps(build_marker_model_evaluation_document(report), sort_keys=True)
        self.assertEqual(first, second)
        self.assertEqual(
            build_marker_model_evaluation_document(report)["version"],
            MARKER_MODEL_EVALUATION_REPORT_VERSION,
        )


class MarkerModelEvaluationOrchestrationTests(unittest.TestCase):
    def test_freeze_video_decodes_once_without_looping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            width, height = 64, 48
            video_path = root / "held_out.mov"
            _write_synthetic_video(video_path, width=width, height=height, frame_count=3)
            detector = mock.Mock()
            detector.detectMarkers.return_value = (None, None, None)
            frozen, summary = freeze_held_out_video_detections(
                video_path,
                detector,
                calibration_width=width,
                calibration_height=height,
                expected_marker_ids=frozenset({0, 1}),
            )
            self.assertEqual(len(frozen.frames), 3)
            self.assertEqual(summary.frame_count, 3)
            self.assertEqual(detector.detectMarkers.call_count, 3)

    def test_shared_frozen_detections_across_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            width, height = 64, 48
            _write_intrinsics(root / "intrinsic.json", width=width, height=height)
            _write_object_model(root / "object_model.json", [0, 1, 2, 3])
            _write_marker_model(root / "marker_a.json", [0, 1, 2, 3])
            _write_marker_model(root / "marker_b.json", [0, 1, 2, 3])
            _write_synthetic_video(root / "held_out.mov", width=width, height=height, frame_count=1)
            (root / "cad.glb").write_bytes((REPO_ROOT / "config/Model/CAD/nexplayground_sim.glb").read_bytes())
            manifest_path = root / "manifest.json"
            _write_manifest(
                manifest_path,
                workspace=root,
                marker_ids=[0, 1, 2, 3],
                candidates=[
                    {
                        "name": "candidate_a",
                        "marker_model": "marker_a.json",
                        "capture_session": "session_a",
                        "solver_variant": "default",
                    },
                    {
                        "name": "candidate_b",
                        "marker_model": "marker_b.json",
                        "capture_session": "session_b",
                        "solver_variant": "default",
                    },
                ],
            )

            from object_apriltag.evaluation.detection_consistency import (
                DetectionConsistencyCandidateResult,
                DetectionConsistencyEvaluation,
                FrozenFrameDetections,
                FrozenVideoDetections,
            )

            fake_detection = DetectionConsistencyEvaluation(
                expected_marker_ids=(0, 1, 2, 3),
                candidates=(
                    DetectionConsistencyCandidateResult(
                        candidate_name="candidate_a",
                        compatible=True,
                        incompatibility_reason=None,
                        summary_px=MetricSummaryPx(0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        eligible_fold_count=0,
                        possible_fold_count=0,
                        solve_failure_count=0,
                        ineligible_fold_count=0,
                        per_marker=(),
                        visible_marker_count_strata=(),
                        per_source_video=(),
                    ),
                    DetectionConsistencyCandidateResult(
                        candidate_name="candidate_b",
                        compatible=True,
                        incompatibility_reason=None,
                        summary_px=MetricSummaryPx(0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        eligible_fold_count=0,
                        possible_fold_count=0,
                        solve_failure_count=0,
                        ineligible_fold_count=0,
                        per_marker=(),
                        visible_marker_count_strata=(),
                        per_source_video=(),
                    ),
                ),
            )
            fake_cad = mock.Mock()
            fake_cad.leave_one_marker_out.all_excluded_summary_mm.rmse_mm = 0.0

            with (
                mock.patch(
                    "object_apriltag.evaluation.runner.load_cad_landmarks",
                    return_value=mock.Mock(landmarks={}),
                ),
                mock.patch(
                    "object_apriltag.evaluation.runner.freeze_held_out_video_detections",
                    return_value=(
                        FrozenVideoDetections(
                            source_video=str(root / "held_out.mov"),
                            frames=(FrozenFrameDetections(detections=()),),
                        ),
                        mock.Mock(
                            source_video=str(root / "held_out.mov"),
                            frame_count=1,
                            unknown_marker_ids=0,
                            duplicate_marker_skips=0,
                            malformed_detections=0,
                        ),
                    ),
                ) as freeze_mock,
                mock.patch(
                    "object_apriltag.evaluation.runner.evaluate_detection_consistency",
                    return_value=fake_detection,
                ) as detection_mock,
                mock.patch(
                    "object_apriltag.evaluation.runner.evaluate_cad_geometry",
                    return_value=fake_cad,
                ),
            ):
                manifest = load_evaluation_manifest(manifest_path)
                report = evaluate_marker_models(manifest)

            self.assertEqual(freeze_mock.call_count, 1)
            self.assertEqual(detection_mock.call_count, 1)
            self.assertEqual(len(report.candidates), 2)


class MarkerModelEvaluationCliTests(unittest.TestCase):
    def test_cli_smoke_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.json"
            fake_report = mock.Mock(spec=MarkerModelEvaluationReport)
            fake_report.version = MARKER_MODEL_EVALUATION_REPORT_VERSION
            fake_report.manifest_path = EXAMPLE_MANIFEST
            fake_report.candidates = ()
            fake_report.rankings = {
                "cad_leave_one_marker_out_rmse_mm": (),
                "detection_held_out_p95_px": (),
            }
            fake_report.grouping = mock.Mock(notes=())
            with (
                mock.patch(
                    "object_apriltag.cli.evaluate_marker_model.evaluate_marker_models_from_manifest",
                    return_value=fake_report,
                ),
                mock.patch(
                    "object_apriltag.cli.evaluate_marker_model.save_marker_model_evaluation_report"
                ) as save_mock,
                mock.patch(
                    "object_apriltag.cli.evaluate_marker_model.format_marker_model_evaluation_console_summary",
                    return_value="summary\n",
                ),
            ):
                exit_code = evaluate_marker_model_main(
                    ["--manifest", str(EXAMPLE_MANIFEST), "--output", str(output_path)]
                )
            self.assertEqual(exit_code, 0)
            save_mock.assert_called_once_with(output_path, fake_report)


if __name__ == "__main__":
    unittest.main()
