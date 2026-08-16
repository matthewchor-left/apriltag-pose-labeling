"""Tests for CAD geometry evaluation metrics."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from object_apriltag.cad import CadLandmarks
from object_apriltag.evaluation.cad_geometry import (
    derive_marker_derived_landmarks,
    evaluate_cad_geometry,
    fit_cad_registration,
    metric_summary_mm,
)
from object_apriltag.evaluation.kabsch import (
    apply_rigid_transform,
    kabsch_rigid_transform,
    validate_rigid_rotation,
)
from object_apriltag.layout import (
    build_marker_layout,
    footprint_corner_with_padding,
    footprint_from_dict,
)


def _square_corners(half: float, z: float = 0.0) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, z],
        "top_right": [half, -half, z],
        "bottom_right": [half, half, z],
        "bottom_left": [-half, half, z],
    }


def _footprint_with_top_left(marker_id: int, top_left: np.ndarray, marker_size: float) -> object:
    top_left = np.asarray(top_left, dtype=np.float64).reshape(3)
    half = marker_size / 2.0
    return footprint_from_dict(
        marker_id,
        {
            "top_left": top_left.tolist(),
            "top_right": (top_left + np.array([marker_size, 0.0, 0.0])).tolist(),
            "bottom_right": (top_left + np.array([marker_size, marker_size, 0.0])).tolist(),
            "bottom_left": (top_left + np.array([0.0, marker_size, 0.0])).tolist(),
        },
    )


def _four_landmark_document() -> dict:
    return {
        "units": "meters",
        "coordinate_frame": "marker_model",
        "keypoints": {
            "a": [9.0, 9.0, 9.0],
            "b": [8.0, 8.0, 8.0],
            "c": [7.0, 7.0, 7.0],
            "d": [6.0, 6.0, 6.0],
        },
        "skeleton": [["a", "b"], ["c", "d"]],
        "keypoint_sources": {
            "a": {"marker_id": 0, "corner": "top_left"},
            "b": {"marker_id": 1, "corner": "top_left"},
            "c": {"marker_id": 2, "corner": "top_left"},
            "d": {"marker_id": 3, "corner": "top_left"},
        },
    }


def _four_landmark_layout(marker_size: float = 0.04, perturb_marker_id: int | None = None) -> object:
    model_points = {
        "a": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "b": np.array([0.1, 0.0, 0.0], dtype=np.float64),
        "c": np.array([0.0, 0.1, 0.0], dtype=np.float64),
        "d": np.array([0.0, 0.0, 0.1], dtype=np.float64),
    }
    if perturb_marker_id is not None:
        name = {0: "a", 1: "b", 2: "c", 3: "d"}[perturb_marker_id]
        model_points[name] = model_points[name] + np.array([0.05, 0.0, 0.0])

    footprints = {
        marker_id: _footprint_with_top_left(marker_id, model_points[name], marker_size)
        for marker_id, name in enumerate(["a", "b", "c", "d"])
    }
    return build_marker_layout(0, marker_size, footprints)


class KabschRigidTransformTests(unittest.TestCase):
    def test_recovers_exact_rigid_transform(self) -> None:
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        translation = np.array([0.2, -0.1, 0.3], dtype=np.float64)
        target = (rotation @ source.T).T + translation

        fitted_rotation, fitted_translation = kabsch_rigid_transform(source, target)
        np.testing.assert_allclose(fitted_rotation, rotation, atol=1e-9)
        np.testing.assert_allclose(fitted_translation, translation, atol=1e-9)
        np.testing.assert_allclose(
            apply_rigid_transform(source, fitted_rotation, fitted_translation),
            target,
            atol=1e-9,
        )

    def test_rejects_reflection_for_proper_rotation(self) -> None:
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        target = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)
        rotation, _ = kabsch_rigid_transform(source, target)
        validation = validate_rigid_rotation(rotation)
        self.assertGreater(validation.determinant, 0.0)
        self.assertTrue(validation.is_proper_rotation)

    def test_does_not_apply_scale(self) -> None:
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        target = source * 2.0
        rotation, translation = kabsch_rigid_transform(source, target)
        transformed = apply_rigid_transform(source, rotation, translation)
        self.assertGreater(float(np.linalg.norm(transformed - target)), 0.5)


class CadGeometryEvaluationTests(unittest.TestCase):
    def test_candidate_landmarks_derived_from_keypoint_sources_not_persisted_keypoints(self) -> None:
        document = _four_landmark_document()
        layout = _four_landmark_layout()
        derived = derive_marker_derived_landmarks(document, layout)

        for name, (marker_id, corner, padding_m) in {
            name: (
                document["keypoint_sources"][name]["marker_id"],
                document["keypoint_sources"][name]["corner"],
                0.0,
            )
            for name in derived
        }.items():
            expected = footprint_corner_with_padding(
                layout.footprints[marker_id],
                corner,
                padding_m,
            )
            np.testing.assert_allclose(derived[name], expected)
            self.assertFalse(np.allclose(derived[name], document["keypoints"][name]))

    def test_exact_alignment_yields_near_zero_cad_disagreement(self) -> None:
        document = _four_landmark_document()
        layout = _four_landmark_layout()
        derived = derive_marker_derived_landmarks(document, layout)
        cad = CadLandmarks(landmarks=derived)

        report = evaluate_cad_geometry(cad, document, layout)
        self.assertLess(report.rigid_fit.summary_mm.rmse_mm, 1e-6)
        self.assertLess(report.leave_one_marker_out.all_excluded_summary_mm.rmse_mm, 1e-6)

    def test_fits_in_memory_registration_from_marker_derived_landmarks(self) -> None:
        document = _four_landmark_document()
        layout = _four_landmark_layout()
        target = derive_marker_derived_landmarks(document, layout)
        rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        translation = np.array([0.2, -0.1, 0.3], dtype=np.float64)
        cad = CadLandmarks(
            landmarks={
                name: rotation.T @ (point - translation)
                for name, point in target.items()
            }
        )

        registration = fit_cad_registration(cad, document, layout)

        self.assertEqual(registration.units, "meters")
        self.assertEqual(registration.source_frame, "cad")
        self.assertEqual(registration.target_frame, "marker_model")
        for name, point in cad.landmarks.items():
            homogeneous = np.append(point, 1.0)
            transformed = (registration.transform_4x4 @ homogeneous)[:3]
            np.testing.assert_allclose(transformed, target[name], atol=1e-9)

    def test_perturbed_marker_localized_by_leave_one_marker_out(self) -> None:
        document = _four_landmark_document()
        clean_layout = _four_landmark_layout()
        clean_cad = CadLandmarks(landmarks=derive_marker_derived_landmarks(document, clean_layout))

        perturbed_layout = _four_landmark_layout(perturb_marker_id=2)
        report = evaluate_cad_geometry(clean_cad, document, perturbed_layout)

        held_out_fold = next(fold for fold in report.leave_one_marker_out.folds if fold.held_out_marker_id == 2)
        reference_fold = next(fold for fold in report.leave_one_marker_out.folds if fold.held_out_marker_id == 0)
        self.assertTrue(held_out_fold.eligible)
        self.assertTrue(reference_fold.eligible)
        self.assertGreater(held_out_fold.summary_mm.max_mm, 40.0)
        self.assertGreater(
            held_out_fold.summary_mm.max_mm,
            reference_fold.summary_mm.max_mm + 30.0,
        )

    def test_degenerate_fold_is_refused_without_crashing(self) -> None:
        document = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "keypoints": {
                "p0": [0.0, 0.0, 0.0],
                "p1": [0.1, 0.0, 0.0],
                "p2": [0.2, 0.0, 0.0],
                "p3": [0.3, 0.0, 0.0],
            },
            "skeleton": [["p0", "p1"]],
            "keypoint_sources": {
                "p0": {"marker_id": 0, "corner": "top_left"},
                "p1": {"marker_id": 1, "corner": "top_left"},
                "p2": {"marker_id": 2, "corner": "top_left"},
                "p3": {"marker_id": 3, "corner": "top_left"},
            },
        }
        marker_size = 0.04
        footprints = {
            marker_id: _footprint_with_top_left(
                marker_id,
                np.array([float(marker_id) * 0.1, 0.0, 0.0]),
                marker_size,
            )
            for marker_id in range(4)
        }
        layout = build_marker_layout(0, marker_size, footprints)
        cad = CadLandmarks(landmarks=derive_marker_derived_landmarks(document, layout))

        report = evaluate_cad_geometry(cad, document, layout)
        self.assertGreater(report.leave_one_marker_out.refused_fold_count, 0)
        for fold in report.leave_one_marker_out.folds:
            if not fold.eligible:
                self.assertIsNotNone(fold.refusal_reason)
                self.assertEqual(fold.per_landmark_cad_disagreement_mm, {})

    def test_pair_and_skeleton_distance_disagreement_reports_mm(self) -> None:
        document = copy.deepcopy(_four_landmark_document())
        layout = _four_landmark_layout(perturb_marker_id=1)
        cad_points = derive_marker_derived_landmarks(document, _four_landmark_layout())
        report = evaluate_cad_geometry(CadLandmarks(landmarks=cad_points), document, layout)

        self.assertEqual(len(report.pair_distance_disagreement.distances), 6)
        self.assertEqual(len(report.skeleton_edge_disagreement.distances), 2)
        self.assertGreater(report.pair_distance_disagreement.summary_mm.max_mm, 0.0)
        self.assertGreater(report.skeleton_edge_disagreement.summary_mm.max_mm, 0.0)

    def test_rejects_missing_cad_landmark_names(self) -> None:
        document = _four_landmark_document()
        layout = _four_landmark_layout()
        derived = derive_marker_derived_landmarks(document, layout)
        del derived["c"]

        with self.assertRaisesRegex(ValueError, "CAD landmarks missing names"):
            evaluate_cad_geometry(CadLandmarks(landmarks=derived), document, layout)

    def test_rejects_non_finite_cad_positions(self) -> None:
        document = _four_landmark_document()
        layout = _four_landmark_layout()
        derived = derive_marker_derived_landmarks(document, layout)
        cad = dict(derived)
        cad["a"] = np.array([float("nan"), 0.0, 0.0])

        with self.assertRaisesRegex(ValueError, "CAD landmark 'a'"):
            evaluate_cad_geometry(cad, document, layout)

    def test_insufficient_retained_landmarks_refuses_all_folds(self) -> None:
        document = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "keypoints": {
                "a": [0.0, 0.0, 0.0],
                "b": [0.1, 0.0, 0.0],
            },
            "keypoint_sources": {
                "a": {"marker_id": 0, "corner": "top_left"},
                "b": {"marker_id": 1, "corner": "top_left"},
            },
        }
        marker_size = 0.04
        footprints = {
            marker_id: _footprint_with_top_left(
                marker_id,
                np.array([float(marker_id) * 0.1, 0.0, 0.0]),
                marker_size,
            )
            for marker_id in range(2)
        }
        layout = build_marker_layout(0, marker_size, footprints)
        cad = CadLandmarks(landmarks=derive_marker_derived_landmarks(document, layout))

        report = evaluate_cad_geometry(cad, document, layout)
        self.assertEqual(report.leave_one_marker_out.eligible_fold_count, 0)
        self.assertEqual(report.leave_one_marker_out.refused_fold_count, 2)
        for fold in report.leave_one_marker_out.folds:
            self.assertIn("insufficient_retained_landmarks", fold.refusal_reason or "")

    def test_leave_one_marker_out_handles_multiple_landmarks_per_marker(self) -> None:
        document = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "keypoints": {
                "m0_a": [0.0, 0.0, 0.0],
                "m0_b": [0.0, 0.0, 0.0],
                "m1": [0.1, 0.0, 0.0],
                "m2": [0.0, 0.1, 0.0],
                "m3": [0.0, 0.0, 0.1],
            },
            "keypoint_sources": {
                "m0_a": {"marker_id": 0, "corner": "top_left"},
                "m0_b": {"marker_id": 0, "corner": "top_right"},
                "m1": {"marker_id": 1, "corner": "top_left"},
                "m2": {"marker_id": 2, "corner": "top_left"},
                "m3": {"marker_id": 3, "corner": "top_left"},
            },
        }
        marker_size = 0.04
        top_lefts = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([0.1, 0.0, 0.0]),
            2: np.array([0.0, 0.1, 0.0]),
            3: np.array([0.0, 0.0, 0.1]),
        }
        footprints = {
            marker_id: _footprint_with_top_left(marker_id, top_lefts[marker_id], marker_size)
            for marker_id in range(4)
        }
        layout = build_marker_layout(0, marker_size, footprints)
        cad = CadLandmarks(landmarks=derive_marker_derived_landmarks(document, layout))

        report = evaluate_cad_geometry(cad, document, layout)
        held_out_fold = next(fold for fold in report.leave_one_marker_out.folds if fold.held_out_marker_id == 0)
        self.assertTrue(held_out_fold.eligible)
        self.assertEqual(held_out_fold.excluded_landmark_names, ("m0_a", "m0_b"))
        self.assertEqual(set(held_out_fold.per_landmark_cad_disagreement_mm), {"m0_a", "m0_b"})

    def test_metric_summary_mm_empty_returns_zeroed_summary(self) -> None:
        summary = metric_summary_mm([])
        self.assertEqual(summary.count, 0)
        self.assertEqual(summary.min_mm, 0.0)
        self.assertEqual(summary.p95_mm, 0.0)


class KabschFiniteInputTests(unittest.TestCase):
    def test_rejects_non_finite_points(self) -> None:
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        target = source.copy()
        target[1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            kabsch_rigid_transform(source, target)


if __name__ == "__main__":
    unittest.main()
