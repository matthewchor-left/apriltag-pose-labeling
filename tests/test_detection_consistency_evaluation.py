"""Tests for held-out detection consistency evaluation."""

from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from object_apriltag.evaluation.detection_consistency import (
    DetectionCandidate,
    FrozenFrameDetections,
    FrozenVideoDetections,
    evaluate_detection_consistency,
    metric_summary_px,
    normalize_frame_detections,
)
from object_apriltag.layout import (
    build_marker_layout,
    footprint_from_dict,
    layout_point_to_object_frame,
)
from object_apriltag.pose import estimate_global_layout_pose


def _camera() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array(
            [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        np.zeros((5, 1), dtype=np.float64),
    )


def _footprint_with_top_left(
    marker_id: int,
    top_left: np.ndarray,
    marker_size: float,
    tilt_x_deg: float = 0.0,
) -> object:
    top_left = np.asarray(top_left, dtype=np.float64).reshape(3)
    deltas = {
        "top_left": np.zeros(3, dtype=np.float64),
        "top_right": np.array([marker_size, 0.0, 0.0], dtype=np.float64),
        "bottom_right": np.array([marker_size, marker_size, 0.0], dtype=np.float64),
        "bottom_left": np.array([0.0, marker_size, 0.0], dtype=np.float64),
    }
    if tilt_x_deg != 0.0:
        angle = np.deg2rad(tilt_x_deg)
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float64,
        )
        deltas = {name: rotation @ offset for name, offset in deltas.items()}
    return footprint_from_dict(
        marker_id,
        {name: (top_left + offset).tolist() for name, offset in deltas.items()},
    )


REAL_CANDIDATE_MARKER_IDS = frozenset({0, 2, 3, 4, 19, 22, 23, 24, 25, 26, 28, 29})


def _multi_marker_layout(
    marker_ids: set[int],
    marker_size: float = 0.04,
) -> object:
    spacing = 0.12
    sorted_ids = sorted(marker_ids)
    tilts_deg = [0.0, 40.0, -35.0, 35.0, -30.0, 25.0]
    footprints = {}
    for index, marker_id in enumerate(sorted_ids):
        row = index // 4
        col = index % 4
        top_left = np.array(
            [col * spacing, row * spacing, float(index) * 0.05],
            dtype=np.float64,
        )
        footprints[marker_id] = _footprint_with_top_left(
            marker_id,
            top_left,
            marker_size,
            tilt_x_deg=tilts_deg[index % len(tilts_deg)],
        )
    return build_marker_layout(sorted_ids[0], marker_size, footprints)


def _four_marker_layout(
    marker_size: float = 0.04,
    perturb_marker_id: int | None = None,
) -> object:
    model_points = {
        0: (np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0),
        1: (np.array([0.1, 0.0, 0.0], dtype=np.float64), 35.0),
        2: (np.array([0.0, 0.1, 0.0], dtype=np.float64), -25.0),
        3: (np.array([0.0, 0.0, 0.1], dtype=np.float64), 20.0),
    }
    if perturb_marker_id is not None:
        point, tilt = model_points[perturb_marker_id]
        model_points[perturb_marker_id] = (
            point + np.array([0.05, 0.0, 0.0]),
            tilt,
        )
    footprints = {
        marker_id: _footprint_with_top_left(marker_id, point, marker_size, tilt_x_deg=tilt)
        for marker_id, (point, tilt) in model_points.items()
    }
    return build_marker_layout(0, marker_size, footprints)


def _project_layout_detections(
    layout,
    marker_ids: list[int],
    object_rvec: np.ndarray,
    object_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[np.ndarray, int]]:
    detections = []
    for marker_id in marker_ids:
        object_points = np.stack(
            [
                layout_point_to_object_frame(point, layout)
                for point in layout.footprints[marker_id].corners()
            ]
        )
        projected, _ = cv2.projectPoints(
            object_points,
            object_rvec,
            object_origin,
            camera_matrix,
            dist_coeffs,
        )
        detections.append(
            (projected.reshape(1, 4, 2).astype(np.float32), marker_id)
        )
    return detections


def _frozen_video(
    layout,
    marker_ids: list[int],
    object_rvec: np.ndarray,
    object_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    source_video: str = "synthetic.mov",
    frame_count: int = 3,
) -> FrozenVideoDetections:
    frame = FrozenFrameDetections(
        detections=tuple(
            _project_layout_detections(
                layout,
                marker_ids,
                object_rvec,
                object_origin,
                camera_matrix,
                dist_coeffs,
            )
        )
    )
    return FrozenVideoDetections(
        source_video=source_video,
        frames=(frame,) * frame_count,
    )


class StrictGlobalPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = _four_marker_layout()
        self.camera_matrix, self.dist_coeffs = _camera()
        self.object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        self.object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)

    def test_global_layout_pose_rejects_single_marker(self) -> None:
        detections = _project_layout_detections(
            self.layout,
            [0],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        self.assertIsNone(
            estimate_global_layout_pose(
                detections,
                self.layout,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )


class DetectionConsistencyEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = _four_marker_layout()
        self.camera_matrix, self.dist_coeffs = _camera()
        self.object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        self.object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)
        self.expected_marker_ids = frozenset({0, 1, 2, 3})
        self.video = _frozen_video(
            self.layout,
            sorted(self.expected_marker_ids),
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )

    def test_correct_layout_beats_perturbed_layout(self) -> None:
        clean = DetectionCandidate(name="clean", layout=self.layout)
        perturbed = DetectionCandidate(
            name="perturbed",
            layout=_four_marker_layout(perturb_marker_id=2),
        )
        report = evaluate_detection_consistency(
            expected_marker_ids=self.expected_marker_ids,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[self.video],
            candidates=[clean, perturbed],
        )
        clean_result = report.candidates[0]
        perturbed_result = report.candidates[1]
        self.assertLess(clean_result.summary_px.p95_px, 1e-3)
        self.assertGreater(perturbed_result.summary_px.p95_px, 5.0)
        self.assertGreater(
            perturbed_result.summary_px.p95_px,
            clean_result.summary_px.p95_px + 5.0,
        )

    def test_held_out_marker_excluded_from_pose_solve(self) -> None:
        captured_training_ids: list[set[int]] = []

        def capture_solve(detections, layout, camera_matrix, dist_coeffs):
            captured_training_ids.append({marker_id for _, marker_id in detections})
            return estimate_global_layout_pose(
                detections, layout, camera_matrix, dist_coeffs
            )

        single_frame_video = _frozen_video(
            self.layout,
            sorted(self.expected_marker_ids),
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
            frame_count=1,
        )
        with mock.patch(
            "object_apriltag.evaluation.detection_consistency.estimate_global_layout_pose",
            side_effect=capture_solve,
        ):
            evaluate_detection_consistency(
                expected_marker_ids=self.expected_marker_ids,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                videos=[single_frame_video],
                candidates=[DetectionCandidate(name="clean", layout=self.layout)],
            )

        expected_training_sets = [
            set(self.expected_marker_ids) - {held_out_marker_id}
            for held_out_marker_id in sorted(self.expected_marker_ids)
        ]
        self.assertEqual(captured_training_ids, expected_training_sets)

    def test_fewer_than_two_training_markers_is_ineligible(self) -> None:
        video = _frozen_video(
            self.layout,
            [0, 1],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
            frame_count=1,
        )
        report = evaluate_detection_consistency(
            expected_marker_ids=self.expected_marker_ids,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[video],
            candidates=[DetectionCandidate(name="clean", layout=self.layout)],
        )
        result = report.candidates[0]
        self.assertEqual(result.possible_fold_count, 2)
        self.assertEqual(result.eligible_fold_count, 0)
        self.assertEqual(result.ineligible_fold_count, 2)
        self.assertEqual(result.summary_px.count, 0)

    def test_solve_failure_accounted_separately(self) -> None:
        with mock.patch(
            "object_apriltag.evaluation.detection_consistency.estimate_global_layout_pose",
            return_value=None,
        ):
            report = evaluate_detection_consistency(
                expected_marker_ids=self.expected_marker_ids,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                videos=[self.video],
                candidates=[DetectionCandidate(name="clean", layout=self.layout)],
            )
        result = report.candidates[0]
        self.assertEqual(result.possible_fold_count, 12)
        self.assertEqual(result.eligible_fold_count, 12)
        self.assertEqual(result.solve_failure_count, 12)
        self.assertEqual(result.summary_px.count, 0)

    def test_identical_frozen_observations_used_for_all_candidates(self) -> None:
        clean = DetectionCandidate(name="clean", layout=self.layout)
        perturbed = DetectionCandidate(
            name="perturbed",
            layout=_four_marker_layout(perturb_marker_id=1),
        )
        report = evaluate_detection_consistency(
            expected_marker_ids=self.expected_marker_ids,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[self.video],
            candidates=[clean, perturbed],
        )
        clean_marker_1 = next(
            marker for marker in report.candidates[0].per_marker if marker.marker_id == 1
        )
        perturbed_marker_1 = next(
            marker for marker in report.candidates[1].per_marker if marker.marker_id == 1
        )
        self.assertEqual(clean_marker_1.possible_fold_count, perturbed_marker_1.possible_fold_count)
        self.assertEqual(clean_marker_1.eligible_fold_count, perturbed_marker_1.eligible_fold_count)

    def test_duplicate_marker_id_drops_marker_entirely(self) -> None:
        base = _project_layout_detections(
            self.layout,
            [2],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )[0]
        duplicate = base[0].copy()
        duplicate[0, 0, 0] += 80.0
        corners_by_id, duplicate_skips, _unknown, _malformed = normalize_frame_detections(
            [base, (duplicate, 2)],
            expected_marker_ids=frozenset({2}),
        )
        self.assertEqual(duplicate_skips, 1)
        self.assertNotIn(2, corners_by_id)

    def test_unknown_and_non_finite_detections_skipped(self) -> None:
        valid = _project_layout_detections(
            self.layout,
            [0],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )[0][0]
        nan_corners = valid.copy()
        nan_corners[0, 0, 0] = float("nan")
        corners_by_id, duplicate_skips, unknown_ids, malformed_skips = normalize_frame_detections(
            [(valid, 0), (valid, 99), (nan_corners, 1)],
            expected_marker_ids=frozenset({0, 1}),
        )
        self.assertEqual(unknown_ids, 1)
        self.assertEqual(duplicate_skips, 0)
        self.assertEqual(malformed_skips, 1)
        self.assertEqual(set(corners_by_id), {0})

    def test_incompatible_candidate_refused(self) -> None:
        wrong_layout = _four_marker_layout()
        wrong_layout = build_marker_layout(
            0,
            0.04,
            {marker_id: wrong_layout.footprints[marker_id] for marker_id in (0, 1, 2)},
        )
        report = evaluate_detection_consistency(
            expected_marker_ids=self.expected_marker_ids,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[self.video],
            candidates=[DetectionCandidate(name="wrong", layout=wrong_layout)],
        )
        result = report.candidates[0]
        self.assertFalse(result.compatible)
        self.assertIn("incompatible_marker_ids", result.incompatibility_reason or "")
        self.assertEqual(result.possible_fold_count, 0)

    def test_empty_video_yields_zero_folds(self) -> None:
        empty_video = FrozenVideoDetections(source_video="empty.mov", frames=())
        report = evaluate_detection_consistency(
            expected_marker_ids=self.expected_marker_ids,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[empty_video],
            candidates=[DetectionCandidate(name="clean", layout=self.layout)],
        )
        result = report.candidates[0]
        self.assertEqual(result.possible_fold_count, 0)
        self.assertEqual(result.summary_px.count, 0)

    def test_aggregation_per_marker_strata_and_video(self) -> None:
        video_a = _frozen_video(
            self.layout,
            sorted(self.expected_marker_ids),
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
            source_video="a.mov",
            frame_count=1,
        )
        video_b = _frozen_video(
            self.layout,
            [0, 1, 2],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
            source_video="b.mov",
            frame_count=1,
        )
        report = evaluate_detection_consistency(
            expected_marker_ids=self.expected_marker_ids,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[video_a, video_b],
            candidates=[DetectionCandidate(name="clean", layout=self.layout)],
        )
        result = report.candidates[0]
        self.assertEqual(len(result.per_marker), 4)
        self.assertEqual(len(result.per_source_video), 2)
        self.assertEqual(
            {stratum.visible_marker_count for stratum in result.visible_marker_count_strata},
            {3, 4},
        )
        self.assertGreater(result.summary_px.count, 0)
        self.assertGreaterEqual(result.summary_px.p95_px, result.summary_px.median_px)

    def test_metric_summary_px_empty_returns_zeroed_summary(self) -> None:
        summary = metric_summary_px([])
        self.assertEqual(summary.count, 0)
        self.assertEqual(summary.p95_px, 0.0)

    def test_full_candidate_marker_set_compatible_with_subset_video_visibility(self) -> None:
        layout = _multi_marker_layout(REAL_CANDIDATE_MARKER_IDS)
        visible_ids = [0, 2, 3]
        video = _frozen_video(
            layout,
            visible_ids,
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
            frame_count=1,
        )
        unknown_detections = [
            (np.full((1, 4, 2), 100.0 + marker_id, dtype=np.float32), marker_id)
            for marker_id in (5, 17, 18, 20)
        ]
        frame = FrozenFrameDetections(
            detections=tuple(video.frames[0].detections + tuple(unknown_detections))
        )
        mixed_video = FrozenVideoDetections(
            source_video="playground_static_4_tag_min_test.mov",
            frames=(frame,),
        )
        report = evaluate_detection_consistency(
            expected_marker_ids=REAL_CANDIDATE_MARKER_IDS,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[mixed_video],
            candidates=[DetectionCandidate(name="full-layout", layout=layout)],
        )
        result = report.candidates[0]
        self.assertTrue(result.compatible)
        self.assertEqual(result.possible_fold_count, 3)
        self.assertEqual(result.eligible_fold_count, 3)
        self.assertEqual(result.summary_px.count, 12)
        invisible_marker = next(marker for marker in result.per_marker if marker.marker_id == 19)
        self.assertEqual(invisible_marker.possible_fold_count, 0)

    def test_duplicate_marker_id_makes_frame_ineligible_for_that_marker(self) -> None:
        layout = _multi_marker_layout({0, 2, 3})
        base_video = _frozen_video(
            layout,
            [0, 2, 3],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
            frame_count=1,
        )
        duplicate_two = _project_layout_detections(
            layout,
            [2],
            self.object_rvec,
            self.object_origin,
            self.camera_matrix,
            self.dist_coeffs,
        )
        shifted_duplicate = duplicate_two[0][0].copy()
        shifted_duplicate[0, 0, 0] += 25.0
        frame = FrozenFrameDetections(
            detections=tuple(base_video.frames[0].detections + (duplicate_two[0], (shifted_duplicate, 2)))
        )
        video = FrozenVideoDetections(source_video="duplicate-id-2.mov", frames=(frame,))
        report = evaluate_detection_consistency(
            expected_marker_ids=frozenset({0, 2, 3}),
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            videos=[video],
            candidates=[DetectionCandidate(name="layout", layout=layout)],
        )
        result = report.candidates[0]
        self.assertEqual(result.possible_fold_count, 2)
        self.assertEqual(result.eligible_fold_count, 0)
        marker_two = next(marker for marker in result.per_marker if marker.marker_id == 2)
        self.assertEqual(marker_two.possible_fold_count, 0)

    def test_projection_failure_counts_as_solve_failure(self) -> None:
        with mock.patch(
            "object_apriltag.evaluation.detection_consistency._held_out_corner_errors_px",
            return_value=[],
        ):
            report = evaluate_detection_consistency(
                expected_marker_ids=self.expected_marker_ids,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                videos=[self.video],
                candidates=[DetectionCandidate(name="clean", layout=self.layout)],
            )
        result = report.candidates[0]
        self.assertEqual(result.solve_failure_count, 12)
        self.assertEqual(result.summary_px.count, 0)


if __name__ == "__main__":
    unittest.main()
