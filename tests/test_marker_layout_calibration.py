"""Synthetic tests for marker layout calibration."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.layout import (
    CORNER_NAMES,
    build_marker_layout,
    footprint_from_dict,
    footprint_orientation,
    load_marker_model,
    marker_layout_to_dict,
    marker_origin_on_object,
    save_marker_model,
)
from object_apriltag.marker_layout_calibration import (
    CalibrationResult,
    CalibrationSettings,
    CalibrationSolveDiagnostics,
    FrameObservation,
    calibrate_marker_layout,
    uniform_marker_sizes,
)
from object_apriltag.marker_layout_calibration.continuous_refinement import (
    recheck_pair_support as _recheck_pair_support,
)
from object_apriltag.marker_layout_calibration.discrete_graph import (
    estimate_pair_consensus,
    make_dropped_pair_edge,
    make_restored_pair_edge,
    maybe_restore_weak_connectivity,
    weak_restore_candidates,
)
from object_apriltag.marker_layout_calibration.input import object_points_by_marker
from object_apriltag.marker_layout_calibration.pose_initialization import (
    reference_gauge_pose,
    restrict_pair_consensus_to_frames,
    synth_pair_observations,
)
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CornerObservation,
    PairConsensus,
)
from object_apriltag.marker_layout_calibration.solve_quality import quality_from_pairs
from object_apriltag.pose import marker_corner_object_points


def _uniform_marker_sizes(marker_ids: Iterable[int], marker_size_m: float) -> dict[int, float]:
    return uniform_marker_sizes(list(marker_ids), marker_size_m)


def _uniform_object_points_by_marker(marker_ids: Iterable[int], marker_size_m: float) -> dict[int, np.ndarray]:
    return object_points_by_marker(_uniform_marker_sizes(marker_ids, marker_size_m))


def _default_camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def reference_gauge_pose(marker_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    half = marker_size_m / 2.0
    top_left = np.array([-half, -half, 0.0], dtype=np.float64)
    top_right = np.array([half, -half, 0.0], dtype=np.float64)
    bottom_right = np.array([half, half, 0.0], dtype=np.float64)
    bottom_left = np.array([-half, half, 0.0], dtype=np.float64)
    rotation = footprint_orientation(top_left, top_right, bottom_left, bottom_right)
    translation = marker_origin_on_object(bottom_left, bottom_right)
    return rotation, translation


def _ground_truth_footprints(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    marker_size_m: float,
) -> dict[int, object]:
    object_points = marker_corner_object_points(marker_size_m)
    footprints = {}
    for marker_id, (rotation, translation) in marker_poses.items():
        payload = {
            corner_name: (rotation @ object_points[corner_index] + translation).tolist()
            for corner_index, corner_name in enumerate(CORNER_NAMES)
        }
        footprints[marker_id] = footprint_from_dict(marker_id, payload)
    return footprints


def synthesize_observations(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    frame_count: int = 25,
    marker_size_m: float = 0.07,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    visible_markers: Callable[[int], Iterable[int]] | None = None,
    noise_std_px: float = 0.0,
    outlier_fraction: float = 0.0,
    outlier_shift_px: float = 50.0,
    seed: int = 0,
) -> list[FrameObservation]:
    camera_matrix, dist_coeffs = camera_matrix or _default_camera()[0], dist_coeffs or _default_camera()[1]
    object_points = marker_corner_object_points(marker_size_m)
    rng = np.random.default_rng(seed)
    observations: list[FrameObservation] = []

    for frame_index in range(frame_count):
        layout_rotation, _ = cv2.Rodrigues(
            np.array([0.1, -0.15 + 0.01 * frame_index, 0.05], dtype=np.float64)
        )
        layout_translation = np.array(
            [0.02, -0.01, 0.6 + 0.002 * frame_index],
            dtype=np.float64,
        )
        marker_ids = (
            list(visible_markers(frame_index))
            if visible_markers is not None
            else list(marker_poses)
        )
        markers: dict[int, np.ndarray] = {}
        for marker_id in marker_ids:
            marker_rotation, marker_translation = marker_poses[marker_id]
            corners: list[np.ndarray] = []
            for corner_index in range(4):
                layout_point = marker_rotation @ object_points[corner_index] + marker_translation
                camera_point = layout_rotation @ layout_point + layout_translation
                projected, _ = cv2.projectPoints(
                    camera_point.reshape(1, 1, 3).astype(np.float32),
                    np.zeros((3, 1), dtype=np.float64),
                    np.zeros((3, 1), dtype=np.float64),
                    camera_matrix,
                    dist_coeffs,
                )
                image_point = projected.reshape(2).astype(np.float64)
                if noise_std_px > 0.0:
                    image_point += rng.normal(0.0, noise_std_px, size=2)
                corners.append(image_point)
            corner_array = np.stack(corners, axis=0)
            if outlier_fraction > 0.0 and rng.random() < outlier_fraction:
                corner_index = int(rng.integers(0, 4))
                corner_array[corner_index] += rng.normal(0.0, outlier_shift_px, size=2)
            markers[int(marker_id)] = corner_array
        observations.append(FrameObservation(frame_id=frame_index, markers=markers))
    return observations


def _shuffle_marker_dict(observations: list[FrameObservation]) -> list[FrameObservation]:
    shuffled: list[FrameObservation] = []
    for observation in observations:
        markers = {
            marker_id: corners.copy()
            for marker_id, corners in reversed(list(observation.markers.items()))
        }
        shuffled.append(FrameObservation(frame_id=observation.frame_id, markers=markers))
    return shuffled


def _apply_constant_marker_noise(
    observations: list[FrameObservation],
    marker_id: int,
    noise_std_px: float,
    *,
    seed: int = 0,
) -> None:
    offset = np.random.default_rng(seed).normal(0.0, noise_std_px, size=(4, 2))
    for observation in observations:
        if marker_id in observation.markers:
            observation.markers[marker_id] = observation.markers[marker_id] + offset


def _rotate_marker_corners(observations: list[FrameObservation], frame_indices: Iterable[int], marker_id: int) -> None:
    for frame_index in frame_indices:
        corners = observations[frame_index].markers[marker_id]
        observations[frame_index].markers[marker_id] = corners[[2, 3, 0, 1]].copy()


def _pair_poses(marker_size_m: float = 0.07) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    reference_rotation, reference_translation = reference_gauge_pose(marker_size_m)
    return {
        0: (reference_rotation, reference_translation),
        1: (reference_rotation, reference_translation + np.array([0.12, 0.0, -0.05], dtype=np.float64)),
    }


def _synth_pair_with_corrupt_frames(
    frame_count: int,
    corrupt_frames: frozenset[int],
    *,
    marker_size_m: float = 0.07,
    varying_corrupt: bool = False,
) -> list[FrameObservation]:
    camera_matrix, dist_coeffs = _default_camera()
    object_points = marker_corner_object_points(marker_size_m)
    return synth_pair_observations(
        frame_count,
        _pair_poses(marker_size_m),
        object_points,
        camera_matrix,
        dist_coeffs,
        corrupt_frames=corrupt_frames,
        corrupt_offset=np.array([0.20, 0.0, -0.08], dtype=np.float64),
        varying_corrupt=varying_corrupt,
    )


def _two_marker_poses(marker_size_m: float = 0.07) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return _pair_poses(marker_size_m)


@contextmanager
def _pruning_refinement_failure_without_weak_recovery(*, block_checkpoints: bool = False):
    """Simulate post-pruning refinement failure that weak-edge restore cannot heal."""
    real_restore = maybe_restore_weak_connectivity

    def _restore_without_post_pruning_weak_recovery(
        pair_consensus,
        weak_pool,
        dropped,
        required_ids,
        reference_marker_id,
        stage,
        *,
        best_effort,
        restored_pair_edges,
    ):
        if stage == "post_pruning":
            best_effort = False
        return real_restore(
            pair_consensus,
            weak_pool,
            dropped,
            required_ids,
            reference_marker_id,
            stage,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )

    weak_restore_patch = mock.patch(
        "object_apriltag.marker_layout_calibration.discrete_graph.maybe_restore_weak_connectivity",
        side_effect=_restore_without_post_pruning_weak_recovery,
    )
    if block_checkpoints:
        with (
            mock.patch(
                "object_apriltag.marker_layout_calibration.continuous_refinement.ContinuousLayoutRefinement._maybe_record_checkpoint",
                return_value=None,
            ),
            weak_restore_patch,
        ):
            yield
    else:
        with weak_restore_patch:
            yield


def _chain_marker_poses(marker_size_m: float = 0.07) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    reference_rotation, reference_translation = reference_gauge_pose(marker_size_m)
    side_rotation, _ = cv2.Rodrigues(np.array([0.0, np.pi / 2.0, 0.0], dtype=np.float64))
    return {
        0: (reference_rotation, reference_translation),
        1: (reference_rotation, reference_translation + np.array([0.12, 0.0, -0.05], dtype=np.float64)),
        2: (
            reference_rotation @ side_rotation,
            reference_translation + np.array([0.12, 0.0, -0.12], dtype=np.float64),
        ),
        3: (
            reference_rotation @ side_rotation,
            reference_translation + np.array([0.24, 0.0, -0.12], dtype=np.float64),
        ),
    }


def _line_marker_poses(
    marker_count: int,
    *,
    marker_size_m: float = 0.07,
    spacing_m: float = 0.12,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    reference_rotation, reference_translation = reference_gauge_pose(marker_size_m)
    return {
        marker_id: (
            reference_rotation,
            reference_translation + np.array([spacing_m * marker_id, 0.0, -0.05 * marker_id], dtype=np.float64),
        )
        for marker_id in range(marker_count)
    }


class MarkerLayoutCalibrationSuccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _calibrate(self, observations: list[FrameObservation], expected_ids: list[int]) -> CalibrationResult:
        return calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=expected_ids,
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

    def test_accepts_mostly_good_frames_and_rejects_inconsistent_assignment_frames(self) -> None:
        observations = _synth_pair_with_corrupt_frames(
            25,
            frozenset({2, 7, 11, 16, 22}),
        )
        marker_poses = _pair_poses(self.marker_size_m)
        result = self._calibrate(observations, [0, 1])

        self.assertIsNone(result.failure_reason)
        assert result.layout is not None
        assert result.quality is not None
        self.assertEqual(result.quality.input_frame_count, 25)
        self.assertEqual(result.quality.rejected_frame_count, 5)
        self.assertEqual(result.quality.accepted_frame_count, 20)
        self.assertEqual(result.quality.frame_count, 20)
        self.assertEqual(result.quality.edges[0].inlier_count, 20)
        recovered_offset = (
            result.layout.footprints[1].bottom_left - result.layout.footprints[0].bottom_left
        )
        expected_offset = marker_poses[1][1] - marker_poses[0][1]
        np.testing.assert_allclose(recovered_offset, expected_offset, atol=1e-2)

    def test_recovers_planar_pair_geometry(self) -> None:
        marker_poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(marker_poses, frame_count=25, marker_size_m=self.marker_size_m)
        result = self._calibrate(observations, [0, 1])

        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.layout)
        assert result.layout is not None
        recovered_offset = (
            result.layout.footprints[1].bottom_left - result.layout.footprints[0].bottom_left
        )
        expected_offset = marker_poses[1][1] - marker_poses[0][1]
        np.testing.assert_allclose(recovered_offset, expected_offset, atol=1e-2)

    def test_reference_marker_gauge_is_fixed(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1])
        assert result.layout is not None
        footprint = result.layout.footprints[0]
        half = self.marker_size_m / 2.0
        np.testing.assert_allclose(footprint.top_left, [-half, -half, 0.0], atol=1e-3)
        np.testing.assert_allclose(footprint.bottom_left, [-half, half, 0.0], atol=1e-3)
        np.testing.assert_allclose(
            (footprint.bottom_left + footprint.bottom_right) / 2.0,
            [0.0, half, 0.0],
            atol=1e-3,
        )

    def test_recovers_non_coplanar_chain_with_interleaved_covisibility(self) -> None:
        marker_poses = _chain_marker_poses(self.marker_size_m)
        ground_truth = _ground_truth_footprints(marker_poses, self.marker_size_m)
        chain_visibility = [(0, 1), (1, 2), (2, 3)]
        frame_count = 25 * len(chain_visibility)
        observations = synthesize_observations(
            marker_poses,
            frame_count=frame_count,
            marker_size_m=self.marker_size_m,
            visible_markers=lambda frame_index: chain_visibility[frame_index % len(chain_visibility)],
        )
        result = self._calibrate(observations, [0, 1, 2, 3])

        self.assertIsNone(result.failure_reason)
        assert result.layout is not None
        for marker_id in (2, 3):
            for corner_name in CORNER_NAMES:
                expected = getattr(ground_truth[marker_id], corner_name)
                actual = getattr(result.layout.footprints[marker_id], corner_name)
                self.assertLess(
                    float(np.linalg.norm(actual - expected)),
                    0.01,
                    msg=f"marker {marker_id} {corner_name}",
                )

    def test_shuffled_marker_dict_order_recovers_equivalent_layout(self) -> None:
        marker_poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(marker_poses, frame_count=25, marker_size_m=self.marker_size_m)
        ordered = self._calibrate(observations, [0, 1])
        shuffled = self._calibrate(_shuffle_marker_dict(observations), [0, 1])

        self.assertIsNone(ordered.failure_reason)
        self.assertIsNone(shuffled.failure_reason)
        assert ordered.layout is not None and shuffled.layout is not None
        for marker_id in (0, 1):
            for corner_name in CORNER_NAMES:
                expected = getattr(ordered.layout.footprints[marker_id], corner_name)
                actual = getattr(shuffled.layout.footprints[marker_id], corner_name)
                np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_edge_support_counts_distinct_frame_ids(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=20,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1])
        assert result.quality is not None
        self.assertEqual(result.quality.edges[0].inlier_count, 20)

        from object_apriltag.marker_layout_calibration.discrete_graph import (
            collect_pair_hypotheses,
            estimate_frame_candidates,
            normalize_observations,
        )

        object_points_by_marker = _uniform_object_points_by_marker([0, 1], self.marker_size_m)
        normalized = normalize_observations(observations, [0, 1])
        frame_candidates = estimate_frame_candidates(
            normalized,
            object_points_by_marker,
            self.camera_matrix,
            self.dist_coeffs,
        )
        hypotheses = collect_pair_hypotheses(frame_candidates, [0, 1])
        pair = (0, 1)
        unique_frames = {frame_index for _, _, frame_index in hypotheses[pair]}
        self.assertGreater(len(hypotheses[pair]), len(unique_frames))
        self.assertEqual(len(unique_frames), 20)

    def test_json_roundtrip_preserves_geometry(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1])
        assert result.layout is not None

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            save_marker_model(path, result.layout)
            loaded = load_marker_model(path)
            self.assertEqual(
                marker_layout_to_dict(loaded),
                marker_layout_to_dict(result.layout),
            )

    def test_quality_report_lists_connected_markers_and_edges(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1])
        assert result.quality is not None
        self.assertEqual(result.quality.connected_marker_ids, frozenset({0, 1}))
        self.assertGreaterEqual(len(result.quality.edges), 1)
        self.assertGreaterEqual(result.quality.edges[0].inlier_count, 20)
        self.assertLess(result.quality.reprojection_rms_px, 0.1)

    def test_sparse_outliers_are_pruned_and_calibration_still_passes(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            outlier_fraction=0.05,
            outlier_shift_px=40.0,
            seed=10,
        )
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.failure_reason)
        assert result.layout is not None
        recovered_offset = (
            result.layout.footprints[1].bottom_left - result.layout.footprints[0].bottom_left
        )
        expected_offset = _two_marker_poses(self.marker_size_m)[1][1] - _two_marker_poses(self.marker_size_m)[0][1]
        np.testing.assert_allclose(recovered_offset, expected_offset, atol=1e-2)


class MarkerLayoutCalibrationRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _calibrate(self, observations: list[FrameObservation], expected_ids: list[int]) -> CalibrationResult:
        return calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=expected_ids,
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

    def test_refuses_when_expected_id_never_observed(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1, 2])
        self.assertIsNone(result.layout)
        self.assertIn("never observed", result.failure_reason or "")

    def test_refuses_when_pair_support_is_below_twenty(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=19,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        self.assertIn("not connected", result.failure_reason or "")

    def test_refuses_when_all_assignment_frames_are_inconsistent(self) -> None:
        observations = _synth_pair_with_corrupt_frames(
            25,
            frozenset(range(25)),
            varying_corrupt=True,
        )
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        self.assertIsNotNone(result.failure_reason)
        assert result.quality is not None
        self.assertEqual(result.quality.input_frame_count, 25)
        self.assertEqual(result.quality.accepted_frame_count, 0)

    def test_refuses_when_too_many_assignment_frames_are_inconsistent(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            seed=4,
        )
        _rotate_marker_corners(observations, range(19, 25), marker_id=1)
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        self.assertIsNotNone(result.failure_reason)
        assert result.quality is not None
        self.assertEqual(result.quality.accepted_frame_count, 0)

    def test_refuses_after_prune_when_pair_support_is_lost(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            seed=1,
        )
        _apply_constant_marker_noise(observations, marker_id=1, noise_std_px=3.0, seed=0)
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        self.assertTrue(
            "supported frames after pruning" in (result.failure_reason or "")
            or "not connected after pruning" in (result.failure_reason or ""),
            result.failure_reason,
        )
        assert result.quality is not None
        self.assertEqual(result.quality.inlier_corner_count, 0)
        assert result.quality.dropped_pair_edges is not None
        post_pruning_drops = [
            edge for edge in result.quality.dropped_pair_edges if edge.stage == "post_pruning"
        ]
        self.assertGreater(len(post_pruning_drops), 0)

    def test_refuses_when_per_marker_reprojection_gate_fails(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            seed=1,
        )
        _apply_constant_marker_noise(observations, marker_id=1, noise_std_px=1.0, seed=0)
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=CalibrationSettings(
                min_inliers_per_edge=20,
                reprojection_rms_gate_px=0.5,
            ),
        )
        self.assertIsNone(result.layout)
        self.assertIn("Marker 1 reprojection RMS", result.failure_reason or "")
        assert result.quality is not None
        self.assertLess(result.quality.reprojection_rms_px, 0.5)
        self.assertGreater(result.quality.per_marker_reprojection_rms_px[1], 0.5)

    @mock.patch("object_apriltag.marker_layout_calibration.continuous_refinement.least_squares")
    def test_bundle_adjustment_failures_are_structured(self, least_squares_mock: mock.Mock) -> None:
        observations = _synth_pair_with_corrupt_frames(25, frozenset({2, 7}))
        least_squares_mock.side_effect = ValueError("singular matrix")
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        assert result.quality is not None
        self.assertIn("Bundle adjustment failed", result.failure_reason or "")
        self.assertEqual(result.quality.reprojection_rms_px, float("inf"))
        self.assertEqual(result.quality.inlier_corner_count, 0)
        self.assertEqual(result.quality.rejected_frame_count, 2)
        assert result.quality.assignment_rejection_records is not None
        self.assertEqual(len(result.quality.assignment_rejection_records), 2)
        assert result.quality.dropped_pair_edges is not None

    @mock.patch("object_apriltag.marker_layout_calibration.continuous_refinement.least_squares")
    def test_bundle_adjustment_failure_preserves_assignment_and_edge_diagnostics(
        self,
        least_squares_mock: mock.Mock,
    ) -> None:
        observations = [
            FrameObservation(frame_id=f"capture-{index}", markers=obs.markers)
            for index, obs in enumerate(_synth_pair_with_corrupt_frames(25, frozenset({2, 7, 11})))
        ]
        least_squares_mock.side_effect = ValueError("singular matrix")
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(result.layout)
        assert result.quality is not None
        assert result.quality.assignment_rejections is not None
        self.assertEqual(result.quality.assignment_rejections.total_rejected, 3)
        assert result.quality.assignment_rejection_records is not None
        self.assertEqual(
            {record.frame_id for record in result.quality.assignment_rejection_records},
            {"capture-2", "capture-7", "capture-11"},
        )
        assert result.quality.dropped_pair_edges is not None

    @mock.patch("object_apriltag.marker_layout_calibration.continuous_refinement.least_squares")
    def test_positive_depth_failure_is_structured(self, least_squares_mock: mock.Mock) -> None:
        from scipy.optimize import OptimizeResult

        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )

        def _return_behind_camera_pose(fun, x0, **kwargs):
            bad = np.asarray(x0, dtype=np.float64).copy()
            bad[-1] = -1.0
            return OptimizeResult(x=bad, success=True, status=1)

        least_squares_mock.side_effect = _return_behind_camera_pose
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        assert result.quality is not None
        self.assertEqual(result.quality.reprojection_rms_px, float("inf"))
        self.assertIn("non-positive depth", result.failure_reason or "")

    def test_refuses_when_expected_ids_are_disconnected(self) -> None:
        marker_poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            marker_poses,
            frame_count=25,
            marker_size_m=self.marker_size_m,
            visible_markers=lambda _: (0, 1),
        )
        result = self._calibrate(observations, [0, 1, 2])
        self.assertIsNone(result.layout)
        self.assertIn("never observed", result.failure_reason or "")

    def test_refuses_when_reprojection_gate_fails(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            noise_std_px=0.05,
            seed=3,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=CalibrationSettings(
                min_inliers_per_edge=20,
                reprojection_rms_gate_px=0.15,
            ),
        )
        self.assertIsNone(result.layout)
        self.assertIn("reprojection RMS", result.failure_reason or "")
        assert result.quality is not None
        self.assertGreater(result.quality.reprojection_rms_px, 0.15)

    def test_structured_failure_has_no_partial_layout(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=19,
            marker_size_m=self.marker_size_m,
        )
        result = self._calibrate(observations, [0, 1])
        self.assertIsNone(result.layout)
        self.assertIsNotNone(result.failure_reason)
        self.assertIsNotNone(result.quality)


class BestEffortCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()

    def test_best_effort_emits_provisional_layout_when_only_quality_gates_fail(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            noise_std_px=0.05,
            seed=3,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=CalibrationSettings(
                min_inliers_per_edge=20,
                reprojection_rms_gate_px=0.15,
            ),
            best_effort=True,
        )
        self.assertIsNotNone(result.layout)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.outcome, "provisional")
        self.assertEqual(result.calibration_policy, "best_effort")
        self.assertTrue(result.failed_quality_gates)
        self.assertIn("reprojection RMS", result.failed_quality_gates[0])
        assert result.quality is not None
        self.assertGreater(result.quality.reprojection_rms_px, 0.15)

    def test_strict_mode_still_refuses_when_quality_gates_fail(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            noise_std_px=0.05,
            seed=3,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=CalibrationSettings(
                min_inliers_per_edge=20,
                reprojection_rms_gate_px=0.15,
            ),
        )
        self.assertIsNone(result.layout)
        self.assertIsNotNone(result.failure_reason)
        self.assertEqual(result.outcome, "refused")
        self.assertEqual(result.calibration_policy, "strict")

    def test_best_effort_still_refuses_when_marker_never_observed(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1, 2],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=CalibrationSettings(min_inliers_per_edge=20),
            best_effort=True,
        )
        self.assertIsNone(result.layout)
        self.assertIn("never observed", result.failure_reason or "")
        self.assertEqual(result.outcome, "refused")

    def test_best_effort_recovers_all_weak_two_marker_graph_below_min_inliers(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=19,
            marker_size_m=self.marker_size_m,
        )
        settings = CalibrationSettings(min_inliers_per_edge=20)
        strict = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=settings,
        )
        self.assertIsNone(strict.layout)
        self.assertEqual(strict.outcome, "refused")

        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=settings,
            best_effort=True,
        )
        self.assertIsNotNone(result.layout)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.calibration_policy, "best_effort")
        self.assertIn(result.outcome, ("accepted", "provisional"))
        assert result.quality is not None
        self.assertEqual(result.quality.connected_marker_ids, frozenset({0, 1}))
        assert result.quality.restored_pair_edges is not None
        restored = next(
            edge for edge in result.quality.restored_pair_edges if edge.marker_pair == (0, 1)
        )
        self.assertEqual(restored.original_reason, "insufficient_observed_frames")
        self.assertEqual(restored.supported_count, 19)
        self.assertEqual(restored.observed_count, 19)

    def test_provisional_marker_model_roundtrips_and_loads(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            noise_std_px=0.05,
            seed=3,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=CalibrationSettings(
                min_inliers_per_edge=20,
                reprojection_rms_gate_px=0.15,
            ),
            best_effort=True,
        )
        assert result.layout is not None
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            save_marker_model(path, result.layout)
            loaded = load_marker_model(path)
            self.assertEqual(
                marker_layout_to_dict(loaded),
                marker_layout_to_dict(result.layout),
            )


class MarkerLayoutCalibrationInputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _minimal_observation(self) -> FrameObservation:
        return FrameObservation(
            frame_id=0,
            markers={
                0: np.zeros((4, 2), dtype=np.float64),
                1: np.ones((4, 2), dtype=np.float64),
            },
        )

    def _assert_structured_refusal(self, result: CalibrationResult, *, reason_substring: str) -> None:
        self.assertIsNone(result.layout)
        self.assertIsNone(result.quality)
        self.assertIsNotNone(result.failure_reason)
        self.assertIn(reason_substring, result.failure_reason or "")

    def test_rejects_empty_expected_ids(self) -> None:
        result = calibrate_marker_layout(
            [],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="empty")

    def test_rejects_reference_not_in_expected_ids(self) -> None:
        result = calibrate_marker_layout(
            [],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[1, 2],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="not in expected_marker_ids")

    def test_rejects_duplicate_expected_marker_ids(self) -> None:
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1, 0],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="duplicates")

    def test_rejects_duplicate_frame_ids(self) -> None:
        observation = self._minimal_observation()
        result = calibrate_marker_layout(
            [observation, observation],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(
            result,
            reason_substring="Duplicate FrameObservation.frame_id",
        )

    def test_rejects_malformed_expected_marker_corners(self) -> None:
        result = calibrate_marker_layout(
            [
                FrameObservation(
                    frame_id=0,
                    markers={0: np.zeros((3, 2), dtype=np.float64), 1: np.ones((4, 2))},
                )
            ],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="Malformed corners")

    def test_rejects_non_finite_expected_marker_corners(self) -> None:
        corners = np.ones((4, 2), dtype=np.float64)
        corners[0, 0] = np.inf
        result = calibrate_marker_layout(
            [
                FrameObservation(
                    frame_id=0,
                    markers={0: corners, 1: np.ones((4, 2), dtype=np.float64)},
                )
            ],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="finite 4x2 points")

    def test_rejects_invalid_camera_matrix_shape(self) -> None:
        bad_matrix = np.zeros((3, 4), dtype=np.float64)
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            bad_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="shape (3, 3)")

    def test_rejects_non_finite_camera_matrix(self) -> None:
        bad_matrix = self.camera_matrix.copy()
        bad_matrix[0, 0] = np.nan
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            bad_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="camera_matrix must contain only finite values")

    def test_rejects_empty_dist_coeffs(self) -> None:
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            self.camera_matrix,
            np.array([], dtype=np.float64),
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="dist_coeffs must not be empty")

    def test_rejects_non_finite_dist_coeffs(self) -> None:
        bad_dist = self.dist_coeffs.copy()
        bad_dist[0, 0] = np.inf
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            self.camera_matrix,
            bad_dist,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
        )
        self._assert_structured_refusal(result, reason_substring="dist_coeffs must contain only finite values")

    def test_rejects_non_positive_marker_size(self) -> None:
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=0.0,
        )
        self._assert_structured_refusal(result, reason_substring="marker_size_m must be a finite positive number")

    def test_rejects_non_finite_marker_size(self) -> None:
        result = calibrate_marker_layout(
            [self._minimal_observation()],
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=float("nan"),
        )
        self._assert_structured_refusal(result, reason_substring="marker_size_m must be a finite positive number")

    def test_rejects_invalid_calibration_settings_fields(self) -> None:
        invalid_settings = (
            CalibrationSettings(min_inliers_per_edge=0),
            CalibrationSettings(max_ba_iterations=0),
            CalibrationSettings(reprojection_rms_gate_px=float("nan")),
            CalibrationSettings(pair_translation_rms_gate_ratio=-0.1),
            CalibrationSettings(pair_rotation_rms_gate_deg=0.0),
            CalibrationSettings(huber_delta_px=float("inf")),
            CalibrationSettings(corner_outlier_px=0.0),
        )
        expected_fragments = (
            "min_inliers_per_edge must be positive",
            "max_ba_iterations must be positive",
            "reprojection_rms_gate_px must be finite and positive",
            "pair_translation_rms_gate_ratio must be finite and positive",
            "pair_rotation_rms_gate_deg must be finite and positive",
            "huber_delta_px must be finite and positive",
            "corner_outlier_px must be finite and positive",
        )
        for settings, fragment in zip(invalid_settings, expected_fragments, strict=True):
            with self.subTest(fragment=fragment):
                result = calibrate_marker_layout(
                    [self._minimal_observation()],
                    self.camera_matrix,
                    self.dist_coeffs,
                    expected_marker_ids=[0, 1],
                    reference_marker_id=0,
                    marker_size_m=self.marker_size_m,
                    settings=settings,
                )
                self._assert_structured_refusal(result, reason_substring=fragment)

    def test_filters_malformed_unknown_marker_corners_without_error(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        for observation in observations:
            observation.markers[99] = np.zeros((3, 2), dtype=np.float64)
            observation.markers[100] = np.full((4, 2), np.nan, dtype=np.float64)

        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.layout)
        self.assertIsNotNone(result.quality)


def _make_pair_consensus(
    marker_a: int,
    marker_b: int,
    frame_indices: Iterable[int],
) -> PairConsensus:
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    frames = tuple(sorted(frame_indices))
    return PairConsensus(
        marker_a=marker_a,
        marker_b=marker_b,
        rotation_ba=rotation,
        translation_ba=translation,
        inlier_frames=frames,
        inlier_hypotheses={frame_index: (rotation, translation) for frame_index in frames},
    )


def _triangle_marker_poses(marker_size_m: float = 0.07) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    reference_rotation, reference_translation = reference_gauge_pose(marker_size_m)
    return {
        0: (reference_rotation, reference_translation),
        1: (reference_rotation, reference_translation + np.array([0.12, 0.0, -0.05], dtype=np.float64)),
        2: (reference_rotation, reference_translation + np.array([0.12, 0.0, -0.12], dtype=np.float64)),
    }


class PairGraphFilteringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = CalibrationSettings(min_inliers_per_edge=20)
        self.expected_ids = [0, 1, 2]
        self.reference_marker_id = 0

    def test_restrict_drops_weak_redundant_edge_when_chain_remains(self) -> None:
        pair_consensus = {
            (0, 1): _make_pair_consensus(0, 1, range(20)),
            (1, 2): _make_pair_consensus(1, 2, range(20)),
            (0, 2): _make_pair_consensus(0, 2, range(10)),
        }
        allowed_frames = frozenset(range(20))

        filtered, failure, dropped = restrict_pair_consensus_to_frames(
            pair_consensus,
            allowed_frames,
            self.expected_ids,
            self.reference_marker_id,
            self.settings,
            marker_sizes_m=_uniform_marker_sizes(self.expected_ids, 0.07),
        )

        self.assertIsNone(failure)
        self.assertEqual(set(filtered), {(0, 1), (1, 2)})
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].marker_pair, (0, 2))
        self.assertEqual(dropped[0].stage, "assignment_support")
        self.assertEqual(dropped[0].reason, "insufficient_support")

    def test_restrict_fails_when_required_bridge_edge_is_dropped(self) -> None:
        pair_consensus = {
            (0, 1): _make_pair_consensus(0, 1, range(10)),
            (1, 2): _make_pair_consensus(1, 2, range(20)),
        }
        allowed_frames = frozenset(range(20))

        filtered, failure, dropped = restrict_pair_consensus_to_frames(
            pair_consensus,
            allowed_frames,
            self.expected_ids,
            self.reference_marker_id,
            self.settings,
            marker_sizes_m=_uniform_marker_sizes(self.expected_ids, 0.07),
        )

        self.assertIsNotNone(failure)
        self.assertIn("not connected", failure or "")
        self.assertEqual(set(filtered), {(1, 2)})
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].marker_pair, (0, 1))
        self.assertEqual(dropped[0].stage, "assignment_support")

    def test_estimate_drops_bad_redundant_edge_when_chain_remains(self) -> None:
        marker_size_m = 0.07
        translation_spread = 0.02
        rotation = np.eye(3, dtype=np.float64)
        good_translation = np.zeros(3, dtype=np.float64)

        def _hypotheses(
            pair: tuple[int, int],
            *,
            bad: bool = False,
        ) -> list[tuple[np.ndarray, np.ndarray, int]]:
            hypotheses: list[tuple[np.ndarray, np.ndarray, int]] = []
            for frame_index in range(25):
                if bad:
                    offset = translation_spread * (1.0 if frame_index % 2 else -1.0)
                    translation = good_translation + np.array([offset, 0.0, 0.0], dtype=np.float64)
                else:
                    translation = good_translation
                hypotheses.append((rotation, translation, frame_index))
            return hypotheses

        pair_hypotheses = {
            (0, 1): _hypotheses((0, 1)),
            (1, 2): _hypotheses((1, 2)),
            (0, 2): _hypotheses((0, 2), bad=True),
        }

        marker_sizes_m = _uniform_marker_sizes(self.expected_ids, marker_size_m)
        filtered, failure, dropped = estimate_pair_consensus(
            pair_hypotheses,
            self.expected_ids,
            self.reference_marker_id,
            marker_sizes_m,
            self.settings,
        )

        self.assertIsNone(failure)
        self.assertEqual(set(filtered), {(0, 1), (1, 2)})
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].marker_pair, (0, 2))
        self.assertEqual(dropped[0].stage, "initial_consensus")

    def test_recheck_drops_weak_redundant_edge_when_chain_remains(self) -> None:
        pair_consensus = {
            (0, 1): _make_pair_consensus(0, 1, range(20)),
            (1, 2): _make_pair_consensus(1, 2, range(20)),
            (0, 2): _make_pair_consensus(0, 2, range(10)),
        }
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=np.zeros(2, dtype=np.float64),
            )
            for frame_index in range(20)
            for marker_id in (0, 1, 2)
            for corner_index in range(4)
        ]
        inlier_mask = np.ones(len(corner_observations), dtype=bool)

        filtered, failure, dropped = _recheck_pair_support(
            pair_consensus,
            corner_observations,
            inlier_mask,
            self.expected_ids,
            self.reference_marker_id,
            self.settings,
            allowed_frames=frozenset(range(20)),
            marker_sizes_m=_uniform_marker_sizes(self.expected_ids, 0.07),
        )

        self.assertIsNone(failure)
        self.assertEqual(set(filtered), {(0, 1), (1, 2)})
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0].marker_pair, (0, 2))
        self.assertEqual(dropped[0].stage, "post_pruning")

    def test_calibration_survives_redundant_weak_triangle_edge(self) -> None:
        marker_size_m = 0.07
        marker_poses = _triangle_marker_poses(marker_size_m)
        observations: list[FrameObservation] = []
        for frame_index in range(40):
            visible = (0, 1) if frame_index < 20 else (1, 2)
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=marker_size_m,
                visible_markers=lambda _, visible=visible: visible,
                seed=frame_index,
            )[0]
            observations.append(
                FrameObservation(frame_id=frame_index, markers=frame_observation.markers)
            )
        for frame_index in range(10):
            triple = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=marker_size_m,
                visible_markers=lambda _: (0, 1, 2),
                seed=100 + frame_index,
            )[0]
            observations[frame_index] = FrameObservation(
                frame_id=frame_index,
                markers={**observations[frame_index].markers, 2: triple.markers[2]},
            )

        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2],
            reference_marker_id=0,
            marker_size_m=marker_size_m,
            settings=self.settings,
        )

        self.assertIsNone(result.failure_reason, result.failure_reason)
        self.assertIsNotNone(result.layout)
        assert result.quality is not None
        self.assertEqual(result.quality.connected_marker_ids, frozenset({0, 1, 2}))
        edge_pairs = {(edge.marker_a, edge.marker_b) for edge in result.quality.edges}
        self.assertIn((0, 1), edge_pairs)
        self.assertIn((1, 2), edge_pairs)
        self.assertNotIn((0, 2), edge_pairs)
        assert result.quality.dropped_pair_edges is not None
        dropped_pairs = {edge.marker_pair for edge in result.quality.dropped_pair_edges}
        self.assertIn((0, 2), dropped_pairs)
        self.assertTrue(
            any(edge.stage == "initial_consensus" for edge in result.quality.dropped_pair_edges)
            or any(edge.stage == "assignment_support" for edge in result.quality.dropped_pair_edges)
            or any(edge.stage == "post_pruning" for edge in result.quality.dropped_pair_edges)
        )


class OptimizationCheckpointRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _prune_failure_observations(self) -> list[FrameObservation]:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
            seed=1,
        )
        _apply_constant_marker_noise(observations, marker_id=1, noise_std_px=3.0, seed=0)
        return observations

    def test_strict_mode_still_refuses_when_pruning_fails(self) -> None:
        result = calibrate_marker_layout(
            self._prune_failure_observations(),
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(result.layout)
        self.assertEqual(result.outcome, "refused")
        self.assertIsNone(result.selected_checkpoint_stage)
        self.assertIsNone(result.failed_refinement_stage)

    def test_best_effort_recovers_pre_pruning_checkpoint_when_pruning_fails(self) -> None:
        with _pruning_refinement_failure_without_weak_recovery():
            result = calibrate_marker_layout(
                self._prune_failure_observations(),
                self.camera_matrix,
                self.dist_coeffs,
                expected_marker_ids=[0, 1],
                reference_marker_id=0,
                marker_size_m=self.marker_size_m,
                settings=self.settings,
                best_effort=True,
            )
        self.assertIsNotNone(result.layout)
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.outcome, "provisional")
        self.assertEqual(result.calibration_policy, "best_effort")
        self.assertEqual(result.selected_checkpoint_stage, "initial_bundle_adjustment")
        self.assertEqual(result.failed_refinement_stage, "post_pruning_refit")
        assert result.quality is not None
        self.assertEqual(result.quality.connected_marker_ids, frozenset({0, 1}))
        self.assertEqual(result.quality.missing_expected_ids, frozenset())

    def test_best_effort_refuses_when_no_valid_complete_checkpoint_exists(self) -> None:
        with _pruning_refinement_failure_without_weak_recovery(block_checkpoints=True):
            result = calibrate_marker_layout(
                self._prune_failure_observations(),
                self.camera_matrix,
                self.dist_coeffs,
                expected_marker_ids=[0, 1],
                reference_marker_id=0,
                marker_size_m=self.marker_size_m,
                settings=self.settings,
                best_effort=True,
            )

        self.assertIsNone(result.layout)
        self.assertEqual(result.outcome, "refused")
        self.assertIsNotNone(result.failure_reason)
        self.assertTrue(
            "supported frames after pruning" in (result.failure_reason or "")
            or "not connected after pruning" in (result.failure_reason or "")
            or "Too few inlier" in (result.failure_reason or "")
        )

    @mock.patch("object_apriltag.marker_layout_calibration.continuous_refinement.least_squares")
    def test_best_effort_rejects_incomplete_checkpoint(
        self,
        least_squares_mock: mock.Mock,
    ) -> None:
        from scipy.optimize import OptimizeResult, least_squares

        observations = self._prune_failure_observations()
        call_count = {"n": 0}

        def _ba_once_then_real(fun, x0, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                bad = np.asarray(x0, dtype=np.float64).copy()
                bad[-1] = -1.0
                return OptimizeResult(x=bad, success=True, status=1)
            return least_squares(fun, x0, **kwargs)

        least_squares_mock.side_effect = _ba_once_then_real

        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )

        self.assertIsNotNone(result.layout)
        self.assertEqual(result.selected_checkpoint_stage, "graph_initialization")
        self.assertEqual(result.failed_refinement_stage, "initial_bundle_adjustment")


class WeakRestoreSupportFloorTests(unittest.TestCase):
    def test_single_frame_consensus_not_restorable_when_observed_count_meets_floor(self) -> None:
        pair = (0, 1)
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        edge = PairConsensus(
            marker_a=0,
            marker_b=1,
            rotation_ba=rotation,
            translation_ba=translation,
            inlier_frames=(0,),
            inlier_hypotheses={0: (rotation, translation)},
        )
        dropped = make_dropped_pair_edge(
            pair,
            "initial_consensus",
            "insufficient_observed_frames",
            observed_count=2,
            supported_count=2,
            required_count=20,
            translation_gate=0.007,
            rotation_gate=5.0,
            edge=edge,
        )

        self.assertEqual(weak_restore_candidates({}, {pair: edge}, [dropped]), [])

    def test_make_restored_pair_edge_reports_consensus_support(self) -> None:
        pair = (0, 1)
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        edge = PairConsensus(
            marker_a=0,
            marker_b=1,
            rotation_ba=rotation,
            translation_ba=translation,
            inlier_frames=(0, 1, 2),
            inlier_hypotheses={
                frame_index: (rotation, translation) for frame_index in (0, 1, 2)
            },
        )
        dropped = make_dropped_pair_edge(
            pair,
            "initial_consensus",
            "insufficient_observed_frames",
            observed_count=10,
            supported_count=10,
            required_count=20,
            translation_gate=0.007,
            rotation_gate=5.0,
            edge=edge,
        )

        restored = make_restored_pair_edge(dropped, edge, "initial_consensus")

        self.assertEqual(restored.supported_count, 3)
        self.assertAlmostEqual(restored.support_fraction, 0.3)
        self.assertEqual(dropped.supported_count, 10)

    def test_weak_restore_ranking_prefers_higher_consensus_support(self) -> None:
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)

        def _edge(marker_a: int, marker_b: int, inlier_frames: tuple[int, ...]) -> PairConsensus:
            return PairConsensus(
                marker_a=marker_a,
                marker_b=marker_b,
                rotation_ba=rotation,
                translation_ba=translation,
                inlier_frames=inlier_frames,
                inlier_hypotheses={
                    frame_index: (rotation, translation) for frame_index in inlier_frames
                },
            )

        edge_01 = _edge(0, 1, tuple(range(8)))
        edge_02 = _edge(0, 2, tuple(range(3)))
        drop_02 = make_dropped_pair_edge(
            (0, 2),
            "initial_consensus",
            "insufficient_observed_frames",
            observed_count=10,
            supported_count=10,
            required_count=20,
            translation_gate=0.007,
            rotation_gate=5.0,
            edge=edge_02,
        )
        drop_01 = make_dropped_pair_edge(
            (0, 1),
            "initial_consensus",
            "insufficient_observed_frames",
            observed_count=10,
            supported_count=10,
            required_count=20,
            translation_gate=0.007,
            rotation_gate=5.0,
            edge=edge_01,
        )
        restored: list = []

        failure = maybe_restore_weak_connectivity(
            {},
            {(0, 1): edge_01, (0, 2): edge_02},
            [drop_02, drop_01],
            [0, 1, 2],
            0,
            "initial_consensus",
            best_effort=True,
            restored_pair_edges=restored,
        )

        self.assertIsNone(failure)
        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].marker_pair, (0, 1))
        self.assertEqual(restored[0].supported_count, 8)
        self.assertAlmostEqual(restored[0].support_fraction, 0.8)
        self.assertEqual(restored[1].marker_pair, (0, 2))
        self.assertEqual(restored[1].supported_count, 3)
        self.assertAlmostEqual(restored[1].support_fraction, 0.3)


class WeakPairConnectivityRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.settings = CalibrationSettings(min_inliers_per_edge=20)
        self.expected_ids = [0, 1, 2]
        self.reference_marker_id = 0

    def _bridge_observations(
        self,
        *,
        bridge_frames: int,
        tail_frames: int,
        rotate_opposite_ippe_frames: Iterable[int] | None = None,
    ) -> list[FrameObservation]:
        marker_poses = _triangle_marker_poses(self.marker_size_m)
        observations: list[FrameObservation] = []
        for frame_index in range(bridge_frames + tail_frames):
            visible = (0, 1) if frame_index < bridge_frames else (1, 2)
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=visible: visible,
                seed=frame_index,
            )[0]
            observations.append(
                FrameObservation(frame_id=frame_index, markers=frame_observation.markers)
            )
        if rotate_opposite_ippe_frames is not None:
            _rotate_marker_corners(observations, rotate_opposite_ippe_frames, 1)
        return observations

    def test_strict_mode_refuses_weak_bridge_pair(self) -> None:
        observations = self._bridge_observations(bridge_frames=10, tail_frames=20)
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=self.expected_ids,
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(result.layout)
        self.assertIsNotNone(result.failure_reason)
        self.assertEqual(result.outcome, "refused")

    def test_best_effort_recovers_weak_bridge_pair(self) -> None:
        observations = self._bridge_observations(bridge_frames=10, tail_frames=20)
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=self.expected_ids,
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.layout)
        assert result.quality is not None
        self.assertEqual(result.quality.connected_marker_ids, frozenset({0, 1, 2}))
        assert result.quality.restored_pair_edges is not None
        restored_pairs = {edge.marker_pair for edge in result.quality.restored_pair_edges}
        self.assertIn((0, 1), restored_pairs)
        restored = next(
            edge for edge in result.quality.restored_pair_edges if edge.marker_pair == (0, 1)
        )
        self.assertEqual(restored.original_reason, "insufficient_observed_frames")
        self.assertEqual(restored.supported_count, 10)
        self.assertEqual(restored.observed_count, 10)
        self.assertAlmostEqual(restored.support_fraction, 1.0)
        self.assertIsNotNone(restored.translation_rms_m)
        self.assertIsNotNone(restored.rotation_rms_deg)

    def test_weak_bridge_recovery_uses_dominant_ippe_mode(self) -> None:
        observations = self._bridge_observations(
            bridge_frames=20,
            tail_frames=20,
            rotate_opposite_ippe_frames=range(10, 20),
        )
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=self.expected_ids,
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.failure_reason)
        assert result.layout is not None and result.quality is not None
        ground_truth = _ground_truth_footprints(_triangle_marker_poses(self.marker_size_m), self.marker_size_m)
        for marker_id in self.expected_ids:
            for corner_name in CORNER_NAMES:
                expected = getattr(ground_truth[marker_id], corner_name)
                actual = getattr(result.layout.footprints[marker_id], corner_name)
                self.assertLess(
                    float(np.linalg.norm(actual - expected)),
                    0.02,
                    msg=f"marker {marker_id} {corner_name}",
                )
        assert result.quality.restored_pair_edges is not None
        restored = next(
            edge for edge in result.quality.restored_pair_edges if edge.marker_pair == (0, 1)
        )
        self.assertLess(restored.supported_count, 20)

    def test_best_effort_refuses_when_raw_graph_disconnected(self) -> None:
        marker_poses = {
            **_triangle_marker_poses(self.marker_size_m),
            3: (
                _triangle_marker_poses(self.marker_size_m)[2][0],
                _triangle_marker_poses(self.marker_size_m)[2][1] + np.array([0.12, 0.0, 0.0], dtype=np.float64),
            ),
        }
        observations: list[FrameObservation] = []
        for frame_index in range(20):
            visible = (0, 1) if frame_index < 10 else (2, 3)
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=visible: visible,
                seed=frame_index,
            )[0]
            observations.append(
                FrameObservation(frame_id=frame_index, markers=frame_observation.markers)
            )
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.layout)
        self.assertIn("raw observations", result.failure_reason or "")
        assert result.quality is not None
        self.assertTrue({2, 3}.issubset(result.quality.missing_expected_ids))

    def test_restrict_best_effort_recovers_weak_bridge(self) -> None:
        pair_consensus = {
            (0, 1): _make_pair_consensus(0, 1, range(10)),
            (1, 2): _make_pair_consensus(1, 2, range(20)),
        }
        filtered, failure, dropped = restrict_pair_consensus_to_frames(
            pair_consensus,
            frozenset(range(20)),
            self.expected_ids,
            self.reference_marker_id,
            self.settings,
            marker_sizes_m=_uniform_marker_sizes(self.expected_ids, self.marker_size_m),
            best_effort=True,
        )
        self.assertIsNone(failure)
        self.assertIn((0, 1), filtered)
        self.assertEqual(len(dropped), 1)

    def test_weak_recovery_restores_only_required_ranked_bridges(self) -> None:
        marker_poses = _line_marker_poses(5, marker_size_m=self.marker_size_m)
        observations: list[FrameObservation] = []
        for frame_index in range(25):
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=(0, 1): visible,
                seed=frame_index,
            )[0]
            observations.append(FrameObservation(frame_id=frame_index, markers=frame_observation.markers))
        for frame_index in range(25, 35):
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=(1, 2): visible,
                seed=frame_index,
            )[0]
            observations.append(FrameObservation(frame_id=frame_index, markers=frame_observation.markers))
        for frame_index in range(35, 45):
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=(2, 3): visible,
                seed=frame_index,
            )[0]
            observations.append(FrameObservation(frame_id=frame_index, markers=frame_observation.markers))
        for frame_index in range(45, 70):
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=(3, 4): visible,
                seed=frame_index,
            )[0]
            observations.append(FrameObservation(frame_id=frame_index, markers=frame_observation.markers))
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2, 3, 4],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.failure_reason)
        assert result.quality is not None and result.quality.restored_pair_edges is not None
        restored_pairs = {edge.marker_pair for edge in result.quality.restored_pair_edges}
        self.assertEqual(restored_pairs, {(1, 2), (2, 3)})
        bridge_12 = next(edge for edge in result.quality.restored_pair_edges if edge.marker_pair == (1, 2))
        bridge_23 = next(edge for edge in result.quality.restored_pair_edges if edge.marker_pair == (2, 3))
        self.assertGreaterEqual(bridge_23.supported_count, bridge_12.supported_count)


class FrameAssignmentFallbackRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _minority_flip_observations(
        self,
        *,
        frame_count: int,
        flip_start: int,
    ) -> list[FrameObservation]:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=frame_count,
            marker_size_m=self.marker_size_m,
            seed=4,
        )
        _rotate_marker_corners(observations, range(flip_start, frame_count), marker_id=1)
        return observations

    def test_strict_assignment_pass_unchanged_without_best_effort(self) -> None:
        observations = self._minority_flip_observations(frame_count=30, flip_start=20)
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        assert result.quality is not None
        self.assertEqual(result.quality.accepted_frame_count, 20)
        self.assertEqual(result.quality.rejected_frame_count, 10)
        self.assertIsNone(result.quality.fallback_assignment_records)

    def test_best_effort_recovers_minority_ippe_flip_frames(self) -> None:
        observations = self._minority_flip_observations(frame_count=30, flip_start=20)
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.failure_reason)
        assert result.layout is not None and result.quality is not None
        self.assertEqual(result.quality.input_frame_count, 30)
        self.assertEqual(result.quality.rejected_frame_count, 0)
        assert result.quality.fallback_assignment_records is not None
        self.assertEqual(len(result.quality.fallback_assignment_records), 10)
        fallback_frame_ids = {
            record.frame_id for record in result.quality.fallback_assignment_records
        }
        self.assertEqual(fallback_frame_ids, set(range(20, 30)))
        for record in result.quality.fallback_assignment_records:
            self.assertGreater(record.disagreement_cost, 0.0)
            self.assertIsNotNone(record.translation_error_m)
            self.assertIsNotNone(record.rotation_error_deg)

    def test_fallback_assignment_follows_consistent_global_branch(self) -> None:
        observations = self._minority_flip_observations(frame_count=30, flip_start=20)
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        assert result.layout is not None
        marker_poses = _two_marker_poses(self.marker_size_m)
        ground_truth = _ground_truth_footprints(marker_poses, self.marker_size_m)
        for marker_id in (0, 1):
            for corner_name in CORNER_NAMES:
                expected = getattr(ground_truth[marker_id], corner_name)
                actual = getattr(result.layout.footprints[marker_id], corner_name)
                self.assertLess(
                    float(np.linalg.norm(actual - expected)),
                    0.02,
                    msg=f"marker {marker_id} {corner_name}",
                )

    def test_best_effort_still_rejects_frames_without_valid_assignment(self) -> None:
        observations = _synth_pair_with_corrupt_frames(
            25,
            frozenset(range(25)),
            varying_corrupt=True,
        )
        result = calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.layout)
        assert result.quality is not None
        self.assertEqual(result.quality.accepted_frame_count, 0)
        self.assertIsNone(result.quality.fallback_assignment_records)


class PartialOutputCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.settings = CalibrationSettings(min_inliers_per_edge=20)
        self.reference_marker_id = 0

    def _raw_disconnected_observations(self) -> list[FrameObservation]:
        marker_poses = {
            **_triangle_marker_poses(self.marker_size_m),
            3: (
                _triangle_marker_poses(self.marker_size_m)[2][0],
                _triangle_marker_poses(self.marker_size_m)[2][1]
                + np.array([0.12, 0.0, 0.0], dtype=np.float64),
            ),
        }
        observations: list[FrameObservation] = []
        for frame_index in range(40):
            visible = (0, 1) if frame_index < 25 else (2, 3)
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=visible: visible,
                seed=frame_index,
            )[0]
            observations.append(
                FrameObservation(frame_id=frame_index, markers=frame_observation.markers)
            )
        return observations

    def test_partial_output_requires_best_effort(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            partial_output=True,
        )
        self.assertIsNone(result.layout)
        self.assertEqual(result.outcome, "refused")
        self.assertIn("best-effort", result.failure_reason or "")

    def test_best_effort_without_partial_output_refuses_incomplete_layout(self) -> None:
        observations = self._raw_disconnected_observations()
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.layout)
        self.assertEqual(result.outcome, "refused")
        self.assertIn("raw observations", result.failure_reason or "")

    def test_partial_output_emits_reference_connected_subset(self) -> None:
        observations = self._raw_disconnected_observations()
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
            partial_output=True,
        )
        self.assertIsNone(result.failure_reason)
        self.assertIsNotNone(result.layout)
        self.assertEqual(result.outcome, "partial")
        assert result.layout is not None
        self.assertEqual(set(result.layout.marker_ids), {0, 1})
        assert result.quality is not None
        self.assertEqual(result.quality.connected_marker_ids, frozenset({0, 1}))
        self.assertEqual(result.quality.missing_expected_ids, frozenset({2, 3}))
        omitted = {record.marker_id: record.reason for record in result.omitted_markers}
        self.assertEqual(set(omitted), {2, 3})
        self.assertEqual(
            omitted[2],
            "not_connected_in_raw_observations",
        )
        self.assertEqual(
            omitted[3],
            "not_connected_in_raw_observations",
        )

    def test_partial_model_roundtrips_and_loads(self) -> None:
        observations = self._raw_disconnected_observations()
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
            partial_output=True,
        )
        assert result.layout is not None
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            save_marker_model(path, result.layout)
            loaded = load_marker_model(path)
            self.assertEqual(
                marker_layout_to_dict(loaded),
                marker_layout_to_dict(result.layout),
            )

    def test_emit_partial_refuses_reference_only_component(self) -> None:
        from object_apriltag.marker_layout_calibration.finalize import emit_partial_calibration_result

        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        result = emit_partial_calibration_result(
            observations,
            *_default_camera(),
            requested_marker_ids=[0, 1, 2],
            connected_ids={0},
            omitted={1: "not_connected_to_reference", 2: "not_connected_to_reference"},
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            marker_sizes_m=uniform_marker_sizes([0, 1, 2], self.marker_size_m),
            settings=self.settings,
            best_effort=True,
        )
        self.assertIsNone(result.layout)
        self.assertEqual(result.outcome, "refused")
        self.assertIn("reference", (result.failure_reason or "").lower())

    def test_partial_output_validates_marker_sizes_for_emitted_subset(self) -> None:
        observations = self._raw_disconnected_observations()
        marker_sizes_m = uniform_marker_sizes([0, 1, 2, 3], self.marker_size_m)
        marker_sizes_m[1] = self.marker_size_m * 1.1
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            marker_sizes_m=marker_sizes_m,
            settings=self.settings,
            best_effort=True,
            partial_output=True,
        )
        assert result.layout is not None
        self.assertAlmostEqual(result.layout.marker_sizes_m[1], marker_sizes_m[1])
        self.assertNotIn(2, result.layout.marker_sizes_m)
        self.assertNotIn(3, result.layout.marker_sizes_m)

    def _observations_with_marker_only_in_rejected_frames(
        self,
        *,
        rejected_marker_id: int,
        rejected_frames: int = 8,
    ) -> list[FrameObservation]:
        marker_poses = _triangle_marker_poses(self.marker_size_m)
        observations: list[FrameObservation] = []
        for frame_index in range(30):
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=(0, 1): visible,
                seed=frame_index,
            )[0]
            observations.append(FrameObservation(frame_id=frame_index, markers=frame_observation.markers))
        for frame_index in range(30, 30 + rejected_frames):
            frame_observation = synthesize_observations(
                marker_poses,
                frame_count=1,
                marker_size_m=self.marker_size_m,
                visible_markers=lambda _, visible=(1, rejected_marker_id): visible,
                seed=frame_index,
            )[0]
            observations.append(
                FrameObservation(frame_id=frame_index, markers=frame_observation.markers)
            )
        _rotate_marker_corners(observations, range(30, 30 + rejected_frames), rejected_marker_id)
        return observations

    def test_strict_refuses_when_marker_missing_from_accepted_frames(self) -> None:
        observations = self._observations_with_marker_only_in_rejected_frames(rejected_marker_id=2)
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1, 2],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        self.assertIsNone(result.layout)
        self.assertEqual(result.outcome, "refused")

    def test_partial_after_missing_accepted_frames_emits_subset(self) -> None:
        from object_apriltag.marker_layout_calibration.finalize import partial_after_missing_accepted_frames_or_refuse

        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        pair_consensus = {
            (0, 1): _make_pair_consensus(0, 1, range(25)),
        }
        quality = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        ).quality
        assert quality is not None
        result = partial_after_missing_accepted_frames_or_refuse(
            observations,
            *_default_camera(),
            pair_consensus,
            quality,
            "Expected marker IDs have no accepted-frame observations after rejection: [2].",
            requested_marker_ids=[0, 1, 2],
            omitted_markers={},
            markers_in_accepted_frames={0, 1},
            missing_after_rejection=[2],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            marker_sizes_m=uniform_marker_sizes([0, 1, 2], self.marker_size_m),
            settings=self.settings,
            best_effort=True,
            partial_output=True,
            anchor_marker_ids=(0, 1),
        )
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.outcome, "partial")
        assert result.layout is not None
        self.assertEqual(set(result.layout.marker_ids), {0, 1})
        omitted = {record.marker_id: record.reason for record in result.omitted_markers}
        self.assertEqual(omitted, {2: "no_accepted_frame_observations"})

    def test_partial_after_missing_accepted_frames_refuses_without_partial_output(self) -> None:
        from object_apriltag.marker_layout_calibration.finalize import partial_after_missing_accepted_frames_or_refuse

        observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        pair_consensus = {(0, 1): _make_pair_consensus(0, 1, range(25))}
        failure_message = "Expected marker IDs have no accepted-frame observations after rejection: [2]."
        result = partial_after_missing_accepted_frames_or_refuse(
            observations,
            *_default_camera(),
            pair_consensus,
            quality_from_pairs(
                pair_consensus,
                [0, 1, 2],
                self.reference_marker_id,
                frozenset({2}),
                input_frame_count=25,
                rejected_frame_count=0,
                accepted_frame_count=25,
                observation_count=0,
            ),
            failure_message,
            requested_marker_ids=[0, 1, 2],
            omitted_markers={},
            markers_in_accepted_frames={0, 1},
            missing_after_rejection=[2],
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            marker_sizes_m=uniform_marker_sizes([0, 1, 2], self.marker_size_m),
            settings=self.settings,
            best_effort=True,
            partial_output=False,
            anchor_marker_ids=(0, 1),
        )
        self.assertIsNone(result.layout)
        self.assertEqual(result.failure_reason, failure_message)


class SaveMarkerModelTests(unittest.TestCase):
    def test_roundtrip_matches_in_memory_layout(self) -> None:
        marker_size_m = 0.07
        half = marker_size_m / 2.0
        footprints = {
            0: footprint_from_dict(
                0,
                {
                    "top_left": [-half, -half, 0.0],
                    "top_right": [half, -half, 0.0],
                    "bottom_right": [half, half, 0.0],
                    "bottom_left": [-half, half, 0.0],
                },
            ),
            1: footprint_from_dict(
                1,
                {
                    "top_left": [-half, -half + 0.08, 0.0],
                    "top_right": [half, -half + 0.08, 0.0],
                    "bottom_right": [half, half + 0.08, 0.0],
                    "bottom_left": [-half, half + 0.08, 0.0],
                },
            ),
        }
        layout = build_marker_layout(0, marker_size_m, footprints)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            save_marker_model(path, layout)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["reference_marker_id"], 0)
            loaded = load_marker_model(path)
            self.assertEqual(loaded.marker_ids, layout.marker_ids)

    def test_refused_calibration_does_not_write_output(self) -> None:
        observations = synthesize_observations(
            _two_marker_poses(0.07),
            frame_count=15,
            marker_size_m=0.07,
        )
        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=0.07,
            settings=CalibrationSettings(min_inliers_per_edge=20),
        )
        self.assertIsNone(result.layout)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            if result.layout is not None:
                save_marker_model(path, result.layout)
            self.assertFalse(path.exists())


class CalibrationSolveDiagnosticsTests(unittest.TestCase):
    EXPECTED_STAGE_KEYS = (
        "ippe_candidate_generation",
        "initial_pair_consensus",
        "strict_assignment",
        "initial_bundle_adjustment",
        "pruning",
        "post_pruning_refit",
    )
    OPTIMIZER_RUN_KEYS = (
        "stage",
        "nfev",
        "njev",
        "status",
        "cost",
        "active_frame_count",
        "inlier_corner_count",
    )

    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)
        self.observations = synthesize_observations(
            _two_marker_poses(self.marker_size_m),
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )

    def test_without_collector_calibration_result_unchanged(self) -> None:
        baseline = calibrate_marker_layout(
            self.observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )
        with_diagnostics = calibrate_marker_layout(
            self.observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            solve_diagnostics=CalibrationSolveDiagnostics(),
        )
        self.assertEqual(baseline.failure_reason, with_diagnostics.failure_reason)
        self.assertEqual(baseline.outcome, with_diagnostics.outcome)
        if baseline.layout is not None and with_diagnostics.layout is not None:
            np.testing.assert_allclose(
                baseline.layout.footprints[1].bottom_left,
                with_diagnostics.layout.footprints[1].bottom_left,
                atol=1e-9,
            )

    def test_collector_records_finite_stage_timings_and_optimizer_runs(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        calibrate_marker_layout(
            self.observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            solve_diagnostics=diagnostics,
        )
        for stage in self.EXPECTED_STAGE_KEYS:
            self.assertIn(stage, diagnostics.solve_stages_seconds)
            value = diagnostics.solve_stages_seconds[stage]
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)
            self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(len(diagnostics.optimizer_runs), 2)
        for run in diagnostics.optimizer_runs:
            self.assertEqual(set(run), set(self.OPTIMIZER_RUN_KEYS))
            self.assertGreaterEqual(run["nfev"], 0)
            self.assertGreaterEqual(run["active_frame_count"], 1)
            self.assertGreaterEqual(run["inlier_corner_count"], 8)
            self.assertIn(run["stage"], ("initial_bundle_adjustment", "post_pruning_refit"))

    def test_best_effort_records_fallback_assignment_stage(self) -> None:
        observations = _synth_pair_with_corrupt_frames(
            30,
            frozenset({2, 7, 11, 16, 22}),
        )
        diagnostics = CalibrationSolveDiagnostics()
        calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
            solve_diagnostics=diagnostics,
        )
        self.assertIn("fallback_assignment", diagnostics.solve_stages_seconds)
        self.assertGreater(diagnostics.solve_stages_seconds["fallback_assignment"], 0.0)


if __name__ == "__main__":
    unittest.main()
