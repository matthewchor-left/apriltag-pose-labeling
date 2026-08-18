"""Focused tests for continuous layout refinement seam."""

from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration import (
    CalibrationResult,
    CalibrationSettings,
    uniform_marker_sizes,
)
from object_apriltag.marker_layout_calibration.discrete_graph import maybe_restore_weak_connectivity
from object_apriltag.marker_layout_calibration.continuous_refinement import (
    ContinuousLayoutRefinement,
    LayoutRefinementContext,
    LayoutSolveState,
    _evaluate_bundle_adjustment_residuals,
    _pack_parameters,
    _unpack_parameters,
    run_bundle_adjustment,
)
from object_apriltag.marker_layout_calibration.solve_primitives import (
    CalibrationSolveDiagnostics,
    CornerObservation,
    PairConsensus,
    build_bundle_adjustment_observation_layout,
    project_corner,
)
from object_apriltag.pose import marker_corner_object_points
from tests.test_marker_layout_calibration import (
    _default_camera,
    _pair_poses,
    _pruning_refinement_failure_without_weak_recovery,
    synthesize_observations,
)


def _two_marker_poses(marker_size_m: float = 0.07):
    return _pair_poses(marker_size_m)


def _layout_frame_poses(frame_count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_index in range(frame_count):
        rotation, _ = cv2.Rodrigues(
            np.array([0.1, -0.15 + 0.01 * frame_index, 0.05], dtype=np.float64)
        )
        translation = np.array(
            [0.02, -0.01, 0.6 + 0.002 * frame_index],
            dtype=np.float64,
        )
        poses.append((rotation, translation))
    return poses


class LayoutSolveStateTests(unittest.TestCase):
    def test_checkpoint_snapshot_isolated_from_live_consensus_mutation(self) -> None:
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
        edge = PairConsensus(
            marker_a=0,
            marker_b=1,
            rotation_ba=rotation,
            translation_ba=translation,
            inlier_frames=(0,),
            inlier_hypotheses={0: (rotation.copy(), translation.copy())},
        )
        pair_consensus = {(0, 1): edge}
        state = LayoutSolveState(
            corner_observations=[],
            marker_poses={0: (rotation, translation)},
            frame_poses=[None],
            inlier_mask=np.zeros(0, dtype=bool),
            pair_consensus=pair_consensus,
        )
        checkpoint = state.checkpoint_snapshot("graph_initialization")
        edge.inlier_hypotheses[0] = (rotation * 2.0, translation + 1.0)
        snap = checkpoint.pair_consensus[(0, 1)]
        self.assertFalse(np.allclose(snap.inlier_hypotheses[0][1], translation + 1.0))


def _minimal_bundle_adjustment_inputs(
    *,
    frame_count: int = 12,
    marker_size_m: float = 0.07,
    seed: int = 2,
) -> tuple[
    list[CornerObservation],
    np.ndarray,
    dict[int, tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray] | None],
    dict[int, np.ndarray],
]:
    poses = _two_marker_poses(marker_size_m)
    observations = synthesize_observations(
        poses,
        frame_count=frame_count,
        marker_size_m=marker_size_m,
        seed=seed,
    )
    corner_observations = [
        CornerObservation(
            frame_index=frame_index,
            marker_id=marker_id,
            corner_index=corner_index,
            image_point=obs.markers[marker_id][corner_index],
        )
        for frame_index, obs in enumerate(observations)
        for marker_id in obs.markers
        for corner_index in range(4)
    ]
    inlier_mask = np.ones(len(corner_observations), dtype=bool)
    marker_poses = {
        marker_id: (rotation.copy(), translation.copy())
        for marker_id, (rotation, translation) in poses.items()
    }
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(observations)
    object_points = {
        marker_id: marker_corner_object_points(marker_size_m) for marker_id in poses
    }
    return corner_observations, inlier_mask, marker_poses, frame_poses, object_points


def _three_marker_poses(marker_size_m: float = 0.07) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses = _two_marker_poses(marker_size_m)
    rotation, translation = poses[1]
    poses[2] = (rotation, translation + np.array([0.08, 0.06, 0.02], dtype=np.float64))
    return poses


def _scalar_bundle_adjustment_residuals(
    params: np.ndarray,
    *,
    corner_observations: list[CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    non_reference_ids: list[int],
    active_frames: list[int],
    reference_marker_id: int,
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Test-only scalar residual evaluator matching pre-vectorization BA semantics."""
    if not np.all(np.isfinite(params)):
        inlier_count = int(np.count_nonzero(inlier_mask))
        return np.full(2 * inlier_count, 1e3, dtype=np.float64)
    marker_state, frame_pose_list = _unpack_parameters(
        params,
        marker_poses,
        frame_poses,
        non_reference_ids,
        active_frames,
        reference_marker_id,
    )
    values: list[float] = []
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        frame_pose = frame_pose_list[observation.frame_index]
        marker_pose = marker_state.get(observation.marker_id)
        if frame_pose is None or marker_pose is None:
            values.extend([1000.0, 1000.0])
            continue
        projected = project_corner(
            observation.corner_index,
            observation.marker_id,
            marker_pose,
            frame_pose,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
        )
        if not np.all(np.isfinite(projected)):
            values.extend([1000.0, 1000.0])
            continue
        delta = projected - observation.image_point
        values.extend(delta.tolist())
    return np.asarray(values, dtype=np.float64)


def _active_frames_from_mask(
    corner_observations: list[CornerObservation],
    inlier_mask: np.ndarray,
) -> list[int]:
    return sorted(
        {
            observation.frame_index
            for observation, keep in zip(corner_observations, inlier_mask, strict=True)
            if keep
        }
    )


class VectorizedResidualEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_zero = _default_camera()
        self.dist_nonzero = np.array(
            [[0.12, -0.04, 0.01, 0.0, 0.0]],
            dtype=np.float64,
        ).T
        self.marker_size_m = 0.07

    def _compare_scalar_and_vectorized(
        self,
        *,
        poses: dict[int, tuple[np.ndarray, np.ndarray]],
        frame_count: int,
        dist_coeffs: np.ndarray,
        seed: int,
        params_perturbation: np.ndarray | None = None,
        frame_poses: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
        inlier_mask: np.ndarray | None = None,
        reference_marker_id: int = 0,
    ) -> None:
        observations = synthesize_observations(
            poses,
            frame_count=frame_count,
            marker_size_m=self.marker_size_m,
            seed=seed,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        if inlier_mask is None:
            inlier_mask = np.ones(len(corner_observations), dtype=bool)
        marker_poses = {
            marker_id: (rotation.copy(), translation.copy())
            for marker_id, (rotation, translation) in poses.items()
        }
        if frame_poses is None:
            frame_poses = _layout_frame_poses(len(observations))
        non_reference_ids = [marker_id for marker_id in poses if marker_id != reference_marker_id]
        active_frames = _active_frames_from_mask(corner_observations, inlier_mask)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m) for marker_id in poses
        }
        params = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
        if params_perturbation is not None:
            params = params + params_perturbation
        layout = build_bundle_adjustment_observation_layout(
            corner_observations,
            inlier_mask,
            object_points,
        )
        marker_state, frame_pose_list = _unpack_parameters(
            params,
            marker_poses,
            frame_poses,
            non_reference_ids,
            active_frames,
            reference_marker_id,
        )
        scalar = _scalar_bundle_adjustment_residuals(
            params,
            corner_observations=corner_observations,
            inlier_mask=inlier_mask,
            marker_poses=marker_poses,
            frame_poses=frame_poses,
            non_reference_ids=non_reference_ids,
            active_frames=active_frames,
            reference_marker_id=reference_marker_id,
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        vectorized, _, _, _ = _evaluate_bundle_adjustment_residuals(
            layout,
            marker_state,
            frame_pose_list,
            self.camera_matrix,
            dist_coeffs,
        )
        penalty_mask = scalar == 1000.0
        np.testing.assert_array_equal(penalty_mask, vectorized == 1000.0)
        finite_mask = ~penalty_mask
        np.testing.assert_allclose(
            scalar[finite_mask],
            vectorized[finite_mask],
            rtol=0.0,
            atol=1e-9,
        )

    def test_scalar_matches_vectorized_without_distortion(self) -> None:
        self._compare_scalar_and_vectorized(
            poses=_two_marker_poses(self.marker_size_m),
            frame_count=10,
            dist_coeffs=self.dist_zero,
            seed=3,
        )

    def test_scalar_matches_vectorized_with_distortion(self) -> None:
        self._compare_scalar_and_vectorized(
            poses=_two_marker_poses(self.marker_size_m),
            frame_count=10,
            dist_coeffs=self.dist_nonzero,
            seed=4,
        )

    def test_scalar_matches_vectorized_multiple_markers_and_frames(self) -> None:
        self._compare_scalar_and_vectorized(
            poses=_three_marker_poses(self.marker_size_m),
            frame_count=8,
            dist_coeffs=self.dist_nonzero,
            seed=5,
        )

    def test_scalar_matches_vectorized_with_reference_marker_fixed(self) -> None:
        self._compare_scalar_and_vectorized(
            poses=_two_marker_poses(self.marker_size_m),
            frame_count=12,
            dist_coeffs=self.dist_zero,
            seed=6,
            reference_marker_id=0,
        )

    def test_nonfinite_params_return_exact_penalty_vector(self) -> None:
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(frame_count=6, marker_size_m=self.marker_size_m, seed=7)
        captured: dict[str, np.ndarray] = {}

        def _capture_residuals(residual_fn, x0, **kwargs):
            nan_params = x0.copy()
            nan_params[0] = np.nan
            captured["nonfinite"] = residual_fn(nan_params)
            return mock.Mock(
                success=True,
                status=0,
                x=x0,
                nfev=1,
                njev=None,
                cost=0.0,
            )

        with mock.patch(
            "object_apriltag.marker_layout_calibration.continuous_refinement.least_squares",
            side_effect=_capture_residuals,
        ):
            run_bundle_adjustment(
                corner_observations,
                inlier_mask,
                marker_poses,
                frame_poses,
                reference_marker_id=0,
                non_reference_ids=[1],
                object_points_by_marker=object_points,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_zero,
                settings=CalibrationSettings(min_inliers_per_edge=20),
            )
        self.assertTrue(np.all(captured["nonfinite"] == 1000.0))

    def test_missing_frame_pose_uses_penalty(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            poses,
            frame_count=6,
            marker_size_m=self.marker_size_m,
            seed=8,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        inlier_mask = np.ones(len(corner_observations), dtype=bool)
        marker_poses = {
            marker_id: (rotation.copy(), translation.copy())
            for marker_id, (rotation, translation) in poses.items()
        }
        frame_poses = _layout_frame_poses(len(observations))
        non_reference_ids = [1]
        active_frames = _active_frames_from_mask(corner_observations, inlier_mask)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m) for marker_id in poses
        }
        params = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
        marker_state, frame_pose_list = _unpack_parameters(
            params,
            marker_poses,
            frame_poses,
            non_reference_ids,
            active_frames,
            0,
        )
        frame_pose_list[0] = None
        layout = build_bundle_adjustment_observation_layout(
            corner_observations,
            inlier_mask,
            object_points,
        )
        scalar_values: list[float] = []
        for observation, keep in zip(corner_observations, inlier_mask, strict=True):
            if not keep:
                continue
            frame_pose = frame_pose_list[observation.frame_index]
            marker_pose = marker_state.get(observation.marker_id)
            if frame_pose is None or marker_pose is None:
                scalar_values.extend([1000.0, 1000.0])
                continue
            projected = project_corner(
                observation.corner_index,
                observation.marker_id,
                marker_pose,
                frame_pose,
                object_points,
                self.camera_matrix,
                self.dist_zero,
            )
            if not np.all(np.isfinite(projected)):
                scalar_values.extend([1000.0, 1000.0])
                continue
            scalar_values.extend((projected - observation.image_point).tolist())
        scalar = np.asarray(scalar_values, dtype=np.float64)
        vectorized, projection_count, projectpoints_count, batched_corners = (
            _evaluate_bundle_adjustment_residuals(
                layout,
                marker_state,
                frame_pose_list,
                self.camera_matrix,
                self.dist_zero,
            )
        )
        np.testing.assert_array_equal(scalar, vectorized)
        frame_zero_corners = sum(
            1 for observation, keep in zip(corner_observations, inlier_mask, strict=True)
            if keep and observation.frame_index == 0
        )
        self.assertEqual(projection_count, layout.observation_count - frame_zero_corners)
        self.assertEqual(projectpoints_count, 1)
        self.assertEqual(batched_corners, layout.observation_count - frame_zero_corners)

    def test_nonfinite_projection_uses_penalty(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        perturbation = np.zeros(6 + 6 * 6, dtype=np.float64)
        perturbation[3:6] = np.array([1e6, 1e6, -1e6], dtype=np.float64)
        self._compare_scalar_and_vectorized(
            poses=poses,
            frame_count=6,
            dist_coeffs=self.dist_zero,
            seed=9,
            params_perturbation=perturbation,
        )

    def test_partial_inlier_mask_preserves_residual_ordering(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            poses,
            frame_count=8,
            marker_size_m=self.marker_size_m,
            seed=10,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in sorted(obs.markers)
            for corner_index in range(4)
        ]
        inlier_mask = np.ones(len(corner_observations), dtype=bool)
        inlier_mask[3:7] = False
        inlier_mask[20:24] = False
        marker_poses = {
            marker_id: (rotation.copy(), translation.copy())
            for marker_id, (rotation, translation) in poses.items()
        }
        frame_poses = _layout_frame_poses(len(observations))
        non_reference_ids = [1]
        active_frames = _active_frames_from_mask(corner_observations, inlier_mask)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m) for marker_id in poses
        }
        params = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
        layout = build_bundle_adjustment_observation_layout(
            corner_observations,
            inlier_mask,
            object_points,
        )
        expected_active = [
            observation
            for observation, keep in zip(corner_observations, inlier_mask, strict=True)
            if keep
        ]
        np.testing.assert_array_equal(layout.marker_ids, [obs.marker_id for obs in expected_active])
        np.testing.assert_array_equal(layout.frame_indices, [obs.frame_index for obs in expected_active])
        scalar = _scalar_bundle_adjustment_residuals(
            params,
            corner_observations=corner_observations,
            inlier_mask=inlier_mask,
            marker_poses=marker_poses,
            frame_poses=frame_poses,
            non_reference_ids=non_reference_ids,
            active_frames=active_frames,
            reference_marker_id=0,
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_zero,
        )
        marker_state, frame_pose_list = _unpack_parameters(
            params,
            marker_poses,
            frame_poses,
            non_reference_ids,
            active_frames,
            0,
        )
        vectorized, _, _, _ = _evaluate_bundle_adjustment_residuals(
            layout,
            marker_state,
            frame_pose_list,
            self.camera_matrix,
            self.dist_zero,
        )
        self.assertEqual(scalar.shape[0], 2 * layout.observation_count)
        np.testing.assert_allclose(scalar, vectorized, rtol=0.0, atol=1e-9)

    def test_missing_marker_pose_uses_penalty(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            poses,
            frame_count=6,
            marker_size_m=self.marker_size_m,
            seed=11,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        inlier_mask = np.ones(len(corner_observations), dtype=bool)
        marker_poses = {
            marker_id: (rotation.copy(), translation.copy())
            for marker_id, (rotation, translation) in poses.items()
        }
        frame_poses = _layout_frame_poses(len(observations))
        non_reference_ids = [1]
        active_frames = _active_frames_from_mask(corner_observations, inlier_mask)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m) for marker_id in poses
        }
        params = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
        marker_state, frame_pose_list = _unpack_parameters(
            params,
            marker_poses,
            frame_poses,
            non_reference_ids,
            active_frames,
            0,
        )
        marker_state.pop(1)
        layout = build_bundle_adjustment_observation_layout(
            corner_observations,
            inlier_mask,
            object_points,
        )
        scalar_values: list[float] = []
        for observation, keep in zip(corner_observations, inlier_mask, strict=True):
            if not keep:
                continue
            frame_pose = frame_pose_list[observation.frame_index]
            marker_pose = marker_state.get(observation.marker_id)
            if frame_pose is None or marker_pose is None:
                scalar_values.extend([1000.0, 1000.0])
                continue
            projected = project_corner(
                observation.corner_index,
                observation.marker_id,
                marker_pose,
                frame_pose,
                object_points,
                self.camera_matrix,
                self.dist_zero,
            )
            if not np.all(np.isfinite(projected)):
                scalar_values.extend([1000.0, 1000.0])
                continue
            scalar_values.extend((projected - observation.image_point).tolist())
        scalar = np.asarray(scalar_values, dtype=np.float64)
        vectorized, projection_count, projectpoints_count, batched_corners = (
            _evaluate_bundle_adjustment_residuals(
                layout,
                marker_state,
                frame_pose_list,
                self.camera_matrix,
                self.dist_zero,
            )
        )
        np.testing.assert_array_equal(scalar, vectorized)
        marker_one_corners = sum(
            1 for observation, keep in zip(corner_observations, inlier_mask, strict=True)
            if keep and observation.marker_id == 1
        )
        self.assertEqual(projection_count, layout.observation_count - marker_one_corners)
        self.assertEqual(projectpoints_count, 1)
        self.assertEqual(batched_corners, layout.observation_count - marker_one_corners)

    def test_all_invalid_poses_skip_opencv(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            poses,
            frame_count=4,
            marker_size_m=self.marker_size_m,
            seed=12,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        inlier_mask = np.ones(len(corner_observations), dtype=bool)
        marker_poses = {
            marker_id: (rotation.copy(), translation.copy())
            for marker_id, (rotation, translation) in poses.items()
        }
        frame_poses = _layout_frame_poses(len(observations))
        non_reference_ids = [1]
        active_frames = _active_frames_from_mask(corner_observations, inlier_mask)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m) for marker_id in poses
        }
        params = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
        marker_state, frame_pose_list = _unpack_parameters(
            params,
            marker_poses,
            frame_poses,
            non_reference_ids,
            active_frames,
            0,
        )
        frame_pose_list = [None] * len(frame_pose_list)
        layout = build_bundle_adjustment_observation_layout(
            corner_observations,
            inlier_mask,
            object_points,
        )
        with mock.patch(
            "object_apriltag.marker_layout_calibration.continuous_refinement.cv2.projectPoints",
            side_effect=AssertionError("projectPoints should not run"),
        ):
            vectorized, projection_count, projectpoints_count, batched_corners = (
                _evaluate_bundle_adjustment_residuals(
                    layout,
                    marker_state,
                    frame_pose_list,
                    self.camera_matrix,
                    self.dist_zero,
                )
            )
        self.assertTrue(np.all(vectorized == 1000.0))
        self.assertEqual(projection_count, 0)
        self.assertEqual(projectpoints_count, 0)
        self.assertEqual(batched_corners, 0)


class BundleAdjustmentEndToEndEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.marker_size_m = 0.07
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def test_vectorized_ba_matches_scalar_residual_trajectory(self) -> None:
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(frame_count=12, marker_size_m=self.marker_size_m)
        non_reference_ids = [1]
        active_frames = _active_frames_from_mask(corner_observations, inlier_mask)
        x0 = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
        layout = build_bundle_adjustment_observation_layout(
            corner_observations,
            inlier_mask,
            object_points,
        )

        def vectorized_residuals(params: np.ndarray) -> np.ndarray:
            marker_state, frame_pose_list = _unpack_parameters(
                params,
                marker_poses,
                frame_poses,
                non_reference_ids,
                active_frames,
                0,
            )
            values, _, _, _ = _evaluate_bundle_adjustment_residuals(
                layout,
                marker_state,
                frame_pose_list,
                self.camera_matrix,
                self.dist_coeffs,
            )
            return values

        probe_params = [x0]
        probe_params.extend(x0 + delta for delta in np.linspace(-0.02, 0.02, 5))
        for params in probe_params:
            scalar = _scalar_bundle_adjustment_residuals(
                params,
                corner_observations=corner_observations,
                inlier_mask=inlier_mask,
                marker_poses=marker_poses,
                frame_poses=frame_poses,
                non_reference_ids=non_reference_ids,
                active_frames=active_frames,
                reference_marker_id=0,
                object_points_by_marker=object_points,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
            )
            vectorized = vectorized_residuals(params)
            np.testing.assert_allclose(scalar, vectorized, rtol=0.0, atol=1e-9)

    def test_full_ba_regression_against_fixed_expected_poses(self) -> None:
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(frame_count=12, marker_size_m=self.marker_size_m, seed=11)
        updated_poses, updated_frames, _, failure = run_bundle_adjustment(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            reference_marker_id=0,
            non_reference_ids=[1],
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            settings=self.settings,
        )
        self.assertIsNone(failure)
        expected_marker_one_translation = np.array(
            [0.11117564, 0.03739392, -0.00609749],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            updated_poses[1][1],
            expected_marker_one_translation,
            atol=5e-4,
        )
        self.assertEqual(len(updated_frames), len(frame_poses))


class RunBundleAdjustmentTests(unittest.TestCase):
    BA_TIMING_SECTOR_KEYS = (
        "setup",
        "least_squares",
        "post",
        "residual_callback_total",
        "residual_unpack",
        "projection_loop",
        "residual_callback_other",
        "least_squares_overhead",
    )
    BA_COUNT_KEYS = (
        "parameter_count",
        "residual_count",
        "residual_callback_invocations",
        "projection_calls",
        "opencv_projectpoints_invocations",
        "batched_corner_count",
    )

    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.marker_size_m = 0.07
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _run_minimal_bundle_adjustment(
        self,
        *,
        diagnostics: CalibrationSolveDiagnostics | None = None,
        stage_name: str | None = "initial_bundle_adjustment",
    ) -> CalibrationSolveDiagnostics | None:
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(marker_size_m=self.marker_size_m)
        _, _, _, failure = run_bundle_adjustment(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            reference_marker_id=0,
            non_reference_ids=[1],
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            settings=self.settings,
            solve_diagnostics=diagnostics,
            stage_name=stage_name,
        )
        self.assertIsNone(failure)
        return diagnostics

    def test_optimizer_run_includes_bundle_adjustment_sector_timings(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        self._run_minimal_bundle_adjustment(diagnostics=diagnostics)
        self.assertEqual(len(diagnostics.optimizer_runs), 1)
        run = diagnostics.optimizer_runs[0]
        self.assertEqual(set(run["timing_seconds"]), set(self.BA_TIMING_SECTOR_KEYS))
        self.assertEqual(set(run["counts"]), set(self.BA_COUNT_KEYS))

    def test_bundle_adjustment_timing_sectors_are_nonnegative_and_account(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        self._run_minimal_bundle_adjustment(diagnostics=diagnostics)
        timing = diagnostics.optimizer_runs[0]["timing_seconds"]
        for key in self.BA_TIMING_SECTOR_KEYS:
            self.assertGreaterEqual(timing[key], 0.0)
            self.assertTrue(np.isfinite(timing[key]))
        self.assertAlmostEqual(
            timing["residual_callback_other"]
            + timing["residual_unpack"]
            + timing["projection_loop"],
            timing["residual_callback_total"],
            places=9,
        )
        self.assertAlmostEqual(
            timing["residual_callback_total"] + timing["least_squares_overhead"],
            timing["least_squares"],
            places=9,
        )

    def test_bundle_adjustment_counts_match_problem_shape(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(frame_count=12, marker_size_m=self.marker_size_m)
        run_bundle_adjustment(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            reference_marker_id=0,
            non_reference_ids=[1],
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            settings=self.settings,
            solve_diagnostics=diagnostics,
            stage_name="initial_bundle_adjustment",
        )
        run = diagnostics.optimizer_runs[0]
        counts = run["counts"]
        inlier_corner_count = int(np.count_nonzero(inlier_mask))
        active_frame_count = len({obs.frame_index for obs, keep in zip(corner_observations, inlier_mask, strict=True) if keep})
        self.assertEqual(counts["parameter_count"], 6 + 6 * active_frame_count)
        self.assertEqual(counts["residual_count"], 2 * inlier_corner_count)
        self.assertGreaterEqual(counts["residual_callback_invocations"], run["nfev"])
        self.assertEqual(
            counts["projection_calls"],
            inlier_corner_count * counts["residual_callback_invocations"],
        )
        self.assertEqual(
            counts["opencv_projectpoints_invocations"],
            counts["residual_callback_invocations"],
        )
        self.assertEqual(
            counts["batched_corner_count"],
            inlier_corner_count * counts["residual_callback_invocations"],
        )

    def test_batched_projectpoints_invocation_count_with_mock(self) -> None:
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(frame_count=8, marker_size_m=self.marker_size_m)
        diagnostics = CalibrationSolveDiagnostics()
        original_project_points = cv2.projectPoints
        call_count = 0

        def _counting_project_points(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_project_points(*args, **kwargs)

        with mock.patch(
            "object_apriltag.marker_layout_calibration.continuous_refinement.cv2.projectPoints",
            side_effect=_counting_project_points,
        ):
            run_bundle_adjustment(
                corner_observations,
                inlier_mask,
                marker_poses,
                frame_poses,
                reference_marker_id=0,
                non_reference_ids=[1],
                object_points_by_marker=object_points,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                settings=self.settings,
                solve_diagnostics=diagnostics,
                stage_name="initial_bundle_adjustment",
            )
        run = diagnostics.optimizer_runs[0]
        counts = run["counts"]
        self.assertEqual(counts["opencv_projectpoints_invocations"], counts["residual_callback_invocations"])
        self.assertEqual(call_count, counts["opencv_projectpoints_invocations"])
        self.assertGreater(counts["opencv_projectpoints_invocations"], 0)
        self.assertLess(counts["opencv_projectpoints_invocations"], counts["projection_calls"])

    def test_least_squares_value_error_records_partial_optimizer_run(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(marker_size_m=self.marker_size_m)
        with mock.patch(
            "object_apriltag.marker_layout_calibration.continuous_refinement.least_squares",
            side_effect=ValueError("synthetic failure"),
        ):
            _, _, _, failure = run_bundle_adjustment(
                corner_observations,
                inlier_mask,
                marker_poses,
                frame_poses,
                reference_marker_id=0,
                non_reference_ids=[1],
                object_points_by_marker=object_points,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                settings=self.settings,
                solve_diagnostics=diagnostics,
                stage_name="initial_bundle_adjustment",
            )
        self.assertIsNotNone(failure)
        self.assertIn("synthetic failure", failure)
        self.assertEqual(len(diagnostics.optimizer_runs), 1)
        run = diagnostics.optimizer_runs[0]
        self.assertIsNone(run["nfev"])
        self.assertIsNone(run["status"])
        self.assertEqual(run["counts"]["projection_calls"], 0)
        self.assertIn("initial_bundle_adjustment", diagnostics.solve_stages_seconds)
        self.assertAlmostEqual(
            diagnostics.solve_stages_seconds["initial_bundle_adjustment"],
            run["timing_seconds"]["least_squares"],
            places=3,
        )

    def test_non_convergence_records_optimizer_run_without_raising(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        (
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            object_points,
        ) = _minimal_bundle_adjustment_inputs(marker_size_m=self.marker_size_m)

        def _non_converged(*_args, **_kwargs):
            x0 = _args[1]
            return mock.Mock(
                success=False,
                status=-1,
                x=x0,
                nfev=2,
                njev=None,
                cost=1.0,
            )

        with mock.patch(
            "object_apriltag.marker_layout_calibration.continuous_refinement.least_squares",
            side_effect=_non_converged,
        ):
            _, _, _, failure = run_bundle_adjustment(
                corner_observations,
                inlier_mask,
                marker_poses,
                frame_poses,
                reference_marker_id=0,
                non_reference_ids=[1],
                object_points_by_marker=object_points,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                settings=self.settings,
                solve_diagnostics=diagnostics,
                stage_name="initial_bundle_adjustment",
            )
        self.assertIsNotNone(failure)
        self.assertIn("did not converge", failure)
        run = diagnostics.optimizer_runs[0]
        self.assertEqual(run["status"], -1)
        self.assertEqual(run["counts"]["projection_calls"], 0)
        self.assertIn("initial_bundle_adjustment", diagnostics.solve_stages_seconds)

    def test_solve_stage_timing_matches_optimizer_least_squares_sector(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        self._run_minimal_bundle_adjustment(diagnostics=diagnostics)
        run = diagnostics.optimizer_runs[0]
        self.assertAlmostEqual(
            diagnostics.solve_stages_seconds[run["stage"]],
            run["timing_seconds"]["least_squares"],
            places=3,
        )

    def test_without_solve_diagnostics_optimizer_runs_stay_empty(self) -> None:
        diagnostics = CalibrationSolveDiagnostics()
        self._run_minimal_bundle_adjustment(diagnostics=None)
        self.assertEqual(diagnostics.optimizer_runs, [])

    def test_reference_marker_pose_stays_fixed(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        reference_rotation, reference_translation = poses[0]
        observations = synthesize_observations(
            poses,
            frame_count=20,
            marker_size_m=self.marker_size_m,
            seed=2,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        inlier_mask = np.ones(len(corner_observations), dtype=bool)
        marker_poses = {marker_id: (rotation.copy(), translation.copy()) for marker_id, (rotation, translation) in poses.items()}
        frame_poses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(observations)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m)
            for marker_id in poses
        }
        updated_poses, _, _, failure = run_bundle_adjustment(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            reference_marker_id=0,
            non_reference_ids=[1],
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            settings=self.settings,
            stage_name="initial_bundle_adjustment",
        )
        self.assertIsNone(failure)
        ref_rotation, ref_translation = updated_poses[0]
        self.assertTrue(np.allclose(ref_rotation, reference_rotation))
        self.assertTrue(np.allclose(ref_translation, reference_translation))


class ContinuousLayoutRefinementSeamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _minimal_context(self) -> LayoutRefinementContext:
        expected_ids = [0, 1]
        marker_sizes_m = uniform_marker_sizes(expected_ids, self.marker_size_m)
        object_points = {
            marker_id: marker_corner_object_points(self.marker_size_m)
            for marker_id in expected_ids
        }
        return LayoutRefinementContext(
            reference_marker_id=0,
            non_reference_ids=[1],
            expected_ids=expected_ids,
            accepted_frames=frozenset(range(20)),
            object_points_by_marker=object_points,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            settings=self.settings,
            marker_sizes_m=marker_sizes_m,
            marker_size_m=self.marker_size_m,
            restored_pair_edges=None,
            input_frame_count=20,
            rejected_frame_count=0,
            accepted_frame_count=20,
            assignment_rejection_summary=None,
            assignment_rejection_records=None,
            dropped_edges=[],
            restore_weak_connectivity=maybe_restore_weak_connectivity,
        )

    def _synth_state(self) -> LayoutSolveState:
        poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            poses,
            frame_count=20,
            marker_size_m=self.marker_size_m,
            seed=1,
        )
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        rotation_ba, translation_ba = poses[1]
        pair_consensus = {
            (0, 1): PairConsensus(
                marker_a=0,
                marker_b=1,
                rotation_ba=rotation_ba,
                translation_ba=translation_ba,
                inlier_frames=tuple(range(20)),
                inlier_hypotheses={frame_index: (rotation_ba, translation_ba) for frame_index in range(20)},
            )
        }
        return LayoutSolveState(
            corner_observations=corner_observations,
            marker_poses={marker_id: (r.copy(), t.copy()) for marker_id, (r, t) in poses.items()},
            frame_poses=_layout_frame_poses(len(observations)),
            inlier_mask=np.ones(len(corner_observations), dtype=bool),
            pair_consensus=pair_consensus,
        )

    def test_success_returns_outcome_without_early_result(self) -> None:
        outcome = ContinuousLayoutRefinement(self._minimal_context()).run(self._synth_state())
        self.assertIsNone(outcome.early_result)
        self.assertGreater(int(np.count_nonzero(outcome.state.inlier_mask)), 0)

    @mock.patch("object_apriltag.marker_layout_calibration.continuous_refinement.least_squares")
    def test_ba_failure_returns_refused_early_result(self, least_squares_mock: mock.Mock) -> None:
        least_squares_mock.side_effect = ValueError("singular matrix")
        outcome = ContinuousLayoutRefinement(self._minimal_context()).run(self._synth_state())
        self.assertIsNotNone(outcome.early_result)
        assert outcome.early_result is not None
        self.assertIsNotNone(outcome.early_result.layout)
        self.assertEqual(outcome.early_result.outcome, "provisional")

    def test_best_effort_recovers_provisional_checkpoint_on_prune_failure(self) -> None:
        poses = _two_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            poses,
            frame_count=25,
            marker_size_m=self.marker_size_m,
            seed=1,
        )
        from tests.test_marker_layout_calibration import _apply_constant_marker_noise

        _apply_constant_marker_noise(observations, marker_id=1, noise_std_px=3.0, seed=0)
        corner_observations = [
            CornerObservation(
                frame_index=frame_index,
                marker_id=marker_id,
                corner_index=corner_index,
                image_point=obs.markers[marker_id][corner_index],
            )
            for frame_index, obs in enumerate(observations)
            for marker_id in obs.markers
            for corner_index in range(4)
        ]
        rotation_ba, translation_ba = poses[1]
        pair_consensus = {
            (0, 1): PairConsensus(
                marker_a=0,
                marker_b=1,
                rotation_ba=rotation_ba,
                translation_ba=translation_ba,
                inlier_frames=tuple(range(25)),
                inlier_hypotheses={frame_index: (rotation_ba, translation_ba) for frame_index in range(25)},
            )
        }
        state = LayoutSolveState(
            corner_observations=corner_observations,
            marker_poses={marker_id: (r.copy(), t.copy()) for marker_id, (r, t) in poses.items()},
            frame_poses=_layout_frame_poses(len(observations)),
            inlier_mask=np.ones(len(corner_observations), dtype=bool),
            pair_consensus=pair_consensus,
        )
        context = self._minimal_context()
        context.accepted_frames = frozenset(range(25))
        context.input_frame_count = 25
        context.accepted_frame_count = 25
        with _pruning_refinement_failure_without_weak_recovery():
            outcome = ContinuousLayoutRefinement(context).run(state)
        self.assertIsNotNone(outcome.early_result)
        assert isinstance(outcome.early_result, CalibrationResult)
        self.assertEqual(outcome.early_result.outcome, "provisional")
        self.assertEqual(outcome.early_result.selected_checkpoint_stage, "initial_bundle_adjustment")
        self.assertEqual(outcome.early_result.failed_refinement_stage, "post_pruning_refit")


if __name__ == "__main__":
    unittest.main()
