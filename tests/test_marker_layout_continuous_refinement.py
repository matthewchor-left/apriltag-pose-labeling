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
    run_bundle_adjustment,
)
from object_apriltag.marker_layout_calibration.solve_primitives import CornerObservation, PairConsensus
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


class RunBundleAdjustmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.marker_size_m = 0.07
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

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

    def _minimal_context(self, *, best_effort: bool = False) -> LayoutRefinementContext:
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
            best_effort=best_effort,
            restored_pair_edges=None,
            input_frame_count=20,
            rejected_frame_count=0,
            accepted_frame_count=20,
            assignment_rejection_summary=None,
            assignment_rejection_records=None,
            fallback_assignment_records=None,
            dropped_edges=[],
            anchor_core_diagnostics=None,
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
        self.assertIsNone(outcome.early_result.layout)
        self.assertIn("Bundle adjustment failed", outcome.early_result.failure_reason or "")

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
        context = self._minimal_context(best_effort=True)
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
