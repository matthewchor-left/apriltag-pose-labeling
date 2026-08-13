"""Behavior tests for live pair-readiness diagnostics during capture."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    FrameObservation,
    LivePairReadinessDiagnostics,
    PairReadinessEdge,
    _PairConsensus,
    _classify_pair_readiness,
    compute_live_pair_readiness,
)
from tests.test_marker_layout_calibration import (
    _chain_marker_poses,
    _default_camera,
    _pair_poses,
    _reference_gauge_pose,
    synthesize_observations,
)


class LivePairReadinessStrongGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)
        self.expected_ids = [0, 1, 2, 3]
        self.reference_marker_id = 0

    def _readiness(self, observations: list[FrameObservation], *, settings=None):
        return compute_live_pair_readiness(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=self.expected_ids,
            reference_marker_id=self.reference_marker_id,
            marker_size_m=self.marker_size_m,
            settings=settings or self.settings,
        )

    def test_strong_connected_graph_reports_passing_pairs_and_full_connectivity(self) -> None:
        marker_poses = _chain_marker_poses(self.marker_size_m)
        chain_visibility = [(0, 1), (1, 2), (2, 3)]
        observations = synthesize_observations(
            marker_poses,
            frame_count=25 * len(chain_visibility),
            marker_size_m=self.marker_size_m,
            visible_markers=lambda frame_index: chain_visibility[frame_index % len(chain_visibility)],
        )

        diagnostics = self._readiness(observations)

        self.assertIsNone(diagnostics.failure_reason)
        self.assertEqual(diagnostics.sample_count, len(observations))
        self.assertEqual(diagnostics.connected_marker_ids, frozenset(self.expected_ids))
        self.assertEqual(diagnostics.missing_marker_ids, frozenset())
        passing_pairs = {
            (edge.marker_a, edge.marker_b)
            for edge in diagnostics.pairs
            if edge.status == "pass"
        }
        self.assertEqual(passing_pairs, {(0, 1), (1, 2), (2, 3)})
        for edge in diagnostics.pairs:
            if edge.status != "pass":
                continue
            self.assertGreaterEqual(edge.raw_covisible_frames, self.settings.min_inliers_per_edge)
            self.assertIsNone(edge.translation_rms_m)
            self.assertIsNone(edge.rotation_rms_deg)

    def test_weak_pair_reports_insufficient_robust_support(self) -> None:
        observations = synthesize_observations(
            _pair_poses(self.marker_size_m),
            frame_count=10,
            marker_size_m=self.marker_size_m,
        )
        diagnostics = compute_live_pair_readiness(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
        )

        edge = diagnostics.pairs[0]
        self.assertEqual((edge.marker_a, edge.marker_b), (0, 1))
        self.assertEqual(edge.status, "weak")
        self.assertEqual(edge.raw_covisible_frames, 10)
        self.assertEqual(edge.robust_inlier_count, 10)
        self.assertIsNone(edge.translation_rms_m)
        self.assertIsNone(edge.rotation_rms_deg)

    def test_failing_rms_reports_fail_status(self) -> None:
        rotation_gate = self.settings.pair_rotation_rms_gate_deg
        base_rotation, _ = cv2.Rodrigues(np.zeros(3, dtype=np.float64))
        seed_translation = np.array([0.12, 0.0, -0.05], dtype=np.float64)
        inlier_hypotheses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for frame_index in range(20):
            sign = 1 if frame_index % 2 == 0 else -1
            rotation, _ = cv2.Rodrigues(
                np.array([0.0, 0.0, sign * np.deg2rad(rotation_gate * 1.001)], dtype=np.float64)
            )
            inlier_hypotheses[frame_index] = (rotation, seed_translation)
        edge = _PairConsensus(
            marker_a=0,
            marker_b=1,
            rotation_ba=base_rotation,
            translation_ba=seed_translation,
            inlier_frames=tuple(range(20)),
            inlier_hypotheses=inlier_hypotheses,
        )

        status, robust_count, translation_rms_m, rotation_rms_deg = _classify_pair_readiness(
            edge,
            self.settings,
            self.marker_size_m,
        )

        self.assertEqual(status, "fail")
        self.assertEqual(robust_count, 20)
        self.assertIsNotNone(translation_rms_m)
        self.assertIsNotNone(rotation_rms_deg)
        self.assertGreater(rotation_rms_deg, rotation_gate)

    def test_redundant_passing_edges_still_yield_full_connectivity(self) -> None:
        ref_rotation, ref_translation = _reference_gauge_pose(self.marker_size_m)
        marker_poses = {
            0: (ref_rotation, ref_translation),
            1: (ref_rotation, ref_translation + np.array([0.12, 0.0, 0.0], dtype=np.float64)),
            2: (ref_rotation, ref_translation + np.array([0.0, 0.12, 0.0], dtype=np.float64)),
            3: (
                ref_rotation,
                ref_translation + np.array([0.12, 0.12, 0.0], dtype=np.float64),
            ),
        }
        observations = synthesize_observations(
            marker_poses,
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )

        diagnostics = self._readiness(observations)

        passing = [edge for edge in diagnostics.pairs if edge.status == "pass"]
        self.assertGreater(len(passing), 3)
        self.assertEqual(diagnostics.connected_marker_ids, frozenset(self.expected_ids))
        self.assertEqual(diagnostics.missing_marker_ids, frozenset())

    def test_disconnected_graph_reports_missing_markers(self) -> None:
        marker_poses = _chain_marker_poses(self.marker_size_m)
        observations = synthesize_observations(
            marker_poses,
            frame_count=25,
            marker_size_m=self.marker_size_m,
            visible_markers=lambda _: (0, 1),
        )

        diagnostics = self._readiness(observations)

        self.assertEqual(diagnostics.connected_marker_ids, frozenset({0, 1}))
        self.assertEqual(diagnostics.missing_marker_ids, frozenset({2, 3}))
        edge = next(edge for edge in diagnostics.pairs if (edge.marker_a, edge.marker_b) == (0, 1))
        self.assertEqual(edge.status, "pass")


class LivePairReadinessHudFormattingTests(unittest.TestCase):
    def test_hud_lines_summarize_graph_and_pairs_deterministically(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import build_pair_readiness_hud_lines
        from object_apriltag.cli.live_pair_readiness_worker import LivePairReadinessView

        diagnostics_pairs = tuple(
            PairReadinessEdge(
                marker_a=marker_a,
                marker_b=marker_b,
                raw_covisible_frames=25,
                robust_inlier_count=25,
                translation_rms_m=0.001,
                rotation_rms_deg=0.5,
                status="pass",
            )
            for marker_a, marker_b in ((0, 1), (0, 2), (1, 2), (2, 3), (0, 3), (1, 3))
        )
        readiness_view = LivePairReadinessView(
            diagnostics=LivePairReadinessDiagnostics(
                pairs=diagnostics_pairs,
                connected_marker_ids=frozenset({0, 1, 2, 3}),
                missing_marker_ids=frozenset(),
                sample_count=25,
            ),
            represented_sample_count=25,
            is_computing=False,
        )
        lines = build_pair_readiness_hud_lines(
            expected_ids=[0, 1, 2, 3],
            visible_ids=[0, 1, 2],
            current_sample_count=25,
            readiness_view=readiness_view,
            reference_marker_id=0,
            max_pair_lines=3,
        )

        self.assertIn("readiness@25 samples", lines[3])
        self.assertIn("markers connected [0, 1, 2, 3] missing []", lines[4])
        self.assertIn("graph: 4/4 connected from ref 0", lines[5])
        self.assertIn("pairs: 6 pass", lines[6])
        self.assertEqual(len([line for line in lines if line.startswith("(")]), 3)
        self.assertTrue(any(line.endswith("...") for line in lines))

    def test_readiness_snapshot_line_marks_stale_compute_state(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import format_readiness_snapshot_line

        self.assertEqual(
            format_readiness_snapshot_line(
                current_sample_count=5,
                represented_sample_count=3,
                is_computing=True,
            ),
            "readiness@3/5 samples (computing...)",
        )
