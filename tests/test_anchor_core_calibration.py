"""Tests for anchor-core marker layout calibration."""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from object_apriltag.layout import CORNER_NAMES
from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    FrameObservation,
    _MarkerCandidate,
    _evaluate_complete_assignment,
    calibrate_marker_layout,
    parse_anchor_marker_ids,
    parse_marker_id_spec,
    resolve_frame_ippe_assignment,
)
from tests.test_marker_layout_calibration import (
    _chain_marker_poses,
    _default_camera,
    _ground_truth_footprints,
    _uniform_marker_sizes,
    _pair_poses,
    synthesize_observations,
)


def _dummy_candidate(marker_id: int, branch: int) -> _MarkerCandidate:
    rotation = np.eye(3, dtype=np.float64)
    if branch % 2 == 1:
        flip, _ = __import__("cv2").Rodrigues(np.array([np.pi, 0.0, 0.0], dtype=np.float64))
        rotation = flip
    translation = np.array([0.0, 0.0, 0.5 + 0.01 * marker_id], dtype=np.float64)
    return _MarkerCandidate(
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=translation.reshape(3, 1),
        rotation=rotation,
        reprojection_rms_px=0.5,
    )


class ParseMarkerIdSpecTests(unittest.TestCase):
    def test_expands_ranges(self) -> None:
        marker_ids, failure = parse_marker_id_spec(["0", "1", "2", "3-10", "11", "12"])
        self.assertIsNone(failure)
        assert marker_ids is not None
        self.assertEqual(marker_ids, list(range(13)))

    def test_rejects_duplicate_ids(self) -> None:
        _, failure = parse_marker_id_spec(["0", "3-5", "4"])
        self.assertIn("duplicates", failure or "")

    def test_rejects_descending_range(self) -> None:
        _, failure = parse_marker_id_spec(["10-3"])
        self.assertIn("ascending", failure or "")


class ParseAnchorMarkerIdsTests(unittest.TestCase):
    def test_none_is_valid(self) -> None:
        anchors, failure = parse_anchor_marker_ids(None, [0, 1, 2], 0)
        self.assertIsNone(anchors)
        self.assertIsNone(failure)

    def test_requires_reference_marker(self) -> None:
        _, failure = parse_anchor_marker_ids([1, 2], [0, 1, 2], 0)
        self.assertIn("reference_marker_id", failure or "")

    def test_requires_subset_of_expected(self) -> None:
        _, failure = parse_anchor_marker_ids([0, 1, 9], [0, 1, 2], 0)
        self.assertIn("subset", failure or "")


class AnchorAssignmentComplexityTests(unittest.TestCase):
    def test_anchor_search_evaluates_two_to_k_not_two_to_m(self) -> None:
        candidates = {
            marker_id: [_dummy_candidate(marker_id, 0), _dummy_candidate(marker_id, 1)]
            for marker_id in range(20)
        }
        pair_consensus = {}
        settings = CalibrationSettings()
        evaluation_count = 0
        original = _evaluate_complete_assignment

        def counting_evaluate(*args, **kwargs):
            nonlocal evaluation_count
            evaluation_count += 1
            return original(*args, **kwargs)

        with mock.patch(
            "object_apriltag.marker_layout_calibration._evaluate_complete_assignment",
            side_effect=counting_evaluate,
        ):
            resolve_frame_ippe_assignment(
                candidates,
                pair_consensus,
                settings,
                _uniform_marker_sizes(range(6), 0.07),
                search_marker_ids=frozenset(range(6)),
            )
        self.assertEqual(evaluation_count, 64)


class AnchorCoreCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _calibrate(
        self,
        observations: list[FrameObservation],
        expected_ids: list[int],
        *,
        anchor_marker_ids: tuple[int, ...] | None,
        anchor_stop_after_expansion: bool = False,
        best_effort: bool = False,
        partial_output: bool = False,
    ):
        return calibrate_marker_layout(
            observations,
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=expected_ids,
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            anchor_marker_ids=anchor_marker_ids,
            anchor_stop_after_expansion=anchor_stop_after_expansion,
            best_effort=best_effort,
            partial_output=partial_output,
        )

    def test_anchors_equal_expected_matches_legacy_success(self) -> None:
        marker_poses = _pair_poses(self.marker_size_m)
        observations = synthesize_observations(
            marker_poses,
            frame_count=25,
            marker_size_m=self.marker_size_m,
        )
        legacy = self._calibrate(observations, [0, 1], anchor_marker_ids=None)
        anchor_all = self._calibrate(observations, [0, 1], anchor_marker_ids=(0, 1))
        self.assertIsNone(legacy.failure_reason)
        self.assertIsNone(anchor_all.failure_reason)
        assert legacy.layout is not None and anchor_all.layout is not None
        for marker_id in (0, 1):
            for corner_name in CORNER_NAMES:
                expected = getattr(legacy.layout.footprints[marker_id], corner_name)
                actual = getattr(anchor_all.layout.footprints[marker_id], corner_name)
                np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_anchor_core_recovers_chain_with_interleaved_covisibility(self) -> None:
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
        result = self._calibrate(observations, [0, 1, 2, 3], anchor_marker_ids=(0, 1))

        self.assertIsNone(result.failure_reason)
        assert result.layout is not None
        assert result.quality is not None
        self.assertIsNotNone(result.quality.anchor_core)
        assert result.quality.anchor_core is not None
        self.assertEqual(result.quality.anchor_core.mode, "anchor_core")
        self.assertEqual(result.quality.anchor_core.configured_anchor_ids, (0, 1))
        self.assertEqual(result.quality.anchor_core.bootstrap.status, "ok")
        if result.quality.assignment_rejection_records is not None:
            self.assertEqual(
                result.quality.rejected_frame_count,
                len(result.quality.assignment_rejection_records),
            )
        for marker_id in (2, 3):
            for corner_name in CORNER_NAMES:
                expected = getattr(ground_truth[marker_id], corner_name)
                actual = getattr(result.layout.footprints[marker_id], corner_name)
                self.assertLess(
                    float(np.linalg.norm(actual - expected)),
                    0.01,
                    msg=f"marker {marker_id} {corner_name}",
                )

    def test_stop_after_expansion_returns_layout_without_bundle_adjustment(self) -> None:
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
        result = self._calibrate(
            observations,
            [0, 1, 2, 3],
            anchor_marker_ids=(0, 1),
            anchor_stop_after_expansion=True,
        )

        self.assertIsNone(result.failure_reason)
        assert result.layout is not None
        assert result.quality is not None
        assert result.quality.anchor_core is not None
        self.assertTrue(result.quality.anchor_core.stopped_after_expansion)
        self.assertEqual(result.quality.observation_count, 0)
        for marker_id in (2, 3):
            for corner_name in CORNER_NAMES:
                expected = getattr(ground_truth[marker_id], corner_name)
                actual = getattr(result.layout.footprints[marker_id], corner_name)
                self.assertLess(
                    float(np.linalg.norm(actual - expected)),
                    0.01,
                    msg=f"marker {marker_id} {corner_name}",
                )

    def test_expansion_failure_reports_anchor_core_without_partial_layout(self) -> None:
        marker_poses = _chain_marker_poses(self.marker_size_m)
        visibility = [(0, 1)] * 25 + [(1, 2)] * 25 + [(2, 3)] * 19
        observations = synthesize_observations(
            marker_poses,
            frame_count=len(visibility),
            marker_size_m=self.marker_size_m,
            visible_markers=lambda frame_index: visibility[frame_index],
        )
        result = self._calibrate(observations, [0, 1, 2, 3], anchor_marker_ids=(0, 1))

        self.assertIsNone(result.layout)
        self.assertIsNotNone(result.failure_reason)
        assert result.quality is not None
        assert result.quality.anchor_core is not None
        self.assertIn(3, result.quality.anchor_core.unresolved_ids)

    def test_twenty_marker_anchor_core_finishes_within_runtime_ceiling(self) -> None:
        marker_poses = _chain_marker_poses(self.marker_size_m)
        for marker_id in range(4, 20):
            base = marker_poses[3]
            marker_poses[marker_id] = (
                base[0],
                base[1] + np.array([0.03 * (marker_id - 3), 0.0, 0.0], dtype=np.float64),
            )
        expected_ids = list(range(20))
        observations = synthesize_observations(
            marker_poses,
            frame_count=30,
            marker_size_m=self.marker_size_m,
            visible_markers=lambda frame_index: tuple(
                expected_ids[max(0, frame_index % 17 - 1) : min(20, frame_index % 17 + 7)]
            ),
        )
        start = time.monotonic()
        result = self._calibrate(
            observations,
            expected_ids,
            anchor_marker_ids=(0, 1, 2, 3, 4, 5),
        )
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 30.0, msg=f"anchor-core solve took {elapsed:.1f}s")
        if result.failure_reason is not None:
            self.assertIn("anchor", result.failure_reason.lower())


if __name__ == "__main__":
    unittest.main()
