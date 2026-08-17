"""Tests for mixed physical AprilTag sizes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.detector import ObjectDetector
from object_apriltag.layout import (
    build_marker_layout,
    footprint_edge_lengths,
    footprint_from_dict,
    layout_point_to_object_frame,
    load_marker_model,
    marker_layout_to_dict,
    resolve_marker_sizes,
    validate_all_footprint_sizes,
)
from object_apriltag.marker_layout_calibration import (
    CalibrationQualityReport,
    CalibrationSettings,
    EdgeDiagnostics,
    FrameObservation,
    calibrate_marker_layout,
    resolve_marker_sizes_for_calibration,
)
from object_apriltag.marker_layout_calibration.finalize import check_quality_gates
from object_apriltag.marker_layout_calibration.input import parse_marker_size_override_spec
from object_apriltag.marker_layout_calibration.solve_quality import pair_translation_gate
from object_apriltag.pose import estimate_global_layout_pose, marker_corner_object_points
from tests.test_marker_layout_calibration import (
    _default_camera,
    reference_gauge_pose,
    _two_marker_poses,
    synthesize_observations,
)


def _square_payload(half: float, z: float = 0.0) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, z],
        "top_right": [half, -half, z],
        "bottom_right": [half, half, z],
        "bottom_left": [-half, half, z],
    }


class MarkerSizeOverrideParserTests(unittest.TestCase):
    def test_parses_single_id_and_range_overrides(self) -> None:
        parsed, failure = parse_marker_size_override_spec(["4:0.03", "10-12:0.025"])
        self.assertIsNone(failure)
        assert parsed is not None
        self.assertEqual(parsed, [([4], 0.03), ([10, 11, 12], 0.025)])

    def test_rejects_overlapping_override_ranges(self) -> None:
        parsed, failure = parse_marker_size_override_spec(["4:0.03", "4-5:0.025"])
        self.assertIsNone(parsed)
        assert failure is not None
        self.assertIn("overlapping", failure)

    def test_rejects_nonpositive_and_nonfinite_sizes(self) -> None:
        for token in ("1:0", "1:-0.01", "1:nan"):
            parsed, failure = parse_marker_size_override_spec([token])
            self.assertIsNone(parsed, msg=token)
            assert failure is not None

    def test_resolve_rejects_out_of_set_override_ids(self) -> None:
        resolved, failure = resolve_marker_sizes_for_calibration(
            [0, 1],
            0.07,
            ["9:0.03"],
        )
        self.assertIsNone(resolved)
        assert failure is not None
        self.assertIn("subset", failure)


class MixedLayoutJsonTests(unittest.TestCase):
    def test_uniform_json_round_trip_unchanged(self) -> None:
        payload = {
            "reference_marker_id": 0,
            "units": "meters",
            "marker_size_m": 0.07,
            "markers": {"0": _square_payload(0.035), "1": _square_payload(0.035, z=0.12)},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            layout = load_marker_model(path)
        self.assertNotIn("size_m", marker_layout_to_dict(layout)["markers"]["0"])
        self.assertEqual(layout.marker_size_for(0), 0.07)
        self.assertEqual(layout.marker_size_for(1), 0.07)

    def test_rejects_non_finite_default_marker_size(self) -> None:
        payload = {
            "reference_marker_id": 0,
            "units": "meters",
            "marker_size_m": "nan",
            "markers": {"0": _square_payload(0.035)},
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "layout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_marker_model(path)
        self.assertIn("positive and finite", str(ctx.exception))

    def test_build_rejects_mismatched_marker_size_map(self) -> None:
        footprints = {
            0: footprint_from_dict(0, _square_payload(0.035)),
            1: footprint_from_dict(1, _square_payload(0.035, z=0.12)),
        }
        with self.assertRaises(ValueError) as ctx:
            build_marker_layout(0, 0.07, footprints, marker_sizes_m={0: 0.07})
        self.assertIn("missing footprint marker IDs", str(ctx.exception))

    def test_resolve_marker_sizes_rejects_invalid_override_values(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            resolve_marker_sizes({0, 1}, 0.07, {1: float("nan")})
        self.assertIn("positive and finite", str(ctx.exception))

    def test_mixed_json_load_save_and_validation(self) -> None:
        default = 0.07
        small = 0.05
        footprints = {
            0: footprint_from_dict(0, _square_payload(default / 2)),
            1: footprint_from_dict(1, _square_payload(small / 2, z=0.12)),
        }
        layout = build_marker_layout(
            0,
            default,
            footprints,
            marker_sizes_m={0: default, 1: small},
        )
        payload = marker_layout_to_dict(layout)
        self.assertNotIn("size_m", payload["markers"]["0"])
        self.assertEqual(payload["markers"]["1"]["size_m"], small)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mixed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_marker_model(path)
        self.assertEqual(loaded.marker_size_for(1), small)
        validate_all_footprint_sizes(loaded.footprints, loaded.marker_sizes_m)


class MixedCalibrationTests(unittest.TestCase):
    def test_recovers_declared_edge_sizes(self) -> None:
        default_size = 0.07
        small_size = 0.05
        marker_sizes_m = {0: default_size, 1: small_size}
        marker_poses = _two_marker_poses(default_size)
        marker_poses[1] = (
            marker_poses[1][0],
            marker_poses[1][1] + np.array([0.12, 0.0, -0.05], dtype=np.float64),
        )
        observations = []
        for frame_index in range(25):
            markers = {}
            for marker_id, size in marker_sizes_m.items():
                object_points = marker_corner_object_points(size)
                rotation, translation = marker_poses[marker_id]
                layout_rotation, _ = cv2.Rodrigues(np.array([0.1, -0.15, 0.05], dtype=np.float64))
                layout_translation = np.array([0.02, -0.01, 0.6], dtype=np.float64)
                corners = []
                for corner_index in range(4):
                    layout_point = rotation @ object_points[corner_index] + translation
                    camera_point = layout_rotation @ layout_point + layout_translation
                    projected, _ = cv2.projectPoints(
                        camera_point.reshape(1, 1, 3).astype(np.float32),
                        np.zeros((3, 1), dtype=np.float64),
                        np.zeros((3, 1), dtype=np.float64),
                        *_default_camera(),
                    )
                    corners.append(projected.reshape(2))
                markers[marker_id] = np.stack(corners, axis=0)
            observations.append(FrameObservation(frame_id=frame_index, markers=markers))

        result = calibrate_marker_layout(
            observations,
            *_default_camera(),
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            marker_size_m=default_size,
            marker_sizes_m=marker_sizes_m,
            settings=CalibrationSettings(min_inliers_per_edge=20),
        )
        self.assertIsNone(result.failure_reason, msg=result.failure_reason)
        assert result.layout is not None
        for marker_id, expected_size in marker_sizes_m.items():
            edges = footprint_edge_lengths(*result.layout.footprints[marker_id].corners())
            for edge in edges:
                self.assertAlmostEqual(edge, expected_size, places=2)


class MixedRuntimeFusionTests(unittest.TestCase):
    def test_fusion_uses_per_marker_sizes(self) -> None:
        default_size = 0.07
        small_size = 0.05
        object_rvec = np.array([0.18, -0.12, 0.07], dtype=np.float64)
        object_origin = np.array([0.02, -0.015, 0.62], dtype=np.float64)
        footprints = {
            0: footprint_from_dict(0, _square_payload(default_size / 2)),
            1: footprint_from_dict(1, _square_payload(small_size / 2, z=0.12)),
            2: footprint_from_dict(2, _square_payload(default_size / 2, z=0.12)),
            3: footprint_from_dict(3, _square_payload(small_size / 2, z=0.24)),
        }
        layout = build_marker_layout(
            0,
            default_size,
            footprints,
            marker_sizes_m={0: default_size, 1: small_size, 2: default_size, 3: small_size},
        )
        camera_matrix, dist_coeffs = _default_camera()

        def projected_corners(marker_id: int) -> np.ndarray:
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
            return projected.reshape(1, 4, 2).astype(np.float32)

        def projected_corners_with_wrong_sizes(marker_id: int) -> np.ndarray:
            size = small_size if marker_id in (0, 2) else default_size
            object_points = marker_corner_object_points(size)
            marker_rotation, _ = cv2.Rodrigues(object_rvec)
            marker_origin = object_origin + marker_rotation @ layout.transforms[
                marker_id
            ].offset
            rvec, _ = cv2.Rodrigues(marker_rotation @ layout.transforms[marker_id].rotation)
            projected, _ = cv2.projectPoints(
                object_points.astype(np.float32),
                rvec,
                marker_origin.reshape(3, 1),
                camera_matrix,
                dist_coeffs,
            )
            return projected.reshape(1, 4, 2).astype(np.float32)

        detections = [
            (projected_corners(marker_id), marker_id) for marker_id in (0, 1, 2, 3)
        ]
        pose = estimate_global_layout_pose(detections, layout, camera_matrix, dist_coeffs)
        self.assertIsNotNone(pose)
        origin, rotation = pose

        swapped = [
            (projected_corners_with_wrong_sizes(marker_id), marker_id)
            for marker_id in (0, 1, 2, 3)
        ]
        swapped_pose = estimate_global_layout_pose(
            swapped, layout, camera_matrix, dist_coeffs
        )
        self.assertIsNotNone(swapped_pose)
        swapped_origin, swapped_rotation = swapped_pose
        assert origin is not None and swapped_origin is not None
        self.assertGreater(np.linalg.norm(origin - swapped_origin), 1e-3)

    def test_detector_rejects_scalar_override_on_mixed_model(self) -> None:
        layout = build_marker_layout(
            0,
            0.07,
            {
                0: footprint_from_dict(0, _square_payload(0.035)),
                1: footprint_from_dict(1, _square_payload(0.025, z=0.12)),
            },
            marker_sizes_m={0: 0.07, 1: 0.05},
        )
        camera_matrix, dist_coeffs = _default_camera()
        with self.assertRaises(ValueError):
            ObjectDetector(
                camera_matrix,
                dist_coeffs,
                marker_model=layout,
                marker_size_m=0.07,
            )


class PairGateScalingTests(unittest.TestCase):
    def test_pair_gate_uses_min_marker_size(self) -> None:
        settings = CalibrationSettings()
        sizes = {0: 0.07, 1: 0.05}
        gate = pair_translation_gate(settings, sizes, (0, 1))
        self.assertAlmostEqual(gate, settings.pair_translation_rms_gate_ratio * 0.05)

    def test_quality_gate_evaluates_each_edge_against_pair_gate(self) -> None:
        settings = CalibrationSettings(pair_translation_rms_gate_ratio=0.1)
        sizes = {0: 0.07, 1: 0.05}
        translation_rms_m = 0.006
        self.assertLess(translation_rms_m, settings.pair_translation_rms_gate_ratio * 0.07)
        self.assertGreater(translation_rms_m, pair_translation_gate(settings, sizes, (0, 1)))
        quality = CalibrationQualityReport(
            reprojection_rms_px=0.0,
            per_marker_reprojection_rms_px={0: 0.0, 1: 0.0},
            edges=(
                EdgeDiagnostics(
                    marker_a=0,
                    marker_b=1,
                    inlier_count=25,
                    translation_rms_m=translation_rms_m,
                    rotation_rms_deg=0.0,
                ),
            ),
            pair_translation_rms_max_m=translation_rms_m,
            pair_rotation_rms_max_deg=0.0,
            frame_count=25,
            observation_count=50,
            inlier_corner_count=200,
            input_frame_count=25,
            rejected_frame_count=0,
            accepted_frame_count=25,
            connected_marker_ids=frozenset({0, 1}),
            missing_expected_ids=frozenset(),
            unused_expected_ids=frozenset(),
        )
        failure = check_quality_gates(quality, settings, sizes, [0, 1])
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertIn("translation RMS", failure)


if __name__ == "__main__":
    unittest.main()
