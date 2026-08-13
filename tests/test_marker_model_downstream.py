"""Downstream acceptance for provisional and partial marker models."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag import ObjectDetector
from object_apriltag.layout import save_marker_model
from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    calibrate_marker_layout,
)
from tests.test_marker_layout_calibration import (
    _default_camera,
    _two_marker_poses,
    synthesize_observations,
)


class MarkerModelDownstreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.marker_size_m = 0.07
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _synthetic_corners(self) -> np.ndarray:
        half = self.marker_size_m / 2.0
        center = np.array([320.0, 240.0])
        scale = 800.0
        return np.array(
            [
                center + scale * np.array([-half, 0.0]),
                center + scale * np.array([half, 0.0]),
                center + scale * np.array([half, self.marker_size_m]),
                center + scale * np.array([-half, self.marker_size_m]),
            ],
            dtype=np.float32,
        ).reshape(1, 4, 2)

    def _raw_disconnected_observations(self):
        from object_apriltag.marker_layout_calibration import FrameObservation
        from tests.test_marker_layout_calibration import (
            _triangle_marker_poses,
            synthesize_observations,
        )

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

    def test_provisional_model_loads_in_object_detector(self) -> None:
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
            detector = ObjectDetector(
                self.camera_matrix,
                self.dist_coeffs,
                marker_model=path,
            )
            pose = detector.fuse([(self._synthetic_corners(), 0)])
            self.assertIsNotNone(pose)

    def test_partial_model_loads_in_object_detector(self) -> None:
        result = calibrate_marker_layout(
            self._raw_disconnected_observations(),
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
            partial_output=True,
        )
        assert result.layout is not None
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            save_marker_model(path, result.layout)
            detector = ObjectDetector(
                self.camera_matrix,
                self.dist_coeffs,
                marker_model=path,
            )
            self.assertEqual(detector.marker_model.marker_ids, {0, 1})
            pose = detector.fuse([(self._synthetic_corners(), 0)])
            self.assertIsNotNone(pose)

    def test_partial_model_passes_inspection_cli(self) -> None:
        from object_apriltag.cli.inspect_marker_model import main

        result = calibrate_marker_layout(
            self._raw_disconnected_observations(),
            self.camera_matrix,
            self.dist_coeffs,
            expected_marker_ids=[0, 1, 2, 3],
            reference_marker_id=0,
            marker_size_m=self.marker_size_m,
            settings=self.settings,
            best_effort=True,
            partial_output=True,
        )
        assert result.layout is not None
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "marker_model.json"
            save_marker_model(path, result.layout)
            argv = [
                "object-inspect-marker-model",
                "--marker-model",
                str(path),
                "--no-visualize",
            ]
            with mock.patch("sys.argv", argv), mock.patch("builtins.print") as print_mock:
                main()
            printed = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
            self.assertIn("Reference marker id: 0", printed)
            self.assertIn("Marker 0", printed)
            self.assertIn("Marker 1", printed)
            self.assertNotIn("Marker 2", printed)


if __name__ == "__main__":
    unittest.main()
