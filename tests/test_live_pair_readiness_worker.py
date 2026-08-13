"""Tests for background live pair-readiness worker."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag.cli.live_pair_readiness_worker import (
    LivePairReadinessWorker,
    snapshot_observations,
)
from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    FrameObservation,
    LivePairReadinessDiagnostics,
)
from tests.test_marker_layout_calibration import _default_camera


def _observation(frame_id: int, marker_id: int = 0, value: float = 0.0) -> FrameObservation:
    corners = np.full((4, 2), value + marker_id, dtype=np.float64)
    return FrameObservation(frame_id=frame_id, markers={marker_id: corners, 1: corners + 10.0})


def _diagnostics_for(sample_count: int, *, tag: str = "") -> LivePairReadinessDiagnostics:
    return LivePairReadinessDiagnostics(
        pairs=(),
        connected_marker_ids=frozenset({0}),
        missing_marker_ids=frozenset({1}),
        sample_count=sample_count,
        failure_reason=tag or None,
    )


class LivePairReadinessWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix, self.dist_coeffs = _default_camera()
        self.settings = CalibrationSettings(min_inliers_per_edge=20)

    def _worker(self, compute_fn) -> LivePairReadinessWorker:
        return LivePairReadinessWorker(
            compute_fn=compute_fn,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            expected_marker_ids=[0, 1],
            reference_marker_id=0,
            settings=self.settings,
        )

    def test_submit_passes_immutable_snapshot_to_compute(self) -> None:
        gate = threading.Event()
        seen_value: list[float] = []

        def compute_fn(observations, *_args, **_kwargs):
            seen_value.append(float(observations[0].markers[0][0, 0]))
            gate.wait(timeout=1.0)
            return _diagnostics_for(len(observations))

        worker = self._worker(compute_fn)
        try:
            observations = [_observation(0, value=1.5)]
            worker.submit(observations)
            observations[0].markers[0][0, 0] = 99.0
            gate.set()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if worker.poll(1).represented_sample_count == 1:
                    break
                time.sleep(0.01)
            self.assertEqual(seen_value, [1.5])
        finally:
            worker.shutdown()

    def test_coalesces_pending_updates_to_newest_snapshot(self) -> None:
        gate = threading.Event()
        compute_lengths: list[int] = []

        def compute_fn(observations, *_args, **_kwargs):
            compute_lengths.append(len(observations))
            if len(compute_lengths) == 1:
                gate.wait(timeout=1.0)
            return _diagnostics_for(len(observations))

        worker = self._worker(compute_fn)
        try:
            worker.submit([_observation(0)])
            time.sleep(0.05)
            worker.submit([_observation(0), _observation(1)])
            worker.submit([_observation(0), _observation(1), _observation(2)])
            gate.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                view = worker.poll(3)
                if view.represented_sample_count == 3 and not view.is_computing:
                    break
                time.sleep(0.01)
            self.assertEqual(compute_lengths[-1], 3)
            self.assertLessEqual(len(compute_lengths), 2)
        finally:
            worker.shutdown()

    def test_compute_exception_surfaces_readable_failure(self) -> None:
        def compute_fn(*_args, **_kwargs):
            raise RuntimeError("synthetic boom")

        worker = self._worker(compute_fn)
        try:
            worker.submit([_observation(0)])
            deadline = time.monotonic() + 1.0
            failure_reason = None
            while time.monotonic() < deadline:
                view = worker.poll(1)
                if view.represented_sample_count == 1:
                    failure_reason = view.diagnostics.failure_reason
                    break
                time.sleep(0.01)
            self.assertIsNotNone(failure_reason)
            assert failure_reason is not None
            self.assertIn("Pair readiness failed", failure_reason)
            self.assertIn("synthetic boom", failure_reason)
        finally:
            worker.shutdown()

    def test_shutdown_returns_without_waiting_for_slow_compute(self) -> None:
        gate = threading.Event()

        def compute_fn(observations, *_args, **_kwargs):
            gate.wait(timeout=2.0)
            return _diagnostics_for(len(observations))

        worker = self._worker(compute_fn)
        worker.submit([_observation(0)])
        time.sleep(0.05)
        start = time.monotonic()
        worker.shutdown(join_timeout=0.0)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.2)
        gate.set()

    def test_snapshot_observations_deep_copies_corners(self) -> None:
        observations = [_observation(0, value=2.0)]
        snapshot = snapshot_observations(observations)
        observations[0].markers[0][0, 0] = 50.0
        self.assertEqual(float(snapshot[0].markers[0][0, 0]), 2.0)


class LivePairReadinessWorkerCaptureIntegrationTests(unittest.TestCase):
    def test_capture_continues_while_readiness_compute_is_blocked(self) -> None:
        from object_apriltag.cli.calibrate_marker_model import run_capture

        gate = threading.Event()
        compute_calls = 0

        def slow_compute(observations, *_args, **_kwargs):
            nonlocal compute_calls
            compute_calls += 1
            if compute_calls == 1:
                gate.wait(timeout=2.0)
            return LivePairReadinessDiagnostics(
                pairs=(),
                connected_marker_ids=frozenset({0}),
                missing_marker_ids=frozenset({1}),
                sample_count=len(observations),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            calibration = tmp_path / "intrinsics.json"
            output = tmp_path / "marker_model.json"
            _write_intrinsics(calibration)

            args = mock.Mock(
                camera=0,
                calibration=calibration,
                dictionary="36h11",
                detection_sensitivity="default",
                marker_size=0.07,
                marker_ids=["0", "1"],
                reference_marker_id=0,
                output=output,
                force=True,
                auto=False,
                sample_rate_hz=10.0,
                min_pair_inliers=20,
                reprojection_rms_gate_px=2.0,
                pair_translation_rms_gate_ratio=0.10,
                pair_rotation_rms_gate_deg=5.0,
                diagnostics_output=None,
                anchor_marker_ids=None,
                anchor_stop_after_expansion=False,
                marker_size_for=None,
            )

            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            capture = mock.MagicMock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, frame.copy())

            detector = mock.MagicMock()
            visible = {0: np.zeros((4, 2)), 1: np.ones((4, 2))}

            def detect_markers(_gray):
                return (
                    [corner.reshape(1, 4, 2) for corner in visible.values()],
                    np.array([[0], [1]], dtype=np.int32),
                    None,
                )

            detector.detectMarkers.side_effect = detect_markers

            wait_keys = [ord("c")] + [0] * 11 + [ord("q")]

            wait_iter = iter(wait_keys)

            with (
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.VideoCapture", return_value=capture),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.build_apriltag_detector",
                    return_value=detector,
                ),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.cv2.waitKey",
                    side_effect=lambda _delay: next(wait_iter, ord("q")),
                ),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.imshow"),
                mock.patch("object_apriltag.cli.calibrate_marker_model.cv2.destroyAllWindows"),
                mock.patch(
                    "object_apriltag.cli.calibrate_marker_model.compute_live_pair_readiness",
                    side_effect=slow_compute,
                ),
            ):
                run_capture(args)
                reads_during_block = capture.read.call_count
                gate.set()

            self.assertGreaterEqual(reads_during_block, 5)
            self.assertGreaterEqual(compute_calls, 1)


def _write_intrinsics(path: Path, *, width: int = 640, height: int = 480) -> None:
    payload = {
        "calibration_source": "test",
        "image_size": [width, height],
        "camera_matrix": [[900.0, 0.0, width / 2.0], [0.0, 900.0, height / 2.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
