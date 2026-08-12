"""Background worker for live pair-readiness diagnostics during capture."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from object_apriltag.marker_layout_calibration import (
    CalibrationSettings,
    FrameObservation,
    LivePairReadinessDiagnostics,
)

ComputeLivePairReadiness = Callable[..., LivePairReadinessDiagnostics]


def snapshot_observations(
    observations: Sequence[FrameObservation],
) -> tuple[FrameObservation, ...]:
    return tuple(
        FrameObservation(
            frame_id=observation.frame_id,
            markers={
                marker_id: corners.copy()
                for marker_id, corners in observation.markers.items()
            },
        )
        for observation in observations
    )


def empty_pair_readiness(
    *,
    sample_count: int,
    expected_marker_ids: list[int],
    reference_marker_id: int,
) -> LivePairReadinessDiagnostics:
    return LivePairReadinessDiagnostics(
        pairs=(),
        connected_marker_ids=frozenset({reference_marker_id}),
        missing_marker_ids=frozenset(set(expected_marker_ids) - {reference_marker_id}),
        sample_count=sample_count,
    )


@dataclass(frozen=True)
class LivePairReadinessView:
    diagnostics: LivePairReadinessDiagnostics
    represented_sample_count: int
    is_computing: bool


class LivePairReadinessWorker:
    """Compute pair readiness off the capture thread with coalesced snapshots."""

    def __init__(
        self,
        *,
        compute_fn: ComputeLivePairReadiness,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        expected_marker_ids: list[int],
        reference_marker_id: int,
        marker_size_m: float,
        settings: CalibrationSettings,
    ) -> None:
        self._compute_fn = compute_fn
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._expected_marker_ids = expected_marker_ids
        self._reference_marker_id = reference_marker_id
        self._marker_size_m = marker_size_m
        self._settings = settings

        self._condition = threading.Condition()
        self._shutdown_requested = False
        self._pending_snapshot: tuple[FrameObservation, ...] | None = None
        self._computing_sample_count: int | None = None
        self._latest_diagnostics = empty_pair_readiness(
            sample_count=0,
            expected_marker_ids=expected_marker_ids,
            reference_marker_id=reference_marker_id,
        )
        self._represented_sample_count = 0

        self._thread = threading.Thread(target=self._run, name="live-pair-readiness", daemon=True)
        self._thread.start()

    def submit(self, observations: Sequence[FrameObservation]) -> None:
        snapshot = snapshot_observations(observations)
        with self._condition:
            self._pending_snapshot = snapshot
            self._condition.notify()

    def poll(self, current_sample_count: int) -> LivePairReadinessView:
        with self._condition:
            is_computing = (
                self._represented_sample_count < current_sample_count
                or self._computing_sample_count is not None
                or self._pending_snapshot is not None
            )
            return LivePairReadinessView(
                diagnostics=self._latest_diagnostics,
                represented_sample_count=self._represented_sample_count,
                is_computing=is_computing,
            )

    def shutdown(self, *, join_timeout: float = 0.0) -> None:
        with self._condition:
            self._shutdown_requested = True
            self._pending_snapshot = None
            self._condition.notify()
        self._thread.join(timeout=join_timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending_snapshot is None and not self._shutdown_requested:
                    self._condition.wait()
                if self._shutdown_requested and self._pending_snapshot is None:
                    return
                snapshot = self._pending_snapshot
                self._pending_snapshot = None
                self._computing_sample_count = len(snapshot)

            diagnostics = self._compute_snapshot(snapshot)

            with self._condition:
                self._latest_diagnostics = diagnostics
                self._represented_sample_count = diagnostics.sample_count
                self._computing_sample_count = None

    def _compute_snapshot(
        self,
        snapshot: tuple[FrameObservation, ...],
    ) -> LivePairReadinessDiagnostics:
        try:
            diagnostics = self._compute_fn(
                list(snapshot),
                self._camera_matrix,
                self._dist_coeffs,
                expected_marker_ids=self._expected_marker_ids,
                reference_marker_id=self._reference_marker_id,
                marker_size_m=self._marker_size_m,
                settings=self._settings,
            )
        except Exception as exc:
            return LivePairReadinessDiagnostics(
                pairs=(),
                connected_marker_ids=frozenset({self._reference_marker_id}),
                missing_marker_ids=frozenset(
                    set(self._expected_marker_ids) - {self._reference_marker_id}
                ),
                sample_count=len(snapshot),
                failure_reason=f"Pair readiness failed: {exc}",
            )

        return LivePairReadinessDiagnostics(
            pairs=diagnostics.pairs,
            connected_marker_ids=diagnostics.connected_marker_ids,
            missing_marker_ids=diagnostics.missing_marker_ids,
            sample_count=len(snapshot),
            failure_reason=diagnostics.failure_reason,
        )
