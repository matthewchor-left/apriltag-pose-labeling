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
    """Deep-copy captured observations for background-thread computation.

    Args:
        observations: Live capture observation list from the main thread.

    Returns:
        Immutable tuple of copied ``FrameObservation`` records.
    """
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
    reference_marker_id: int | None,
) -> LivePairReadinessDiagnostics:
    """Build a placeholder readiness snapshot before any samples are captured.

    Args:
        sample_count: Current captured sample count.
        expected_marker_ids: Full list of expected marker IDs.
        reference_marker_id: Reference marker assumed connected at startup, or
            ``None`` when reference selection is deferred.

    Returns:
        Diagnostics with only the reference marker connected and all others missing,
        or all markers missing when reference selection is deferred.
    """
    if reference_marker_id is None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(expected_marker_ids),
            sample_count=sample_count,
        )
    return LivePairReadinessDiagnostics(
        pairs=(),
        connected_marker_ids=frozenset({reference_marker_id}),
        missing_marker_ids=frozenset(set(expected_marker_ids) - {reference_marker_id}),
        sample_count=sample_count,
    )


@dataclass(frozen=True)
class LivePairReadinessView:
    """Snapshot of pair-readiness diagnostics for HUD polling.

    Attributes:
        diagnostics: Latest pair-readiness graph diagnostics.
        represented_sample_count: Sample count the snapshot was computed from.
        is_computing: Whether a newer snapshot is pending or in flight.
    """

    diagnostics: LivePairReadinessDiagnostics
    represented_sample_count: int
    is_computing: bool


class LivePairReadinessWorker:
    """Compute pair readiness off the capture thread with coalesced snapshots.

    Submits deep-copied observation lists to a background thread so the live
    preview loop is not blocked by pair-readiness graph computation.
    """

    def __init__(
        self,
        *,
        compute_fn: ComputeLivePairReadiness,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        expected_marker_ids: list[int],
        reference_marker_id: int | None,
        settings: CalibrationSettings,
    ) -> None:
        """Start the background readiness worker thread.

        Args:
            compute_fn: Callable compatible with ``compute_live_pair_readiness``.
            camera_matrix: Camera intrinsics matrix.
            dist_coeffs: Distortion coefficients.
            expected_marker_ids: Full list of expected marker IDs.
            reference_marker_id: Reference marker for connectivity reporting, or
                ``None`` to infer from raw co-visibility during computation.
            settings: Calibration thresholds used for pair-readiness gates.
        """
        self._compute_fn = compute_fn
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._expected_marker_ids = expected_marker_ids
        self._reference_marker_id = reference_marker_id
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
        """Queue a deep-copied observation snapshot for background computation.

        Coalesces rapid submissions: only the latest pending snapshot is retained.

        Args:
            observations: Current live capture observation list.
        """
        snapshot = snapshot_observations(observations)
        with self._condition:
            self._pending_snapshot = snapshot
            self._condition.notify()

    def poll(self, current_sample_count: int) -> LivePairReadinessView:
        """Return the latest readiness snapshot for HUD display.

        Args:
            current_sample_count: Number of captured observations on the main thread.

        Returns:
            View with latest diagnostics, represented sample count, and computing flag.
        """
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
        """Signal the worker thread to exit and optionally wait for it.

        Args:
            join_timeout: Seconds to wait for the background thread to finish.
        """
        with self._condition:
            self._shutdown_requested = True
            self._pending_snapshot = None
            self._condition.notify()
        self._thread.join(timeout=join_timeout)

    def _run(self) -> None:
        """Background loop: compute readiness for the latest coalesced snapshot."""
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
        """Run pair-readiness computation on one observation snapshot.

        Args:
            snapshot: Deep-copied observations from the capture thread.

        Returns:
            Pair-readiness diagnostics for ``snapshot``, or a failure placeholder when
            computation raises.
        """
        try:
            diagnostics = self._compute_fn(
                list(snapshot),
                self._camera_matrix,
                self._dist_coeffs,
                expected_marker_ids=self._expected_marker_ids,
                reference_marker_id=self._reference_marker_id,
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
