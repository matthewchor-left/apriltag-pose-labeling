"""Held-out moving-video detection consistency evaluation.

Frozen per-frame detections are reused for every candidate layout. For each
visible expected marker, pose is solved from all other visible expected markers
only (strict global RANSAC/LM, no single-marker fallback). Held-out footprint
corners are projected and compared to observed corners in pixels.

Duplicate marker IDs in one frame: the marker ID is dropped entirely so
scoring never depends on detector ordering. Extra duplicates are counted in
frame diagnostics only.
Unknown marker IDs (not in expected_marker_ids): ignored for solve and scoring.
Non-finite corners: that detection entry is skipped.
Empty videos: zero possible folds with zeroed summaries.
Incompatible candidates (layout.marker_ids != expected_marker_ids): refused with
zeroed metrics and compatible=False.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from object_apriltag.evaluation.types import (
    DetectionConsistencyCandidateResult,
    DetectionConsistencyEvaluation,
    LeaveOneMarkerDetectionFold,
    MetricSummaryPx,
    PerMarkerDetectionSummary,
    SourceVideoDetectionSummary,
    VisibleMarkerCountStratum,
)
from object_apriltag.layout import MarkerLayout, layout_point_to_camera
from object_apriltag.pose import Detection, estimate_strict_global_pose

_MIN_TRAINING_MARKERS = 2


@dataclass(frozen=True)
class FrozenFrameDetections:
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class FrozenVideoDetections:
    source_video: str
    frames: tuple[FrozenFrameDetections, ...]


@dataclass(frozen=True)
class DetectionCandidate:
    name: str
    layout: MarkerLayout


def metric_summary_px(values_px: Sequence[float] | np.ndarray) -> MetricSummaryPx:
    array = np.asarray(values_px, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return MetricSummaryPx(
            count=0,
            min_px=0.0,
            median_px=0.0,
            rmse_px=0.0,
            p95_px=0.0,
            max_px=0.0,
        )
    return MetricSummaryPx(
        count=int(array.size),
        min_px=float(np.min(array)),
        median_px=float(np.median(array)),
        rmse_px=float(np.sqrt(np.mean(array * array))),
        p95_px=float(np.percentile(array, 95)),
        max_px=float(np.max(array)),
    )


def normalize_frame_detections(
    detections: Sequence[Detection],
    *,
    expected_marker_ids: frozenset[int],
) -> tuple[dict[int, np.ndarray], int, int, int]:
    """Return usable corners by marker ID, duplicate skips, unknown IDs, malformed skips."""
    corners_by_id: dict[int, np.ndarray] = {}
    duplicate_dropped: set[int] = set()
    duplicate_skips = 0
    unknown_ids = 0
    malformed_skips = 0
    for corners, marker_id in detections:
        if marker_id not in expected_marker_ids:
            unknown_ids += 1
            continue
        if marker_id in duplicate_dropped:
            duplicate_skips += 1
            continue
        try:
            detected = np.asarray(corners, dtype=np.float64).reshape(4, 2)
        except ValueError:
            malformed_skips += 1
            continue
        if not np.all(np.isfinite(detected)):
            malformed_skips += 1
            continue
        if marker_id in corners_by_id:
            duplicate_skips += 1
            del corners_by_id[marker_id]
            duplicate_dropped.add(marker_id)
            continue
        corners_by_id[marker_id] = detected
    return corners_by_id, duplicate_skips, unknown_ids, malformed_skips


def evaluate_detection_consistency(
    *,
    expected_marker_ids: frozenset[int] | set[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    videos: Sequence[FrozenVideoDetections],
    candidates: Sequence[DetectionCandidate],
) -> DetectionConsistencyEvaluation:
    expected = frozenset(expected_marker_ids)
    candidate_results = tuple(
        _evaluate_candidate(
            candidate=candidate,
            expected_marker_ids=expected,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            videos=videos,
        )
        for candidate in candidates
    )
    return DetectionConsistencyEvaluation(
        expected_marker_ids=tuple(sorted(expected)),
        candidates=candidate_results,
    )


def _evaluate_candidate(
    *,
    candidate: DetectionCandidate,
    expected_marker_ids: frozenset[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    videos: Sequence[FrozenVideoDetections],
) -> DetectionConsistencyCandidateResult:
    layout_ids = candidate.layout.marker_ids
    if layout_ids != set(expected_marker_ids):
        extra = sorted(layout_ids - expected_marker_ids)
        missing = sorted(expected_marker_ids - layout_ids)
        return _incompatible_candidate_result(
            candidate.name,
            reason=(
                f"incompatible_marker_ids: layout has {sorted(layout_ids)}, "
                f"expected {sorted(expected_marker_ids)}; "
                f"extra={extra}, missing={missing}."
            ),
        )

    folds: list[LeaveOneMarkerDetectionFold] = []
    for video in videos:
        for frame_index, frame in enumerate(video.frames):
            corners_by_id, _, _, _ = normalize_frame_detections(
                frame.detections,
                expected_marker_ids=expected_marker_ids,
            )
            visible_marker_ids = sorted(corners_by_id)
            visible_count = len(visible_marker_ids)
            for held_out_marker_id in visible_marker_ids:
                training_marker_ids = [
                    marker_id
                    for marker_id in visible_marker_ids
                    if marker_id != held_out_marker_id
                ]
                if len(training_marker_ids) < _MIN_TRAINING_MARKERS:
                    folds.append(
                        LeaveOneMarkerDetectionFold(
                            source_video=video.source_video,
                            frame_index=frame_index,
                            held_out_marker_id=held_out_marker_id,
                            visible_marker_count=visible_count,
                            eligible=False,
                            refusal_reason=(
                                f"insufficient_training_markers: need >= {_MIN_TRAINING_MARKERS}, "
                                f"got {len(training_marker_ids)}."
                            ),
                            solve_failed=False,
                            corner_errors_px=None,
                            summary_px=None,
                        )
                    )
                    continue

                training_detections = [
                    (corners_by_id[marker_id].reshape(1, 4, 2).astype(np.float32), marker_id)
                    for marker_id in training_marker_ids
                ]
                origin, rotation = estimate_strict_global_pose(
                    training_detections,
                    candidate.layout,
                    camera_matrix,
                    dist_coeffs,
                )
                if origin is None or rotation is None:
                    folds.append(
                        LeaveOneMarkerDetectionFold(
                            source_video=video.source_video,
                            frame_index=frame_index,
                            held_out_marker_id=held_out_marker_id,
                            visible_marker_count=visible_count,
                            eligible=True,
                            refusal_reason=None,
                            solve_failed=True,
                            corner_errors_px=None,
                            summary_px=None,
                        )
                    )
                    continue

                try:
                    corner_errors = _held_out_corner_errors_px(
                        observed_corners=corners_by_id[held_out_marker_id],
                        held_out_marker_id=held_out_marker_id,
                        layout=candidate.layout,
                        object_rotation=rotation,
                        object_origin=origin,
                        camera_matrix=camera_matrix,
                        dist_coeffs=dist_coeffs,
                    )
                except cv2.error:
                    folds.append(
                        LeaveOneMarkerDetectionFold(
                            source_video=video.source_video,
                            frame_index=frame_index,
                            held_out_marker_id=held_out_marker_id,
                            visible_marker_count=visible_count,
                            eligible=True,
                            refusal_reason=None,
                            solve_failed=True,
                            corner_errors_px=None,
                            summary_px=None,
                        )
                    )
                    continue
                if len(corner_errors) != 4:
                    folds.append(
                        LeaveOneMarkerDetectionFold(
                            source_video=video.source_video,
                            frame_index=frame_index,
                            held_out_marker_id=held_out_marker_id,
                            visible_marker_count=visible_count,
                            eligible=True,
                            refusal_reason=None,
                            solve_failed=True,
                            corner_errors_px=None,
                            summary_px=None,
                        )
                    )
                    continue
                fold_summary = metric_summary_px(corner_errors)
                folds.append(
                    LeaveOneMarkerDetectionFold(
                        source_video=video.source_video,
                        frame_index=frame_index,
                        held_out_marker_id=held_out_marker_id,
                        visible_marker_count=visible_count,
                        eligible=True,
                        refusal_reason=None,
                        solve_failed=False,
                        corner_errors_px=tuple(corner_errors),
                        summary_px=fold_summary,
                    )
                )

    return _aggregate_candidate_result(candidate.name, folds, expected_marker_ids)


def _held_out_corner_errors_px(
    *,
    observed_corners: np.ndarray,
    held_out_marker_id: int,
    layout: MarkerLayout,
    object_rotation: np.ndarray,
    object_origin: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    footprint = layout.footprints[held_out_marker_id]
    observed = np.asarray(observed_corners, dtype=np.float64).reshape(4, 2)
    zero_rvec = np.zeros(3, dtype=np.float64)
    zero_tvec = np.zeros(3, dtype=np.float64)
    errors: list[float] = []
    for index, point_layout in enumerate(footprint.corners()):
        camera_point = layout_point_to_camera(
            point_layout, object_rotation, object_origin, layout
        )
        try:
            projected, _ = cv2.projectPoints(
                camera_point.reshape(1, 1, 3).astype(np.float64),
                zero_rvec,
                zero_tvec,
                camera_matrix,
                dist_coeffs,
            )
        except cv2.error:
            return []
        projected_xy = projected.reshape(2)
        if not np.all(np.isfinite(projected_xy)):
            return []
        errors.append(float(np.linalg.norm(projected_xy - observed[index])))
    return errors


def _aggregate_candidate_result(
    candidate_name: str,
    folds: Sequence[LeaveOneMarkerDetectionFold],
    expected_marker_ids: frozenset[int],
) -> DetectionConsistencyCandidateResult:
    all_corner_errors: list[float] = []
    for fold in folds:
        if fold.eligible and not fold.solve_failed and fold.corner_errors_px is not None:
            all_corner_errors.extend(fold.corner_errors_px)

    possible_fold_count = len(folds)
    eligible_folds = [fold for fold in folds if fold.eligible]
    eligible_fold_count = len(eligible_folds)
    solve_failure_count = sum(1 for fold in eligible_folds if fold.solve_failed)
    ineligible_fold_count = possible_fold_count - eligible_fold_count

    per_marker = tuple(
        _per_marker_summary(marker_id, folds)
        for marker_id in sorted(expected_marker_ids)
    )
    strata = tuple(
        _visible_marker_count_stratum(visible_count, folds)
        for visible_count in sorted({fold.visible_marker_count for fold in folds})
    )
    per_source_video = tuple(
        _source_video_summary(source_video, folds)
        for source_video in sorted({fold.source_video for fold in folds})
    )

    return DetectionConsistencyCandidateResult(
        candidate_name=candidate_name,
        compatible=True,
        incompatibility_reason=None,
        summary_px=metric_summary_px(all_corner_errors),
        eligible_fold_count=eligible_fold_count,
        possible_fold_count=possible_fold_count,
        solve_failure_count=solve_failure_count,
        ineligible_fold_count=ineligible_fold_count,
        per_marker=per_marker,
        visible_marker_count_strata=strata,
        per_source_video=per_source_video,
    )


def _per_marker_summary(
    marker_id: int,
    folds: Sequence[LeaveOneMarkerDetectionFold],
) -> PerMarkerDetectionSummary:
    marker_folds = [fold for fold in folds if fold.held_out_marker_id == marker_id]
    return _summarize_fold_group(marker_id, marker_folds)


def _visible_marker_count_stratum(
    visible_marker_count: int,
    folds: Sequence[LeaveOneMarkerDetectionFold],
) -> VisibleMarkerCountStratum:
    stratum_folds = [
        fold for fold in folds if fold.visible_marker_count == visible_marker_count
    ]
    summary = _summarize_fold_group(None, stratum_folds)
    return VisibleMarkerCountStratum(
        visible_marker_count=visible_marker_count,
        summary_px=summary.summary_px,
        eligible_fold_count=summary.eligible_fold_count,
        possible_fold_count=summary.possible_fold_count,
        solve_failure_count=summary.solve_failure_count,
        ineligible_fold_count=summary.ineligible_fold_count,
    )


def _source_video_summary(
    source_video: str,
    folds: Sequence[LeaveOneMarkerDetectionFold],
) -> SourceVideoDetectionSummary:
    video_folds = [fold for fold in folds if fold.source_video == source_video]
    summary = _summarize_fold_group(None, video_folds)
    return SourceVideoDetectionSummary(
        source_video=source_video,
        summary_px=summary.summary_px,
        eligible_fold_count=summary.eligible_fold_count,
        possible_fold_count=summary.possible_fold_count,
        solve_failure_count=summary.solve_failure_count,
        ineligible_fold_count=summary.ineligible_fold_count,
    )


def _summarize_fold_group(
    marker_id: int | None,
    folds: Sequence[LeaveOneMarkerDetectionFold],
) -> PerMarkerDetectionSummary:
    corner_errors: list[float] = []
    for fold in folds:
        if fold.eligible and not fold.solve_failed and fold.corner_errors_px is not None:
            corner_errors.extend(fold.corner_errors_px)
    possible_fold_count = len(folds)
    eligible_folds = [fold for fold in folds if fold.eligible]
    return PerMarkerDetectionSummary(
        marker_id=marker_id if marker_id is not None else -1,
        summary_px=metric_summary_px(corner_errors),
        eligible_fold_count=len(eligible_folds),
        possible_fold_count=possible_fold_count,
        solve_failure_count=sum(1 for fold in eligible_folds if fold.solve_failed),
        ineligible_fold_count=possible_fold_count - len(eligible_folds),
    )


def _incompatible_candidate_result(
    candidate_name: str,
    *,
    reason: str,
) -> DetectionConsistencyCandidateResult:
    empty_summary = metric_summary_px([])
    return DetectionConsistencyCandidateResult(
        candidate_name=candidate_name,
        compatible=False,
        incompatibility_reason=reason,
        summary_px=empty_summary,
        eligible_fold_count=0,
        possible_fold_count=0,
        solve_failure_count=0,
        ineligible_fold_count=0,
        per_marker=(),
        visible_marker_count_strata=(),
        per_source_video=(),
    )
