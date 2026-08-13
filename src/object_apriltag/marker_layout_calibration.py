"""Multi-view marker layout calibration from co-visible AprilTag corners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from object_apriltag.layout import (
    CORNER_NAMES,
    MarkerFootprint,
    MarkerLayout,
    build_marker_layout,
    footprint_from_dict,
    footprint_orientation,
    marker_origin_on_object,
    resolve_marker_sizes,
)
from object_apriltag.pose import marker_corner_object_points

MarkerPair = tuple[int, int]


@dataclass(frozen=True)
class FrameObservation:
    """One camera sample with expected marker corners in OpenCV order."""

    frame_id: str | int
    markers: dict[int, np.ndarray]


@dataclass(frozen=True)
class CalibrationSettings:
    min_inliers_per_edge: int = 20
    reprojection_rms_gate_px: float = 2.0
    pair_translation_rms_gate_ratio: float = 0.10
    pair_rotation_rms_gate_deg: float = 5.0
    huber_delta_px: float = 1.0
    corner_outlier_px: float = 3.0
    max_ba_iterations: int = 50


@dataclass(frozen=True)
class EdgeDiagnostics:
    marker_a: int
    marker_b: int
    inlier_count: int
    translation_rms_m: float
    rotation_rms_deg: float


@dataclass(frozen=True)
class CalibrationQualityReport:
    reprojection_rms_px: float
    per_marker_reprojection_rms_px: dict[int, float]
    edges: tuple[EdgeDiagnostics, ...]
    pair_translation_rms_max_m: float
    pair_rotation_rms_max_deg: float
    frame_count: int
    observation_count: int
    inlier_corner_count: int
    input_frame_count: int
    rejected_frame_count: int
    accepted_frame_count: int
    connected_marker_ids: frozenset[int]
    missing_expected_ids: frozenset[int]
    unused_expected_ids: frozenset[int]
    assignment_rejections: AssignmentRejectionSummary | None = None
    assignment_rejection_records: tuple[FrameAssignmentRejectionRecord, ...] | None = None
    dropped_pair_edges: tuple[DroppedPairEdge, ...] | None = None
    restored_pair_edges: tuple[RestoredPairEdge, ...] | None = None
    anchor_core: "AnchorCoreDiagnostics | None" = None


@dataclass(frozen=True)
class MarkerExpansionRecord:
    marker_id: int
    status: str
    support_frames: int = 0
    reason: str | None = None
    stage: str = "expand"


@dataclass(frozen=True)
class AnchorCoreBootstrapDiagnostics:
    status: str
    frames_considered: int
    frames_accepted: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class AnchorCoreDiagnostics:
    mode: str
    configured_anchor_ids: tuple[int, ...]
    bootstrap: AnchorCoreBootstrapDiagnostics
    expansion: tuple[MarkerExpansionRecord, ...]
    final_solved_ids: frozenset[int]
    unresolved_ids: frozenset[int]
    stopped_after_expansion: bool = False


@dataclass(frozen=True)
class QualityGateFailure:
    category: Literal["strict", "connectivity", "data"]
    message: str


@dataclass(frozen=True)
class CalibrationResult:
    layout: MarkerLayout | None
    quality: CalibrationQualityReport | None
    failure_reason: str | None
    outcome: Literal["accepted", "provisional", "refused"] | None = None
    calibration_policy: Literal["strict", "best_effort"] = "strict"
    failed_quality_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome is not None:
            return
        if self.layout is not None and self.failure_reason is None:
            resolved: Literal["accepted", "provisional", "refused"] = (
                "provisional" if self.failed_quality_gates else "accepted"
            )
        else:
            resolved = "refused"
        object.__setattr__(self, "outcome", resolved)


@dataclass(frozen=True)
class PairReadinessEdge:
    marker_a: int
    marker_b: int
    raw_covisible_frames: int
    robust_inlier_count: int
    translation_rms_m: float | None
    rotation_rms_deg: float | None
    status: str


@dataclass(frozen=True)
class LivePairReadinessDiagnostics:
    pairs: tuple[PairReadinessEdge, ...]
    connected_marker_ids: frozenset[int]
    missing_marker_ids: frozenset[int]
    sample_count: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class FrameAssignmentRejection:
    reason: str
    marker_pair: MarkerPair | None = None
    translation_error_m: float | None = None
    rotation_error_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None


@dataclass(frozen=True)
class FrameAssignmentResult:
    assignment: dict[int, _MarkerCandidate] | None
    rejection: FrameAssignmentRejection | None


@dataclass(frozen=True)
class AssignmentRejectionCauseCount:
    reason: str
    marker_pair: MarkerPair | None
    count: int


@dataclass(frozen=True)
class MeasurementDistribution:
    min: float | None
    median: float | None
    p95: float | None
    max: float | None


@dataclass(frozen=True)
class FrameAssignmentRejectionRecord:
    frame_index: int
    frame_id: str | int
    visible_marker_ids: tuple[int, ...]
    reason: str
    marker_pair: MarkerPair | None = None
    translation_error_m: float | None = None
    rotation_error_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None


@dataclass(frozen=True)
class AssignmentRejectionCauseStats:
    reason: str
    marker_pair: MarkerPair | None
    count: int
    sample_frame_ids: tuple[str | int, ...]
    translation_error_m: MeasurementDistribution | None
    rotation_error_deg: MeasurementDistribution | None
    translation_gate_m: float | None
    rotation_gate_deg: float | None
    translation_error_ratio: MeasurementDistribution | None
    rotation_error_ratio: MeasurementDistribution | None


@dataclass(frozen=True)
class AssignmentRejectionSummary:
    total_rejected: int
    by_reason: tuple[tuple[str, int], ...]
    by_pair: tuple[tuple[MarkerPair, int], ...]
    top_causes: tuple[AssignmentRejectionCauseCount, ...]
    by_cause: tuple[AssignmentRejectionCauseStats, ...] = ()


@dataclass(frozen=True)
class DroppedPairEdge:
    marker_a: int
    marker_b: int
    stage: str
    reason: str
    observed_count: int
    supported_count: int
    required_count: int
    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None

    @property
    def marker_pair(self) -> MarkerPair:
        return (self.marker_a, self.marker_b)


@dataclass(frozen=True)
class RestoredPairEdge:
    marker_a: int
    marker_b: int
    stage: str
    original_stage: str
    original_reason: str
    observed_count: int
    supported_count: int
    required_count: int
    support_fraction: float
    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    translation_gate_m: float | None = None
    rotation_gate_deg: float | None = None

    @property
    def marker_pair(self) -> MarkerPair:
        return (self.marker_a, self.marker_b)


@dataclass(frozen=True)
class _MarkerCandidate:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray
    reprojection_rms_px: float


@dataclass(frozen=True)
class _PairConsensus:
    marker_a: int
    marker_b: int
    rotation_ba: np.ndarray
    translation_ba: np.ndarray
    inlier_frames: tuple[int, ...]
    inlier_hypotheses: dict[int, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class _CornerObservation:
    frame_index: int
    marker_id: int
    corner_index: int
    image_point: np.ndarray


def _validate_settings(settings: CalibrationSettings) -> str | None:
    if settings.min_inliers_per_edge <= 0:
        return "CalibrationSettings.min_inliers_per_edge must be positive."
    if settings.max_ba_iterations <= 0:
        return "CalibrationSettings.max_ba_iterations must be positive."
    positive_float_fields = (
        ("reprojection_rms_gate_px", settings.reprojection_rms_gate_px),
        ("pair_translation_rms_gate_ratio", settings.pair_translation_rms_gate_ratio),
        ("pair_rotation_rms_gate_deg", settings.pair_rotation_rms_gate_deg),
        ("huber_delta_px", settings.huber_delta_px),
        ("corner_outlier_px", settings.corner_outlier_px),
    )
    for field_name, value in positive_float_fields:
        if not np.isfinite(value) or value <= 0.0:
            return f"CalibrationSettings.{field_name} must be finite and positive."
    return None


def parse_marker_id_spec(tokens: Sequence[str]) -> tuple[list[int] | None, str | None]:
    """Parse marker ID CLI tokens, expanding inclusive ranges such as ``3-10``."""
    if not tokens:
        return None, "must not be empty."

    marker_ids: list[int] = []
    seen: set[int] = set()
    duplicates: set[int] = set()
    for token in tokens:
        token = str(token).strip()
        if not token:
            return None, "must not contain empty tokens."
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text or not end_text:
                return None, f"invalid range token {token!r}."
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                return None, f"range token must use integers: {token!r}."
            if start > end:
                return None, f"range token must be ascending: {token!r}."
            values = range(start, end + 1)
        else:
            try:
                values = [int(token)]
            except ValueError:
                return None, f"token must be an integer or range: {token!r}."
        for marker_id in values:
            if marker_id in seen:
                duplicates.add(marker_id)
            seen.add(marker_id)
            marker_ids.append(marker_id)

    if duplicates:
        return None, f"contains duplicates: {sorted(duplicates)}."
    return sorted(marker_ids), None


def _parse_expected_marker_ids(
    expected_marker_ids: Sequence[int],
    reference_marker_id: int,
) -> tuple[list[int] | None, str | None]:
    try:
        marker_ids = [int(marker_id) for marker_id in expected_marker_ids]
    except (TypeError, ValueError):
        return None, "expected_marker_ids must contain integer marker IDs."
    if not marker_ids:
        return None, "expected_marker_ids is empty."

    seen: set[int] = set()
    duplicates: set[int] = set()
    for marker_id in marker_ids:
        if marker_id in seen:
            duplicates.add(marker_id)
        seen.add(marker_id)
    if duplicates:
        return None, f"expected_marker_ids contains duplicates: {sorted(duplicates)}."

    expected_ids = sorted(marker_ids)
    if int(reference_marker_id) not in seen:
        return None, f"reference_marker_id {reference_marker_id} is not in expected_marker_ids."
    return expected_ids, None


def parse_anchor_marker_ids(
    anchor_marker_ids: Sequence[int] | None,
    expected_ids: Sequence[int],
    reference_marker_id: int,
) -> tuple[tuple[int, ...] | None, str | None]:
    """Validate explicit anchor-core marker IDs against the expected layout."""
    if anchor_marker_ids is None:
        return None, None
    try:
        anchors = [int(marker_id) for marker_id in anchor_marker_ids]
    except (TypeError, ValueError):
        return None, "anchor_marker_ids must contain integer marker IDs."
    if len(anchors) < 2:
        return None, "anchor_marker_ids must contain at least two marker IDs."
    if len(anchors) != len(set(anchors)):
        duplicates = sorted({marker_id for marker_id in anchors if anchors.count(marker_id) > 1})
        return None, f"anchor_marker_ids contains duplicates: {duplicates}."
    expected_set = set(int(marker_id) for marker_id in expected_ids)
    missing = sorted(set(anchors) - expected_set)
    if missing:
        return None, f"anchor_marker_ids are not subset of expected_marker_ids; extra {missing}."
    if int(reference_marker_id) not in anchors:
        return None, f"reference_marker_id {reference_marker_id} must appear in anchor_marker_ids."
    return tuple(sorted(anchors)), None


def _validate_marker_size(marker_size_m: float) -> str | None:
    if not np.isfinite(marker_size_m) or marker_size_m <= 0.0:
        return "marker_size_m must be a finite positive number."
    return None


def uniform_marker_sizes(expected_ids: Sequence[int], marker_size_m: float) -> dict[int, float]:
    return {int(marker_id): float(marker_size_m) for marker_id in expected_ids}


def _validate_marker_sizes(
    marker_sizes_m: Mapping[int, float],
    expected_ids: Sequence[int],
) -> str | None:
    expected_set = {int(marker_id) for marker_id in expected_ids}
    if set(marker_sizes_m) != expected_set:
        missing = sorted(expected_set - set(marker_sizes_m))
        extra = sorted(set(marker_sizes_m) - expected_set)
        if missing:
            return f"marker_sizes_m missing expected marker IDs: {missing}."
        return f"marker_sizes_m contains unexpected marker IDs: {extra}."
    for marker_id, size in marker_sizes_m.items():
        failure = _validate_marker_size(size)
        if failure is not None:
            return f"marker_sizes_m[{marker_id}]: {failure}"
    return None


def _object_points_by_marker(marker_sizes_m: Mapping[int, float]) -> dict[int, np.ndarray]:
    by_size: dict[float, np.ndarray] = {}
    result: dict[int, np.ndarray] = {}
    for marker_id, size in marker_sizes_m.items():
        if size not in by_size:
            by_size[size] = marker_corner_object_points(size).astype(np.float64)
        result[int(marker_id)] = by_size[size]
    return result


def _parse_marker_id_range_token(token: str) -> tuple[list[int] | None, str | None]:
    token = str(token).strip()
    if not token:
        return None, "must not be empty."
    if "-" in token:
        start_text, end_text = token.split("-", 1)
        if not start_text or not end_text:
            return None, f"invalid range token {token!r}."
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            return None, f"range token must use integers: {token!r}."
        if start > end:
            return None, f"range token must be ascending: {token!r}."
        return list(range(start, end + 1)), None
    try:
        return [int(token)], None
    except ValueError:
        return None, f"token must be an integer or range: {token!r}."


def parse_marker_size_override_spec(
    tokens: Sequence[str],
) -> tuple[list[tuple[list[int], float]] | None, str | None]:
    if not tokens:
        return [], None
    overrides: list[tuple[list[int], float]] = []
    used_ids: set[int] = set()
    for token in tokens:
        token = str(token).strip()
        if ":" not in token:
            return None, f"invalid override token {token!r}; expected ID_OR_RANGE:SIZE."
        id_part, size_part = token.rsplit(":", 1)
        if not id_part or not size_part:
            return None, f"invalid override token {token!r}."
        marker_ids, parse_failure = _parse_marker_id_range_token(id_part)
        if parse_failure is not None:
            return None, parse_failure
        assert marker_ids is not None
        try:
            size = float(size_part)
        except ValueError:
            return None, f"override size must be a number in {token!r}."
        if not np.isfinite(size) or size <= 0.0:
            return None, f"override size must be positive and finite in {token!r}."
        overlap = sorted(set(marker_ids) & used_ids)
        if overlap:
            return None, f"overlapping marker size overrides for IDs {overlap}."
        used_ids.update(marker_ids)
        overrides.append((marker_ids, size))
    return overrides, None


def resolve_marker_sizes_for_calibration(
    expected_ids: Sequence[int],
    default_size_m: float,
    override_tokens: Sequence[str] | None = None,
) -> tuple[dict[int, float] | None, str | None]:
    default_failure = _validate_marker_size(default_size_m)
    if default_failure is not None:
        return None, default_failure
    overrides: dict[int, float] = {}
    if override_tokens:
        parsed, parse_failure = parse_marker_size_override_spec(override_tokens)
        if parse_failure is not None:
            return None, parse_failure
        assert parsed is not None
        for marker_ids, size in parsed:
            for marker_id in marker_ids:
                overrides[marker_id] = size
        unknown = sorted(set(overrides) - {int(marker_id) for marker_id in expected_ids})
        if unknown:
            return None, f"marker size overrides are not subset of marker IDs; extra {unknown}."
    try:
        resolved = resolve_marker_sizes(set(expected_ids), default_size_m, overrides)
    except ValueError as exc:
        return None, str(exc)
    sizes_failure = _validate_marker_sizes(resolved, expected_ids)
    if sizes_failure is not None:
        return None, sizes_failure
    return resolved, None


def _validate_camera_inputs(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    try:
        matrix = np.asarray(camera_matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, "camera_matrix must be a numeric 3x3 matrix."
    if matrix.shape != (3, 3):
        return None, None, f"camera_matrix must have shape (3, 3), got {matrix.shape}."
    if not np.all(np.isfinite(matrix)):
        return None, None, "camera_matrix must contain only finite values."

    try:
        distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    except (TypeError, ValueError):
        return None, None, "dist_coeffs must be numeric."
    if distortion.size == 0:
        return None, None, "dist_coeffs must not be empty."
    if not np.all(np.isfinite(distortion)):
        return None, None, "dist_coeffs must contain only finite values."
    return matrix, distortion, None


def _parse_marker_corners(
    corners: np.ndarray,
    frame_id: str | int,
    marker_id: int,
) -> tuple[np.ndarray | None, str | None]:
    try:
        array = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError):
        return None, (
            f"Malformed corners for frame {frame_id!r}, marker {marker_id}: "
            "corners must be numeric."
        )
    if array.size != 8:
        return None, (
            f"Malformed corners for frame {frame_id!r}, marker {marker_id}: "
            f"expected 4x2 corners, got {array.size} values."
        )
    array = array.reshape(4, 2)
    if array.shape != (4, 2) or not np.all(np.isfinite(array)):
        return None, (
            f"Malformed corners for frame {frame_id!r}, marker {marker_id}: "
            "corners must be finite 4x2 points."
        )
    return array, None


def _validate_observations(
    observations: Sequence[FrameObservation],
    expected_ids: list[int],
) -> str | None:
    expected_set = set(expected_ids)
    seen_frame_ids: set[str | int] = set()
    for observation in observations:
        if observation.frame_id in seen_frame_ids:
            return f"Duplicate FrameObservation.frame_id: {observation.frame_id!r}."
        seen_frame_ids.add(observation.frame_id)
        for marker_id, corners in observation.markers.items():
            marker_id = int(marker_id)
            if marker_id not in expected_set:
                continue
            _, failure = _parse_marker_corners(corners, observation.frame_id, marker_id)
            if failure is not None:
                return failure
    return None


def calibrate_marker_layout(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    expected_marker_ids: Sequence[int],
    reference_marker_id: int,
    marker_size_m: float,
    settings: CalibrationSettings | None = None,
    anchor_marker_ids: Sequence[int] | None = None,
    anchor_stop_after_expansion: bool = False,
    marker_sizes_m: Mapping[int, float] | None = None,
    best_effort: bool = False,
) -> CalibrationResult:
    """Estimate a connected marker layout or refuse with a structured reason."""
    settings = settings or CalibrationSettings()
    settings_failure = _validate_settings(settings)
    if settings_failure is not None:
        return CalibrationResult(None, None, settings_failure)

    expected_ids, expected_failure = _parse_expected_marker_ids(
        expected_marker_ids,
        reference_marker_id,
    )
    if expected_failure is not None:
        return CalibrationResult(None, None, expected_failure)

    anchor_ids, anchor_failure = parse_anchor_marker_ids(
        anchor_marker_ids,
        expected_ids,
        reference_marker_id,
    )
    if anchor_failure is not None:
        return CalibrationResult(None, None, anchor_failure)

    if anchor_stop_after_expansion and anchor_ids is None:
        return CalibrationResult(
            None,
            None,
            "--anchor-stop-after-expansion requires explicit anchor_marker_ids.",
        )

    marker_size_failure = _validate_marker_size(marker_size_m)
    if marker_size_failure is not None:
        return CalibrationResult(None, None, marker_size_failure)

    if marker_sizes_m is None:
        marker_sizes_m = uniform_marker_sizes(expected_ids, marker_size_m)
    else:
        sizes_failure = _validate_marker_sizes(marker_sizes_m, expected_ids)
        if sizes_failure is not None:
            return CalibrationResult(None, None, sizes_failure)

    camera_matrix, dist_coeffs, camera_failure = _validate_camera_inputs(
        camera_matrix,
        dist_coeffs,
    )
    if camera_failure is not None:
        return CalibrationResult(None, None, camera_failure)

    observations_failure = _validate_observations(observations, expected_ids)
    if observations_failure is not None:
        return CalibrationResult(None, None, observations_failure)

    object_points_by_marker = _object_points_by_marker(marker_sizes_m)

    normalized_observations = _normalize_observations(observations, expected_ids)
    if not normalized_observations:
        missing = frozenset(expected_ids)
        return CalibrationResult(
            None,
            _empty_quality(missing, frozenset(), input_frame_count=len(observations)),
            f"No usable observations for expected marker IDs; missing {sorted(missing)}.",
        )

    observed_ids = {
        marker_id
        for _, markers in normalized_observations
        for marker_id in markers
    }
    never_observed = sorted(set(expected_ids) - observed_ids)
    if never_observed:
        return CalibrationResult(
            None,
            _empty_quality(
                frozenset(never_observed),
                frozenset(observed_ids),
                input_frame_count=len(normalized_observations),
            ),
            f"Expected marker IDs never observed: {never_observed}.",
        )

    frame_candidates = _estimate_frame_candidates(
        normalized_observations,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
    )
    if not frame_candidates:
        return CalibrationResult(
            None,
            _empty_quality(
                frozenset(expected_ids),
                frozenset(),
                input_frame_count=len(normalized_observations),
            ),
            "No valid IPPE marker poses found in any frame.",
        )

    pair_hypotheses = _collect_pair_hypotheses(frame_candidates, expected_ids)
    raw_pair_counts = _raw_covisible_pair_counts(normalized_observations)
    raw_connected = _connected_marker_ids_from_pairs(raw_pair_counts.keys(), reference_marker_id)
    raw_missing = sorted(set(expected_ids) - raw_connected)
    if raw_missing:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                {},
                expected_ids,
                reference_marker_id,
                frozenset(raw_missing),
                input_frame_count=len(normalized_observations),
                rejected_frame_count=0,
                accepted_frame_count=0,
                observation_count=0,
            ),
            (
                f"Expected marker IDs are not connected in raw observations; "
                f"missing {raw_missing}."
            ),
        )

    restored_pair_edges: list[RestoredPairEdge] = []
    use_legacy_assignment = anchor_ids is None or set(anchor_ids) == set(expected_ids)
    anchor_core_diagnostics: AnchorCoreDiagnostics | None = None
    preinitialized_marker_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None

    if use_legacy_assignment:
        pair_consensus, pair_failure, dropped_pair_edges = _estimate_pair_consensus(
            pair_hypotheses,
            expected_ids,
            reference_marker_id,
            marker_sizes_m,
            settings,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )
    else:
        assert anchor_ids is not None
        (
            anchor_assigned,
            rejected_frames,
            assignment_rejections,
            pair_consensus,
            preinitialized_marker_poses,
            anchor_drops,
            anchor_core_diagnostics,
            anchor_failure,
        ) = _assign_and_initialize_anchor_core(
            frame_candidates,
            pair_hypotheses,
            normalized_observations,
            expected_ids,
            anchor_ids,
            reference_marker_id,
            marker_sizes_m,
            settings,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
            stop_after_expansion=anchor_stop_after_expansion,
            best_effort=best_effort,
            restored_pair_edges=restored_pair_edges,
        )
        dropped_edges = list(anchor_drops)
        if anchor_failure is not None or pair_consensus is None or anchor_assigned is None:
            assignment_rejection_records = build_assignment_rejection_records(
                normalized_observations,
                rejected_frames,
                assignment_rejections,
            )
            assignment_rejection_summary = summarize_assignment_rejection_records(
                assignment_rejection_records
            )
            quality = _quality_from_pairs(
                pair_consensus or {},
                expected_ids,
                reference_marker_id,
                _missing_from_graph(pair_consensus or {}, expected_ids, reference_marker_id),
                input_frame_count=len(normalized_observations),
                rejected_frame_count=len(rejected_frames),
                accepted_frame_count=len(anchor_assigned or {}),
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                anchor_core=anchor_core_diagnostics,
            )
            return CalibrationResult(None, quality, anchor_failure)
        assigned_candidates = anchor_assigned
        assignment_rejection_records = build_assignment_rejection_records(
            normalized_observations,
            rejected_frames,
            assignment_rejections,
        )
        assignment_rejection_summary = summarize_assignment_rejection_records(
            assignment_rejection_records
        )
        if anchor_stop_after_expansion:
            assert preinitialized_marker_poses is not None
            assert pair_consensus is not None
            marker_poses = preinitialized_marker_poses
            footprints = _footprints_from_poses(marker_poses, marker_sizes_m)
            if set(footprints) != set(expected_ids):
                absent = sorted(set(expected_ids) - set(footprints))
                return CalibrationResult(
                    None,
                    _quality_from_pairs(
                        pair_consensus,
                        expected_ids,
                        reference_marker_id,
                        frozenset(absent),
                        input_frame_count=len(normalized_observations),
                        rejected_frame_count=len(rejected_frames),
                        accepted_frame_count=len(
                            {
                                frame_index
                                for frame_index, assignment in assigned_candidates.items()
                                if len(assignment) >= 2
                            }
                        ),
                        observation_count=0,
                        assignment_rejections=assignment_rejection_summary,
                        assignment_rejection_records=assignment_rejection_records,
                        dropped_pair_edges=tuple(dropped_edges),
                        restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                        anchor_core=anchor_core_diagnostics,
                    ),
                    (
                        "Expansion-only layout did not produce all expected marker "
                        f"footprints; missing {absent}."
                    ),
                )
            layout = build_marker_layout(
                reference_marker_id=reference_marker_id,
                marker_size_m=marker_size_m,
                footprints=footprints,
                marker_sizes_m=dict(marker_sizes_m),
            )
            quality = _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                _missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=len(normalized_observations),
                rejected_frame_count=len(rejected_frames),
                accepted_frame_count=len(
                    {
                        frame_index
                        for frame_index, assignment in assigned_candidates.items()
                        if len(assignment) >= 2
                    }
                ),
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                anchor_core=anchor_core_diagnostics,
            )
            return CalibrationResult(layout, quality, None)
        input_frame_count = len(normalized_observations)
        rejected_frame_count = len(rejected_frames)
        accepted_frames = frozenset(
            frame_index
            for frame_index, assignment in assigned_candidates.items()
            if len(assignment) >= 2
        )
        accepted_frame_count = len(accepted_frames)
        pair_consensus, assignment_support_failure, assignment_drops = (
            _restrict_pair_consensus_to_frames(
                pair_consensus,
                accepted_frames,
                expected_ids,
                reference_marker_id,
                settings,
                marker_sizes_m=marker_sizes_m,
                best_effort=best_effort,
                restored_pair_edges=restored_pair_edges,
            )
        )
        dropped_edges.extend(assignment_drops)
        if assignment_support_failure is not None:
            return CalibrationResult(
                None,
                _quality_from_pairs(
                    pair_consensus,
                    expected_ids,
                    reference_marker_id,
                    _missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                    input_frame_count=input_frame_count,
                    rejected_frame_count=rejected_frame_count,
                    accepted_frame_count=accepted_frame_count,
                    observation_count=0,
                    assignment_rejections=assignment_rejection_summary,
                    assignment_rejection_records=assignment_rejection_records,
                    dropped_pair_edges=tuple(dropped_edges),
                    restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                    anchor_core=anchor_core_diagnostics,
                ),
                assignment_support_failure,
            )
        markers_in_accepted_frames = _markers_in_frame_indices(
            normalized_observations,
            accepted_frames,
        )
        missing_after_rejection = sorted(set(expected_ids) - markers_in_accepted_frames)
        if missing_after_rejection:
            return CalibrationResult(
                None,
                _quality_from_pairs(
                    pair_consensus,
                    expected_ids,
                    reference_marker_id,
                    frozenset(missing_after_rejection),
                    input_frame_count=input_frame_count,
                    rejected_frame_count=rejected_frame_count,
                    accepted_frame_count=accepted_frame_count,
                    observation_count=0,
                    assignment_rejections=assignment_rejection_summary,
                    assignment_rejection_records=assignment_rejection_records,
                    dropped_pair_edges=tuple(dropped_edges),
                    restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                    anchor_core=anchor_core_diagnostics,
                ),
                (
                    "Expected marker IDs have no accepted-frame observations after "
                    f"anchor-core expansion: {missing_after_rejection}."
                ),
            )
        assert preinitialized_marker_poses is not None
        marker_poses = preinitialized_marker_poses
        frame_poses = _initialize_frame_poses(
            assigned_candidates,
            marker_poses,
            len(normalized_observations),
        )
        corner_observations = _build_corner_observations(normalized_observations, expected_ids)
        inlier_mask = _mask_corner_observations_for_frames(corner_observations, accepted_frames)
        non_reference_ids = [
            marker_id for marker_id in expected_ids if marker_id != reference_marker_id
        ]
        marker_poses, frame_poses, inlier_mask, ba_failure = _run_bundle_adjustment(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            reference_marker_id,
            non_reference_ids,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
            settings,
        )
        if ba_failure is not None:
            missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
            return CalibrationResult(
                None,
                _quality_from_pairs(
                    pair_consensus,
                    expected_ids,
                    reference_marker_id,
                    missing,
                    input_frame_count=input_frame_count,
                    rejected_frame_count=rejected_frame_count,
                    accepted_frame_count=accepted_frame_count,
                    observation_count=len(corner_observations),
                    assignment_rejections=assignment_rejection_summary,
                    assignment_rejection_records=assignment_rejection_records,
                    dropped_pair_edges=tuple(dropped_edges),
                    restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                    anchor_core=anchor_core_diagnostics,
                ),
                ba_failure,
            )
        assignment_hypotheses = _collect_assignment_pair_hypotheses(
            assigned_candidates,
            frozenset(expected_ids),
        )
        frozen_frames = _freeze_assigned_frame_candidates(frame_candidates, assigned_candidates)
        pair_consensus = _pair_consensus_from_assignment_hypotheses(
            _collect_pair_hypotheses(frozen_frames, expected_ids),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )
        marker_poses, frame_poses, inlier_mask, pair_consensus, prune_failure, pruning_drops = (
            _prune_and_refit(
                corner_observations,
                inlier_mask,
                marker_poses,
                frame_poses,
                reference_marker_id,
                non_reference_ids,
                expected_ids,
                pair_consensus,
                accepted_frames,
                object_points_by_marker,
                camera_matrix,
                dist_coeffs,
                settings,
                marker_sizes_m,
                best_effort=best_effort,
                restored_pair_edges=restored_pair_edges,
            )
        )
        dropped_edges.extend(pruning_drops)
        if prune_failure is not None:
            missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
            return CalibrationResult(
                None,
                _quality_from_pairs(
                    pair_consensus,
                    expected_ids,
                    reference_marker_id,
                    missing,
                    input_frame_count=input_frame_count,
                    rejected_frame_count=rejected_frame_count,
                    accepted_frame_count=_covisible_frame_count(corner_observations, inlier_mask),
                    observation_count=int(np.count_nonzero(inlier_mask)),
                    assignment_rejections=assignment_rejection_summary,
                    assignment_rejection_records=assignment_rejection_records,
                    dropped_pair_edges=tuple(dropped_edges),
                    restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
                    anchor_core=anchor_core_diagnostics,
                ),
                prune_failure,
            )
        connected_ids = _connected_marker_ids(pair_consensus, reference_marker_id)
        missing_ids = frozenset(set(expected_ids) - connected_ids)
        final_accepted_frame_count = _covisible_frame_count(corner_observations, inlier_mask)
        quality = _build_quality_report(
            corner_observations,
            inlier_mask,
            marker_poses,
            frame_poses,
            pair_consensus,
            expected_ids,
            reference_marker_id,
            missing_ids,
            input_frame_count,
            rejected_frame_count,
            final_accepted_frame_count,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
            assignment_rejections=assignment_rejection_summary,
            assignment_rejection_records=assignment_rejection_records,
            dropped_pair_edges=tuple(dropped_edges),
            restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            anchor_core=anchor_core_diagnostics,
        )
        gate_failure = _check_quality_gates(quality, settings, marker_sizes_m, expected_ids)
        finalized = _finalize_solved_calibration(
            marker_poses,
            quality,
            settings,
            marker_sizes_m,
            expected_ids,
            reference_marker_id,
            marker_size_m,
            missing_ids,
            gate_failure=gate_failure,
            best_effort=best_effort,
        )
        if finalized is not None:
            return finalized
        layout = build_marker_layout(
            reference_marker_id=reference_marker_id,
            marker_size_m=marker_size_m,
            footprints=_footprints_from_poses(marker_poses, marker_sizes_m),
            marker_sizes_m=dict(marker_sizes_m),
        )
        return _accepted_calibration_result(layout, quality, best_effort=best_effort)

    dropped_edges = list(dropped_pair_edges)
    pair_failure = pair_failure if use_legacy_assignment else None
    if pair_failure is not None:
        missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing,
                input_frame_count=len(normalized_observations),
                rejected_frame_count=0,
                accepted_frame_count=0,
                observation_count=0,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            pair_failure,
        )

    assigned_candidates, rejected_frames, assignment_rejections = _assign_ippe_candidates(
        frame_candidates,
        pair_consensus,
        settings,
        marker_sizes_m,
    )
    assignment_rejection_records = build_assignment_rejection_records(
        normalized_observations,
        rejected_frames,
        assignment_rejections,
    )
    assignment_rejection_summary = summarize_assignment_rejection_records(assignment_rejection_records)
    input_frame_count = len(normalized_observations)
    rejected_frame_count = len(rejected_frames)
    accepted_frames = frozenset(assigned_candidates)
    accepted_frame_count = len(accepted_frames)
    if accepted_frame_count == 0:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                _missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=0,
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            "No frames with assignable IPPE candidates remain after rejecting inconsistent samples.",
        )

    pair_consensus, assignment_support_failure, assignment_drops = _restrict_pair_consensus_to_frames(
        pair_consensus,
        accepted_frames,
        expected_ids,
        reference_marker_id,
        settings,
        marker_sizes_m=marker_sizes_m,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    dropped_edges.extend(assignment_drops)
    if assignment_support_failure is not None:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                _missing_from_graph(pair_consensus, expected_ids, reference_marker_id),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            assignment_support_failure,
        )

    markers_in_accepted_frames = _markers_in_frame_indices(normalized_observations, accepted_frames)
    missing_after_rejection = sorted(set(expected_ids) - markers_in_accepted_frames)
    if missing_after_rejection:
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                frozenset(missing_after_rejection),
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=0,
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            f"Expected marker IDs have no accepted-frame observations after rejection: {missing_after_rejection}.",
        )

    ref_rotation, ref_translation = _reference_gauge_pose(marker_sizes_m[reference_marker_id])
    marker_poses = _initialize_marker_poses(
        reference_marker_id,
        ref_rotation,
        ref_translation,
        expected_ids,
        pair_consensus,
    )
    frame_poses = _initialize_frame_poses(
        assigned_candidates,
        marker_poses,
        len(normalized_observations),
    )

    corner_observations = _build_corner_observations(normalized_observations, expected_ids)
    inlier_mask = _mask_corner_observations_for_frames(corner_observations, accepted_frames)
    non_reference_ids = [marker_id for marker_id in expected_ids if marker_id != reference_marker_id]

    marker_poses, frame_poses, inlier_mask, ba_failure = _run_bundle_adjustment(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_ids,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        settings,
    )
    if ba_failure is not None:
        missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing,
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=accepted_frame_count,
                observation_count=len(corner_observations),
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            ba_failure,
        )

    marker_poses, frame_poses, inlier_mask, pair_consensus, prune_failure, pruning_drops = _prune_and_refit(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_ids,
        expected_ids,
        pair_consensus,
        accepted_frames,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        settings,
        marker_sizes_m,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    dropped_edges.extend(pruning_drops)
    if prune_failure is not None:
        missing = _missing_from_graph(pair_consensus, expected_ids, reference_marker_id)
        return CalibrationResult(
            None,
            _quality_from_pairs(
                pair_consensus,
                expected_ids,
                reference_marker_id,
                missing,
                input_frame_count=input_frame_count,
                rejected_frame_count=rejected_frame_count,
                accepted_frame_count=_covisible_frame_count(corner_observations, inlier_mask),
                observation_count=int(np.count_nonzero(inlier_mask)),
                assignment_rejections=assignment_rejection_summary,
                assignment_rejection_records=assignment_rejection_records,
                dropped_pair_edges=tuple(dropped_edges),
                restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
            ),
            prune_failure,
        )

    connected_ids = _connected_marker_ids(pair_consensus, reference_marker_id)
    missing_ids = frozenset(set(expected_ids) - connected_ids)
    final_accepted_frame_count = _covisible_frame_count(corner_observations, inlier_mask)
    quality = _build_quality_report(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        pair_consensus,
        expected_ids,
        reference_marker_id,
        missing_ids,
        input_frame_count,
        rejected_frame_count,
        final_accepted_frame_count,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        assignment_rejections=assignment_rejection_summary,
        assignment_rejection_records=assignment_rejection_records,
        dropped_pair_edges=tuple(dropped_edges),
        restored_pair_edges=tuple(restored_pair_edges) if restored_pair_edges else None,
    )

    gate_failure = _check_quality_gates(quality, settings, marker_sizes_m, expected_ids)
    finalized = _finalize_solved_calibration(
        marker_poses,
        quality,
        settings,
        marker_sizes_m,
        expected_ids,
        reference_marker_id,
        marker_size_m,
        missing_ids,
        gate_failure=gate_failure,
        best_effort=best_effort,
    )
    if finalized is not None:
        return finalized
    layout = build_marker_layout(
        reference_marker_id=reference_marker_id,
        marker_size_m=marker_size_m,
        footprints=_footprints_from_poses(marker_poses, marker_sizes_m),
        marker_sizes_m=dict(marker_sizes_m),
    )
    return _accepted_calibration_result(layout, quality, best_effort=best_effort)


def _accepted_calibration_result(
    layout: MarkerLayout,
    quality: CalibrationQualityReport,
    *,
    best_effort: bool,
) -> CalibrationResult:
    return CalibrationResult(
        layout,
        quality,
        None,
        outcome="accepted",
        calibration_policy="best_effort" if best_effort else "strict",
    )


def _finalize_solved_calibration(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    quality: CalibrationQualityReport,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    expected_ids: list[int],
    reference_marker_id: int,
    marker_size_m: float,
    missing_ids: frozenset[int],
    *,
    gate_failure: str | None,
    best_effort: bool,
) -> CalibrationResult | None:
    policy: Literal["strict", "best_effort"] = "best_effort" if best_effort else "strict"
    gate_failures = _collect_quality_gate_failures(
        quality,
        settings,
        marker_sizes_m,
        expected_ids,
    )
    failed_gate_messages = tuple(failure.message for failure in gate_failures)

    if gate_failures:
        if (
            best_effort
            and all(failure.category == "strict" for failure in gate_failures)
            and not missing_ids
        ):
            footprints = _footprints_from_poses(marker_poses, marker_sizes_m)
            if set(footprints) != set(expected_ids):
                absent = sorted(set(expected_ids) - set(footprints))
                return CalibrationResult(
                    None,
                    quality,
                    f"Calibration did not produce all expected marker footprints; missing {absent}.",
                    outcome="refused",
                    calibration_policy=policy,
                    failed_quality_gates=failed_gate_messages,
                )
            layout = build_marker_layout(
                reference_marker_id=reference_marker_id,
                marker_size_m=marker_size_m,
                footprints=footprints,
                marker_sizes_m=dict(marker_sizes_m),
            )
            return CalibrationResult(
                layout,
                quality,
                None,
                outcome="provisional",
                calibration_policy=policy,
                failed_quality_gates=failed_gate_messages,
            )
        return CalibrationResult(
            None,
            quality,
            gate_failure or gate_failures[0].message,
            outcome="refused",
            calibration_policy=policy,
            failed_quality_gates=failed_gate_messages,
        )

    if missing_ids:
        return CalibrationResult(
            None,
            quality,
            f"Expected marker IDs are not connected to reference: {sorted(missing_ids)}.",
            outcome="refused",
            calibration_policy=policy,
        )

    footprints = _footprints_from_poses(marker_poses, marker_sizes_m)
    if set(footprints) != set(expected_ids):
        absent = sorted(set(expected_ids) - set(footprints))
        return CalibrationResult(
            None,
            quality,
            f"Calibration did not produce all expected marker footprints; missing {absent}.",
            outcome="refused",
            calibration_policy=policy,
        )
    return None


def _normalize_observations(
    observations: Sequence[FrameObservation],
    expected_ids: list[int],
) -> list[tuple[str | int, dict[int, np.ndarray]]]:
    expected_set = set(expected_ids)
    normalized: list[tuple[str | int, dict[int, np.ndarray]]] = []
    for observation in observations:
        markers: dict[int, np.ndarray] = {}
        for marker_id, corners in observation.markers.items():
            marker_id = int(marker_id)
            if marker_id not in expected_set:
                continue
            array, failure = _parse_marker_corners(corners, observation.frame_id, marker_id)
            if failure is not None or array is None:
                continue
            markers[marker_id] = array
        if len(markers) >= 2:
            normalized.append((observation.frame_id, markers))
    return normalized


def _estimate_frame_candidates(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[int, dict[int, list[_MarkerCandidate]]]]:
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]] = []
    for frame_index, (_, markers) in enumerate(observations):
        candidates: dict[int, list[_MarkerCandidate]] = {}
        for marker_id, image_points in markers.items():
            marker_candidates = _ippe_candidates(
                object_points_by_marker[marker_id],
                image_points,
                camera_matrix,
                dist_coeffs,
            )
            if marker_candidates:
                candidates[marker_id] = marker_candidates
        if len(candidates) >= 2:
            frame_candidates.append((frame_index, candidates))
    return frame_candidates


def _ippe_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[_MarkerCandidate]:
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or rvecs is None or tvecs is None:
        return []

    candidates: list[_MarkerCandidate] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rotation, _ = cv2.Rodrigues(rvec)
        if not _is_marker_facing_camera(rotation):
            continue
        rms = _reprojection_rms(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        candidates.append(
            _MarkerCandidate(
                rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
                tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
                rotation=rotation.astype(np.float64),
                reprojection_rms_px=rms,
            )
        )
    return candidates


def _is_marker_facing_camera(rotation: np.ndarray) -> bool:
    normal = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(normal[2]) < 0.0


def _reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3).astype(np.float32),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(image_points.reshape(-1, 2) - projected, axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def _relative_marker_transform(
    parent: _MarkerCandidate,
    child: _MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points in child marker frame to parent marker frame."""
    rotation = parent.rotation.T @ child.rotation
    translation = parent.rotation.T @ (child.tvec - parent.tvec)
    return rotation, translation


def _transform_high_in_low(
    low: _MarkerCandidate,
    high: _MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points in the high marker frame to the low marker frame."""
    return _relative_marker_transform(low, high)


def _pair_translation_gate(
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    pair: MarkerPair,
) -> float:
    return settings.pair_translation_rms_gate_ratio * min(
        marker_sizes_m[pair[0]],
        marker_sizes_m[pair[1]],
    )


def compute_live_pair_readiness(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    expected_marker_ids: Sequence[int],
    reference_marker_id: int,
    settings: CalibrationSettings | None = None,
) -> LivePairReadinessDiagnostics:
    """Estimate co-visibility pair strength and graph readiness (topology only)."""
    settings = settings or CalibrationSettings()
    settings_failure = _validate_settings(settings)
    if settings_failure is not None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(),
            sample_count=len(observations),
            failure_reason=settings_failure,
        )

    expected_ids, expected_failure = _parse_expected_marker_ids(
        expected_marker_ids,
        reference_marker_id,
    )
    if expected_failure is not None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(),
            sample_count=len(observations),
            failure_reason=expected_failure,
        )

    camera_matrix, dist_coeffs, camera_failure = _validate_camera_inputs(
        camera_matrix,
        dist_coeffs,
    )
    if camera_failure is not None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(),
            sample_count=len(observations),
            failure_reason=camera_failure,
        )

    observations_failure = _validate_observations(observations, expected_ids)
    if observations_failure is not None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(),
            sample_count=len(observations),
            failure_reason=observations_failure,
        )

    normalized_observations = _normalize_observations(observations, expected_ids)
    raw_pair_counts = _raw_covisible_pair_counts(normalized_observations)
    if not raw_pair_counts:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset({reference_marker_id}),
            missing_marker_ids=frozenset(expected_ids) - {reference_marker_id},
            sample_count=len(observations),
        )

    pair_reports: list[PairReadinessEdge] = []
    passing_pairs: list[MarkerPair] = []
    for pair, raw_count in sorted(raw_pair_counts.items()):
        status = "pass" if raw_count >= settings.min_inliers_per_edge else "weak"
        if status == "pass":
            passing_pairs.append(pair)
        pair_reports.append(
            PairReadinessEdge(
                marker_a=pair[0],
                marker_b=pair[1],
                raw_covisible_frames=raw_count,
                robust_inlier_count=raw_count,
                translation_rms_m=None,
                rotation_rms_deg=None,
                status=status,
            )
        )

    connected = frozenset(
        _connected_marker_ids_from_pairs(passing_pairs, reference_marker_id)
    )
    missing = frozenset(set(expected_ids) - connected)
    return LivePairReadinessDiagnostics(
        pairs=tuple(pair_reports),
        connected_marker_ids=connected,
        missing_marker_ids=missing,
        sample_count=len(observations),
    )


def _connected_marker_ids_from_pairs(
    pairs: Iterable[MarkerPair],
    reference_marker_id: int,
) -> set[int]:
    graph: dict[int, set[int]] = {}
    for marker_a, marker_b in pairs:
        graph.setdefault(marker_a, set()).add(marker_b)
        graph.setdefault(marker_b, set()).add(marker_a)

    connected = {reference_marker_id}
    queue = [reference_marker_id]
    while queue:
        current = queue.pop()
        for neighbor in graph.get(current, set()):
            if neighbor not in connected:
                connected.add(neighbor)
                queue.append(neighbor)
    return connected


def _raw_covisible_pair_counts(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
) -> dict[MarkerPair, int]:
    counts: dict[MarkerPair, int] = {}
    for _, markers in observations:
        marker_ids = sorted(markers)
        for index_a, marker_low in enumerate(marker_ids):
            for marker_high in marker_ids[index_a + 1 :]:
                counts[(marker_low, marker_high)] = counts.get((marker_low, marker_high), 0) + 1
    return counts


def _best_pair_consensus(
    pair: MarkerPair,
    hypotheses: list[tuple[np.ndarray, np.ndarray, int]],
    translation_gate: float,
    rotation_gate: float,
) -> _PairConsensus | None:
    if not hypotheses:
        return None

    best_frames: dict[int, int] = {}
    best_rotation = np.eye(3, dtype=np.float64)
    best_translation = np.zeros(3, dtype=np.float64)
    for seed_index, (seed_rotation, seed_translation, _) in enumerate(hypotheses):
        candidate_frames = _inlier_frames_for_seed(
            hypotheses,
            seed_index,
            translation_gate,
            rotation_gate,
        )
        if len(candidate_frames) > len(best_frames):
            best_frames = candidate_frames
            best_rotation = seed_rotation
            best_translation = seed_translation

    if not best_frames:
        return None

    selected_hypotheses = {
        frame_index: hypotheses[hypothesis_index][:2]
        for frame_index, hypothesis_index in best_frames.items()
    }
    inlier_rotations = [values[0] for values in selected_hypotheses.values()]
    inlier_translations = np.stack([values[1] for values in selected_hypotheses.values()], axis=0)
    best_rotation = _average_rotations(inlier_rotations)
    best_translation = np.mean(inlier_translations, axis=0)
    return _PairConsensus(
        marker_a=pair[0],
        marker_b=pair[1],
        rotation_ba=best_rotation,
        translation_ba=best_translation,
        inlier_frames=tuple(sorted(selected_hypotheses)),
        inlier_hypotheses=selected_hypotheses,
    )


def _classify_pair_readiness(
    edge: _PairConsensus | None,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    pair: MarkerPair,
) -> tuple[str, int, float | None, float | None]:
    robust_count = len(edge.inlier_frames) if edge is not None else 0
    if edge is None or robust_count < settings.min_inliers_per_edge:
        return "weak", robust_count, None, None

    diagnostics = _edge_diagnostics(pair, edge)
    translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
    rotation_gate = settings.pair_rotation_rms_gate_deg
    if (
        diagnostics.translation_rms_m <= translation_gate
        and diagnostics.rotation_rms_deg <= rotation_gate
    ):
        return (
            "pass",
            diagnostics.inlier_count,
            diagnostics.translation_rms_m,
            diagnostics.rotation_rms_deg,
        )
    return (
        "fail",
        diagnostics.inlier_count,
        diagnostics.translation_rms_m,
        diagnostics.rotation_rms_deg,
    )


def _collect_pair_hypotheses(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    expected_ids: list[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    expected_set = set(expected_ids)
    for frame_index, candidates in frame_candidates:
        marker_ids = sorted(marker_id for marker_id in candidates if marker_id in expected_set)
        for index_a, marker_low in enumerate(marker_ids):
            for marker_high in marker_ids[index_a + 1 :]:
                pair = (marker_low, marker_high)
                for candidate_low in candidates[marker_low]:
                    for candidate_high in candidates[marker_high]:
                        rotation_ba, translation_ba = _transform_high_in_low(
                            candidate_low,
                            candidate_high,
                        )
                        hypotheses.setdefault(pair, []).append(
                            (rotation_ba, translation_ba, frame_index)
                        )
    return hypotheses


def _estimate_pair_consensus(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    expected_ids: list[int],
    reference_marker_id: int,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    *,
    connectivity_ids: Sequence[int] | None = None,
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[dict[MarkerPair, _PairConsensus], str | None, tuple[DroppedPairEdge, ...]]:
    rotation_gate = settings.pair_rotation_rms_gate_deg
    consensus: dict[MarkerPair, _PairConsensus] = {}
    weak_pool: dict[MarkerPair, _PairConsensus] = {}
    dropped: list[DroppedPairEdge] = []

    for pair, hypotheses in pair_hypotheses.items():
        translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
        unique_frames = {frame_index for _, _, frame_index in hypotheses}
        observed_count = len(unique_frames)
        edge = _best_pair_consensus(pair, hypotheses, translation_gate, rotation_gate)
        if edge is not None:
            weak_pool[pair] = edge
        if observed_count < settings.min_inliers_per_edge:
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "initial_consensus",
                    "insufficient_observed_frames",
                    observed_count=observed_count,
                    supported_count=observed_count,
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue

        if edge is None or len(edge.inlier_frames) < settings.min_inliers_per_edge:
            supported_count = len(edge.inlier_frames) if edge is not None else 0
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "initial_consensus",
                    "insufficient_inlier_frames",
                    observed_count=observed_count,
                    supported_count=supported_count,
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        consensus[pair] = edge

    filtered: dict[MarkerPair, _PairConsensus] = {}
    for pair, edge in consensus.items():
        translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
        diagnostics = _edge_diagnostics(pair, edge)
        if diagnostics.inlier_count < settings.min_inliers_per_edge:
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "initial_consensus",
                    "insufficient_inlier_frames",
                    observed_count=len(edge.inlier_frames),
                    supported_count=diagnostics.inlier_count,
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        if diagnostics.translation_rms_m > translation_gate:
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "initial_consensus",
                    "translation_rms_gate",
                    observed_count=len(edge.inlier_frames),
                    supported_count=diagnostics.inlier_count,
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        if diagnostics.rotation_rms_deg > rotation_gate:
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "initial_consensus",
                    "rotation_rms_gate",
                    observed_count=len(edge.inlier_frames),
                    supported_count=diagnostics.inlier_count,
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        filtered[pair] = edge

    required_ids = list(connectivity_ids) if connectivity_ids is not None else expected_ids
    failure = _maybe_restore_weak_connectivity(
        filtered,
        weak_pool,
        dropped,
        required_ids,
        reference_marker_id,
        "initial_consensus",
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    if failure is not None:
        return filtered, failure, tuple(dropped)

    return filtered, None, tuple(dropped)


def _inlier_frames_for_seed(
    hypotheses: list[tuple[np.ndarray, np.ndarray, int]],
    seed_index: int,
    translation_gate: float,
    rotation_gate: float,
) -> dict[int, int]:
    seed_rotation, seed_translation, _ = hypotheses[seed_index]
    inlier_frames: dict[int, int] = {}
    for hypothesis_index, (rotation, translation, frame_index) in enumerate(hypotheses):
        if (
            np.linalg.norm(translation - seed_translation) > translation_gate
            or _rotation_geodesic_deg(rotation, seed_rotation) > rotation_gate
        ):
            continue
        current_index = inlier_frames.get(frame_index)
        if current_index is None:
            inlier_frames[frame_index] = hypothesis_index
            continue
        current_rotation, current_translation, _ = hypotheses[current_index]
        current_cost = (
            np.linalg.norm(current_translation - seed_translation) / max(translation_gate, 1e-9)
            + _rotation_geodesic_deg(current_rotation, seed_rotation) / max(rotation_gate, 1e-9)
        )
        candidate_cost = (
            np.linalg.norm(translation - seed_translation) / max(translation_gate, 1e-9)
            + _rotation_geodesic_deg(rotation, seed_rotation) / max(rotation_gate, 1e-9)
        )
        if candidate_cost < current_cost:
            inlier_frames[frame_index] = hypothesis_index
    return inlier_frames


def _connected_marker_ids(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    reference_marker_id: int,
) -> set[int]:
    graph: dict[int, set[int]] = {}
    for marker_a, marker_b in pair_consensus:
        graph.setdefault(marker_a, set()).add(marker_b)
        graph.setdefault(marker_b, set()).add(marker_a)

    connected = {reference_marker_id}
    queue = [reference_marker_id]
    while queue:
        current = queue.pop()
        for neighbor in graph.get(current, set()):
            if neighbor not in connected:
                connected.add(neighbor)
                queue.append(neighbor)
    return connected


def _missing_from_graph(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
) -> frozenset[int]:
    connected = _connected_marker_ids(pair_consensus, reference_marker_id)
    return frozenset(set(expected_ids) - connected)


def _assign_ippe_candidates(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    search_marker_ids: frozenset[int] | None = None,
) -> tuple[
    dict[int, dict[int, _MarkerCandidate]],
    tuple[int, ...],
    tuple[FrameAssignmentRejection, ...],
]:
    assigned: dict[int, dict[int, _MarkerCandidate]] = {}
    rejected_frames: list[int] = []
    rejections: list[FrameAssignmentRejection] = []
    for frame_index, candidates in frame_candidates:
        result = resolve_frame_ippe_assignment(
            candidates,
            pair_consensus,
            settings,
            marker_sizes_m,
            search_marker_ids=search_marker_ids,
        )
        if result.assignment is None:
            rejected_frames.append(frame_index)
            rejections.append(
                result.rejection
                or FrameAssignmentRejection(reason="no_constrained_pair")
            )
            continue
        assigned[frame_index] = result.assignment
    return assigned, tuple(rejected_frames), tuple(rejections)


def resolve_frame_ippe_assignment(
    candidates: dict[int, list[_MarkerCandidate]],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    search_marker_ids: frozenset[int] | None = None,
) -> FrameAssignmentResult:
    if search_marker_ids is not None:
        marker_ids = sorted(marker_id for marker_id in candidates if marker_id in search_marker_ids)
    else:
        marker_ids = sorted(candidates)
    if len(marker_ids) < 2:
        return FrameAssignmentResult(
            assignment=None,
            rejection=FrameAssignmentRejection(reason="insufficient_anchors_visible"),
        )
    holder = _new_assignment_search_holder(settings, marker_sizes_m)
    _search_assignments(marker_ids, candidates, pair_consensus, {}, 0, holder)
    assignment = holder["assignment"]
    if assignment is not None:
        return FrameAssignmentResult(assignment=dict(assignment), rejection=None)
    return FrameAssignmentResult(
        assignment=None,
        rejection=_rejection_from_assignment_search_holder(holder),
    )


def _new_assignment_search_holder(
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
) -> dict[str, object]:
    return {
        "score": float("-inf"),
        "assignment": None,
        "saw_constrained_pair": False,
        "worst_key": None,
        "worst_pair": None,
        "worst_translation_error": 0.0,
        "worst_rotation_error": 0.0,
        "worst_translation_fail": False,
        "worst_rotation_fail": False,
        "worst_translation_gate": 0.0,
        "marker_sizes_m": dict(marker_sizes_m),
        "settings": settings,
        "rotation_gate": settings.pair_rotation_rms_gate_deg,
    }


def _rejection_from_assignment_search_holder(holder: dict[str, object]) -> FrameAssignmentRejection:
    if not holder["saw_constrained_pair"]:
        return FrameAssignmentRejection(reason="no_constrained_pair")

    worst_pair = holder["worst_pair"]
    if worst_pair is None:
        return FrameAssignmentRejection(reason="no_constrained_pair")

    marker_sizes_m = holder["marker_sizes_m"]
    assert isinstance(marker_sizes_m, dict)
    translation_gate = float(holder.get("worst_translation_gate", 0.0))
    rotation_gate = float(holder["rotation_gate"])
    worst_translation_error = float(holder["worst_translation_error"])
    worst_rotation_error = float(holder["worst_rotation_error"])
    worst_translation_fail = bool(holder["worst_translation_fail"])
    worst_rotation_fail = bool(holder["worst_rotation_fail"])
    if worst_translation_fail and worst_rotation_fail:
        primary_reason = (
            "translation_gate"
            if (worst_translation_error / translation_gate)
            >= (worst_rotation_error / rotation_gate)
            else "rotation_gate"
        )
    elif worst_translation_fail:
        primary_reason = "translation_gate"
    else:
        primary_reason = "rotation_gate"
    return FrameAssignmentRejection(
        reason=primary_reason,
        marker_pair=worst_pair,
        translation_error_m=worst_translation_error,
        rotation_error_deg=worst_rotation_error,
        translation_gate_m=translation_gate,
        rotation_gate_deg=rotation_gate,
    )


def _evaluate_complete_assignment(
    assignment: dict[int, _MarkerCandidate],
    marker_ids: list[int],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    rotation_gate: float,
) -> tuple[
    float | None,
    bool,
    tuple[float, MarkerPair, int] | None,
    MarkerPair | None,
    float,
    float,
    bool,
    bool,
    float,
]:
    constrained_edges = 0
    total_cost = 0.0
    assignment_valid = True
    worst_key: tuple[float, MarkerPair, int] | None = None
    worst_pair: MarkerPair | None = None
    worst_translation_error = 0.0
    worst_rotation_error = 0.0
    worst_translation_fail = False
    worst_rotation_fail = False
    worst_translation_gate = 0.0

    for index_a, marker_low in enumerate(marker_ids):
        for marker_high in marker_ids[index_a + 1 :]:
            pair = (marker_low, marker_high)
            edge = pair_consensus.get(pair)
            if edge is None:
                continue
            translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
            constrained_edges += 1
            rotation_ba, translation_ba = _transform_high_in_low(
                assignment[marker_low],
                assignment[marker_high],
            )
            translation_error = float(np.linalg.norm(translation_ba - edge.translation_ba))
            rotation_error = _rotation_geodesic_deg(rotation_ba, edge.rotation_ba)
            translation_fail = translation_error > translation_gate
            rotation_fail = rotation_error > rotation_gate
            if translation_fail or rotation_fail:
                assignment_valid = False
                translation_exceedance = translation_error / translation_gate
                rotation_exceedance = rotation_error / rotation_gate
                exceedance = max(translation_exceedance, rotation_exceedance)
                reason_rank = 0 if translation_exceedance >= rotation_exceedance else 1
                candidate_key = (exceedance, pair, reason_rank)
                if worst_key is None or candidate_key > worst_key:
                    worst_key = candidate_key
                    worst_pair = pair
                    worst_translation_error = translation_error
                    worst_rotation_error = rotation_error
                    worst_translation_fail = translation_fail
                    worst_rotation_fail = rotation_fail
                    worst_translation_gate = translation_gate
                continue
            total_cost += (translation_error / translation_gate) ** 2 + (
                rotation_error / rotation_gate
            ) ** 2

    if constrained_edges == 0 or not assignment_valid:
        return (
            None,
            constrained_edges > 0,
            worst_key,
            worst_pair,
            worst_translation_error,
            worst_rotation_error,
            worst_translation_fail,
            worst_rotation_fail,
            worst_translation_gate,
        )
    return (
        -total_cost,
        True,
        worst_key,
        worst_pair,
        worst_translation_error,
        worst_rotation_error,
        worst_translation_fail,
        worst_rotation_fail,
        worst_translation_gate,
    )


def _merge_assignment_violation_into_holder(
    holder: dict[str, object],
    has_constrained_pair: bool,
    worst_key: tuple[float, MarkerPair, int] | None,
    worst_pair: MarkerPair | None,
    worst_translation_error: float,
    worst_rotation_error: float,
    worst_translation_fail: bool,
    worst_rotation_fail: bool,
    worst_translation_gate: float,
) -> None:
    if has_constrained_pair:
        holder["saw_constrained_pair"] = True
    if worst_key is None:
        return
    current_worst_key = holder["worst_key"]
    if current_worst_key is None or worst_key > current_worst_key:
        holder["worst_key"] = worst_key
        holder["worst_pair"] = worst_pair
        holder["worst_translation_error"] = worst_translation_error
        holder["worst_rotation_error"] = worst_rotation_error
        holder["worst_translation_fail"] = worst_translation_fail
        holder["worst_rotation_fail"] = worst_rotation_fail
        holder["worst_translation_gate"] = worst_translation_gate


def summarize_assignment_rejections(
    rejections: Sequence[FrameAssignmentRejection],
) -> AssignmentRejectionSummary:
    if rejections and not isinstance(rejections[0], FrameAssignmentRejection):
        raise TypeError(
            "summarize_assignment_rejections expects FrameAssignmentRejection inputs; "
            "use summarize_assignment_rejection_records for FrameAssignmentRejectionRecord."
        )
    by_reason: dict[str, int] = {}
    by_pair: dict[MarkerPair, int] = {}
    cause_counts: dict[tuple[str, MarkerPair | None], int] = {}
    for rejection in rejections:
        by_reason[rejection.reason] = by_reason.get(rejection.reason, 0) + 1
        if rejection.marker_pair is not None:
            by_pair[rejection.marker_pair] = by_pair.get(rejection.marker_pair, 0) + 1
        cause_key = (rejection.reason, rejection.marker_pair)
        cause_counts[cause_key] = cause_counts.get(cause_key, 0) + 1
    top_causes = tuple(
        AssignmentRejectionCauseCount(reason=reason, marker_pair=pair, count=count)
        for (reason, pair), count in sorted(
            cause_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1] or (-1, -1)),
        )
    )
    return AssignmentRejectionSummary(
        total_rejected=len(rejections),
        by_reason=tuple(sorted(by_reason.items())),
        by_pair=tuple(sorted(by_pair.items())),
        top_causes=top_causes,
        by_cause=(),
    )


def summarize_assignment_rejection_records(
    records: Sequence[FrameAssignmentRejectionRecord],
) -> AssignmentRejectionSummary:
    if records and not isinstance(records[0], FrameAssignmentRejectionRecord):
        raise TypeError(
            "summarize_assignment_rejection_records expects FrameAssignmentRejectionRecord inputs; "
            "use summarize_assignment_rejections for FrameAssignmentRejection."
        )
    by_reason: dict[str, int] = {}
    by_pair: dict[MarkerPair, int] = {}
    cause_counts: dict[tuple[str, MarkerPair | None], int] = {}
    cause_groups: dict[tuple[str, MarkerPair | None], list[FrameAssignmentRejectionRecord]] = {}
    for record in records:
        by_reason[record.reason] = by_reason.get(record.reason, 0) + 1
        if record.marker_pair is not None:
            by_pair[record.marker_pair] = by_pair.get(record.marker_pair, 0) + 1
        cause_key = (record.reason, record.marker_pair)
        cause_counts[cause_key] = cause_counts.get(cause_key, 0) + 1
        cause_groups.setdefault(cause_key, []).append(record)
    top_causes = tuple(
        AssignmentRejectionCauseCount(reason=reason, marker_pair=pair, count=count)
        for (reason, pair), count in sorted(
            cause_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1] or (-1, -1)),
        )
    )
    by_cause: list[AssignmentRejectionCauseStats] = []
    for (reason, pair), group in sorted(
        cause_groups.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1] or (-1, -1)),
    ):
        sorted_group = sorted(group, key=lambda record: record.frame_index)
        translation_errors = [
            value
            for value in (record.translation_error_m for record in group)
            if value is not None
        ]
        rotation_errors = [
            value
            for value in (record.rotation_error_deg for record in group)
            if value is not None
        ]
        translation_gate = next(
            (record.translation_gate_m for record in group if record.translation_gate_m is not None),
            None,
        )
        rotation_gate = next(
            (record.rotation_gate_deg for record in group if record.rotation_gate_deg is not None),
            None,
        )
        translation_ratios = [
            error / translation_gate
            for error in translation_errors
            if translation_gate is not None and translation_gate > 0.0
        ]
        rotation_ratios = [
            error / rotation_gate
            for error in rotation_errors
            if rotation_gate is not None and rotation_gate > 0.0
        ]
        by_cause.append(
            AssignmentRejectionCauseStats(
                reason=reason,
                marker_pair=pair,
                count=len(group),
                sample_frame_ids=tuple(
                    record.frame_id for record in sorted_group[:_ASSIGNMENT_REJECTION_SAMPLE_FRAME_IDS]
                ),
                translation_error_m=_measurement_distribution(translation_errors),
                rotation_error_deg=_measurement_distribution(rotation_errors),
                translation_gate_m=_json_safe_float(translation_gate),
                rotation_gate_deg=_json_safe_float(rotation_gate),
                translation_error_ratio=_measurement_distribution(translation_ratios),
                rotation_error_ratio=_measurement_distribution(rotation_ratios),
            )
        )
    return AssignmentRejectionSummary(
        total_rejected=len(records),
        by_reason=tuple(sorted(by_reason.items())),
        by_pair=tuple(sorted(by_pair.items())),
        top_causes=top_causes,
        by_cause=tuple(by_cause),
    )


_ASSIGNMENT_REJECTION_SAMPLE_FRAME_IDS = 10


def build_assignment_rejection_records(
    normalized_observations: Sequence[tuple[str | int, dict[int, np.ndarray]]],
    rejected_frame_indices: Sequence[int],
    rejections: Sequence[FrameAssignmentRejection],
) -> tuple[FrameAssignmentRejectionRecord, ...]:
    records: list[FrameAssignmentRejectionRecord] = []
    for frame_index, rejection in zip(rejected_frame_indices, rejections, strict=True):
        frame_id, markers = normalized_observations[frame_index]
        visible_marker_ids = tuple(sorted(int(marker_id) for marker_id in markers))
        records.append(
            FrameAssignmentRejectionRecord(
                frame_index=frame_index,
                frame_id=frame_id,
                visible_marker_ids=visible_marker_ids,
                reason=rejection.reason,
                marker_pair=rejection.marker_pair,
                translation_error_m=_json_safe_float(rejection.translation_error_m),
                rotation_error_deg=_json_safe_float(rejection.rotation_error_deg),
                translation_gate_m=_json_safe_float(rejection.translation_gate_m),
                rotation_gate_deg=_json_safe_float(rejection.rotation_gate_deg),
            )
        )
    return tuple(records)


def _json_safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _measurement_distribution(values: Sequence[float]) -> MeasurementDistribution | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return MeasurementDistribution(
        min=_json_safe_float(float(np.min(finite))),
        median=_json_safe_float(float(np.median(finite))),
        p95=_json_safe_float(float(np.percentile(finite, 95))),
        max=_json_safe_float(float(np.max(finite))),
    )


def _make_dropped_pair_edge(
    pair: MarkerPair,
    stage: str,
    reason: str,
    *,
    observed_count: int,
    supported_count: int,
    required_count: int,
    translation_gate: float,
    rotation_gate: float,
    edge: _PairConsensus | None = None,
) -> DroppedPairEdge:
    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    if edge is not None:
        diagnostics = _edge_diagnostics(pair, edge)
        translation_rms_m = _json_safe_float(diagnostics.translation_rms_m)
        rotation_rms_deg = _json_safe_float(diagnostics.rotation_rms_deg)
    return DroppedPairEdge(
        marker_a=pair[0],
        marker_b=pair[1],
        stage=stage,
        reason=reason,
        observed_count=observed_count,
        supported_count=supported_count,
        required_count=required_count,
        translation_rms_m=translation_rms_m,
        rotation_rms_deg=rotation_rms_deg,
        translation_gate_m=_json_safe_float(translation_gate),
        rotation_gate_deg=_json_safe_float(rotation_gate),
    )


def _make_restored_pair_edge(dropped: DroppedPairEdge, stage: str) -> RestoredPairEdge:
    return RestoredPairEdge(
        marker_a=dropped.marker_a,
        marker_b=dropped.marker_b,
        stage=stage,
        original_stage=dropped.stage,
        original_reason=dropped.reason,
        observed_count=dropped.observed_count,
        supported_count=dropped.supported_count,
        required_count=dropped.required_count,
        support_fraction=dropped.supported_count / max(dropped.observed_count, 1),
        translation_rms_m=dropped.translation_rms_m,
        rotation_rms_deg=dropped.rotation_rms_deg,
        translation_gate_m=dropped.translation_gate_m,
        rotation_gate_deg=dropped.rotation_gate_deg,
    )


def _weak_edge_rank_key(
    supported_count: int,
    observed_count: int,
    rotation_rms_deg: float | None,
    translation_rms_m: float | None,
) -> tuple[float, float, float, float]:
    fraction = supported_count / max(observed_count, 1)
    rotation = rotation_rms_deg if rotation_rms_deg is not None else float("inf")
    translation = translation_rms_m if translation_rms_m is not None else float("inf")
    return (-supported_count, -fraction, rotation, translation)


def _connectivity_failure_message(
    stage: str,
    reference_marker_id: int,
    missing: list[int],
) -> str:
    if stage == "initial_consensus":
        return (
            f"Expected marker IDs are not connected to reference {reference_marker_id}; "
            f"missing {missing}."
        )
    if stage == "assignment_support":
        return (
            f"Expected marker IDs are not connected after rejecting assignment frames; "
            f"missing {missing}."
        )
    if stage == "post_pruning":
        return f"Expected marker IDs are not connected after pruning; missing {missing}."
    return f"Expected marker IDs are not connected; missing {missing}."


def _weak_restore_candidates(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    weak_pool: dict[MarkerPair, _PairConsensus],
    dropped: Sequence[DroppedPairEdge],
) -> list[tuple[_PairConsensus, DroppedPairEdge]]:
    candidates: list[tuple[_PairConsensus, DroppedPairEdge]] = []
    seen: set[MarkerPair] = set()
    for drop in dropped:
        pair = drop.marker_pair
        if pair in pair_consensus or pair in seen:
            continue
        edge = weak_pool.get(pair)
        if edge is None or not edge.inlier_frames:
            continue
        candidates.append((edge, drop))
        seen.add(pair)
    return candidates


def _maybe_restore_weak_connectivity(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    weak_pool: dict[MarkerPair, _PairConsensus],
    dropped: list[DroppedPairEdge],
    required_ids: list[int],
    reference_marker_id: int,
    stage: str,
    *,
    best_effort: bool,
    restored_pair_edges: list[RestoredPairEdge] | None,
) -> str | None:
    connected = _connected_marker_ids(pair_consensus, reference_marker_id)
    missing = sorted(set(required_ids) - connected)
    if not missing:
        return None
    if not best_effort:
        return _connectivity_failure_message(stage, reference_marker_id, missing)
    if not pair_consensus:
        return _connectivity_failure_message(stage, reference_marker_id, missing)

    restore_candidates = _weak_restore_candidates(pair_consensus, weak_pool, dropped)
    sorted_candidates = sorted(
        restore_candidates,
        key=lambda item: _weak_edge_rank_key(
            item[1].supported_count,
            item[1].observed_count,
            item[1].rotation_rms_deg,
            item[1].translation_rms_m,
        ),
    )
    remaining = list(sorted_candidates)
    restored_records: list[RestoredPairEdge] = []

    # ponytail: greedy first-ranked bridging edge per iteration; upgrade path is
    # union-find + global min-cost bridge set if multi-component graphs get larger.
    while not set(required_ids).issubset(_connected_marker_ids(pair_consensus, reference_marker_id)):
        connected = _connected_marker_ids(pair_consensus, reference_marker_id)
        chosen_index: int | None = None
        for index, (edge, drop) in enumerate(remaining):
            marker_a, marker_b = drop.marker_pair
            if marker_a in connected and marker_b in connected:
                continue
            if marker_a not in connected and marker_b not in connected:
                continue
            chosen_index = index
            break
        if chosen_index is None:
            break
        edge, drop = remaining.pop(chosen_index)
        pair_consensus[drop.marker_pair] = edge
        restored_records.append(_make_restored_pair_edge(drop, stage))

    if restored_pair_edges is not None:
        restored_pair_edges.extend(restored_records)

    missing = sorted(set(required_ids) - _connected_marker_ids(pair_consensus, reference_marker_id))
    if missing:
        return _connectivity_failure_message(stage, reference_marker_id, missing)
    return None


@dataclass(frozen=True)
class _MarkerPoseHypothesis:
    rotation: np.ndarray
    translation: np.ndarray
    frame_index: int
    candidate: _MarkerCandidate


def _filter_pair_hypotheses_to_markers(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    marker_ids: frozenset[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    return {
        pair: hypotheses
        for pair, hypotheses in pair_hypotheses.items()
        if pair[0] in marker_ids and pair[1] in marker_ids
    }


def _relative_pose_high_in_low(
    low_rotation: np.ndarray,
    low_translation: np.ndarray,
    high_rotation: np.ndarray,
    high_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = low_rotation.T @ high_rotation
    translation = low_rotation.T @ (high_translation - low_translation)
    return rotation, translation


def _pair_consensus_from_assignment_hypotheses(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[MarkerPair, _PairConsensus]:
    rotation_gate = settings.pair_rotation_rms_gate_deg
    consensus: dict[MarkerPair, _PairConsensus] = {}
    for pair, hypotheses in pair_hypotheses.items():
        translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
        edge = _best_pair_consensus(pair, hypotheses, translation_gate, rotation_gate)
        if edge is not None and len(edge.inlier_frames) >= settings.min_inliers_per_edge:
            diagnostics = _edge_diagnostics(pair, edge)
            if diagnostics.translation_rms_m <= translation_gate and diagnostics.rotation_rms_deg <= rotation_gate:
                consensus[pair] = edge
                continue
        if marker_poses is None:
            continue
        marker_low, marker_high = pair
        if marker_low not in marker_poses or marker_high not in marker_poses:
            continue
        inlier_frames = tuple(sorted({frame_index for _, _, frame_index in hypotheses}))
        if len(inlier_frames) < settings.min_inliers_per_edge:
            continue
        low_rotation, low_translation = marker_poses[marker_low]
        high_rotation, high_translation = marker_poses[marker_high]
        rotation_ba, translation_ba = _relative_pose_high_in_low(
            low_rotation,
            low_translation,
            high_rotation,
            high_translation,
        )
        hypotheses_by_frame = {
            frame_index: (rotation, translation)
            for rotation, translation, frame_index in hypotheses
        }
        consensus[pair] = _PairConsensus(
            marker_a=marker_low,
            marker_b=marker_high,
            rotation_ba=rotation_ba,
            translation_ba=translation_ba,
            inlier_frames=inlier_frames,
            inlier_hypotheses={
                frame_index: hypotheses_by_frame[frame_index]
                for frame_index in inlier_frames
                if frame_index in hypotheses_by_frame
            },
        )
    return consensus


def _collect_assignment_pair_hypotheses(
    assigned_candidates: dict[int, dict[int, _MarkerCandidate]],
    marker_ids: frozenset[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    for frame_index, assignment in assigned_candidates.items():
        visible = sorted(marker_id for marker_id in assignment if marker_id in marker_ids)
        for index_a, marker_low in enumerate(visible):
            for marker_high in visible[index_a + 1 :]:
                pair = (marker_low, marker_high)
                rotation_ba, translation_ba = _transform_high_in_low(
                    assignment[marker_low],
                    assignment[marker_high],
                )
                hypotheses.setdefault(pair, []).append(
                    (rotation_ba, translation_ba, frame_index)
                )
    return hypotheses


def _marker_pose_from_frame_and_candidate(
    layout_rotation: np.ndarray,
    layout_translation: np.ndarray,
    candidate: _MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    marker_rotation = layout_rotation.T @ candidate.rotation
    marker_translation = layout_rotation.T @ (candidate.tvec.reshape(3) - layout_translation)
    return marker_rotation, marker_translation


def _frame_pose_from_solved_assignment(
    assignment: dict[int, _MarkerCandidate],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    solved_ids: frozenset[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    estimates: list[tuple[np.ndarray, np.ndarray]] = []
    for marker_id, candidate in assignment.items():
        if marker_id not in solved_ids or marker_id not in marker_poses:
            continue
        marker_rotation, marker_translation = marker_poses[marker_id]
        layout_rotation = candidate.rotation @ marker_rotation.T
        layout_translation = candidate.tvec.reshape(3) - layout_rotation @ marker_translation
        estimates.append((layout_rotation, layout_translation))
    if len(estimates) < 1:
        return None
    return _average_poses(estimates)


def _frame_pose_from_known_marker_candidates(
    candidates: dict[int, list[_MarkerCandidate]],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    solved_ids: frozenset[int],
) -> tuple[tuple[np.ndarray, np.ndarray], dict[int, _MarkerCandidate]] | None:
    selected: dict[int, _MarkerCandidate] = {}
    estimates: list[tuple[np.ndarray, np.ndarray]] = []
    for marker_id in sorted(candidates):
        if marker_id not in solved_ids or marker_id not in marker_poses:
            continue
        marker_rotation, marker_translation = marker_poses[marker_id]
        best_candidate = min(candidates[marker_id], key=lambda candidate: candidate.reprojection_rms_px)
        layout_rotation = best_candidate.rotation @ marker_rotation.T
        layout_translation = best_candidate.tvec.reshape(3) - layout_rotation @ marker_translation
        selected[marker_id] = best_candidate
        estimates.append((layout_rotation, layout_translation))
    if not estimates:
        return None
    return _average_poses(estimates), selected


def _expand_markers_hierarchically(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    assigned_candidates: dict[int, dict[int, _MarkerCandidate]],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    solved_ids: frozenset[int],
    expected_ids: list[int],
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, dict[int, _MarkerCandidate]],
    tuple[MarkerExpansionRecord, ...],
    frozenset[int],
]:
    rotation_gate = settings.pair_rotation_rms_gate_deg
    poses = dict(marker_poses)
    assignments = {
        frame_index: dict(assignment)
        for frame_index, assignment in assigned_candidates.items()
    }
    expansion_records: list[MarkerExpansionRecord] = []
    solved = set(solved_ids)
    expected_set = set(expected_ids)

    while True:
        unsolved = sorted(
            marker_id
            for marker_id in (expected_set - solved)
            if not any(
                record.marker_id == marker_id and record.status == "rejected"
                for record in expansion_records
            )
        )
        if not unsolved:
            break
        progress = False
        candidates_by_frame = {frame_index: candidates for frame_index, candidates in frame_candidates}
        for marker_id in unsolved:
            hypotheses: list[_MarkerPoseHypothesis] = []
            for frame_index, candidates in candidates_by_frame.items():
                if marker_id not in candidates:
                    continue
                frame_pose_result = _frame_pose_from_known_marker_candidates(
                    candidates,
                    poses,
                    frozenset(solved),
                )
                if frame_pose_result is None:
                    continue
                layout_rotation, layout_translation = frame_pose_result[0]
                selected_solved = frame_pose_result[1]
                assignments.setdefault(frame_index, {}).update(selected_solved)
                for candidate in candidates[marker_id]:
                    rotation, translation = _marker_pose_from_frame_and_candidate(
                        layout_rotation,
                        layout_translation,
                        candidate,
                    )
                    hypotheses.append(
                        _MarkerPoseHypothesis(
                            rotation=rotation,
                            translation=translation,
                            frame_index=frame_index,
                            candidate=candidate,
                        )
                    )
            if not hypotheses:
                continue
            translation_gate = _pair_translation_gate(
                settings,
                marker_sizes_m,
                (marker_id, marker_id),
            )
            relative_hypotheses = [
                (hypothesis.rotation, hypothesis.translation, hypothesis.frame_index)
                for hypothesis in hypotheses
            ]
            edge = _best_pair_consensus(
                (marker_id, marker_id),
                relative_hypotheses,
                translation_gate,
                rotation_gate,
            )
            if edge is None or len(edge.inlier_frames) < settings.min_inliers_per_edge:
                expansion_records.append(
                    MarkerExpansionRecord(
                        marker_id=marker_id,
                        status="rejected",
                        support_frames=len(edge.inlier_frames) if edge is not None else 0,
                        reason="insufficient_support",
                    )
                )
                continue
            diagnostics = _edge_diagnostics((marker_id, marker_id), edge)
            if diagnostics.translation_rms_m > translation_gate:
                expansion_records.append(
                    MarkerExpansionRecord(
                        marker_id=marker_id,
                        status="rejected",
                        support_frames=diagnostics.inlier_count,
                        reason="translation_rms_gate",
                    )
                )
                continue
            if diagnostics.rotation_rms_deg > rotation_gate:
                expansion_records.append(
                    MarkerExpansionRecord(
                        marker_id=marker_id,
                        status="rejected",
                        support_frames=diagnostics.inlier_count,
                        reason="rotation_rms_gate",
                    )
                )
                continue
            poses[marker_id] = (edge.rotation_ba.copy(), edge.translation_ba.copy())
            solved.add(marker_id)
            progress = True
            expansion_records.append(
                MarkerExpansionRecord(
                    marker_id=marker_id,
                    status="accepted",
                    support_frames=diagnostics.inlier_count,
                )
            )
            hypotheses_by_frame = {
                hypothesis.frame_index: hypothesis for hypothesis in hypotheses
            }
            for frame_index in edge.inlier_frames:
                hypothesis = hypotheses_by_frame.get(frame_index)
                if hypothesis is None:
                    continue
                assignments.setdefault(frame_index, {})[marker_id] = hypothesis.candidate
        if not progress:
            break

    unresolved = frozenset(expected_set - solved)
    for marker_id in sorted(unresolved):
        if not any(record.marker_id == marker_id for record in expansion_records):
            expansion_records.append(
                MarkerExpansionRecord(
                    marker_id=marker_id,
                    status="rejected",
                    support_frames=0,
                    reason="unreachable_from_anchor_core",
                )
            )
    return poses, assignments, tuple(expansion_records), unresolved


def _freeze_assigned_frame_candidates(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    assigned_candidates: dict[int, dict[int, _MarkerCandidate]],
) -> list[tuple[int, dict[int, list[_MarkerCandidate]]]]:
    frozen: list[tuple[int, dict[int, list[_MarkerCandidate]]]] = []
    for frame_index, candidates in frame_candidates:
        assignment = assigned_candidates.get(frame_index)
        if assignment is None:
            continue
        frozen_candidates = {
            marker_id: [assignment[marker_id]]
            for marker_id in assignment
            if marker_id in candidates
        }
        if len(frozen_candidates) >= 2:
            frozen.append((frame_index, frozen_candidates))
    return frozen


def _assign_and_initialize_anchor_core(
    frame_candidates: list[tuple[int, dict[int, list[_MarkerCandidate]]]],
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    normalized_observations: list[tuple[str | int, dict[int, np.ndarray]]],
    expected_ids: list[int],
    anchor_ids: tuple[int, ...],
    reference_marker_id: int,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    stop_after_expansion: bool = False,
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[
    dict[int, dict[int, _MarkerCandidate]] | None,
    tuple[int, ...],
    tuple[FrameAssignmentRejection, ...],
    dict[MarkerPair, _PairConsensus] | None,
    dict[int, tuple[np.ndarray, np.ndarray]] | None,
    list[DroppedPairEdge],
    AnchorCoreDiagnostics,
    str | None,
]:
    anchor_set = frozenset(anchor_ids)
    dropped_edges: list[DroppedPairEdge] = []
    anchor_hypotheses = _filter_pair_hypotheses_to_markers(pair_hypotheses, anchor_set)
    anchor_consensus, anchor_pair_failure, anchor_drops = _estimate_pair_consensus(
        anchor_hypotheses,
        expected_ids,
        reference_marker_id,
        marker_sizes_m,
        settings,
        connectivity_ids=anchor_ids,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    dropped_edges.extend(anchor_drops)
    bootstrap = AnchorCoreBootstrapDiagnostics(
        status="failed",
        frames_considered=len(frame_candidates),
        frames_accepted=0,
        failure_reason=anchor_pair_failure,
    )
    anchor_core = AnchorCoreDiagnostics(
        mode="anchor_core",
        configured_anchor_ids=anchor_ids,
        bootstrap=bootstrap,
        expansion=(),
        final_solved_ids=frozenset(),
        unresolved_ids=frozenset(expected_ids),
    )
    if anchor_pair_failure is not None:
        return None, (), (), None, None, dropped_edges, anchor_core, (
            f"Anchor core bootstrap failed: {anchor_pair_failure}"
        )

    assigned_candidates, rejected_frames, assignment_rejections = _assign_ippe_candidates(
        frame_candidates,
        anchor_consensus,
        settings,
        marker_sizes_m,
        search_marker_ids=anchor_set,
    )
    accepted_frames = frozenset(assigned_candidates)
    bootstrap = AnchorCoreBootstrapDiagnostics(
        status="failed" if not accepted_frames else "ok",
        frames_considered=len(frame_candidates),
        frames_accepted=len(accepted_frames),
        failure_reason=None if accepted_frames else "no_anchor_assignable_frames",
    )
    anchor_core = AnchorCoreDiagnostics(
        mode="anchor_core",
        configured_anchor_ids=anchor_ids,
        bootstrap=bootstrap,
        expansion=(),
        final_solved_ids=frozenset(anchor_set) if accepted_frames else frozenset(),
        unresolved_ids=frozenset(set(expected_ids) - anchor_set),
    )
    if not accepted_frames:
        return (
            assigned_candidates,
            rejected_frames,
            assignment_rejections,
            anchor_consensus,
            None,
            dropped_edges,
            anchor_core,
            "No frames with assignable anchor IPPE candidates remain after bootstrap.",
        )

    ref_rotation, ref_translation = _reference_gauge_pose(marker_sizes_m[reference_marker_id])
    marker_poses = _initialize_marker_poses(
        reference_marker_id,
        ref_rotation,
        ref_translation,
        list(anchor_ids),
        anchor_consensus,
    )
    frame_poses = _initialize_frame_poses(
        assigned_candidates,
        marker_poses,
        len(normalized_observations),
    )
    anchor_corner_observations = _build_corner_observations(
        normalized_observations,
        list(anchor_ids),
    )
    inlier_mask = _mask_corner_observations_for_frames(anchor_corner_observations, accepted_frames)
    non_reference_anchors = [
        marker_id for marker_id in anchor_ids if marker_id != reference_marker_id
    ]
    marker_poses, frame_poses, inlier_mask, ba_failure = _run_bundle_adjustment(
        anchor_corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_anchors,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        settings,
    )
    if ba_failure is not None:
        bootstrap = AnchorCoreBootstrapDiagnostics(
            status="failed",
            frames_considered=len(frame_candidates),
            frames_accepted=len(accepted_frames),
            failure_reason=ba_failure,
        )
        anchor_core = AnchorCoreDiagnostics(
            mode="anchor_core",
            configured_anchor_ids=anchor_ids,
            bootstrap=bootstrap,
            expansion=(),
            final_solved_ids=frozenset(anchor_set),
            unresolved_ids=frozenset(set(expected_ids) - anchor_set),
        )
        return (
            assigned_candidates,
            rejected_frames,
            assignment_rejections,
            anchor_consensus,
            marker_poses,
            dropped_edges,
            anchor_core,
            f"Anchor core mini bundle adjustment failed: {ba_failure}",
        )

    marker_poses, assignments, expansion_records, unresolved = _expand_markers_hierarchically(
        frame_candidates,
        assigned_candidates,
        marker_poses,
        anchor_set,
        expected_ids,
        settings,
        marker_sizes_m,
    )
    if unresolved:
        anchor_core = AnchorCoreDiagnostics(
            mode="anchor_core",
            configured_anchor_ids=anchor_ids,
            bootstrap=AnchorCoreBootstrapDiagnostics(
                status="ok",
                frames_considered=len(frame_candidates),
                frames_accepted=len(accepted_frames),
            ),
            expansion=expansion_records,
            final_solved_ids=frozenset(marker_poses),
            unresolved_ids=unresolved,
        )
        return (
            assignments,
            rejected_frames,
            assignment_rejections,
            None,
            marker_poses,
            dropped_edges,
            anchor_core,
            f"Anchor core expansion could not solve all expected markers; missing {sorted(unresolved)}.",
        )

    if stop_after_expansion:
        expected_set = frozenset(expected_ids)
        pair_consensus = _pair_consensus_from_assignment_hypotheses(
            _collect_assignment_pair_hypotheses(assignments, expected_set),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )
        anchor_core = AnchorCoreDiagnostics(
            mode="anchor_core",
            configured_anchor_ids=anchor_ids,
            bootstrap=AnchorCoreBootstrapDiagnostics(
                status="ok",
                frames_considered=len(frame_candidates),
                frames_accepted=len(accepted_frames),
            ),
            expansion=expansion_records,
            final_solved_ids=frozenset(marker_poses),
            unresolved_ids=frozenset(),
            stopped_after_expansion=True,
        )
        return (
            assignments,
            rejected_frames,
            assignment_rejections,
            pair_consensus,
            marker_poses,
            dropped_edges,
            anchor_core,
            None,
        )

    expected_set = frozenset(expected_ids)
    seed_frames = _freeze_assigned_frame_candidates(frame_candidates, assignments)
    seed_hypotheses = _collect_pair_hypotheses(seed_frames, expected_ids)
    seed_consensus, seed_failure, seed_drops = _estimate_pair_consensus(
        seed_hypotheses,
        expected_ids,
        reference_marker_id,
        marker_sizes_m,
        settings,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    dropped_edges.extend(seed_drops)
    if seed_failure is not None:
        seed_consensus = _pair_consensus_from_assignment_hypotheses(
            _collect_assignment_pair_hypotheses(assignments, expected_set),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )

    assignments, rejected_frames, assignment_rejections = _assign_ippe_candidates(
        frame_candidates,
        seed_consensus,
        settings,
        marker_sizes_m,
        search_marker_ids=expected_set,
    )

    frozen_frames = _freeze_assigned_frame_candidates(frame_candidates, assignments)
    frozen_hypotheses = _collect_pair_hypotheses(frozen_frames, expected_ids)
    pair_consensus, pair_failure, post_drops = _estimate_pair_consensus(
        frozen_hypotheses,
        expected_ids,
        reference_marker_id,
        marker_sizes_m,
        settings,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    dropped_edges.extend(post_drops)
    if pair_failure is not None:
        pair_consensus = _pair_consensus_from_assignment_hypotheses(
            _collect_assignment_pair_hypotheses(assignments, expected_set),
            settings,
            marker_sizes_m,
            marker_poses=marker_poses,
        )
        connected = _connected_marker_ids(pair_consensus, reference_marker_id)
        missing = sorted(set(expected_ids) - connected)
        if missing:
            anchor_core = AnchorCoreDiagnostics(
                mode="anchor_core",
                configured_anchor_ids=anchor_ids,
                bootstrap=AnchorCoreBootstrapDiagnostics(
                    status="ok",
                    frames_considered=len(frame_candidates),
                    frames_accepted=len(accepted_frames),
                ),
                expansion=expansion_records,
                final_solved_ids=frozenset(marker_poses),
                unresolved_ids=frozenset(missing),
            )
            return (
                assignments,
                rejected_frames,
                assignment_rejections,
                pair_consensus,
                marker_poses,
                dropped_edges,
                anchor_core,
                (
                    f"Expected marker IDs are not connected after anchor-core expansion; "
                    f"missing {missing}."
                ),
            )

    anchor_core = AnchorCoreDiagnostics(
        mode="anchor_core",
        configured_anchor_ids=anchor_ids,
        bootstrap=AnchorCoreBootstrapDiagnostics(
            status="ok",
            frames_considered=len(frame_candidates),
            frames_accepted=len(accepted_frames),
        ),
        expansion=expansion_records,
        final_solved_ids=frozenset(marker_poses),
        unresolved_ids=frozenset(),
    )
    return (
        assignments,
        rejected_frames,
        assignment_rejections,
        pair_consensus,
        marker_poses,
        dropped_edges,
        anchor_core,
        None,
    )


def _restrict_pair_consensus_to_frames(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    allowed_frames: frozenset[int],
    expected_ids: list[int],
    reference_marker_id: int,
    settings: CalibrationSettings,
    *,
    marker_sizes_m: Mapping[int, float],
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[dict[MarkerPair, _PairConsensus], str | None, tuple[DroppedPairEdge, ...]]:
    rotation_gate = settings.pair_rotation_rms_gate_deg
    updated: dict[MarkerPair, _PairConsensus] = {}
    weak_pool: dict[MarkerPair, _PairConsensus] = {}
    dropped: list[DroppedPairEdge] = []
    for pair, edge in pair_consensus.items():
        translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
        supported_frames = tuple(
            sorted(frame_index for frame_index in edge.inlier_frames if frame_index in allowed_frames)
        )
        if len(supported_frames) < settings.min_inliers_per_edge:
            if supported_frames:
                weak_pool[pair] = _PairConsensus(
                    marker_a=edge.marker_a,
                    marker_b=edge.marker_b,
                    rotation_ba=edge.rotation_ba,
                    translation_ba=edge.translation_ba,
                    inlier_frames=supported_frames,
                    inlier_hypotheses={
                        frame_index: edge.inlier_hypotheses[frame_index]
                        for frame_index in supported_frames
                    },
                )
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "assignment_support",
                    "insufficient_support",
                    observed_count=len(edge.inlier_frames),
                    supported_count=len(supported_frames),
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        updated[pair] = _PairConsensus(
            marker_a=edge.marker_a,
            marker_b=edge.marker_b,
            rotation_ba=edge.rotation_ba,
            translation_ba=edge.translation_ba,
            inlier_frames=supported_frames,
            inlier_hypotheses={
                frame_index: edge.inlier_hypotheses[frame_index] for frame_index in supported_frames
            },
        )

    failure = _maybe_restore_weak_connectivity(
        updated,
        weak_pool,
        dropped,
        expected_ids,
        reference_marker_id,
        "assignment_support",
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    if failure is not None:
        return updated, failure, tuple(dropped)
    return updated, None, tuple(dropped)


def _markers_in_frame_indices(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    frame_indices: frozenset[int],
) -> set[int]:
    markers: set[int] = set()
    for frame_index, (_, markers_in_frame) in enumerate(observations):
        if frame_index in frame_indices:
            markers.update(markers_in_frame)
    return markers


def _mask_corner_observations_for_frames(
    corner_observations: list[_CornerObservation],
    allowed_frames: frozenset[int],
) -> np.ndarray:
    return np.array(
        [observation.frame_index in allowed_frames for observation in corner_observations],
        dtype=bool,
    )


def _covisible_frame_count(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> int:
    return len(_covisible_frames_from_inliers(corner_observations, inlier_mask))


def _covisible_frames_from_inliers(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> frozenset[int]:
    complete = _complete_markers_per_frame(corner_observations, inlier_mask)
    return frozenset(
        frame_index
        for frame_index, marker_ids in complete.items()
        if len(marker_ids) >= 2
    )


def _search_assignments(
    marker_ids: list[int],
    candidates: dict[int, list[_MarkerCandidate]],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    current: dict[int, _MarkerCandidate],
    index: int,
    holder: dict[str, object],
) -> None:
    if index == len(marker_ids):
        (
            score,
            has_constrained_pair,
            worst_key,
            worst_pair,
            worst_translation_error,
            worst_rotation_error,
            worst_translation_fail,
            worst_rotation_fail,
            worst_translation_gate,
        ) = _evaluate_complete_assignment(
            current,
            marker_ids,
            pair_consensus,
            holder["marker_sizes_m"],  # type: ignore[arg-type]
            holder["settings"],  # type: ignore[arg-type]
            float(holder["rotation_gate"]),
        )
        _merge_assignment_violation_into_holder(
            holder,
            has_constrained_pair,
            worst_key,
            worst_pair,
            worst_translation_error,
            worst_rotation_error,
            worst_translation_fail,
            worst_rotation_fail,
            worst_translation_gate,
        )
        if score is not None and score > float(holder["score"]):
            holder["score"] = score
            holder["assignment"] = dict(current)
        return

    marker_id = marker_ids[index]
    for candidate in candidates[marker_id]:
        current[marker_id] = candidate
        _search_assignments(
            marker_ids,
            candidates,
            pair_consensus,
            current,
            index + 1,
            holder,
        )
    current.pop(marker_id, None)


def _reference_gauge_pose(marker_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    half = marker_size_m / 2.0
    top_left = np.array([-half, -half, 0.0], dtype=np.float64)
    top_right = np.array([half, -half, 0.0], dtype=np.float64)
    bottom_right = np.array([half, half, 0.0], dtype=np.float64)
    bottom_left = np.array([-half, half, 0.0], dtype=np.float64)
    rotation = footprint_orientation(top_left, top_right, bottom_left, bottom_right)
    translation = marker_origin_on_object(bottom_left, bottom_right)
    return rotation, translation


def _initialize_marker_poses(
    reference_marker_id: int,
    ref_rotation: np.ndarray,
    ref_translation: np.ndarray,
    expected_ids: list[int],
    pair_consensus: dict[MarkerPair, _PairConsensus],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {
        reference_marker_id: (ref_rotation.copy(), ref_translation.copy())
    }
    graph: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for pair, edge in pair_consensus.items():
        marker_a, marker_b = pair
        graph.setdefault(marker_a, []).append((marker_b, edge.rotation_ba, edge.translation_ba))
        graph.setdefault(marker_b, []).append(
            (
                marker_a,
                edge.rotation_ba.T,
                -edge.rotation_ba.T @ edge.translation_ba,
            )
        )

    queue = [reference_marker_id]
    while queue:
        parent_id = queue.pop()
        parent_rotation, parent_translation = poses[parent_id]
        for child_id, rotation_cp, translation_cp in graph.get(parent_id, []):
            if child_id in poses or child_id not in expected_ids:
                continue
            child_rotation = parent_rotation @ rotation_cp
            child_translation = parent_rotation @ translation_cp + parent_translation
            poses[child_id] = (child_rotation, child_translation)
            queue.append(child_id)
    return poses


def _initialize_frame_poses(
    assigned_candidates: dict[int, dict[int, _MarkerCandidate]],
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_count: int,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None] = [None] * frame_count
    for frame_index, assignment in assigned_candidates.items():
        estimates: list[tuple[np.ndarray, np.ndarray]] = []
        for marker_id, candidate in assignment.items():
            if marker_id not in marker_poses:
                continue
            marker_rotation, marker_translation = marker_poses[marker_id]
            layout_rotation = candidate.rotation @ marker_rotation.T
            layout_translation = candidate.tvec - layout_rotation @ marker_translation
            estimates.append((layout_rotation, layout_translation))
        if estimates:
            frame_poses[frame_index] = _average_poses(estimates)
    return frame_poses


def _average_poses(
    poses: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    rotations = [rotation for rotation, _ in poses]
    translations = np.stack([translation for _, translation in poses], axis=0)
    return _average_rotations(rotations), np.mean(translations, axis=0)


def _average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0].copy()
    quaternions = [_rotation_matrix_to_quaternion(rotation) for rotation in rotations]
    reference = quaternions[0]
    aligned = [quaternion if np.dot(quaternion, reference) >= 0.0 else -quaternion for quaternion in quaternions]
    mean = np.mean(aligned, axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 0.0:
        return rotations[0].copy()
    return _quaternion_to_rotation_matrix(mean / norm)


def _build_corner_observations(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    expected_ids: list[int],
) -> list[_CornerObservation]:
    expected_set = set(expected_ids)
    corner_observations: list[_CornerObservation] = []
    for frame_index, (_, markers) in enumerate(observations):
        for marker_id, corners in markers.items():
            if marker_id not in expected_set:
                continue
            for corner_index in range(4):
                corner_observations.append(
                    _CornerObservation(
                        frame_index=frame_index,
                        marker_id=marker_id,
                        corner_index=corner_index,
                        image_point=corners[corner_index],
                    )
                )
    return corner_observations


def _run_bundle_adjustment(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    reference_marker_id: int,
    non_reference_ids: list[int],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settings: CalibrationSettings,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray] | None],
    np.ndarray,
    str | None,
]:
    active_frames = sorted(
        {
            observation.frame_index
            for observation, keep in zip(corner_observations, inlier_mask, strict=True)
            if keep
        }
    )
    if not active_frames:
        return marker_poses, frame_poses, inlier_mask, "Bundle adjustment has no active frames."
    if not non_reference_ids:
        return marker_poses, frame_poses, inlier_mask, None

    x0 = _pack_parameters(marker_poses, frame_poses, non_reference_ids, active_frames)
    if not np.all(np.isfinite(x0)):
        return marker_poses, frame_poses, inlier_mask, "Bundle adjustment initial parameters are non-finite."

    jac_sparsity = _build_jac_sparsity(
        corner_observations,
        inlier_mask,
        non_reference_ids,
        active_frames,
        reference_marker_id,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(params)):
            return np.full(jac_sparsity.shape[0], 1e3, dtype=np.float64)
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
            projected = _project_corner(
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

    try:
        result = least_squares(
            residuals,
            x0,
            jac_sparsity=jac_sparsity,
            loss="huber",
            f_scale=settings.huber_delta_px,
            max_nfev=max(settings.max_ba_iterations * len(x0), len(x0) + 1),
        )
    except ValueError as exc:
        return marker_poses, frame_poses, inlier_mask, f"Bundle adjustment failed: {exc}"

    if not result.success or not np.all(np.isfinite(result.x)):
        return (
            marker_poses,
            frame_poses,
            inlier_mask,
            f"Bundle adjustment did not converge (status={result.status}).",
        )

    marker_poses, frame_poses = _unpack_parameters(
        result.x,
        marker_poses,
        frame_poses,
        non_reference_ids,
        active_frames,
        reference_marker_id,
    )
    depth_failure = _positive_depth_failure(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points_by_marker,
    )
    if depth_failure is not None:
        return marker_poses, frame_poses, inlier_mask, depth_failure
    return marker_poses, frame_poses, inlier_mask, None


def _prune_and_refit(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    reference_marker_id: int,
    non_reference_ids: list[int],
    expected_ids: list[int],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    accepted_frames: frozenset[int],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    *,
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray] | None],
    np.ndarray,
    dict[MarkerPair, _PairConsensus],
    str | None,
    tuple[DroppedPairEdge, ...],
]:
    errors = _corner_errors(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
    )
    pruned = inlier_mask & (errors <= settings.corner_outlier_px)
    pruned = _drop_frames_without_covisibility(corner_observations, pruned)
    pruned = _mask_corner_observations_for_frames(corner_observations, accepted_frames) & pruned
    if int(np.count_nonzero(pruned)) < 8:
        return marker_poses, frame_poses, inlier_mask, pair_consensus, (
            "Too few inlier corners remain after pruning."
        ), ()

    remaining_frames = _covisible_frames_from_inliers(corner_observations, pruned)
    updated_consensus, support_failure, dropped_edges = _recheck_pair_support(
        pair_consensus,
        corner_observations,
        pruned,
        expected_ids,
        reference_marker_id,
        settings,
        allowed_frames=remaining_frames,
        marker_sizes_m=marker_sizes_m,
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    if support_failure is not None:
        return marker_poses, frame_poses, pruned, updated_consensus, support_failure, dropped_edges

    marker_poses, frame_poses, pruned, ba_failure = _run_bundle_adjustment(
        corner_observations,
        pruned,
        marker_poses,
        frame_poses,
        reference_marker_id,
        non_reference_ids,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
        settings,
    )
    if ba_failure is not None:
        return marker_poses, frame_poses, pruned, updated_consensus, ba_failure, dropped_edges
    return marker_poses, frame_poses, pruned, updated_consensus, None, dropped_edges


def _complete_markers_per_frame(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> dict[int, set[int]]:
    corner_counts: dict[tuple[int, int], int] = {}
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        key = (observation.frame_index, observation.marker_id)
        corner_counts[key] = corner_counts.get(key, 0) + 1
    complete: dict[int, set[int]] = {}
    for (frame_index, marker_id), count in corner_counts.items():
        if count == 4:
            complete.setdefault(frame_index, set()).add(marker_id)
    return complete


def _drop_frames_without_covisibility(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
) -> np.ndarray:
    updated = inlier_mask.copy()
    while True:
        complete = _complete_markers_per_frame(corner_observations, updated)
        valid_frames = {
            frame_index
            for frame_index, marker_ids in complete.items()
            if len(marker_ids) >= 2
        }
        changed = False
        for index, observation in enumerate(corner_observations):
            if updated[index] and observation.frame_index not in valid_frames:
                updated[index] = False
                changed = True
        if not changed:
            break
    return updated


def _recheck_pair_support(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    expected_ids: list[int],
    reference_marker_id: int,
    settings: CalibrationSettings,
    allowed_frames: frozenset[int] | None = None,
    *,
    marker_sizes_m: Mapping[int, float],
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[dict[MarkerPair, _PairConsensus], str | None, tuple[DroppedPairEdge, ...]]:
    rotation_gate = settings.pair_rotation_rms_gate_deg
    complete = _complete_markers_per_frame(corner_observations, inlier_mask)
    updated: dict[MarkerPair, _PairConsensus] = {}
    weak_pool: dict[MarkerPair, _PairConsensus] = {}
    dropped: list[DroppedPairEdge] = []
    for pair, edge in pair_consensus.items():
        translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
        marker_low, marker_high = pair
        supported_frames = tuple(
            sorted(
                frame_index
                for frame_index in edge.inlier_frames
                if (allowed_frames is None or frame_index in allowed_frames)
                and frame_index in complete
                and marker_low in complete[frame_index]
                and marker_high in complete[frame_index]
            )
        )
        if len(supported_frames) < settings.min_inliers_per_edge:
            if supported_frames:
                selected_hypotheses = {
                    frame_index: edge.inlier_hypotheses[frame_index]
                    for frame_index in supported_frames
                }
                weak_pool[pair] = _PairConsensus(
                    marker_a=marker_low,
                    marker_b=marker_high,
                    rotation_ba=edge.rotation_ba,
                    translation_ba=edge.translation_ba,
                    inlier_frames=supported_frames,
                    inlier_hypotheses=selected_hypotheses,
                )
            dropped.append(
                _make_dropped_pair_edge(
                    pair,
                    "post_pruning",
                    "insufficient_support",
                    observed_count=len(edge.inlier_frames),
                    supported_count=len(supported_frames),
                    required_count=settings.min_inliers_per_edge,
                    translation_gate=translation_gate,
                    rotation_gate=rotation_gate,
                    edge=edge,
                )
            )
            continue
        selected_hypotheses = {
            frame_index: edge.inlier_hypotheses[frame_index]
            for frame_index in supported_frames
        }
        updated[pair] = _PairConsensus(
            marker_a=marker_low,
            marker_b=marker_high,
            rotation_ba=edge.rotation_ba,
            translation_ba=edge.translation_ba,
            inlier_frames=supported_frames,
            inlier_hypotheses=selected_hypotheses,
        )

    failure = _maybe_restore_weak_connectivity(
        updated,
        weak_pool,
        dropped,
        expected_ids,
        reference_marker_id,
        "post_pruning",
        best_effort=best_effort,
        restored_pair_edges=restored_pair_edges,
    )
    if failure is not None:
        return updated, failure, tuple(dropped)
    return updated, None, tuple(dropped)


def _build_jac_sparsity(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    non_reference_ids: list[int],
    active_frames: list[int],
    reference_marker_id: int,
) -> lil_matrix:
    marker_param_index = {marker_id: index for index, marker_id in enumerate(non_reference_ids)}
    frame_param_index = {frame_index: index for index, frame_index in enumerate(active_frames)}
    num_marker_params = 6 * len(non_reference_ids)
    num_frame_params = 6 * len(active_frames)
    num_params = num_marker_params + num_frame_params
    num_residuals = 2 * int(np.count_nonzero(inlier_mask))
    sparsity = lil_matrix((num_residuals, num_params), dtype=int)
    row = 0
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        if observation.marker_id != reference_marker_id:
            marker_offset = 6 * marker_param_index[observation.marker_id]
            sparsity[row : row + 2, marker_offset : marker_offset + 6] = 1
        frame_offset = num_marker_params + 6 * frame_param_index[observation.frame_index]
        sparsity[row : row + 2, frame_offset : frame_offset + 6] = 1
        row += 2
    return sparsity


def _positive_depth_failure(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    object_points_by_marker: dict[int, np.ndarray],
    min_depth_m: float = 1e-4,
) -> str | None:
    for observation, keep in zip(corner_observations, inlier_mask, strict=True):
        if not keep:
            continue
        frame_pose = frame_poses[observation.frame_index]
        marker_pose = marker_poses.get(observation.marker_id)
        if frame_pose is None or marker_pose is None:
            return "Bundle adjustment produced a frame or marker pose with missing state."
        marker_rotation, marker_translation = marker_pose
        frame_rotation, frame_translation = frame_pose
        object_points = object_points_by_marker[observation.marker_id]
        point_layout = marker_rotation @ object_points[observation.corner_index] + marker_translation
        point_camera = frame_rotation @ point_layout + frame_translation
        if not np.all(np.isfinite(point_camera)) or float(point_camera[2]) <= min_depth_m:
            return (
                f"Bundle adjustment produced non-positive depth for marker "
                f"{observation.marker_id} in frame {observation.frame_index}."
            )
    return None


def _pack_parameters(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    non_reference_ids: list[int],
    active_frames: list[int],
) -> np.ndarray:
    values: list[float] = []
    for marker_id in non_reference_ids:
        rotation, translation = marker_poses[marker_id]
        rvec, _ = cv2.Rodrigues(rotation)
        values.extend(rvec.reshape(3).tolist())
        values.extend(translation.reshape(3).tolist())
    for frame_index in active_frames:
        frame_pose = frame_poses[frame_index]
        if frame_pose is None:
            values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
            continue
        rotation, translation = frame_pose
        rvec, _ = cv2.Rodrigues(rotation)
        values.extend(rvec.reshape(3).tolist())
        values.extend(translation.reshape(3).tolist())
    return np.asarray(values, dtype=np.float64)


def _unpack_parameters(
    params: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    non_reference_ids: list[int],
    active_frames: list[int],
    reference_marker_id: int,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray] | None]]:
    marker_state = dict(marker_poses)
    offset = 0
    for marker_id in non_reference_ids:
        rvec = params[offset : offset + 3]
        translation = params[offset + 3 : offset + 6]
        offset += 6
        rotation, _ = cv2.Rodrigues(rvec)
        marker_state[marker_id] = (rotation, translation)
    marker_state[reference_marker_id] = marker_poses[reference_marker_id]

    updated_frame_poses = list(frame_poses)
    for frame_index in active_frames:
        rvec = params[offset : offset + 3]
        translation = params[offset + 3 : offset + 6]
        offset += 6
        rotation, _ = cv2.Rodrigues(rvec)
        updated_frame_poses[frame_index] = (rotation, translation)
    return marker_state, updated_frame_poses


def _project_corner(
    corner_index: int,
    marker_id: int,
    marker_pose: tuple[np.ndarray, np.ndarray],
    frame_pose: tuple[np.ndarray, np.ndarray],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    marker_rotation, marker_translation = marker_pose
    frame_rotation, frame_translation = frame_pose
    object_points = object_points_by_marker[marker_id]
    point_layout = marker_rotation @ object_points[corner_index] + marker_translation
    point_camera = frame_rotation @ point_layout + frame_translation
    projected, _ = cv2.projectPoints(
        point_camera.reshape(1, 1, 3).astype(np.float32),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(2)


def _corner_errors(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    errors = np.zeros(len(corner_observations), dtype=np.float64)
    for index, (observation, keep) in enumerate(zip(corner_observations, inlier_mask, strict=True)):
        if not keep:
            errors[index] = np.inf
            continue
        frame_pose = frame_poses[observation.frame_index]
        marker_pose = marker_poses.get(observation.marker_id)
        if frame_pose is None or marker_pose is None:
            errors[index] = np.inf
            continue
        projected = _project_corner(
            observation.corner_index,
            observation.marker_id,
            marker_pose,
            frame_pose,
            object_points_by_marker,
            camera_matrix,
            dist_coeffs,
        )
        errors[index] = float(np.linalg.norm(projected - observation.image_point))
    return errors


def _footprints_from_poses(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    marker_sizes_m: Mapping[int, float],
) -> dict[int, MarkerFootprint]:
    object_points_by_marker = _object_points_by_marker(marker_sizes_m)
    footprints: dict[int, MarkerFootprint] = {}
    for marker_id, (rotation, translation) in marker_poses.items():
        object_points = object_points_by_marker[marker_id]
        payload = {}
        for corner_index, corner_name in enumerate(CORNER_NAMES):
            point = rotation @ object_points[corner_index] + translation
            payload[corner_name] = point.tolist()
        footprints[marker_id] = footprint_from_dict(marker_id, payload)
    return footprints


def _edge_diagnostics(
    pair: MarkerPair,
    edge: _PairConsensus,
) -> EdgeDiagnostics:
    translations: list[float] = []
    rotations: list[float] = []
    for frame_index in edge.inlier_frames:
        rotation, translation = edge.inlier_hypotheses[frame_index]
        translations.append(float(np.linalg.norm(translation - edge.translation_ba)))
        rotations.append(_rotation_geodesic_deg(rotation, edge.rotation_ba))
    return EdgeDiagnostics(
        marker_a=pair[0],
        marker_b=pair[1],
        inlier_count=len(edge.inlier_frames),
        translation_rms_m=float(np.sqrt(np.mean(np.square(translations))) if translations else 0.0),
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(rotations))) if rotations else 0.0),
    )


def _build_quality_report(
    corner_observations: list[_CornerObservation],
    inlier_mask: np.ndarray,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    frame_poses: list[tuple[np.ndarray, np.ndarray] | None],
    pair_consensus: dict[MarkerPair, _PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
    missing_ids: frozenset[int],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    assignment_rejections: AssignmentRejectionSummary | None = None,
    assignment_rejection_records: tuple[FrameAssignmentRejectionRecord, ...] | None = None,
    dropped_pair_edges: tuple[DroppedPairEdge, ...] | None = None,
    restored_pair_edges: tuple[RestoredPairEdge, ...] | None = None,
    anchor_core: AnchorCoreDiagnostics | None = None,
) -> CalibrationQualityReport:
    errors = _corner_errors(
        corner_observations,
        inlier_mask,
        marker_poses,
        frame_poses,
        object_points_by_marker,
        camera_matrix,
        dist_coeffs,
    )
    finite_errors = errors[np.isfinite(errors) & inlier_mask]
    per_marker: dict[int, list[float]] = {}
    for observation, error, keep in zip(corner_observations, errors, inlier_mask, strict=True):
        if not keep or not np.isfinite(error):
            continue
        per_marker.setdefault(observation.marker_id, []).append(error)
    per_marker_rms = {
        marker_id: float(np.sqrt(np.mean(np.square(values))))
        for marker_id, values in per_marker.items()
    }
    edge_reports = [_edge_diagnostics(pair, edge) for pair, edge in sorted(pair_consensus.items())]
    connected = frozenset(_connected_marker_ids(pair_consensus, reference_marker_id))
    observed_ids = {
        observation.marker_id
        for observation, keep in zip(corner_observations, inlier_mask, strict=True)
        if keep
    }
    final_frame_count = _covisible_frame_count(corner_observations, inlier_mask)
    return CalibrationQualityReport(
        reprojection_rms_px=float(np.sqrt(np.mean(np.square(finite_errors))) if finite_errors.size else float("inf")),
        per_marker_reprojection_rms_px=per_marker_rms,
        edges=tuple(edge_reports),
        pair_translation_rms_max_m=max((edge.translation_rms_m for edge in edge_reports), default=0.0),
        pair_rotation_rms_max_deg=max((edge.rotation_rms_deg for edge in edge_reports), default=0.0),
        frame_count=final_frame_count,
        observation_count=len(corner_observations),
        inlier_corner_count=int(np.count_nonzero(inlier_mask)),
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        connected_marker_ids=connected,
        missing_expected_ids=missing_ids,
        unused_expected_ids=frozenset(set(expected_ids) - observed_ids),
        assignment_rejections=assignment_rejections,
        assignment_rejection_records=assignment_rejection_records,
        dropped_pair_edges=dropped_pair_edges,
        restored_pair_edges=restored_pair_edges,
        anchor_core=anchor_core,
    )


def _collect_quality_gate_failures(
    quality: CalibrationQualityReport,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    expected_ids: list[int],
) -> tuple[QualityGateFailure, ...]:
    failures: list[QualityGateFailure] = []
    if quality.reprojection_rms_px > settings.reprojection_rms_gate_px:
        failures.append(
            QualityGateFailure(
                "strict",
                (
                    f"Global reprojection RMS {quality.reprojection_rms_px:.3f} px exceeds "
                    f"{settings.reprojection_rms_gate_px:.3f} px gate."
                ),
            )
        )
    for marker_id in expected_ids:
        marker_rms = quality.per_marker_reprojection_rms_px.get(marker_id)
        if marker_rms is None:
            failures.append(
                QualityGateFailure(
                    "data",
                    f"Marker {marker_id} has no inlier reprojection samples after calibration.",
                )
            )
            continue
        if marker_rms > settings.reprojection_rms_gate_px:
            failures.append(
                QualityGateFailure(
                    "strict",
                    (
                        f"Marker {marker_id} reprojection RMS {marker_rms:.3f} px exceeds "
                        f"{settings.reprojection_rms_gate_px:.3f} px gate."
                    ),
                )
            )
    for edge in quality.edges:
        pair = (edge.marker_a, edge.marker_b)
        translation_gate = _pair_translation_gate(settings, marker_sizes_m, pair)
        if edge.translation_rms_m > translation_gate:
            failures.append(
                QualityGateFailure(
                    "strict",
                    (
                        f"Pair ({pair[0]},{pair[1]}) translation RMS {edge.translation_rms_m:.4f} m exceeds "
                        f"{translation_gate:.4f} m gate."
                    ),
                )
            )
    if quality.pair_rotation_rms_max_deg > settings.pair_rotation_rms_gate_deg:
        failures.append(
            QualityGateFailure(
                "strict",
                (
                    f"Pair rotation RMS {quality.pair_rotation_rms_max_deg:.2f} deg exceeds "
                    f"{settings.pair_rotation_rms_gate_deg:.2f} deg gate."
                ),
            )
        )
    if quality.missing_expected_ids:
        failures.append(
            QualityGateFailure(
                "connectivity",
                f"Missing expected marker IDs: {sorted(quality.missing_expected_ids)}.",
            )
        )
    return tuple(failures)


def _check_quality_gates(
    quality: CalibrationQualityReport,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    expected_ids: list[int],
) -> str | None:
    failures = _collect_quality_gate_failures(quality, settings, marker_sizes_m, expected_ids)
    return failures[0].message if failures else None


def _empty_quality(
    missing_expected_ids: frozenset[int],
    connected_marker_ids: frozenset[int],
    input_frame_count: int = 0,
    rejected_frame_count: int = 0,
    accepted_frame_count: int = 0,
) -> CalibrationQualityReport:
    return CalibrationQualityReport(
        reprojection_rms_px=float("inf"),
        per_marker_reprojection_rms_px={},
        edges=(),
        pair_translation_rms_max_m=float("inf"),
        pair_rotation_rms_max_deg=float("inf"),
        frame_count=0,
        observation_count=0,
        inlier_corner_count=0,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        connected_marker_ids=connected_marker_ids,
        missing_expected_ids=missing_expected_ids,
        unused_expected_ids=frozenset(),
    )


def _quality_from_pairs(
    pair_consensus: dict[MarkerPair, _PairConsensus],
    expected_ids: list[int],
    reference_marker_id: int,
    missing_ids: frozenset[int],
    input_frame_count: int,
    rejected_frame_count: int,
    accepted_frame_count: int,
    observation_count: int,
    assignment_rejections: AssignmentRejectionSummary | None = None,
    assignment_rejection_records: tuple[FrameAssignmentRejectionRecord, ...] | None = None,
    dropped_pair_edges: tuple[DroppedPairEdge, ...] | None = None,
    restored_pair_edges: tuple[RestoredPairEdge, ...] | None = None,
    anchor_core: AnchorCoreDiagnostics | None = None,
) -> CalibrationQualityReport:
    edge_reports = [_edge_diagnostics(pair, edge) for pair, edge in sorted(pair_consensus.items())]
    return CalibrationQualityReport(
        reprojection_rms_px=float("inf"),
        per_marker_reprojection_rms_px={},
        edges=tuple(edge_reports),
        pair_translation_rms_max_m=max((edge.translation_rms_m for edge in edge_reports), default=0.0),
        pair_rotation_rms_max_deg=max((edge.rotation_rms_deg for edge in edge_reports), default=0.0),
        frame_count=accepted_frame_count,
        observation_count=observation_count,
        inlier_corner_count=0,
        input_frame_count=input_frame_count,
        rejected_frame_count=rejected_frame_count,
        accepted_frame_count=accepted_frame_count,
        connected_marker_ids=frozenset(_connected_marker_ids(pair_consensus, reference_marker_id)),
        missing_expected_ids=missing_ids,
        unused_expected_ids=frozenset(set(expected_ids) - _connected_marker_ids(pair_consensus, reference_marker_id)),
        assignment_rejections=assignment_rejections,
        assignment_rejection_records=assignment_rejection_records,
        dropped_pair_edges=dropped_pair_edges,
        restored_pair_edges=restored_pair_edges,
        anchor_core=anchor_core,
    )


def _rotation_geodesic_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        return np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    if rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        return np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
    return np.array(
        [
            (rotation[1, 0] - rotation[0, 1]) / s,
            (rotation[0, 2] + rotation[2, 0]) / s,
            (rotation[1, 2] + rotation[2, 1]) / s,
            0.25 * s,
        ],
        dtype=np.float64,
    )


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _synth_marker_corners(
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    marker_ids: Sequence[int],
    layout_rotation: np.ndarray,
    layout_translation: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[int, np.ndarray]:
    markers: dict[int, np.ndarray] = {}
    for marker_id in marker_ids:
        marker_rotation, marker_translation = marker_poses[marker_id]
        image_corners = []
        for corner_index in range(4):
            layout_point = marker_rotation @ object_points[corner_index] + marker_translation
            camera_point = layout_rotation @ layout_point + layout_translation
            projected, _ = cv2.projectPoints(
                camera_point.reshape(1, 1, 3).astype(np.float32),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                camera_matrix,
                dist_coeffs,
            )
            image_corners.append(projected.reshape(2))
        markers[marker_id] = np.stack(image_corners, axis=0)
    return markers


def _synth_pair_observations(
    num_frames: int,
    marker_poses: dict[int, tuple[np.ndarray, np.ndarray]],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    corrupt_frames: frozenset[int] = frozenset(),
    corrupt_offset: np.ndarray | None = None,
    varying_corrupt: bool = False,
) -> list[FrameObservation]:
    observations: list[FrameObservation] = []
    base_wrong_offset = (
        np.array([0.20, 0.0, -0.08], dtype=np.float64)
        if corrupt_offset is None
        else np.asarray(corrupt_offset, dtype=np.float64)
    )
    for frame_index in range(num_frames):
        layout_rotation, _ = cv2.Rodrigues(np.array([0.1, -0.15 + 0.002 * frame_index, 0.05]))
        layout_translation = np.array([0.02, -0.01, 0.6 + 0.002 * frame_index], dtype=np.float64)
        frame_poses = dict(marker_poses)
        if frame_index in corrupt_frames:
            offset = base_wrong_offset
            if varying_corrupt:
                offset = base_wrong_offset + np.array([0.01 * frame_index, 0.0, 0.0])
            frame_poses[1] = (marker_poses[1][0], marker_poses[0][1] + offset)
        markers = _synth_marker_corners(
            frame_poses,
            (1, 0),
            layout_rotation,
            layout_translation,
            object_points,
            camera_matrix,
            dist_coeffs,
        )
        observations.append(FrameObservation(frame_id=frame_index, markers=markers))
    return observations


def _self_check() -> None:
    """Minimal synthetic sanity check for import-time regression."""
    marker_size = 0.07
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    object_points = marker_corner_object_points(marker_size)
    settings = CalibrationSettings(min_inliers_per_edge=20)

    ref_rotation, ref_translation = _reference_gauge_pose(marker_size)
    pair_poses = {
        0: (ref_rotation, ref_translation),
        1: (ref_rotation, ref_translation + np.array([0.12, 0.0, -0.05])),
    }

    mostly_good = calibrate_marker_layout(
        _synth_pair_observations(
            25,
            pair_poses,
            object_points,
            camera_matrix,
            dist_coeffs,
            corrupt_frames=frozenset({2, 7, 11, 16, 22}),
            corrupt_offset=np.array([0.20, 0.0, -0.08]),
        ),
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=marker_size,
        settings=settings,
    )
    assert mostly_good.failure_reason is None, mostly_good.failure_reason
    assert mostly_good.layout is not None
    assert mostly_good.quality is not None
    assert mostly_good.quality.input_frame_count == 25
    assert mostly_good.quality.rejected_frame_count == 5
    assert mostly_good.quality.accepted_frame_count == 20

    all_bad = calibrate_marker_layout(
        _synth_pair_observations(
            25,
            pair_poses,
            object_points,
            camera_matrix,
            dist_coeffs,
            corrupt_frames=frozenset(range(25)),
            varying_corrupt=True,
        ),
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=marker_size,
        settings=settings,
    )
    assert all_bad.layout is None
    assert all_bad.failure_reason is not None
    assert all_bad.quality is not None
    assert all_bad.quality.input_frame_count == 25
    assert all_bad.quality.accepted_frame_count == 0

    marker_poses = {
        0: (ref_rotation, ref_translation),
        1: (ref_rotation, ref_translation + np.array([0.12, 0.0, -0.05])),
        2: (ref_rotation, ref_translation + np.array([0.24, 0.0, -0.10])),
        3: (ref_rotation, ref_translation + np.array([0.36, 0.0, -0.15])),
    }

    observations: list[FrameObservation] = []
    chain_pairs = [(0, 1), (1, 2), (2, 3)]
    frame_index = 0
    for marker_a, marker_b in chain_pairs:
        for _ in range(25):
            layout_rotation, _ = cv2.Rodrigues(np.array([0.1, -0.15 + 0.002 * frame_index, 0.05]))
            layout_translation = np.array([0.02, -0.01, 0.6 + 0.002 * frame_index], dtype=np.float64)
            markers = _synth_marker_corners(
                marker_poses,
                (marker_b, marker_a),
                layout_rotation,
                layout_translation,
                object_points,
                camera_matrix,
                dist_coeffs,
            )
            observations.append(FrameObservation(frame_id=frame_index, markers=markers))
            frame_index += 1

    result = calibrate_marker_layout(
        observations,
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1, 2, 3],
        reference_marker_id=0,
        marker_size_m=marker_size,
        settings=settings,
    )
    assert result.failure_reason is None, result.failure_reason
    assert result.layout is not None
    for edge in result.quality.edges if result.quality else ():
        assert edge.inlier_count >= 20
    recovered = result.layout.footprints[3].bottom_left - result.layout.footprints[0].bottom_left
    assert np.linalg.norm(recovered - np.array([0.36, 0.0, -0.15])) < 0.02


def _input_boundary_self_check() -> None:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    valid_obs = [
        FrameObservation(
            frame_id=0,
            markers={
                0: np.zeros((4, 2), dtype=np.float64),
                1: np.ones((4, 2), dtype=np.float64),
            },
        )
    ]

    duplicate_ids = calibrate_marker_layout(
        valid_obs,
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1, 0],
        reference_marker_id=0,
        marker_size_m=0.07,
    )
    assert duplicate_ids.failure_reason is not None
    assert "duplicates" in duplicate_ids.failure_reason

    duplicate_frames = calibrate_marker_layout(
        [valid_obs[0], valid_obs[0]],
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=0.07,
    )
    assert duplicate_frames.failure_reason is not None
    assert "Duplicate FrameObservation.frame_id" in duplicate_frames.failure_reason

    bad_corners = calibrate_marker_layout(
        [
            FrameObservation(
                frame_id=0,
                markers={0: np.zeros((3, 2)), 1: np.ones((4, 2))},
            )
        ],
        camera_matrix,
        dist_coeffs,
        expected_marker_ids=[0, 1],
        reference_marker_id=0,
        marker_size_m=0.07,
    )
    assert bad_corners.failure_reason is not None
    assert "Malformed corners" in bad_corners.failure_reason


if __name__ == "__main__":
    _self_check()
    _input_boundary_self_check()
    print("marker_layout_calibration self-check passed")
