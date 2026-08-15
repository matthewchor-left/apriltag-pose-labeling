"""Settings, marker IDs/sizes, and observation validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from object_apriltag.layout import resolve_marker_sizes
from object_apriltag.pose import marker_corner_object_points

from object_apriltag.marker_layout_calibration.types import CalibrationSettings, FrameObservation

def validate_settings(settings: CalibrationSettings) -> str | None:
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


def parse_expected_marker_ids(
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


def validate_marker_size(marker_size_m: float) -> str | None:
    if not np.isfinite(marker_size_m) or marker_size_m <= 0.0:
        return "marker_size_m must be a finite positive number."
    return None


def uniform_marker_sizes(expected_ids: Sequence[int], marker_size_m: float) -> dict[int, float]:
    return {int(marker_id): float(marker_size_m) for marker_id in expected_ids}


def validate_marker_sizes(
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
        failure = validate_marker_size(size)
        if failure is not None:
            return f"marker_sizes_m[{marker_id}]: {failure}"
    return None


def object_points_by_marker(marker_sizes_m: Mapping[int, float]) -> dict[int, np.ndarray]:
    by_size: dict[float, np.ndarray] = {}
    result: dict[int, np.ndarray] = {}
    for marker_id, size in marker_sizes_m.items():
        if size not in by_size:
            by_size[size] = marker_corner_object_points(size).astype(np.float64)
        result[int(marker_id)] = by_size[size]
    return result


def parse_marker_id_range_token(token: str) -> tuple[list[int] | None, str | None]:
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
        marker_ids, parse_failure = parse_marker_id_range_token(id_part)
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
    default_failure = validate_marker_size(default_size_m)
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
    sizes_failure = validate_marker_sizes(resolved, expected_ids)
    if sizes_failure is not None:
        return None, sizes_failure
    return resolved, None


def validate_camera_inputs(
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


def parse_marker_corners(
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


def validate_observations(
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
            _, failure = parse_marker_corners(corners, observation.frame_id, marker_id)
            if failure is not None:
                return failure
    return None

