"""Automatic reference marker selection for marker layout calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from object_apriltag.marker_layout_calibration.discrete_graph import (
    connected_marker_ids_from_pairs,
)
from object_apriltag.marker_layout_calibration.types import MarkerPair


def count_connected_keypoint_sources(
    component: frozenset[int],
    keypoint_sources: Mapping[str, tuple[int, str, float]],
) -> int:
    """Count keypoint sources whose marker ID lies in a connected component."""
    return sum(1 for _, (marker_id, _, _) in keypoint_sources.items() if marker_id in component)


def select_reference_marker(
    pairs: Iterable[MarkerPair],
    expected_ids: Sequence[int],
    keypoint_sources: Mapping[str, tuple[int, str, float]],
) -> int:
    """Pick a reference marker maximizing connected keypoint-source coverage.

    For each candidate root, breadth-first expansion over ``pairs`` defines the
    connected component. The winning marker maximizes the number of
    ``keypoint_sources`` entries whose marker ID falls in that component, then
    component size, then lowest marker ID for stability.

    Args:
        pairs: Undirected marker-pair edges used for connectivity analysis.
        expected_ids: Marker IDs eligible for reference selection.
        keypoint_sources: Object-model keypoint derivation map.

    Returns:
        Selected reference marker ID.

    Raises:
        ValueError: When ``expected_ids`` is empty.
    """
    pair_list = list(pairs)
    candidates = sorted({int(marker_id) for marker_id in expected_ids})
    if not candidates:
        raise ValueError("expected_ids is empty.")

    best_marker = candidates[0]
    best_score = (-1, -1, -1, candidates[0])
    for marker_id in candidates:
        component = frozenset(connected_marker_ids_from_pairs(pair_list, marker_id))
        keypoint_tags = count_connected_keypoint_sources(component, keypoint_sources)
        tags_on_root = sum(
            1 for _, (source_marker_id, _, _) in keypoint_sources.items() if source_marker_id == marker_id
        )
        score = (keypoint_tags, tags_on_root, len(component), -marker_id)
        if score > best_score:
            best_score = score
            best_marker = marker_id
    return best_marker
