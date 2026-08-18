"""Tests for automatic reference marker selection."""

from __future__ import annotations

import unittest

from object_apriltag.marker_layout_calibration.reference_selection import (
    select_reference_marker,
)


class ReferenceSelectionTests(unittest.TestCase):
    def test_selects_reference_with_most_connected_keypoint_sources(self) -> None:
        pairs = [(19, 22), (22, 23), (23, 28), (0, 2)]
        expected_ids = [0, 2, 19, 22, 23, 28]
        keypoint_sources = {
            "left": (19, "top_right", 0.004),
            "front": (22, "top_right", 0.004),
            "right-center": (22, "top_right", 0.004),
            "back": (28, "top_right", 0.004),
        }
        # All roots reach four tags; marker 22 wins because two sources use it directly.
        self.assertEqual(
            select_reference_marker(pairs, expected_ids, keypoint_sources),
            22,
        )

    def test_tie_breaks_by_keypoint_sources_on_root(self) -> None:
        pairs = [(19, 22), (22, 23)]
        expected_ids = [19, 22, 23]
        keypoint_sources = {
            "left": (19, "top_left", 0.0),
            "front": (22, "top_left", 0.0),
            "right": (22, "top_left", 0.0),
        }
        self.assertEqual(
            select_reference_marker(pairs, expected_ids, keypoint_sources),
            22,
        )

    def test_falls_back_to_largest_component_without_keypoint_sources(self) -> None:
        pairs = [(5, 6), (6, 12), (0, 1)]
        expected_ids = [0, 1, 5, 6, 12]
        self.assertEqual(
            select_reference_marker(pairs, expected_ids, {}),
            5,
        )

    def test_skips_isolated_expected_id_when_pair_graph_has_edges(self) -> None:
        pairs = [(22, 23), (23, 24)]
        expected_ids = [19, 22, 23, 24]
        keypoint_sources = {
            "isolated": (19, "top_left", 0.0),
            "connected": (22, "top_left", 0.0),
        }
        self.assertEqual(
            select_reference_marker(pairs, expected_ids, keypoint_sources),
            22,
        )


if __name__ == "__main__":
    unittest.main()
