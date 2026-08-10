"""Inspect marker layout transforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from paddle_aruco.layout import CORNER_NAMES, DEFAULT_MARKER_LAYOUT_PATH, footprint_edge_lengths, load_marker_layout
from paddle_aruco.viz import render_marker_layout_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect marker layout transforms.")
    parser.add_argument("--layout", type=Path, default=DEFAULT_MARKER_LAYOUT_PATH)
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    layout = load_marker_layout(args.layout)
    print(f"Layout: {args.layout}")
    print(f"Reference marker id: {layout.reference_marker_id}")
    print(f"Units: {layout.units}")
    print(f"Marker size: {layout.marker_size_m:.4f} m")

    for marker_id in sorted(layout.footprints):
        footprint = layout.footprints[marker_id]
        transform = layout.transforms[marker_id]
        top, right, bottom, left = footprint_edge_lengths(*footprint.corners())
        print(f"\nMarker {marker_id}")
        for corner_name in CORNER_NAMES:
            point = footprint.corners_by_name()[corner_name]
            print(f"  {corner_name}: {point.round(6).tolist()}")
        print(f"  edges (top/right/bottom/left): {top:.4f} / {right:.4f} / {bottom:.4f} / {left:.4f} m")
        print(f"  offset (marker frame): {transform.offset.round(6).tolist()}")
        print(f"  rotation det: {np.linalg.det(transform.rotation):.6f}")

    if args.visualize:
        plot_bgr = render_marker_layout_plot(layout)
        cv2.imshow("Marker layout", plot_bgr)
        print("\nMarker layout plot open. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
