"""Inspect marker model transforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from object_apriltag.layout import CORNER_NAMES, footprint_edge_lengths, load_marker_model
from object_apriltag.viz import render_marker_model_plot


def main() -> None:
    """Print marker-model geometry and optionally open a static diagram window.

    Raises:
        RuntimeError: ``--marker-model`` file is missing or unreadable.
    """
    parser = argparse.ArgumentParser(
        description="Inspect marker model transforms.",
        epilog=(
            "Terms:\n"
            "  marker model diagram  Static plot of marker footprints in model coordinates.\n"
            "  (Not the live camera view — use object-detect for that.)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--marker-model",
        type=Path,
        required=True,
        help="Marker model JSON (sticker footprint positions) to inspect.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Open a marker model diagram window (footprints in model coordinates). "
            "--no-visualize: print corner coordinates and transforms to the terminal only."
        ),
    )
    args = parser.parse_args()

    marker_model = load_marker_model(args.marker_model)
    print(f"Marker model: {args.marker_model}")
    print(f"Reference marker id: {marker_model.reference_marker_id}")
    if marker_model.anchor_marker_ids is not None:
        print(f"Anchor marker ids: {list(marker_model.anchor_marker_ids)}")
    print(f"Units: {marker_model.units}")
    print(f"Default marker size: {marker_model.marker_size_m:.4f} m")

    for marker_id in sorted(marker_model.footprints):
        footprint = marker_model.footprints[marker_id]
        transform = marker_model.transforms[marker_id]
        resolved_size = marker_model.marker_size_for(marker_id)
        size_note = (
            f"{resolved_size:.4f} m (override)"
            if resolved_size != marker_model.marker_size_m
            else f"{resolved_size:.4f} m (default)"
        )
        top, right, bottom, left = footprint_edge_lengths(*footprint.corners())
        print(f"\nMarker {marker_id}")
        print(f"  resolved size: {size_note}")
        for corner_name in CORNER_NAMES:
            point = footprint.corners_by_name()[corner_name]
            print(f"  {corner_name}: {point.round(6).tolist()}")
        print(f"  edges (top/right/bottom/left): {top:.4f} / {right:.4f} / {bottom:.4f} / {left:.4f} m")
        print(f"  offset (marker frame): {transform.offset.round(6).tolist()}")
        print(f"  rotation det: {np.linalg.det(transform.rotation):.6f}")

    if args.visualize:
        plot_bgr = render_marker_model_plot(marker_model)
        cv2.imshow("Marker model", plot_bgr)
        print("\nMarker model diagram open. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
