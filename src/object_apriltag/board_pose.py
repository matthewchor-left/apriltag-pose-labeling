"""ChArUco detection helpers shared by calibration CLIs."""

from __future__ import annotations

import numpy as np


def parse_charuco_detection(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return matching (N,2) corners and (N,) ids, or None when unusable."""
    if charuco_ids is None:
        return None
    ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    if ids_flat.shape[0] == 0:
        return None
    if charuco_corners is None:
        return None
    corners_flat = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    if corners_flat.shape[0] != ids_flat.shape[0]:
        return None
    return corners_flat, ids_flat


def charuco_corners_consistent(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> bool:
    return parse_charuco_detection(charuco_corners, charuco_ids) is not None


def charuco_draw_arrays(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Arrays shaped for OpenCV drawDetectedCornersCharuco."""
    parsed = parse_charuco_detection(charuco_corners, charuco_ids)
    if parsed is None:
        return None
    corners_flat, ids_flat = parsed
    return corners_flat.reshape(-1, 1, 2), ids_flat.reshape(-1, 1)
