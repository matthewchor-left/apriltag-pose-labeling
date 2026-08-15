"""Semi-transparent CAD mesh silhouette overlays."""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from object_apriltag.cad import CadModel, CadRegistration
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import (
    OBJECT_AXIS_FLIP,
    MarkerLayout,
    object_reference_origin,
    object_reference_orientation,
)

_CAMERA_NEAR_M = 1e-4

_PART_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (60, 76, 231),
    (80, 180, 60),
    (40, 170, 220),
    (130, 70, 210),
    (50, 140, 200),
    (90, 120, 30),
    (180, 90, 60),
    (200, 60, 120),
)


def part_color_bgr(part_name: str) -> tuple[int, int, int]:
    """Return a stable BGR color for a CAD mesh part name."""
    index = zlib.crc32(part_name.encode("utf-8")) % len(_PART_COLORS_BGR)
    return _PART_COLORS_BGR[index]


def cad_points_to_layout(points_cad: np.ndarray, registration: CadRegistration) -> np.ndarray:
    """Map CAD-frame points into marker_model/layout coordinates."""
    points = np.asarray(points_cad, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = (registration.transform_4x4 @ homogeneous.T).T[:, :3]
    if not np.all(np.isfinite(transformed)):
        raise ValueError("CAD registration produced non-finite layout coordinates.")
    return transformed


def layout_points_to_camera(
    points_layout: np.ndarray,
    pose: ObjectPose,
    marker_model: MarkerLayout,
) -> np.ndarray:
    """Map layout-frame points into the fused object camera frame."""
    layout = np.asarray(points_layout, dtype=np.float64).reshape(-1, 3)
    origin = object_reference_origin(marker_model)
    orientation = object_reference_orientation(marker_model)
    points_object = (OBJECT_AXIS_FLIP @ orientation.T @ (layout - origin).T).T
    camera_points = (pose.rotation @ points_object.T).T + pose.origin.reshape(1, 3)
    return camera_points


def project_camera_points(
    points_camera: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project camera-frame points through intrinsics and distortion."""
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(
        points,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def render_cad_model_view(
    frame_shape: tuple[int, ...],
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_model: MarkerLayout,
    cad_model: CadModel,
    registration: CadRegistration,
) -> np.ndarray:
    """Render an opaque CAD silhouette on a pure-black background.

    Part-level painter ordering is used; exact inter-part occlusion would require
    a z-buffer. Material colors come from each part's glTF baseColorFactor.
    """
    height, width = _frame_shape_hw(frame_shape)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    parts = _collect_visible_parts(
        pose=pose,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_model=marker_model,
        cad_model=cad_model,
        registration=registration,
        color_mode="material",
    )
    if not parts:
        return frame

    _paint_parts_opaque(frame, parts)
    return frame


def _frame_shape_hw(frame_shape: tuple[int, ...]) -> tuple[int, int]:
    if len(frame_shape) == 2:
        return int(frame_shape[0]), int(frame_shape[1])
    if len(frame_shape) >= 2:
        return int(frame_shape[0]), int(frame_shape[1])
    raise ValueError("frame_shape must be (height, width) or a camera frame with shape (H, W, 3).")


def draw_cad_model_overlay(
    frame: np.ndarray,
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_model: MarkerLayout,
    cad_model: CadModel,
    registration: CadRegistration,
    *,
    alpha: float = 0.35,
) -> None:
    """Draw a semi-transparent colored CAD silhouette onto ``frame`` in place."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("CAD overlay frame must be a BGR image with shape (H, W, 3).")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"CAD overlay alpha must be in (0, 1], got {alpha}.")

    parts = _collect_visible_parts(
        pose=pose,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        marker_model=marker_model,
        cad_model=cad_model,
        registration=registration,
        color_mode="debug",
    )
    if not parts:
        return

    _paint_parts_alpha(frame, parts, alpha=alpha)


def _paint_parts_alpha(
    frame: np.ndarray,
    parts: list[tuple[float, tuple[int, int, int], np.ndarray]],
    *,
    alpha: float,
) -> None:
    # Part-level painter ordering keeps this diagnostic overlay cheap. Exact
    # inter-part occlusion would require a z-buffer.
    parts.sort(key=lambda item: item[0], reverse=True)
    height, width = frame.shape[:2]
    lower = np.array([-2 * width, -2 * height], dtype=np.float64)
    upper = np.array([3 * width, 3 * height], dtype=np.float64)
    for _, color, image_triangles in parts:
        polygons = np.rint(np.clip(image_triangles, lower, upper)).astype(np.int32)
        x0 = max(0, int(polygons[:, :, 0].min()))
        y0 = max(0, int(polygons[:, :, 1].min()))
        x1 = min(width - 1, int(polygons[:, :, 0].max()))
        y1 = min(height - 1, int(polygons[:, :, 1].max()))
        if x0 > x1 or y0 > y1:
            continue

        offset = np.array([x0, y0], dtype=np.int32)
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, polygons - offset, 255, lineType=cv2.LINE_AA)
        covered = mask > 0
        if not np.any(covered):
            continue

        coverage = alpha * mask[covered, None].astype(np.float64) / 255.0
        frame_region = frame[y0 : y1 + 1, x0 : x1 + 1]
        base = frame_region[covered].astype(np.float64)
        frame_region[covered] = (
            (1.0 - coverage) * base + coverage * np.asarray(color)
        ).astype(np.uint8)


def _paint_parts_opaque(
    frame: np.ndarray,
    parts: list[tuple[float, tuple[int, int, int], np.ndarray]],
) -> None:
    # Part-level painter ordering keeps this side view cheap. Exact inter-part
    # occlusion would require a z-buffer.
    parts.sort(key=lambda item: item[0], reverse=True)
    height, width = frame.shape[:2]
    lower = np.array([-2 * width, -2 * height], dtype=np.float64)
    upper = np.array([3 * width, 3 * height], dtype=np.float64)
    for _, color, image_triangles in parts:
        polygons = np.rint(np.clip(image_triangles, lower, upper)).astype(np.int32)
        x0 = max(0, int(polygons[:, :, 0].min()))
        y0 = max(0, int(polygons[:, :, 1].min()))
        x1 = min(width - 1, int(polygons[:, :, 0].max()))
        y1 = min(height - 1, int(polygons[:, :, 1].max()))
        if x0 > x1 or y0 > y1:
            continue

        offset = np.array([x0, y0], dtype=np.int32)
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, polygons - offset, 255, lineType=cv2.LINE_AA)
        covered = mask > 0
        if not np.any(covered):
            continue

        frame_region = frame[y0 : y1 + 1, x0 : x1 + 1]
        frame_region[covered] = np.asarray(color, dtype=np.uint8)


def _collect_visible_parts(
    *,
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_model: MarkerLayout,
    cad_model: CadModel,
    registration: CadRegistration,
    color_mode: str = "debug",
) -> list[tuple[float, tuple[int, int, int], np.ndarray]]:
    if color_mode not in ("debug", "material"):
        raise ValueError(f"Unsupported CAD part color mode {color_mode!r}.")

    collected: list[tuple[float, tuple[int, int, int], np.ndarray]] = []
    for part in cad_model.parts:
        layout_points = cad_points_to_layout(part.vertices, registration)
        camera_points = layout_points_to_camera(layout_points, pose, marker_model)
        image_points = project_camera_points(camera_points, camera_matrix, dist_coeffs)
        triangle_depths = camera_points[part.triangles, 2]
        image_triangles = image_points[part.triangles]
        visible = np.all(triangle_depths > _CAMERA_NEAR_M, axis=1)
        visible &= np.all(np.isfinite(image_triangles), axis=(1, 2))
        if not np.any(visible):
            continue
        color = (
            part.material_color_bgr
            if color_mode == "material"
            else part_color_bgr(part.name)
        )
        collected.append(
            (
                float(np.mean(triangle_depths[visible])),
                color,
                image_triangles[visible],
            )
        )
    return collected
