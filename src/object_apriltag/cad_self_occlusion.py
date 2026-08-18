"""CAD mesh self-occlusion tests for YOLO pose keypoint visibility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from object_apriltag.cad import CadModel, CadRegistration
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import MarkerLayout, camera_point_to_layout_point

# Landmarks are authored on mesh vertices in meters; 1 mm is the maximum snap distance.
LANDMARK_VERTEX_MATCH_TOLERANCE_M = 1e-3

# Ignore ray hits within 1 µm of the segment start (camera origin).
RAY_SEGMENT_ORIGIN_EPSILON_M = 1e-6

# CAD intersections within 5 mm of the landmark endpoint belong to the landmark
# neighborhood and do not mark the keypoint occluded (intentional visibility tolerance).
RAY_SEGMENT_TARGET_VISIBILITY_TOLERANCE_M = 0.005

# Numerical slack for segment-length comparisons only (not a visibility tolerance).
_RAY_SEGMENT_DISTANCE_FP_EPSILON = 1e-9
_RAY_PARALLEL_EPSILON = 1e-12

YOLO_KEYPOINT_OCCLUDED = 1
YOLO_KEYPOINT_VISIBLE = 2


@dataclass(frozen=True)
class CadSelfOcclusionContext:
    """Precomputed CAD mesh topology and landmark-to-vertex associations."""

    vertices: np.ndarray
    triangles: np.ndarray
    landmark_vertex_indices: np.ndarray
    vertex_triangle_indices: tuple[frozenset[int], ...]


def build_cad_self_occlusion_context(
    cad_model: CadModel,
    cad_landmarks: Mapping[str, np.ndarray],
    landmark_names: Sequence[str],
    *,
    match_tolerance_m: float = LANDMARK_VERTEX_MATCH_TOLERANCE_M,
) -> CadSelfOcclusionContext:
    """Associate required landmarks with mesh vertices and precompute incidence data."""
    vertices, triangles = _concatenate_mesh_parts(cad_model)
    if vertices.size == 0 or triangles.size == 0:
        raise ValueError("CAD model must contain triangle mesh geometry for self-occlusion.")

    vertex_triangle_indices = _build_vertex_triangle_incidence(len(vertices), triangles)
    landmark_vertex_indices = np.empty(len(landmark_names), dtype=np.int64)
    for index, name in enumerate(landmark_names):
        if name not in cad_landmarks:
            raise ValueError(f"CAD landmark {name!r} is missing from loaded landmarks.")
        position = np.asarray(cad_landmarks[name], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError(f"CAD landmark {name!r} must be a finite 3D position.")
        distances = np.linalg.norm(vertices - position, axis=1)
        # ponytail: ties pick the lowest vertex index; coincident vertices across parts are expected.
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        if nearest_distance > match_tolerance_m:
            raise ValueError(
                f"CAD landmark {name!r} has no mesh vertex within {match_tolerance_m:g} m "
                f"(nearest {nearest_distance:.6g} m)."
            )
        landmark_vertex_indices[index] = nearest_index

    return CadSelfOcclusionContext(
        vertices=vertices,
        triangles=triangles,
        landmark_vertex_indices=landmark_vertex_indices,
        vertex_triangle_indices=vertex_triangle_indices,
    )


def layout_point_to_cad(point_layout: np.ndarray, registration: CadRegistration) -> np.ndarray:
    """Map one marker-model/layout point into CAD coordinates."""
    point = np.asarray(point_layout, dtype=np.float64).reshape(3)
    homogeneous = np.append(point, 1.0)
    transformed = np.linalg.inv(registration.transform_4x4) @ homogeneous
    result = transformed[:3]
    if not np.all(np.isfinite(result)):
        raise ValueError("CAD registration inverse produced non-finite coordinates.")
    return result


def camera_origin_in_cad(
    pose: ObjectPose,
    marker_model: MarkerLayout,
    registration: CadRegistration,
) -> np.ndarray:
    """Return the OpenCV camera origin expressed in CAD coordinates."""
    origin_layout = camera_point_to_layout_point(
        np.zeros(3, dtype=np.float64),
        pose.rotation,
        pose.origin,
        marker_model,
    )
    return layout_point_to_cad(origin_layout, registration)


def classify_cad_keypoint_visibility(
    context: CadSelfOcclusionContext,
    landmark_positions_cad: np.ndarray,
    camera_origin_cad: np.ndarray,
) -> np.ndarray:
    """Classify each landmark as visible (2) or CAD-self-occluded (1)."""
    points = np.asarray(landmark_positions_cad, dtype=np.float64).reshape(-1, 3)
    origin = np.asarray(camera_origin_cad, dtype=np.float64).reshape(3)
    if points.shape[0] != len(context.landmark_vertex_indices):
        raise ValueError(
            "landmark_positions_cad row count must match the occlusion context landmark count."
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(origin)):
        raise ValueError("Camera origin and landmark positions must be finite.")

    visibility = np.full(points.shape[0], YOLO_KEYPOINT_VISIBLE, dtype=np.int32)
    for index, target in enumerate(points):
        skip_triangles = context.vertex_triangle_indices[int(context.landmark_vertex_indices[index])]
        if _segment_blocked_by_triangles(
            origin,
            target,
            context.vertices,
            context.triangles,
            skip_triangle_indices=skip_triangles,
            origin_epsilon=RAY_SEGMENT_ORIGIN_EPSILON_M,
            target_visibility_tolerance_m=RAY_SEGMENT_TARGET_VISIBILITY_TOLERANCE_M,
        ):
            visibility[index] = YOLO_KEYPOINT_OCCLUDED
    return visibility


def _concatenate_mesh_parts(cad_model: CadModel) -> tuple[np.ndarray, np.ndarray]:
    vertex_blocks: list[np.ndarray] = []
    triangle_blocks: list[np.ndarray] = []
    vertex_offset = 0
    for part in cad_model.parts:
        vertices = np.asarray(part.vertices, dtype=np.float64).reshape(-1, 3)
        triangles = np.asarray(part.triangles, dtype=np.int64).reshape(-1, 3)
        if vertices.size == 0 or triangles.size == 0:
            continue
        vertex_blocks.append(vertices)
        triangle_blocks.append(triangles + vertex_offset)
        vertex_offset += len(vertices)
    if not vertex_blocks:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64)
    return np.vstack(vertex_blocks), np.vstack(triangle_blocks)


def _build_vertex_triangle_incidence(
    vertex_count: int,
    triangles: np.ndarray,
) -> tuple[frozenset[int], ...]:
    incidence: list[set[int]] = [set() for _ in range(vertex_count)]
    for triangle_index, (v0, v1, v2) in enumerate(triangles):
        incidence[int(v0)].add(triangle_index)
        incidence[int(v1)].add(triangle_index)
        incidence[int(v2)].add(triangle_index)
    return tuple(frozenset(indices) for indices in incidence)


def _segment_blocked_by_triangles(
    origin: np.ndarray,
    target: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    skip_triangle_indices: frozenset[int],
    origin_epsilon: float,
    target_visibility_tolerance_m: float,
) -> bool:
    """Return whether a non-incident triangle blocks the open camera-to-landmark segment.

    A hit at distance ``t`` from the camera blocks when::

        origin_epsilon < t < segment_length - target_visibility_tolerance_m

    Hits at or inside the landmark neighborhood (within ``target_visibility_tolerance_m``
    of the endpoint) are ignored.
    """
    segment = target - origin
    segment_length = float(np.linalg.norm(segment))
    if segment_length <= origin_epsilon + target_visibility_tolerance_m:
        return False

    direction = segment / segment_length
    active_mask = np.ones(len(triangles), dtype=bool)
    if skip_triangle_indices:
        active_mask[list(skip_triangle_indices)] = False
    active_triangles = triangles[active_mask]
    if active_triangles.size == 0:
        return False

    v0 = vertices[active_triangles[:, 0]]
    v1 = vertices[active_triangles[:, 1]]
    v2 = vertices[active_triangles[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, h)
    parallel = np.abs(determinant) < _RAY_PARALLEL_EPSILON
    safe_determinant = np.where(parallel, 1.0, determinant)
    inverse_determinant = 1.0 / safe_determinant

    origin_to_v0 = origin - v0
    barycentric_u = inverse_determinant * np.einsum("ij,ij->i", origin_to_v0, h)
    crossed = np.cross(origin_to_v0, edge1)
    barycentric_v = inverse_determinant * np.einsum("ij,j->i", crossed, direction)
    distance_along_segment = inverse_determinant * np.einsum("ij,ij->i", edge2, crossed)

    hit = (
        ~parallel
        & (barycentric_u >= 0.0)
        & (barycentric_u <= 1.0)
        & (barycentric_v >= 0.0)
        & (barycentric_u + barycentric_v <= 1.0)
        & (distance_along_segment > origin_epsilon)
        & (
            distance_along_segment + _RAY_SEGMENT_DISTANCE_FP_EPSILON
            < segment_length - target_visibility_tolerance_m
        )
    )
    return bool(np.any(hit))
