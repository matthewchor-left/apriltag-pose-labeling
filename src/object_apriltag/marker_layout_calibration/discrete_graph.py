"""IPPE candidates, pair consensus, connectivity repair, and live readiness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import cv2
import numpy as np

from object_apriltag.marker_layout_calibration.input import (
    parse_expected_marker_ids,
    parse_marker_corners,
    validate_camera_inputs,
    validate_observations,
    validate_settings,
)
from object_apriltag.marker_layout_calibration.solve_primitives import (
    MarkerCandidate,
    PairConsensus,
    average_rotations,
    connected_marker_ids,
    rotation_geodesic_deg,
)
from object_apriltag.marker_layout_calibration.solve_quality import edge_diagnostics, pair_translation_gate
from object_apriltag.marker_layout_calibration.types import (
    CalibrationSettings,
    DroppedPairEdge,
    FrameObservation,
    LivePairReadinessDiagnostics,
    MarkerPair,
    PairReadinessEdge,
    RestoredPairEdge,
)

# single-frame weak edges are not cross-frame consensus; raise if marker
# graphs grow and pairwise quorum needs tuning beyond this floor.
_BEST_EFFORT_WEAK_EDGE_MIN_SUPPORT = 2

def normalize_observations(
    observations: Sequence[FrameObservation],
    expected_ids: list[int],
) -> list[tuple[str | int, dict[int, np.ndarray]]]:
    """Parse and filter observations to expected markers with co-visibility.

    Args:
        observations: Raw per-frame marker corner observations.
        expected_ids: Marker IDs retained during normalization.

    Returns:
        Frames with at least two valid expected markers, each mapping marker ID
        to a ``(4, 2)`` corner array.
    """
    expected_set = set(expected_ids)
    normalized: list[tuple[str | int, dict[int, np.ndarray]]] = []
    for observation in observations:
        markers: dict[int, np.ndarray] = {}
        for marker_id, corners in observation.markers.items():
            marker_id = int(marker_id)
            if marker_id not in expected_set:
                continue
            array, failure = parse_marker_corners(corners, observation.frame_id, marker_id)
            if failure is not None or array is None:
                continue
            markers[marker_id] = array
        if len(markers) >= 2:
            normalized.append((observation.frame_id, markers))
    return normalized


def estimate_frame_candidates(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
    object_points_by_marker: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[tuple[int, dict[int, list[MarkerCandidate]]]]:
    """Run IPPE per marker and retain frames with sufficient candidates.

    Args:
        observations: Normalized corner observations indexed by frame.
        object_points_by_marker: Object-frame corner coordinates per marker.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.

    Returns:
        List of ``(frame_index, candidates)`` for frames where at least two
        markers produced facing-camera IPPE candidates.
    """
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]] = []
    for frame_index, (_, markers) in enumerate(observations):
        candidates: dict[int, list[MarkerCandidate]] = {}
        for marker_id, image_points in markers.items():
            marker_candidates = ippe_candidates(
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


def ippe_candidates(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[MarkerCandidate]:
    """Solve IPPE PnP and retain facing-camera hypotheses with reprojection RMS.

    Args:
        object_points: Marker corner coordinates in the marker frame, shape ``(4, 3)``.
        image_points: Observed corner pixels, shape ``(4, 2)``.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.

    Returns:
        IPPE candidates whose marker +Z normal points toward the camera, each
        annotated with per-solution reprojection RMS in pixels.
    """
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or rvecs is None or tvecs is None:
        return []

    candidates: list[MarkerCandidate] = []
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rotation, _ = cv2.Rodrigues(rvec)
        if not is_marker_facing_camera(rotation):
            continue
        rms = reprojection_rms(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        candidates.append(
            MarkerCandidate(
                rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
                tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
                rotation=rotation.astype(np.float64),
                reprojection_rms_px=rms,
            )
        )
    return candidates


def is_marker_facing_camera(rotation: np.ndarray) -> bool:
    """Return whether the marker +Z axis points toward the camera.

    Args:
        rotation: Marker rotation in the camera frame, shape ``(3, 3)``.

    Returns:
        True when the transformed +Z normal has negative camera Z (toward the camera).
    """
    normal = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return float(normal[2]) < 0.0


def reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """Compute RMS reprojection error for a pose hypothesis.

    Args:
        object_points: Marker corner coordinates in the marker frame.
        image_points: Observed corner pixels.
        rvec: Rotation vector for ``cv2.projectPoints``.
        tvec: Translation vector for ``cv2.projectPoints``.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.

    Returns:
        Root-mean-square pixel distance between observed and projected corners.
    """
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


def relative_marker_transform(
    parent: MarkerCandidate,
    child: MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points from the child marker frame into the parent marker frame.

    Args:
        parent: Parent (low) marker IPPE candidate in the camera frame.
        child: Child (high) marker IPPE candidate in the camera frame.

    Returns:
        Tuple ``(rotation, translation)`` expressing the child pose in the parent
        marker frame.
    """
    rotation = parent.rotation.T @ child.rotation
    translation = parent.rotation.T @ (child.tvec - parent.tvec)
    return rotation, translation


def transform_high_in_low(
    low: MarkerCandidate,
    high: MarkerCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Map points from the high marker frame into the low marker frame.

    Args:
        low: Low-marker IPPE candidate in the camera frame.
        high: High-marker IPPE candidate in the camera frame.

    Returns:
        Tuple ``(rotation_ba, translation_ba)`` for the high marker expressed in
        the low marker frame.
    """
    return relative_marker_transform(low, high)


def compute_live_pair_readiness(
    observations: Sequence[FrameObservation],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    expected_marker_ids: Sequence[int],
    reference_marker_id: int | None,
    settings: CalibrationSettings | None = None,
) -> LivePairReadinessDiagnostics:
    """Estimate co-visibility pair strength and marker-graph readiness.

    Args:
        observations: Raw per-frame marker corner observations.
        camera_matrix: Camera intrinsics matrix.
        dist_coeffs: Camera distortion coefficients.
        expected_marker_ids: Marker IDs expected in the layout.
        reference_marker_id: Root marker for connectivity analysis, or ``None`` to
            pick the marker with the largest raw co-visibility component.
        settings: Calibration settings; defaults apply when omitted.

    Returns:
        Diagnostics with per-pair raw co-visibility counts, connected marker set,
        and missing IDs relative to the reference.

    Notes:
        Topology-only analysis: no pose estimation or pair RMS gating beyond
        ``min_inliers_per_edge`` on raw co-visibility counts.
    """
    settings = settings or CalibrationSettings()
    settings_failure = validate_settings(settings)
    if settings_failure is not None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(),
            sample_count=len(observations),
            failure_reason=settings_failure,
        )

    expected_ids, expected_failure = parse_expected_marker_ids(
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

    camera_matrix, dist_coeffs, camera_failure = validate_camera_inputs(
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

    observations_failure = validate_observations(observations, expected_ids)
    if observations_failure is not None:
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=frozenset(),
            missing_marker_ids=frozenset(),
            sample_count=len(observations),
            failure_reason=observations_failure,
        )

    normalized_observations = normalize_observations(observations, expected_ids)
    raw_pair_counts = raw_covisible_pair_counts(normalized_observations)
    if reference_marker_id is None:
        if raw_pair_counts:
            from object_apriltag.marker_layout_calibration.reference_selection import (
                select_reference_marker,
            )

            reference_marker_id = select_reference_marker(
                raw_pair_counts.keys(),
                expected_ids,
                {},
            )
        elif expected_ids:
            reference_marker_id = min(expected_ids)
    if not raw_pair_counts:
        connected = frozenset({reference_marker_id}) if reference_marker_id is not None else frozenset()
        missing = (
            frozenset(set(expected_ids) - connected)
            if reference_marker_id is not None
            else frozenset(expected_ids)
        )
        return LivePairReadinessDiagnostics(
            pairs=(),
            connected_marker_ids=connected,
            missing_marker_ids=missing,
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
        connected_marker_ids_from_pairs(passing_pairs, reference_marker_id)
    )
    missing = frozenset(set(expected_ids) - connected)
    return LivePairReadinessDiagnostics(
        pairs=tuple(pair_reports),
        connected_marker_ids=connected,
        missing_marker_ids=missing,
        sample_count=len(observations),
    )


def largest_connected_component_from_pairs(
    pairs: Iterable[MarkerPair],
    candidate_ids: Iterable[int],
) -> set[int]:
    """Return the largest connected component among ``candidate_ids`` in the pair graph.

    Args:
        pairs: Undirected marker-pair edges.
        candidate_ids: Marker IDs eligible for component membership.

    Returns:
        Marker IDs in the largest connected component; empty when no candidates remain.
    """
    candidates = {int(marker_id) for marker_id in candidate_ids}
    if not candidates:
        return set()

    adjacency: dict[int, set[int]] = {marker_id: set() for marker_id in candidates}
    for marker_a, marker_b in pairs:
        if marker_a in candidates and marker_b in candidates:
            adjacency[marker_a].add(marker_b)
            adjacency[marker_b].add(marker_a)

    visited: set[int] = set()
    largest: set[int] = set()
    for start in sorted(candidates):
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(sorted(adjacency[current] - visited))
        if len(component) > len(largest):
            largest = component
    return largest


def connected_marker_ids_from_pairs(
    pairs: Iterable[MarkerPair],
    reference_marker_id: int,
) -> set[int]:
    """Collect markers reachable from the reference along passing pair edges.

    Args:
        pairs: Marker pairs treated as connected edges.
        reference_marker_id: Root marker for breadth-first expansion.

    Returns:
        Set of marker IDs connected to ``reference_marker_id`` via ``pairs``.
    """
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


def raw_covisible_pair_counts(
    observations: list[tuple[str | int, dict[int, np.ndarray]]],
) -> dict[MarkerPair, int]:
    """Count frames where each marker pair co-appears.

    Args:
        observations: Normalized corner observations indexed by frame.

    Returns:
        Per-pair counts of frames where both markers are visible (no pose required).
    """
    counts: dict[MarkerPair, int] = {}
    for _, markers in observations:
        marker_ids = sorted(markers)
        for index_a, marker_low in enumerate(marker_ids):
            for marker_high in marker_ids[index_a + 1 :]:
                counts[(marker_low, marker_high)] = counts.get((marker_low, marker_high), 0) + 1
    return counts


def best_pair_consensus(
    pair: MarkerPair,
    hypotheses: list[tuple[np.ndarray, np.ndarray, int]],
    translation_gate: float,
    rotation_gate: float,
) -> PairConsensus | None:
    """Select robust low-to-high consensus from per-frame hypotheses.

    Args:
        pair: Low-to-high marker ID pair ``(marker_a, marker_b)``.
        hypotheses: List of ``(rotation_ba, translation_ba, frame_index)`` entries.
        translation_gate: Maximum allowed translation deviation from the seed, in meters.
        rotation_gate: Maximum allowed rotation deviation from the seed, in degrees.

    Returns:
        Consensus edge with quaternion-averaged rotation and mean translation over
        inlier frames, or ``None`` when no seed yields inliers.

    Notes:
        Each frame contributes at most one hypothesis; seed selection maximizes
        inlier frame count before averaging.
    """
    if not hypotheses:
        return None

    best_frames: dict[int, int] = {}
    best_rotation = np.eye(3, dtype=np.float64)
    best_translation = np.zeros(3, dtype=np.float64)
    for seed_index, (seed_rotation, seed_translation, _) in enumerate(hypotheses):
        candidate_frames = inlier_frames_for_seed(
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
    best_rotation = average_rotations(inlier_rotations)
    best_translation = np.mean(inlier_translations, axis=0)
    return PairConsensus(
        marker_a=pair[0],
        marker_b=pair[1],
        rotation_ba=best_rotation,
        translation_ba=best_translation,
        inlier_frames=tuple(sorted(selected_hypotheses)),
        inlier_hypotheses=selected_hypotheses,
    )


def classify_pair_readiness(
    edge: PairConsensus | None,
    settings: CalibrationSettings,
    marker_sizes_m: Mapping[int, float],
    pair: MarkerPair,
) -> tuple[str, int, float | None, float | None]:
    """Classify pair readiness using inlier count and RMS gates.

    Args:
        edge: Robust pair consensus, or ``None`` when consensus failed.
        settings: Calibration gates for inlier count and pair RMS.
        marker_sizes_m: Physical edge lengths keyed by marker ID.
        pair: Low-to-high marker ID pair.

    Returns:
        Tuple of status label (``pass``, ``weak``, or ``fail``), robust inlier
        count, translation RMS in meters, and rotation RMS in degrees.
    """
    robust_count = len(edge.inlier_frames) if edge is not None else 0
    if edge is None or robust_count < settings.min_inliers_per_edge:
        return "weak", robust_count, None, None

    diagnostics = edge_diagnostics(pair, edge)
    translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
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


def collect_pair_hypotheses(
    frame_candidates: list[tuple[int, dict[int, list[MarkerCandidate]]]],
    expected_ids: list[int],
) -> dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]]:
    """Enumerate IPPE cross-product pair hypotheses per co-visible frame.

    Args:
        frame_candidates: Per-frame IPPE candidate pools.
        expected_ids: Marker IDs included when forming pairs.

    Returns:
        Per-pair lists of low-to-high relative transforms with frame indices for
        every IPPE candidate cross-product in co-visible frames.
    """
    hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    expected_set = set(expected_ids)
    for frame_index, candidates in frame_candidates:
        marker_ids = sorted(marker_id for marker_id in candidates if marker_id in expected_set)
        for index_a, marker_low in enumerate(marker_ids):
            for marker_high in marker_ids[index_a + 1 :]:
                pair = (marker_low, marker_high)
                for candidate_low in candidates[marker_low]:
                    for candidate_high in candidates[marker_high]:
                        rotation_ba, translation_ba = transform_high_in_low(
                            candidate_low,
                            candidate_high,
                        )
                        hypotheses.setdefault(pair, []).append(
                            (rotation_ba, translation_ba, frame_index)
                        )
    return hypotheses


def estimate_pair_consensus(
    pair_hypotheses: dict[MarkerPair, list[tuple[np.ndarray, np.ndarray, int]]],
    expected_ids: list[int],
    reference_marker_id: int,
    marker_sizes_m: Mapping[int, float],
    settings: CalibrationSettings,
    *,
    connectivity_ids: Sequence[int] | None = None,
    best_effort: bool = False,
    restored_pair_edges: list[RestoredPairEdge] | None = None,
) -> tuple[dict[MarkerPair, PairConsensus], str | None, tuple[DroppedPairEdge, ...]]:
    """Gate pair hypotheses into consensus with optional weak-edge restoration.

    Args:
        pair_hypotheses: Per-pair relative-transform hypotheses.
        expected_ids: Full set of marker IDs targeted by calibration.
        reference_marker_id: Root marker for connectivity checks.
        marker_sizes_m: Physical edge lengths keyed by marker ID.
        settings: Calibration gates for inlier count and pair RMS.
        connectivity_ids: Marker subset for connectivity when restoring weak edges;
            defaults to ``expected_ids``.
        best_effort: When true, restore weak edges to bridge disconnected components.
        restored_pair_edges: Optional list extended with restoration audit records.

    Returns:
        Tuple of gated pair consensus, optional connectivity failure message, and
        dropped-edge audit records.
    """
    rotation_gate = settings.pair_rotation_rms_gate_deg
    consensus: dict[MarkerPair, PairConsensus] = {}
    weak_pool: dict[MarkerPair, PairConsensus] = {}
    dropped: list[DroppedPairEdge] = []

    for pair, hypotheses in pair_hypotheses.items():
        translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
        unique_frames = {frame_index for _, _, frame_index in hypotheses}
        observed_count = len(unique_frames)
        edge = best_pair_consensus(pair, hypotheses, translation_gate, rotation_gate)
        if edge is not None:
            weak_pool[pair] = edge
        if observed_count < settings.min_inliers_per_edge:
            dropped.append(
                make_dropped_pair_edge(
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
                make_dropped_pair_edge(
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

    filtered: dict[MarkerPair, PairConsensus] = {}
    for pair, edge in consensus.items():
        translation_gate = pair_translation_gate(settings, marker_sizes_m, pair)
        diagnostics = edge_diagnostics(pair, edge)
        if diagnostics.inlier_count < settings.min_inliers_per_edge:
            dropped.append(
                make_dropped_pair_edge(
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
                make_dropped_pair_edge(
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
                make_dropped_pair_edge(
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
    failure = maybe_restore_weak_connectivity(
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


def inlier_frames_for_seed(
    hypotheses: list[tuple[np.ndarray, np.ndarray, int]],
    seed_index: int,
    translation_gate: float,
    rotation_gate: float,
) -> dict[int, int]:
    """Select inlier frames for a seed hypothesis under translation/rotation gates.

    Args:
        hypotheses: List of ``(rotation, translation, frame_index)`` pair hypotheses.
        seed_index: Index of the seed hypothesis in ``hypotheses``.
        translation_gate: Maximum translation deviation from the seed, in meters.
        rotation_gate: Maximum rotation deviation from the seed, in degrees.

    Returns:
        Mapping from frame index to the hypothesis index chosen as inlier for that
        frame (lowest gate-normalized cost when multiple hypotheses qualify).
    """
    seed_rotation, seed_translation, _ = hypotheses[seed_index]
    inlier_frames: dict[int, int] = {}
    for hypothesis_index, (rotation, translation, frame_index) in enumerate(hypotheses):
        if (
            np.linalg.norm(translation - seed_translation) > translation_gate
            or rotation_geodesic_deg(rotation, seed_rotation) > rotation_gate
        ):
            continue
        current_index = inlier_frames.get(frame_index)
        if current_index is None:
            inlier_frames[frame_index] = hypothesis_index
            continue
        current_rotation, current_translation, _ = hypotheses[current_index]
        current_cost = (
            np.linalg.norm(current_translation - seed_translation) / max(translation_gate, 1e-9)
            + rotation_geodesic_deg(current_rotation, seed_rotation) / max(rotation_gate, 1e-9)
        )
        candidate_cost = (
            np.linalg.norm(translation - seed_translation) / max(translation_gate, 1e-9)
            + rotation_geodesic_deg(rotation, seed_rotation) / max(rotation_gate, 1e-9)
        )
        if candidate_cost < current_cost:
            inlier_frames[frame_index] = hypothesis_index
    return inlier_frames





def _json_safe_float(value: float | None) -> float | None:
    """Convert a float to a JSON-safe finite value.

    Args:
        value: Input scalar or ``None``.

    Returns:
        Finite ``float`` value, or ``None`` when input is missing or non-finite.
    """
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def make_dropped_pair_edge(
    pair: MarkerPair,
    stage: str,
    reason: str,
    *,
    observed_count: int,
    supported_count: int,
    required_count: int,
    translation_gate: float,
    rotation_gate: float,
    edge: PairConsensus | None = None,
) -> DroppedPairEdge:
    """Record why a pair edge failed gating.

    Args:
        pair: Low-to-high marker ID pair.
        stage: Pipeline stage label for the drop event.
        reason: Machine-readable drop reason.
        observed_count: Frames where the pair was observed.
        supported_count: Frames or inliers supporting the best weak consensus.
        required_count: Minimum support required by settings.
        translation_gate: Translation RMS gate applied, in meters.
        rotation_gate: Rotation RMS gate applied, in degrees.
        edge: Optional best weak consensus used to populate RMS fields.

    Returns:
        Dropped pair-edge audit record with optional RMS diagnostics.
    """
    translation_rms_m: float | None = None
    rotation_rms_deg: float | None = None
    if edge is not None:
        diagnostics = edge_diagnostics(pair, edge)
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


def weak_edge_consensus_support(edge: PairConsensus) -> int:
    """Return inlier frame count used as weak-edge support during repair.

    Args:
        edge: Weak pair consensus candidate.

    Returns:
        Number of inlier frames in ``edge``.
    """
    return len(edge.inlier_frames)


def make_restored_pair_edge(
    dropped: DroppedPairEdge,
    edge: PairConsensus,
    stage: str,
) -> RestoredPairEdge:
    """Build an audit record when a dropped weak edge is reinstated.

    Args:
        dropped: Original dropped-edge record for the pair.
        edge: Weak consensus edge restored into the graph.
        stage: Pipeline stage where restoration occurred.

    Returns:
        Restored pair-edge audit record with support fraction and original drop
        metadata.
    """
    consensus_support = weak_edge_consensus_support(edge)
    return RestoredPairEdge(
        marker_a=dropped.marker_a,
        marker_b=dropped.marker_b,
        stage=stage,
        original_stage=dropped.stage,
        original_reason=dropped.reason,
        observed_count=dropped.observed_count,
        supported_count=consensus_support,
        required_count=dropped.required_count,
        support_fraction=consensus_support / max(dropped.observed_count, 1),
        translation_rms_m=dropped.translation_rms_m,
        rotation_rms_deg=dropped.rotation_rms_deg,
        translation_gate_m=dropped.translation_gate_m,
        rotation_gate_deg=dropped.rotation_gate_deg,
    )


def weak_edge_rank_key(
    supported_count: int,
    observed_count: int,
    rotation_rms_deg: float | None,
    translation_rms_m: float | None,
) -> tuple[float, float, float, float]:
    """Build a sort key for ranking weak-edge restoration candidates.

    Args:
        supported_count: Inlier or supported frame count for the weak edge.
        observed_count: Total frames where the pair was observed.
        rotation_rms_deg: Rotation RMS in degrees, or ``None`` if unknown.
        translation_rms_m: Translation RMS in meters, or ``None`` if unknown.

    Returns:
        Tuple sort key preferring more inliers, higher support fraction, then lower
        RMS (unknown RMS sorts last).
    """
    fraction = supported_count / max(observed_count, 1)
    rotation = rotation_rms_deg if rotation_rms_deg is not None else float("inf")
    translation = translation_rms_m if translation_rms_m is not None else float("inf")
    return (-supported_count, -fraction, rotation, translation)


def connectivity_failure_message(
    stage: str,
    reference_marker_id: int,
    missing: list[int],
) -> str:
    """Format a connectivity failure message for a pipeline stage.

    Args:
        stage: Pipeline stage label (for example ``initial_consensus``).
        reference_marker_id: Root marker used in connectivity analysis.
        missing: Marker IDs not connected to the reference.

    Returns:
        Human-readable connectivity failure string.
    """
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


def meets_best_effort_weak_support_floor(supported_count: int) -> bool:
    """Check whether weak-edge support meets the best-effort restoration floor.

    Args:
        supported_count: Inlier or supported frame count for a weak edge.

    Returns:
        True when ``supported_count`` is at least the configured weak-edge minimum.
    """
    return supported_count >= _BEST_EFFORT_WEAK_EDGE_MIN_SUPPORT


def weak_restore_candidates(
    pair_consensus: dict[MarkerPair, PairConsensus],
    weak_pool: dict[MarkerPair, PairConsensus],
    dropped: Sequence[DroppedPairEdge],
) -> list[tuple[PairConsensus, DroppedPairEdge]]:
    """List dropped pairs with weak consensus meeting the best-effort support floor.

    Args:
        pair_consensus: Currently accepted pair consensus edges.
        weak_pool: Best weak consensus per pair from initial gating.
        dropped: Dropped-edge audit records from gating.

    Returns:
        List of ``(weak_edge, drop_record)`` pairs eligible for connectivity repair.
    """
    candidates: list[tuple[PairConsensus, DroppedPairEdge]] = []
    seen: set[MarkerPair] = set()
    for drop in dropped:
        pair = drop.marker_pair
        if pair in pair_consensus or pair in seen:
            continue
        edge = weak_pool.get(pair)
        if edge is None or not edge.inlier_frames:
            continue
        if not meets_best_effort_weak_support_floor(weak_edge_consensus_support(edge)):
            continue
        candidates.append((edge, drop))
        seen.add(pair)
    return candidates


def maybe_restore_weak_connectivity(
    pair_consensus: dict[MarkerPair, PairConsensus],
    weak_pool: dict[MarkerPair, PairConsensus],
    dropped: list[DroppedPairEdge],
    required_ids: list[int],
    reference_marker_id: int,
    stage: str,
    *,
    best_effort: bool,
    restored_pair_edges: list[RestoredPairEdge] | None,
) -> str | None:
    """Restore weak edges until required markers connect to the reference.

    Args:
        pair_consensus: Accepted pair consensus edges (updated in place on restore).
        weak_pool: Best weak consensus per pair from initial gating.
        dropped: Dropped-edge audit records used to find restore candidates.
        required_ids: Marker IDs that must connect to ``reference_marker_id``.
        reference_marker_id: Root marker for connectivity checks.
        stage: Pipeline stage label recorded on restoration audit entries.
        best_effort: When false, return a failure message without restoring edges.
        restored_pair_edges: Optional list extended with restoration audit records.

    Returns:
        Connectivity failure message when required markers remain disconnected,
        or ``None`` on success.

    Notes:
        Greedy restoration picks the highest-ranked bridging weak edge each
        iteration until ``required_ids`` connect or no bridge remains.
    """
    connected = connected_marker_ids(pair_consensus, reference_marker_id)
    missing = sorted(set(required_ids) - connected)
    if not missing:
        return None
    if not best_effort:
        return connectivity_failure_message(stage, reference_marker_id, missing)

    restore_candidates = weak_restore_candidates(pair_consensus, weak_pool, dropped)
    sorted_candidates = sorted(
        restore_candidates,
        key=lambda item: weak_edge_rank_key(
            weak_edge_consensus_support(item[0]),
            item[1].observed_count,
            item[1].rotation_rms_deg,
            item[1].translation_rms_m,
        ),
    )
    remaining = list(sorted_candidates)
    restored_records: list[RestoredPairEdge] = []

    # greedy first-ranked bridging edge per iteration; upgrade path is
    # union-find + global min-cost bridge set if multi-component graphs get larger.
    while not set(required_ids).issubset(connected_marker_ids(pair_consensus, reference_marker_id)):
        connected = connected_marker_ids(pair_consensus, reference_marker_id)
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
        restored_records.append(make_restored_pair_edge(drop, edge, stage))

    if restored_pair_edges is not None:
        restored_pair_edges.extend(restored_records)

    missing = sorted(set(required_ids) - connected_marker_ids(pair_consensus, reference_marker_id))
    if missing:
        return connectivity_failure_message(stage, reference_marker_id, missing)
    return None


