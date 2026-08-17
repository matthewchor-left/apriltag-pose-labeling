"""CAD landmark geometry evaluation against marker-derived landmarks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from object_apriltag.cad import CadLandmarks, CadRegistration
from object_apriltag.evaluation.kabsch import (
    apply_rigid_transform,
    kabsch_rigid_transform,
    validate_rigid_rotation,
)
from object_apriltag.evaluation.types import (
    CadGeometryEvaluation,
    DistanceCadDisagreement,
    DistanceCadDisagreementReport,
    LandmarkCadDisagreement,
    LeaveOneMarkerCadPrediction,
    LeaveOneMarkerCadPredictionFold,
    MetricSummaryMm,
    RigidCadFit,
)
from object_apriltag.layout import MarkerLayout, footprint_corner_with_padding
from object_apriltag.object_model_edit import parse_keypoint_sources

_MIN_RETAINED_LANDMARKS = 3
_DEGENERACY_EIGENVALUE_RATIO = 1e-6


def fit_cad_registration(
    cad_landmarks: CadLandmarks | Mapping[str, np.ndarray],
    object_model_document: dict[str, Any],
    marker_layout: MarkerLayout,
) -> CadRegistration:
    """Fit an in-memory CAD-to-marker_model registration from named landmarks.

    Args:
        cad_landmarks: CAD landmark positions keyed by name.
        object_model_document: Object model JSON with ``keypoint_sources``.
        marker_layout: Marker layout used to derive marker-frame landmarks.

    Returns:
        ``CadRegistration`` mapping CAD coordinates into marker_model frame.

    Raises:
        ValueError: If too few landmarks are available, names are missing, values
            are non-finite, or point sets are degenerate.
    """
    cad_by_name = (
        dict(cad_landmarks.landmarks)
        if isinstance(cad_landmarks, CadLandmarks)
        else dict(cad_landmarks)
    )
    marker_by_name = derive_marker_derived_landmarks(object_model_document, marker_layout)
    landmark_names = tuple(sorted(marker_by_name))
    if len(landmark_names) < _MIN_RETAINED_LANDMARKS:
        raise ValueError(
            f"CAD registration requires at least {_MIN_RETAINED_LANDMARKS} "
            f"keypoint_sources; found {len(landmark_names)}."
        )
    missing_cad = sorted(set(landmark_names) - set(cad_by_name))
    if missing_cad:
        raise ValueError(f"CAD landmarks missing names required by keypoint_sources: {missing_cad}.")
    _require_finite_landmarks(cad_by_name, label="CAD", names=landmark_names)
    _require_finite_landmarks(marker_by_name, label="Marker-derived", names=landmark_names)

    cad_points = np.asarray([cad_by_name[name] for name in landmark_names], dtype=np.float64)
    marker_points = np.asarray(
        [marker_by_name[name] for name in landmark_names],
        dtype=np.float64,
    )
    if _is_degenerate_point_set(
        cad_points,
        degeneracy_eigenvalue_ratio=_DEGENERACY_EIGENVALUE_RATIO,
    ):
        raise ValueError("CAD registration landmarks are degenerate (coincident or collinear).")
    if _is_degenerate_point_set(
        marker_points,
        degeneracy_eigenvalue_ratio=_DEGENERACY_EIGENVALUE_RATIO,
    ):
        raise ValueError(
            "Marker-derived registration landmarks are degenerate (coincident or collinear)."
        )

    rotation, translation = kabsch_rigid_transform(cad_points, marker_points)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return CadRegistration(
        units="meters",
        source_frame="cad",
        target_frame="marker_model",
        transform_4x4=transform,
    )


def derive_marker_derived_landmarks(
    object_model_document: dict[str, Any],
    marker_layout: MarkerLayout,
) -> dict[str, np.ndarray]:
    """Compute marker-derived landmark positions from keypoint_sources and layout.

    Args:
        object_model_document: Object model JSON with ``keypoint_sources``.
        marker_layout: Marker layout with footprint geometry.

    Returns:
        Landmark positions in marker_model coordinates keyed by name.

    Raises:
        ValueError: If a ``keypoint_sources`` entry references a missing marker ID.
    """
    sources = parse_keypoint_sources(object_model_document)
    landmarks: dict[str, np.ndarray] = {}
    for landmark_name, (marker_id, corner, padding_m) in sources.items():
        if marker_id not in marker_layout.footprints:
            raise ValueError(
                f"keypoint_sources.{landmark_name} references marker {marker_id}, "
                f"which is not present in the marker layout."
            )
        footprint = marker_layout.footprints[marker_id]
        landmarks[landmark_name] = footprint_corner_with_padding(footprint, corner, padding_m)
    return landmarks


def evaluate_cad_geometry(
    cad_landmarks: CadLandmarks | Mapping[str, np.ndarray],
    object_model_document: dict[str, Any],
    marker_layout: MarkerLayout,
    *,
    min_retained_landmarks: int = _MIN_RETAINED_LANDMARKS,
    degeneracy_eigenvalue_ratio: float = _DEGENERACY_EIGENVALUE_RATIO,
) -> CadGeometryEvaluation:
    """Compare CAD landmarks to marker-derived landmarks across geometry metrics.

    Args:
        cad_landmarks: CAD landmark positions keyed by name.
        object_model_document: Object model JSON with ``keypoint_sources``.
        marker_layout: Marker layout used to derive marker-frame landmarks.
        min_retained_landmarks: Minimum landmarks required for leave-one-out folds.
        degeneracy_eigenvalue_ratio: Eigenvalue ratio threshold for degeneracy checks.

    Returns:
        Full CAD geometry evaluation including rigid fit, distance disagreements,
        and leave-one-marker-out prediction.

    Raises:
        ValueError: If required landmark names are missing or values are non-finite.
    """
    cad_by_name = (
        dict(cad_landmarks.landmarks)
        if isinstance(cad_landmarks, CadLandmarks)
        else dict(cad_landmarks)
    )
    marker_derived_by_name = derive_marker_derived_landmarks(object_model_document, marker_layout)
    sources = parse_keypoint_sources(object_model_document)
    landmark_names = tuple(sorted(marker_derived_by_name))
    missing_cad = sorted(set(landmark_names) - set(cad_by_name))
    if missing_cad:
        raise ValueError(f"CAD landmarks missing names required by keypoint_sources: {missing_cad}.")
    _require_finite_landmarks(cad_by_name, label="CAD", names=landmark_names)
    _require_finite_landmarks(marker_derived_by_name, label="Marker-derived", names=landmark_names)

    cad_points = np.asarray([cad_by_name[name] for name in landmark_names], dtype=np.float64)
    marker_points = np.asarray(
        [marker_derived_by_name[name] for name in landmark_names],
        dtype=np.float64,
    )

    rigid_fit = _evaluate_rigid_fit(cad_points, marker_points, landmark_names)
    pair_report = _evaluate_distance_disagreements(
        cad_by_name,
        marker_derived_by_name,
        _all_pairs(landmark_names),
    )
    skeleton_edges = _parse_skeleton_edges(object_model_document, set(landmark_names))
    skeleton_report = _evaluate_distance_disagreements(
        cad_by_name,
        marker_derived_by_name,
        skeleton_edges,
    )
    leave_one_out = _evaluate_leave_one_marker_out(
        cad_by_name=cad_by_name,
        marker_derived_by_name=marker_derived_by_name,
        sources=sources,
        min_retained_landmarks=min_retained_landmarks,
        degeneracy_eigenvalue_ratio=degeneracy_eigenvalue_ratio,
    )

    return CadGeometryEvaluation(
        landmark_names=landmark_names,
        cad_landmarks_m=_points_to_tuple_dict(cad_by_name, landmark_names),
        marker_derived_landmarks_m=_points_to_tuple_dict(marker_derived_by_name, landmark_names),
        rigid_fit=rigid_fit,
        pair_distance_disagreement=pair_report,
        skeleton_edge_disagreement=skeleton_report,
        leave_one_marker_out=leave_one_out,
    )


def metric_summary_mm(values_mm: Sequence[float] | np.ndarray) -> MetricSummaryMm:
    """Summarize millimeter error samples.

    Args:
        values_mm: Scalar error samples in millimeters.

    Returns:
        Min, median, RMSE, P95, and max over samples; zeroed summary when empty.
    """
    array = np.asarray(values_mm, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return MetricSummaryMm(
            count=0,
            min_mm=0.0,
            median_mm=0.0,
            rmse_mm=0.0,
            p95_mm=0.0,
            max_mm=0.0,
        )
    return MetricSummaryMm(
        count=int(array.size),
        min_mm=float(np.min(array)),
        median_mm=float(np.median(array)),
        rmse_mm=float(np.sqrt(np.mean(array * array))),
        p95_mm=float(np.percentile(array, 95)),
        max_mm=float(np.max(array)),
    )


def _evaluate_rigid_fit(
    cad_points: np.ndarray,
    marker_points: np.ndarray,
    landmark_names: tuple[str, ...],
) -> RigidCadFit:
    """Fit and score a global rigid alignment between CAD and marker points.

    Args:
        cad_points: CAD landmark positions as ``(N, 3)`` array.
        marker_points: Marker-derived positions as ``(N, 3)`` array.
        landmark_names: Landmark names in row order.

    Returns:
        Rigid fit with per-landmark and aggregate disagreement in millimeters.
    """
    rotation, translation = kabsch_rigid_transform(cad_points, marker_points)
    transformed = apply_rigid_transform(cad_points, rotation, translation)
    errors_m = transformed - marker_points
    errors_mm = errors_m * 1000.0
    norms_mm = np.linalg.norm(errors_mm, axis=1)
    per_landmark = tuple(
        LandmarkCadDisagreement(
            landmark_name=name,
            cad_disagreement_mm=float(norm),
            error_mm=tuple(float(value) for value in error),
        )
        for name, norm, error in zip(landmark_names, norms_mm, errors_mm, strict=True)
    )
    validation = validate_rigid_rotation(rotation)
    return RigidCadFit(
        rotation=_matrix_to_tuple(rotation),
        translation_m=tuple(float(value) for value in translation),
        rotation_validation=validation,
        per_landmark=per_landmark,
        summary_mm=metric_summary_mm(norms_mm),
    )


def _evaluate_distance_disagreements(
    cad_by_name: Mapping[str, np.ndarray],
    marker_derived_by_name: Mapping[str, np.ndarray],
    edges: Sequence[tuple[str, str]],
) -> DistanceCadDisagreementReport:
    """Compare pairwise edge lengths between CAD and marker-derived landmarks.

    Args:
        cad_by_name: CAD landmark positions keyed by name.
        marker_derived_by_name: Marker-derived positions keyed by name.
        edges: Landmark name pairs defining edges to compare.

    Returns:
        Per-edge distance disagreements with aggregate summary in millimeters.
    """
    disagreements: list[DistanceCadDisagreement] = []
    for start_name, end_name in edges:
        cad_distance_mm = float(
            np.linalg.norm(cad_by_name[end_name] - cad_by_name[start_name]) * 1000.0
        )
        marker_distance_mm = float(
            np.linalg.norm(marker_derived_by_name[end_name] - marker_derived_by_name[start_name])
            * 1000.0
        )
        disagreements.append(
            DistanceCadDisagreement(
                start_landmark=start_name,
                end_landmark=end_name,
                cad_distance_mm=cad_distance_mm,
                marker_derived_distance_mm=marker_distance_mm,
                cad_disagreement_mm=abs(cad_distance_mm - marker_distance_mm),
            )
        )
    summary_values = [entry.cad_disagreement_mm for entry in disagreements]
    return DistanceCadDisagreementReport(
        distances=tuple(disagreements),
        summary_mm=metric_summary_mm(summary_values),
    )


def _evaluate_leave_one_marker_out(
    *,
    cad_by_name: Mapping[str, np.ndarray],
    marker_derived_by_name: Mapping[str, np.ndarray],
    sources: Mapping[str, tuple[int, str, float]],
    min_retained_landmarks: int,
    degeneracy_eigenvalue_ratio: float,
) -> LeaveOneMarkerCadPrediction:
    """Run leave-one-marker-out CAD prediction folds.

    Args:
        cad_by_name: CAD landmark positions keyed by name.
        marker_derived_by_name: Marker-derived positions keyed by name.
        sources: Parsed ``keypoint_sources`` mapping names to marker metadata.
        min_retained_landmarks: Minimum landmarks required to attempt a fold.
        degeneracy_eigenvalue_ratio: Eigenvalue ratio threshold for degeneracy checks.

    Returns:
        Leave-one-marker-out prediction metrics across all marker IDs.
    """
    marker_ids = sorted({marker_id for marker_id, _, _ in sources.values()})
    folds: list[LeaveOneMarkerCadPredictionFold] = []
    all_excluded_errors_mm: list[float] = []

    for held_out_marker_id in marker_ids:
        excluded_names = tuple(
            sorted(name for name, (marker_id, _, _) in sources.items() if marker_id == held_out_marker_id)
        )
        retained_names = tuple(
            sorted(name for name, (marker_id, _, _) in sources.items() if marker_id != held_out_marker_id)
        )
        refusal_reason: str | None = None
        eligible = True
        if len(retained_names) < min_retained_landmarks:
            eligible = False
            refusal_reason = (
                f"insufficient_retained_landmarks: need >= {min_retained_landmarks}, "
                f"got {len(retained_names)}."
            )
        elif _is_degenerate_point_set(
            np.asarray([marker_derived_by_name[name] for name in retained_names], dtype=np.float64),
            degeneracy_eigenvalue_ratio=degeneracy_eigenvalue_ratio,
        ):
            eligible = False
            refusal_reason = "degenerate_retained_landmarks."

        per_landmark_errors: dict[str, float] = {}
        fold_summary: MetricSummaryMm | None = None
        if eligible:
            retained_cad = np.asarray([cad_by_name[name] for name in retained_names], dtype=np.float64)
            retained_marker = np.asarray(
                [marker_derived_by_name[name] for name in retained_names],
                dtype=np.float64,
            )
            rotation, translation = kabsch_rigid_transform(retained_cad, retained_marker)
            for name in excluded_names:
                predicted = apply_rigid_transform(cad_by_name[name], rotation, translation)
                error_mm = float(np.linalg.norm(predicted - marker_derived_by_name[name]) * 1000.0)
                per_landmark_errors[name] = error_mm
                all_excluded_errors_mm.append(error_mm)
            fold_summary = metric_summary_mm(list(per_landmark_errors.values()))

        folds.append(
            LeaveOneMarkerCadPredictionFold(
                held_out_marker_id=held_out_marker_id,
                eligible=eligible,
                refusal_reason=refusal_reason,
                excluded_landmark_names=excluded_names,
                retained_landmark_count=len(retained_names),
                per_landmark_cad_disagreement_mm=per_landmark_errors,
                summary_mm=fold_summary,
            )
        )

    eligible_fold_count = sum(1 for fold in folds if fold.eligible)
    return LeaveOneMarkerCadPrediction(
        folds=tuple(folds),
        eligible_fold_count=eligible_fold_count,
        refused_fold_count=len(folds) - eligible_fold_count,
        all_excluded_summary_mm=metric_summary_mm(all_excluded_errors_mm),
    )


def _is_degenerate_point_set(
    points: np.ndarray,
    *,
    degeneracy_eigenvalue_ratio: float,
) -> bool:
    """Return whether a 3D point set is too degenerate for Kabsch alignment.

    Args:
        points: Point set as ``(N, 3)`` array.
        degeneracy_eigenvalue_ratio: Minimum eigenvalue ratio for a non-degenerate axis.

    Returns:
        ``True`` when fewer than two principal spread axes are present.

    Notes:
        Three coplanar points are acceptable; Kabsch needs two non-degenerate
        principal directions.
    """
    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points_array.shape[0] < 3:
        return True
    centered = points_array - points_array.mean(axis=0)
    if np.allclose(centered, 0.0):
        return True
    covariance = centered.T @ centered / points_array.shape[0]
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    largest = eigenvalues[-1]
    if largest <= 0.0:
        return True
    # Kabsch needs two non-degenerate principal directions; three coplanar points are fine.
    spread_axes = int(np.sum(eigenvalues / largest >= degeneracy_eigenvalue_ratio))
    return spread_axes < 2


def _parse_skeleton_edges(
    object_model_document: dict[str, Any],
    landmark_names: set[str],
) -> tuple[tuple[str, str], ...]:
    """Parse skeleton edges that reference known landmark names.

    Args:
        object_model_document: Object model JSON with optional ``skeleton`` list.
        landmark_names: Landmark names present in the evaluation.

    Returns:
        Skeleton edges whose endpoints are both in ``landmark_names``.

    Raises:
        ValueError: If a skeleton entry is not a two-element name list.
    """
    raw = object_model_document.get("skeleton")
    if not isinstance(raw, list) or not raw:
        return tuple()
    edges: list[tuple[str, str]] = []
    for index, edge in enumerate(raw):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"skeleton[{index}] must be [start, end] keypoint names.")
        start_name, end_name = str(edge[0]), str(edge[1])
        if start_name not in landmark_names or end_name not in landmark_names:
            continue
        edges.append((start_name, end_name))
    return tuple(edges)


def _all_pairs(names: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Enumerate all unordered landmark name pairs.

    Args:
        names: Landmark names to pair.

    Returns:
        Tuple of ``(left, right)`` pairs with ``left`` before ``right`` in input order.
    """
    ordered = list(names)
    pairs: list[tuple[str, str]] = []
    for left_index, left_name in enumerate(ordered):
        for right_name in ordered[left_index + 1 :]:
            pairs.append((left_name, right_name))
    return tuple(pairs)


def _require_finite_landmarks(
    points_by_name: Mapping[str, np.ndarray],
    *,
    label: str,
    names: Sequence[str],
) -> None:
    """Require that named landmarks are finite 3D positions.

    Args:
        points_by_name: Landmark positions keyed by name.
        label: Label prefix for error messages.
        names: Landmark names to validate.

    Raises:
        ValueError: If any named landmark is not a finite 3D position.
    """
    for name in names:
        point = np.asarray(points_by_name[name], dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} landmark {name!r} must be a finite 3D position.")


def _points_to_tuple_dict(
    points_by_name: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> dict[str, tuple[float, float, float]]:
    """Convert named ndarray landmarks to JSON-serializable float tuples.

    Args:
        points_by_name: Landmark positions keyed by name.
        names: Landmark names to export.

    Returns:
        Positions keyed by name as ``(x, y, z)`` float tuples.
    """
    return {
        name: tuple(float(value) for value in points_by_name[name])
        for name in names
    }


def _matrix_to_tuple(matrix: np.ndarray) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Convert a 3x3 ndarray to nested float tuples.

    Args:
        matrix: Rotation or other 3x3 matrix.

    Returns:
        Matrix as nested ``(row, col)`` float tuples.
    """
    array = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in array)
