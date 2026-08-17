"""Rigid Kabsch alignment (rotation + translation, det(R)=+1, no scale)."""

from __future__ import annotations

import numpy as np

from object_apriltag.evaluation.types import RigidRotationValidation


def kabsch_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a rigid transform mapping source points onto target points.

    Uses the Kabsch algorithm with no scale: ``target ≈ R @ source + t`` and
    ``det(R) = +1``.

    Args:
        source_points: Source points as ``(N, 3)`` array.
        target_points: Target points as ``(N, 3)`` array.

    Returns:
        Tuple of ``(rotation, translation)`` where rotation is ``(3, 3)`` and
        translation is ``(3,)``.

    Raises:
        ValueError: If shapes differ, no points are provided, or values are non-finite.
    """
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    if source.shape[0] < 1:
        raise ValueError("At least one point pair is required for Kabsch alignment.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("source_points and target_points must contain only finite values.")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u_matrix, _, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def validate_rigid_rotation(rotation: np.ndarray) -> RigidRotationValidation:
    """Validate that a matrix is a proper rotation.

    Args:
        rotation: Candidate rotation matrix as ``(3, 3)`` array.

    Returns:
        Validation record with determinant, orthonormality error, and pass flag.
    """
    rotation_matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    determinant = float(np.linalg.det(rotation_matrix))
    orthonormality_error = float(np.linalg.norm(rotation_matrix.T @ rotation_matrix - np.eye(3)))
    is_proper_rotation = determinant > 0.0 and orthonormality_error < 1e-6
    return RigidRotationValidation(
        determinant=determinant,
        orthonormality_frobenius_error=orthonormality_error,
        is_proper_rotation=is_proper_rotation,
    )


def apply_rigid_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to 3D points.

    Args:
        points: Input points as ``(N, 3)`` array.
        rotation: Rotation matrix as ``(3, 3)`` array.
        translation: Translation vector as ``(3,)`` array.

    Returns:
        Transformed points as ``(N, 3)`` array.
    """
    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rotation_matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation_vector = np.asarray(translation, dtype=np.float64).reshape(3)
    return (rotation_matrix @ points_array.T).T + translation_vector
