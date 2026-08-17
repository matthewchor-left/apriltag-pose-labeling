"""Versioned evaluation manifest loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from object_apriltag.calibration import config_dir
from object_apriltag.layout import MarkerLayout, load_marker_model
from object_apriltag.object_model_edit import parse_keypoint_sources

EVALUATION_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class DetectorConfig:
    """AprilTag detector settings from the evaluation manifest.

    Attributes:
        dictionary: OpenCV ArUco dictionary name.
        sensitivity: Detector sensitivity preset.
    """

    dictionary: str
    sensitivity: str


@dataclass(frozen=True)
class HeldOutVideo:
    """Held-out evaluation video declared in the manifest.

    Attributes:
        path: Absolute path to the video file.
        held_out: Whether the video is declared held out from calibration.
    """

    path: Path
    held_out: bool


@dataclass(frozen=True)
class CalibrationSource:
    """Optional calibration video provenance for a candidate.

    Attributes:
        video: Path to the calibration video, or ``None`` when unspecified.
    """

    video: Path | None


@dataclass(frozen=True)
class EvaluationCandidate:
    """One marker-model candidate declared in the evaluation manifest.

    Attributes:
        name: Unique candidate identifier.
        marker_model_path: Path to the candidate marker layout JSON.
        capture_session: Capture session label for grouping.
        solver_variant: Solver variant label for grouping.
        calibration_source: Optional calibration video provenance.
    """

    name: str
    marker_model_path: Path
    capture_session: str
    solver_variant: str
    calibration_source: CalibrationSource


@dataclass(frozen=True)
class EvaluationManifest:
    """Parsed and validated evaluation manifest.

    Attributes:
        manifest_path: Absolute path to the manifest JSON file.
        version: Manifest schema version.
        cad_model: Path to the CAD landmark model.
        object_model: Path to the object model with ``keypoint_sources``.
        intrinsics: Path to camera intrinsics JSON.
        detector: AprilTag detector configuration.
        held_out_videos: Videos declared held out from calibration.
        candidates: Marker-model candidates to evaluate.
    """

    manifest_path: Path
    version: int
    cad_model: Path
    object_model: Path
    intrinsics: Path
    detector: DetectorConfig
    held_out_videos: tuple[HeldOutVideo, ...]
    candidates: tuple[EvaluationCandidate, ...]


def repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute path to the repository root (parent of the config directory).
    """
    return config_dir().parent


def repo_relative_path(path: Path) -> str:
    """Convert an absolute path to a repo-relative POSIX path when possible.

    Args:
        path: Path to relativize.

    Returns:
        Repo-relative POSIX path, or the resolved absolute POSIX path when the
        input lies outside the repository.
    """
    resolved = path.resolve()
    root = repo_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_manifest_path(path: str | Path, *, manifest_path: Path) -> Path:
    """Resolve manifest paths relative to the repository root.

    Args:
        path: Relative or absolute path from the manifest.
        manifest_path: Absolute path to the manifest file (unused; kept for API
            symmetry with callers).

    Returns:
        Resolved absolute path.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    _ = manifest_path
    return (repo_root() / candidate).resolve()


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    """Load and validate an evaluation manifest JSON file.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        Parsed ``EvaluationManifest``.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest is malformed or uses an unsupported version.
    """
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation manifest must be a JSON object.")

    version = payload.get("manifest_version")
    if version != EVALUATION_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest_version {version!r}; expected {EVALUATION_MANIFEST_VERSION}."
        )

    cad_model = resolve_manifest_path(_require_path(payload, "cad_model"), manifest_path=manifest_path)
    object_model = resolve_manifest_path(
        _require_path(payload, "object_model"),
        manifest_path=manifest_path,
    )
    intrinsics = resolve_manifest_path(_require_path(payload, "intrinsics"), manifest_path=manifest_path)
    detector = _parse_detector(payload.get("detector"))
    held_out_videos = _parse_held_out_videos(payload.get("held_out_videos"), manifest_path=manifest_path)
    candidates = _parse_candidates(payload.get("candidates"), manifest_path=manifest_path)
    if not candidates:
        raise ValueError("Evaluation manifest must declare at least one candidate.")

    return EvaluationManifest(
        manifest_path=manifest_path,
        version=int(version),
        cad_model=cad_model,
        object_model=object_model,
        intrinsics=intrinsics,
        detector=detector,
        held_out_videos=held_out_videos,
        candidates=candidates,
    )


def load_candidate_layouts(
    manifest: EvaluationManifest,
) -> tuple[dict[str, MarkerLayout], frozenset[int]]:
    """Load marker layouts for all manifest candidates and verify marker ID parity.

    Args:
        manifest: Evaluation manifest with candidate marker model paths.

    Returns:
        Tuple of layouts keyed by candidate name and the shared expected marker
        ID set.

    Raises:
        FileNotFoundError: If a candidate marker model file is missing.
        ValueError: If candidate layouts disagree on marker IDs.
    """
    layouts: dict[str, MarkerLayout] = {}
    marker_id_sets: list[frozenset[int]] = []
    for candidate in manifest.candidates:
        if not candidate.marker_model_path.is_file():
            raise FileNotFoundError(
                f"Candidate {candidate.name!r} marker model not found: {candidate.marker_model_path}."
            )
        layout = load_marker_model(candidate.marker_model_path)
        layouts[candidate.name] = layout
        marker_id_sets.append(frozenset(layout.marker_ids))
    expected_marker_ids = marker_id_sets[0]
    for index, marker_ids in enumerate(marker_id_sets[1:], start=1):
        if marker_ids != expected_marker_ids:
            other = manifest.candidates[index]
            raise ValueError(
                f"Candidate {other.name!r} marker IDs {sorted(marker_ids)} do not match "
                f"expected set {sorted(expected_marker_ids)}."
            )
    return layouts, expected_marker_ids


def validate_object_model_correspondence(
    object_model_document: dict[str, Any],
    expected_marker_ids: frozenset[int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Verify object-model keypoint sources align with candidate marker layouts.

    Args:
        object_model_document: Parsed object model JSON.
        expected_marker_ids: Marker IDs present in all candidate layouts.

    Returns:
        Tuple of sorted landmark names and sorted expected marker IDs.

    Raises:
        ValueError: If ``keypoint_sources`` reference marker IDs missing from layouts.
    """
    sources = parse_keypoint_sources(object_model_document)
    landmark_names = tuple(sorted(sources))
    referenced_marker_ids = frozenset(marker_id for marker_id, _, _ in sources.values())
    missing_from_layout = sorted(referenced_marker_ids - expected_marker_ids)
    if missing_from_layout:
        raise ValueError(
            "object_model keypoint_sources reference marker IDs missing from candidate layouts: "
            f"{missing_from_layout}."
        )
    return landmark_names, tuple(sorted(expected_marker_ids))


def _parse_detector(raw: Any) -> DetectorConfig:
    """Parse the manifest ``detector`` object.

    Args:
        raw: Raw JSON value for ``detector``.

    Returns:
        Parsed detector configuration.

    Raises:
        ValueError: If ``raw`` is not a valid detector object.
    """
    if not isinstance(raw, dict):
        raise ValueError("manifest.detector must be an object with dictionary and sensitivity.")
    dictionary = _require_non_empty_string(raw, "dictionary")
    sensitivity = _require_non_empty_string(raw, "sensitivity")
    return DetectorConfig(dictionary=dictionary, sensitivity=sensitivity)


def _parse_held_out_videos(raw: Any, *, manifest_path: Path) -> tuple[HeldOutVideo, ...]:
    """Parse the manifest ``held_out_videos`` array.

    Args:
        raw: Raw JSON value for ``held_out_videos``.
        manifest_path: Absolute path to the manifest for relative path resolution.

    Returns:
        Tuple of held-out video entries.

    Raises:
        ValueError: If the array is empty or entries are malformed.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest.held_out_videos must be a non-empty array.")
    videos: list[HeldOutVideo] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"held_out_videos[{index}] must be an object.")
        held_out = entry.get("held_out")
        if held_out is not True:
            raise ValueError(
                f"held_out_videos[{index}] must declare held_out: true; "
                "held-out status is a user-provided reproducibility contract."
            )
        path = resolve_manifest_path(_require_path(entry, "path"), manifest_path=manifest_path)
        videos.append(HeldOutVideo(path=path, held_out=True))
    return tuple(videos)


def _parse_candidates(raw: Any, *, manifest_path: Path) -> tuple[EvaluationCandidate, ...]:
    """Parse the manifest ``candidates`` array.

    Args:
        raw: Raw JSON value for ``candidates``.
        manifest_path: Absolute path to the manifest for relative path resolution.

    Returns:
        Tuple of evaluation candidates.

    Raises:
        ValueError: If the array is empty, entries are malformed, or names duplicate.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest.candidates must be a non-empty array.")
    names: set[str] = set()
    candidates: list[EvaluationCandidate] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"candidates[{index}] must be an object.")
        name = _require_non_empty_string(entry, "name")
        if name in names:
            raise ValueError(f"Duplicate candidate name {name!r}.")
        names.add(name)
        marker_model_path = resolve_manifest_path(
            _require_path(entry, "marker_model"),
            manifest_path=manifest_path,
        )
        capture_session = _require_non_empty_string(entry, "capture_session")
        solver_variant = _require_non_empty_string(entry, "solver_variant")
        calibration_source = _parse_calibration_source(
            entry.get("calibration_source"),
            index=index,
            manifest_path=manifest_path,
        )
        candidates.append(
            EvaluationCandidate(
                name=name,
                marker_model_path=marker_model_path,
                capture_session=capture_session,
                solver_variant=solver_variant,
                calibration_source=calibration_source,
            )
        )
    return tuple(candidates)


def _parse_calibration_source(
    raw: Any,
    *,
    index: int,
    manifest_path: Path,
) -> CalibrationSource:
    """Parse one candidate's ``calibration_source`` object.

    Args:
        raw: Raw JSON value for ``calibration_source``, or ``None``.
        index: Candidate index for error messages.
        manifest_path: Absolute path to the manifest for relative path resolution.

    Returns:
        Parsed calibration source.

    Raises:
        ValueError: If ``raw`` is present but not a valid object.
    """
    if raw is None:
        return CalibrationSource(video=None)
    if not isinstance(raw, dict):
        raise ValueError(f"candidates[{index}].calibration_source must be an object when present.")
    video_raw = raw.get("video")
    if video_raw is None:
        return CalibrationSource(video=None)
    if not isinstance(video_raw, str) or not video_raw.strip():
        raise ValueError(f"candidates[{index}].calibration_source.video must be a non-empty path.")
    return CalibrationSource(
        video=resolve_manifest_path(video_raw.strip(), manifest_path=manifest_path)
    )


def _require_path(payload: dict[str, Any], field: str) -> str:
    """Require a non-empty path string field from a manifest object.

    Args:
        payload: Parent JSON object.
        field: Field name to read.

    Returns:
        Trimmed path string.

    Raises:
        ValueError: If the field is missing or not a non-empty string.
    """
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest.{field} must be a non-empty path string.")
    return value.strip()


def _require_non_empty_string(payload: dict[str, Any], field: str) -> str:
    """Require a non-empty string field from a manifest object.

    Args:
        payload: Parent JSON object.
        field: Field name to read.

    Returns:
        Trimmed string value.

    Raises:
        ValueError: If the field is missing or not a non-empty string.
    """
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
