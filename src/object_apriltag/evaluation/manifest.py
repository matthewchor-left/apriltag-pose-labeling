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
    dictionary: str
    sensitivity: str


@dataclass(frozen=True)
class HeldOutVideo:
    path: Path
    held_out: bool


@dataclass(frozen=True)
class CalibrationSource:
    video: Path | None


@dataclass(frozen=True)
class EvaluationCandidate:
    name: str
    marker_model_path: Path
    capture_session: str
    solver_variant: str
    calibration_source: CalibrationSource


@dataclass(frozen=True)
class EvaluationManifest:
    manifest_path: Path
    version: int
    cad_model: Path
    object_model: Path
    intrinsics: Path
    detector: DetectorConfig
    held_out_videos: tuple[HeldOutVideo, ...]
    candidates: tuple[EvaluationCandidate, ...]


def repo_root() -> Path:
    return config_dir().parent


def repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    root = repo_root().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_manifest_path(path: str | Path, *, manifest_path: Path) -> Path:
    """Resolve manifest paths relative to the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    _ = manifest_path
    return (repo_root() / candidate).resolve()


def load_evaluation_manifest(path: str | Path) -> EvaluationManifest:
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
    if not isinstance(raw, dict):
        raise ValueError("manifest.detector must be an object with dictionary and sensitivity.")
    dictionary = _require_non_empty_string(raw, "dictionary")
    sensitivity = _require_non_empty_string(raw, "sensitivity")
    return DetectorConfig(dictionary=dictionary, sensitivity=sensitivity)


def _parse_held_out_videos(raw: Any, *, manifest_path: Path) -> tuple[HeldOutVideo, ...]:
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
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest.{field} must be a non-empty path string.")
    return value.strip()


def _require_non_empty_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()
