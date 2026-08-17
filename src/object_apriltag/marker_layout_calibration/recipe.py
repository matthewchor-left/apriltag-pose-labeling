"""Strict Calibration Recipe parsing for workspace ``config.json`` files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from object_apriltag.frame_source import FrameSource
from object_apriltag.marker_layout_calibration.input import (
    parse_anchor_marker_ids,
    parse_marker_id_spec,
    validate_marker_size,
)
from object_apriltag.marker_layout_calibration.types import CalibrationSettings
from object_apriltag.object_model_edit import parse_keypoint_sources

CALIBRATION_RECIPE_VERSION = 1

BENCHMARK_FRAME_SELECTION_UNIFORM = "uniform"
BENCHMARK_FRAME_SELECTION_SHARPEST = "sharpest"
INTERACTIVE_CAPTURE_MANUAL = "manual"
INTERACTIVE_CAPTURE_AUTO = "auto"
INTERACTIVE_PREVIEW_NONE = "none"
INTERACTIVE_PREVIEW_KEYPOINT_SOURCES = "keypoint_sources"
SOLVER_POLICY_STRICT = "strict"
SOLVER_POLICY_BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class CalibrationRecipePaths:
    """Fixed artifact paths derived from a workspace ``config.json`` location."""

    config_path: Path
    marker_model_path: Path
    object_model_path: Path
    diagnostics_path: Path


@dataclass(frozen=True)
class BenchmarkExecution:
    """Headless benchmark execution settings."""

    sample_rate_hz: float
    frame_selection: Literal["uniform", "sharpest"]


@dataclass(frozen=True)
class InteractiveExecution:
    """Live or looping-video interactive capture settings."""

    capture: Literal["manual", "auto"]
    sample_rate_hz: float | None
    preview: Literal["none", "keypoint_sources"]


@dataclass(frozen=True)
class CalibrationRecipe:
    """Parsed and validated Calibration Recipe."""

    paths: CalibrationRecipePaths
    source: FrameSource
    intrinsics_path: Path
    dictionary: str
    sensitivity: str
    expected_marker_ids: tuple[int, ...]
    marker_sizes_m: dict[int, float]
    reference_marker_id: int
    default_marker_size_m: float
    anchor_marker_ids: tuple[int, ...] | None
    execution: BenchmarkExecution | InteractiveExecution
    settings: CalibrationSettings
    policy: Literal["strict", "best_effort"]
    anchor_stop_after_expansion: bool
    partial_output: bool
    keypoint_sources: dict[str, tuple[int, str, float]]
    skeleton: tuple[tuple[str, str], ...]


def load_calibration_recipe(config_path: str | Path) -> CalibrationRecipe:
    """Load and validate a Calibration Recipe JSON file.

    Args:
        config_path: Path to ``config.json`` in a Calibration Workspace.

    Returns:
        Parsed ``CalibrationRecipe``.

    Raises:
        FileNotFoundError: When the config file does not exist.
        ValueError: When the JSON is malformed or fails validation.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Calibration recipe not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Calibration recipe must be a JSON object.")
    return _parse_recipe_payload(payload, path)


def _parse_recipe_payload(payload: dict[str, Any], config_path: Path) -> CalibrationRecipe:
    """Parse a recipe JSON object after ``config_version`` validation."""
    _require_exact_keys(
        payload,
        "config",
        frozenset(
            {
                "config_version",
                "inputs",
                "detector",
                "markers",
                "execution",
                "solver",
                "object_model",
            }
        ),
    )
    version = payload["config_version"]
    if isinstance(version, bool) or version != CALIBRATION_RECIPE_VERSION:
        raise ValueError(
            f"Unsupported config_version {version!r}; expected {CALIBRATION_RECIPE_VERSION}."
        )

    workspace_dir = config_path.parent
    paths = CalibrationRecipePaths(
        config_path=config_path,
        marker_model_path=workspace_dir / "marker_model.json",
        object_model_path=workspace_dir / "object_model.json",
        diagnostics_path=workspace_dir / "diagnostics.json",
    )

    source, intrinsics_path = _parse_inputs(payload["inputs"], workspace_dir=workspace_dir)
    dictionary, sensitivity = _parse_detector(payload["detector"])
    (
        expected_ids,
        marker_sizes_m,
        reference_marker_id,
        default_marker_size_m,
        anchor_marker_ids,
    ) = _parse_markers(payload["markers"])
    execution = _parse_execution(payload["execution"])
    settings, policy, anchor_stop, partial_output = _parse_solver(
        payload["solver"],
        anchor_marker_ids=anchor_marker_ids,
    )
    keypoint_sources, skeleton = _parse_object_model(
        payload["object_model"],
        expected_marker_ids=expected_ids,
    )

    return CalibrationRecipe(
        paths=paths,
        source=source,
        intrinsics_path=intrinsics_path,
        dictionary=dictionary,
        sensitivity=sensitivity,
        expected_marker_ids=expected_ids,
        marker_sizes_m=marker_sizes_m,
        reference_marker_id=reference_marker_id,
        default_marker_size_m=default_marker_size_m,
        anchor_marker_ids=anchor_marker_ids,
        execution=execution,
        settings=settings,
        policy=policy,
        anchor_stop_after_expansion=anchor_stop,
        partial_output=partial_output,
        keypoint_sources=keypoint_sources,
        skeleton=skeleton,
    )


def _resolve_config_path(path_text: str, *, workspace_dir: Path, field: str) -> Path:
    """Resolve a path string relative to the workspace directory."""
    stripped = path_text.strip()
    if not stripped:
        raise ValueError(f"{field} must be a non-empty path string.")
    candidate = Path(stripped)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace_dir / candidate).resolve()
    if not resolved.is_file():
        raise ValueError(f"{field} path does not exist: {resolved}")
    return resolved


def _parse_source_value(raw: Any, *, workspace_dir: Path) -> FrameSource:
    """Parse a recipe source string into a camera index or video path."""
    if not isinstance(raw, str):
        raise ValueError("inputs.source must be a non-empty string.")
    stripped = raw.strip()
    if not stripped:
        raise ValueError("inputs.source must be a non-empty string.")
    try:
        return int(stripped)
    except ValueError:
        return _resolve_config_path(stripped, workspace_dir=workspace_dir, field="inputs.source")


def _parse_inputs(raw: Any, *, workspace_dir: Path) -> tuple[FrameSource, Path]:
    """Parse the recipe ``inputs`` section."""
    if not isinstance(raw, dict):
        raise ValueError("inputs must be an object.")
    _require_exact_keys(raw, "inputs", frozenset({"source", "intrinsics"}))
    source = _parse_source_value(raw["source"], workspace_dir=workspace_dir)
    intrinsics = _require_non_empty_string(raw["intrinsics"], "inputs.intrinsics")
    intrinsics_path = _resolve_config_path(
        intrinsics,
        workspace_dir=workspace_dir,
        field="inputs.intrinsics",
    )
    return source, intrinsics_path


def _parse_detector(raw: Any) -> tuple[str, str]:
    """Parse the recipe ``detector`` section."""
    if not isinstance(raw, dict):
        raise ValueError("detector must be an object.")
    _require_exact_keys(raw, "detector", frozenset({"dictionary", "sensitivity"}))
    dictionary = _require_non_empty_string(raw["dictionary"], "detector.dictionary")
    sensitivity = _require_non_empty_string(raw["sensitivity"], "detector.sensitivity")
    if sensitivity not in ("default", "relaxed", "aggressive"):
        raise ValueError(
            "detector.sensitivity must be 'default', 'relaxed', or 'aggressive'."
        )
    return dictionary, sensitivity


def _parse_marker_id_entries(raw_ids: Any, field: str) -> list[int]:
    """Parse a marker group's ``ids`` array of integers and range strings."""
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(f"{field} must be a non-empty array.")
    marker_ids: list[int] = []
    for index, entry in enumerate(raw_ids):
        entry_field = f"{field}[{index}]"
        if isinstance(entry, bool):
            raise ValueError(f"{entry_field} must be an integer or range string.")
        if isinstance(entry, int):
            marker_ids.append(entry)
            continue
        if isinstance(entry, str):
            token = entry.strip()
            if not token:
                raise ValueError(f"{entry_field} must not be empty.")
            parsed, failure = parse_marker_id_spec([token])
            if failure is not None:
                raise ValueError(f"{entry_field}: {failure}")
            assert parsed is not None
            marker_ids.extend(parsed)
            continue
        raise ValueError(f"{entry_field} must be an integer or range string.")
    return marker_ids


def _parse_markers(raw: Any) -> tuple[
    tuple[int, ...],
    dict[int, float],
    int,
    float,
    tuple[int, ...] | None,
]:
    """Parse marker inventory groups, reference marker, and optional anchors."""
    if not isinstance(raw, dict):
        raise ValueError("markers must be an object.")
    _require_exact_keys(
        raw,
        "markers",
        frozenset({"reference_marker_id", "groups", "anchor_marker_ids"}),
    )
    reference_marker_id = _parse_marker_id(raw["reference_marker_id"], "markers.reference_marker_id")
    groups_raw = raw["groups"]
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError("markers.groups must be a non-empty array.")

    used_ids: set[int] = set()
    marker_sizes_m: dict[int, float] = {}
    default_marker_size_m: float | None = None

    for index, group in enumerate(groups_raw):
        field = f"markers.groups[{index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{field} must be an object.")
        _require_exact_keys(group, field, frozenset({"ids", "size_m"}))
        group_ids = _parse_marker_id_entries(group["ids"], f"{field}.ids")
        duplicates = sorted(
            marker_id for marker_id in set(group_ids) if group_ids.count(marker_id) > 1
        )
        if duplicates:
            raise ValueError(f"{field} contains duplicate marker IDs: {duplicates}.")
        overlap = sorted(set(group_ids) & used_ids)
        if overlap:
            raise ValueError(f"{field} overlaps existing marker IDs: {overlap}.")
        size_m = _parse_positive_float(group["size_m"], f"{field}.size_m")
        size_failure = validate_marker_size(size_m)
        if size_failure is not None:
            raise ValueError(f"{field}.size_m {size_failure}")
        for marker_id in group_ids:
            used_ids.add(marker_id)
            marker_sizes_m[marker_id] = size_m
        if default_marker_size_m is None:
            default_marker_size_m = size_m

    if not used_ids:
        raise ValueError("markers.groups must declare at least one marker ID.")
    if reference_marker_id not in used_ids:
        raise ValueError(
            f"markers.reference_marker_id {reference_marker_id} is not present in markers.groups."
        )
    assert default_marker_size_m is not None

    expected_ids = tuple(sorted(used_ids))
    anchor_marker_ids = _parse_anchor_marker_ids_field(
        raw["anchor_marker_ids"],
        expected_ids=expected_ids,
        reference_marker_id=reference_marker_id,
    )
    return expected_ids, marker_sizes_m, reference_marker_id, default_marker_size_m, anchor_marker_ids


def _parse_anchor_marker_ids_field(
    raw: Any,
    *,
    expected_ids: tuple[int, ...],
    reference_marker_id: int,
) -> tuple[int, ...] | None:
    """Parse explicit anchor marker IDs or ``null`` for automatic anchor selection."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("markers.anchor_marker_ids must be null or an array of marker IDs.")
    tokens: list[str] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, bool):
            raise ValueError(f"markers.anchor_marker_ids[{index}] must be an integer or range string.")
        if isinstance(entry, int):
            tokens.append(str(entry))
            continue
        if isinstance(entry, str) and entry.strip():
            tokens.append(entry.strip())
            continue
        raise ValueError(
            f"markers.anchor_marker_ids[{index}] must be an integer or range string."
        )
    if not tokens:
        raise ValueError("markers.anchor_marker_ids must not be an empty array; use null instead.")
    parsed, failure = parse_marker_id_spec(tokens)
    if failure is not None:
        raise ValueError(f"markers.anchor_marker_ids: {failure}")
    assert parsed is not None
    anchor_ids, anchor_failure = parse_anchor_marker_ids(
        parsed,
        expected_ids,
        reference_marker_id,
    )
    if anchor_failure is not None:
        raise ValueError(anchor_failure)
    return anchor_ids


def _parse_execution(raw: Any) -> BenchmarkExecution | InteractiveExecution:
    """Parse discriminated ``execution`` settings."""
    if not isinstance(raw, dict):
        raise ValueError("execution must be an object.")
    mode = raw.get("mode")
    if mode == "benchmark":
        _require_exact_keys(
            raw,
            "execution",
            frozenset({"mode", "sample_rate_hz", "frame_selection"}),
        )
        sample_rate_hz = _parse_positive_float(raw["sample_rate_hz"], "execution.sample_rate_hz")
        frame_selection = raw["frame_selection"]
        if frame_selection not in (
            BENCHMARK_FRAME_SELECTION_UNIFORM,
            BENCHMARK_FRAME_SELECTION_SHARPEST,
        ):
            raise ValueError(
                "execution.frame_selection must be "
                f"'{BENCHMARK_FRAME_SELECTION_UNIFORM}' or "
                f"'{BENCHMARK_FRAME_SELECTION_SHARPEST}'."
            )
        return BenchmarkExecution(
            sample_rate_hz=sample_rate_hz,
            frame_selection=frame_selection,
        )
    if mode == "interactive":
        capture = raw.get("capture")
        if capture not in (INTERACTIVE_CAPTURE_MANUAL, INTERACTIVE_CAPTURE_AUTO):
            raise ValueError(
                "execution.capture must be "
                f"'{INTERACTIVE_CAPTURE_MANUAL}' or '{INTERACTIVE_CAPTURE_AUTO}'."
            )
        if capture == INTERACTIVE_CAPTURE_AUTO:
            _require_exact_keys(
                raw,
                "execution",
                frozenset({"mode", "capture", "preview", "sample_rate_hz"}),
            )
            sample_rate_hz = _parse_positive_float(
                raw["sample_rate_hz"],
                "execution.sample_rate_hz",
            )
        else:
            _require_exact_keys(
                raw,
                "execution",
                frozenset({"mode", "capture", "preview"}),
            )
            sample_rate_hz = None
        preview = raw["preview"]
        if preview not in (INTERACTIVE_PREVIEW_NONE, INTERACTIVE_PREVIEW_KEYPOINT_SOURCES):
            raise ValueError(
                "execution.preview must be "
                f"'{INTERACTIVE_PREVIEW_NONE}' or '{INTERACTIVE_PREVIEW_KEYPOINT_SOURCES}'."
            )
        return InteractiveExecution(
            capture=capture,
            sample_rate_hz=sample_rate_hz,
            preview=preview,
        )
    raise ValueError("execution.mode must be 'benchmark' or 'interactive'.")


def _parse_solver(
    raw: Any,
    *,
    anchor_marker_ids: tuple[int, ...] | None,
) -> tuple[CalibrationSettings, Literal["strict", "best_effort"], bool, bool]:
    """Parse solver policy, anchor-stop, partial behavior, and quality settings."""
    if not isinstance(raw, dict):
        raise ValueError("solver must be an object.")
    required = frozenset(
        {
            "policy",
            "anchor_stop_after_expansion",
            "partial_output",
            "min_inliers_per_edge",
            "reprojection_rms_gate_px",
            "pair_translation_rms_gate_ratio",
            "pair_rotation_rms_gate_deg",
            "huber_delta_px",
            "corner_outlier_px",
            "max_ba_iterations",
        }
    )
    _require_exact_keys(raw, "solver", required)

    policy_raw = raw["policy"]
    if policy_raw not in (SOLVER_POLICY_STRICT, SOLVER_POLICY_BEST_EFFORT):
        raise ValueError(
            f"solver.policy must be '{SOLVER_POLICY_STRICT}' or '{SOLVER_POLICY_BEST_EFFORT}'."
        )
    policy: Literal["strict", "best_effort"] = policy_raw

    anchor_stop = _parse_bool(raw["anchor_stop_after_expansion"], "solver.anchor_stop_after_expansion")
    partial_output = _parse_bool(raw["partial_output"], "solver.partial_output")

    if anchor_stop and anchor_marker_ids is None:
        raise ValueError(
            "solver.anchor_stop_after_expansion requires explicit markers.anchor_marker_ids."
        )
    if anchor_stop and policy == SOLVER_POLICY_BEST_EFFORT:
        raise ValueError(
            "solver.anchor_stop_after_expansion cannot be used with solver.policy 'best_effort'."
        )
    if partial_output and policy != SOLVER_POLICY_BEST_EFFORT:
        raise ValueError("solver.partial_output requires solver.policy 'best_effort'.")

    settings = CalibrationSettings(
        min_inliers_per_edge=_parse_positive_int(raw["min_inliers_per_edge"], "solver.min_inliers_per_edge"),
        reprojection_rms_gate_px=_parse_positive_float(
            raw["reprojection_rms_gate_px"],
            "solver.reprojection_rms_gate_px",
        ),
        pair_translation_rms_gate_ratio=_parse_positive_float(
            raw["pair_translation_rms_gate_ratio"],
            "solver.pair_translation_rms_gate_ratio",
        ),
        pair_rotation_rms_gate_deg=_parse_positive_float(
            raw["pair_rotation_rms_gate_deg"],
            "solver.pair_rotation_rms_gate_deg",
        ),
        huber_delta_px=_parse_positive_float(raw["huber_delta_px"], "solver.huber_delta_px"),
        corner_outlier_px=_parse_positive_float(raw["corner_outlier_px"], "solver.corner_outlier_px"),
        max_ba_iterations=_parse_positive_int(raw["max_ba_iterations"], "solver.max_ba_iterations"),
    )
    return settings, policy, anchor_stop, partial_output


def _parse_object_model(
    raw: Any,
    *,
    expected_marker_ids: tuple[int, ...],
) -> tuple[dict[str, tuple[int, str, float]], tuple[tuple[str, str], ...]]:
    """Parse object-model definition and cross-validate skeleton coverage."""
    if not isinstance(raw, dict):
        raise ValueError("object_model must be an object.")
    _require_exact_keys(raw, "object_model", frozenset({"keypoint_sources", "skeleton"}))
    sources_raw = raw["keypoint_sources"]
    if not isinstance(sources_raw, dict) or not sources_raw:
        raise ValueError("object_model.keypoint_sources must be a non-empty object.")
    for name, source in sources_raw.items():
        field = f"object_model.keypoint_sources.{name}"
        if not isinstance(source, dict):
            raise ValueError(f"{field} must be an object.")
        unknown = sorted(set(source) - {"marker_id", "corner", "padding_mm"})
        if unknown:
            raise ValueError(f"{field} has unknown fields: {unknown}.")
    skeleton = _parse_skeleton(raw["skeleton"], "object_model.skeleton")

    sources = parse_keypoint_sources({"keypoint_sources": sources_raw})
    skeleton_names = _skeleton_keypoint_names(skeleton)
    missing_sources = sorted(set(skeleton_names) - set(sources.keys()))
    if missing_sources:
        raise ValueError(
            "object_model skeleton keypoints must have keypoint_sources entries: "
            f"missing {missing_sources}."
        )

    inventory = set(expected_marker_ids)
    outside_inventory = sorted(
        {marker_id for marker_id, _, _ in sources.values()} - inventory
    )
    if outside_inventory:
        raise ValueError(
            "object_model.keypoint_sources reference marker IDs outside markers.groups: "
            f"{outside_inventory}."
        )
    return sources, skeleton


def _skeleton_keypoint_names(skeleton: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """Collect unique skeleton endpoint names in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for start_name, end_name in skeleton:
        for name in (start_name, end_name):
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    return tuple(ordered)


def _parse_skeleton(raw: Any, field: str) -> tuple[tuple[str, str], ...]:
    """Parse a skeleton edge list."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field} must be a non-empty array.")
    edges: list[tuple[str, str]] = []
    for index, edge in enumerate(raw):
        edge_field = f"{field}[{index}]"
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"{edge_field} must be [start, end] keypoint names.")
        if not isinstance(edge[0], str) or not isinstance(edge[1], str):
            raise ValueError(f"{edge_field} keypoint names must be strings.")
        start_name = edge[0].strip()
        end_name = edge[1].strip()
        if not start_name or not end_name:
            raise ValueError(f"{edge_field} keypoint names must be non-empty.")
        edges.append((start_name, end_name))
    return tuple(edges)


def _parse_marker_id(raw: Any, field: str) -> int:
    """Parse a single marker ID from JSON."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be an integer marker id.")
    return int(raw)


def _parse_bool(raw: Any, field: str) -> bool:
    """Parse a strict JSON boolean."""
    if raw is not True and raw is not False:
        raise ValueError(f"{field} must be true or false.")
    return raw


def _parse_positive_int(raw: Any, field: str) -> int:
    """Parse a positive integer setting."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{field} must be a positive integer.")
    if raw <= 0:
        raise ValueError(f"{field} must be positive.")
    return int(raw)


def _parse_positive_float(raw: Any, field: str) -> float:
    """Parse a finite positive float setting."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{field} must be a positive number.")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and positive.")
    return value


def _require_non_empty_string(raw: Any, field: str) -> str:
    """Require a non-empty string field."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return raw.strip()


def _require_exact_keys(payload: dict[str, Any], field: str, allowed: frozenset[str]) -> None:
    """Reject unknown or missing keys in a strict JSON object."""
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"{field} has unknown fields: {unknown}.")
    missing = sorted(allowed - set(payload.keys()))
    if missing:
        raise ValueError(f"{field} is missing required fields: {missing}.")


