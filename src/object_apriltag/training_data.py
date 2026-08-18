"""YOLO pose training sample generation from fused object pose and CAD geometry."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from object_apriltag.cad import CadLandmarks, CadModel, CadRegistration, load_cad_landmarks
from object_apriltag.detector import ObjectDetector, ObjectPose
from object_apriltag.frame_source import (
    FrameSource,
    format_frame_source,
    is_camera_source,
    open_frame_source,
    read_frame,
)
from object_apriltag.layout import MarkerLayout
from object_apriltag.viz.cad_overlay import (
    CAMERA_NEAR_M,
    cad_points_to_layout,
    layout_points_to_camera,
    project_cad_mesh_silhouette_bounds,
    project_camera_points,
)

YOLO_LANDMARK_NAMES: tuple[str, ...] = (
    "back-center",
    "back-left-center",
    "back-right-center",
    "front-center",
    "front-left-center",
    "front-right-center",
    "left-center",
    "right-center",
    "top-back-center",
    "top-back-left",
    "top-back-right",
    "top-center",
    "top-front-center",
    "top-front-left",
    "top-front-right",
    "top-left-center",
    "top-right-center",
)
YOLO_CLASS_ID = 0
YOLO_KEYPOINT_VISIBILITY = 2
YOLO_FIELD_COUNT = 56
JPEG_QUALITY = 95
DATASET_SPLITS = frozenset({"train", "val"})
RUN_REPORT_SCHEMA_VERSION = 1
DATA_YAML_TEXT = (
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: nexplayground\n"
    "kpt_shape: [17, 3]\n"
)


class SampleRejection(str, Enum):
    """Reason a frame failed training-sample acceptance after pose fusion."""

    LANDMARKS = "landmarks"
    BBOX = "bbox"


@dataclass
class FrameLabel:
    """Pixel-space bounding box and keypoints for one accepted frame."""

    bbox_xyxy: tuple[float, float, float, float]
    keypoints_xy: np.ndarray


@dataclass
class RejectionCounts:
    """Per-run rejection counters."""

    no_pose: int = 0
    landmarks: int = 0
    bbox: int = 0


@dataclass
class DatasetGenerationReport:
    """Provenance and counters for one Dataset Generation Run."""

    run_name: str
    split: str
    sample_rate_hz: float
    inputs: dict[str, Any]
    status: str = "completed"
    error: str | None = None
    frames_processed: int = 0
    samples_saved: int = 0
    rejections: RejectionCounts = field(default_factory=RejectionCounts)
    samples: list[dict[str, str]] = field(default_factory=list)


def require_positive_sample_rate(sample_rate_hz: float) -> None:
    """Validate dataset sampling rate."""
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise RuntimeError("--sample-rate-hz must be finite and positive.")


def jpeg_encode_params() -> list[int]:
    """Return OpenCV JPEG encode parameters for Training Sample images."""
    return [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]


def require_yolo_landmarks(cad_landmarks: CadLandmarks | Mapping[str, np.ndarray]) -> None:
    """Ensure all fixed YOLO landmark names are present in CAD landmarks."""
    landmarks = (
        cad_landmarks.landmarks
        if isinstance(cad_landmarks, CadLandmarks)
        else dict(cad_landmarks)
    )
    missing = sorted(set(YOLO_LANDMARK_NAMES) - set(landmarks))
    if missing:
        raise ValueError(f"GLB file is missing required CAD landmarks: {missing}.")


def load_required_yolo_landmarks(path: Path | str) -> CadLandmarks:
    """Load CAD landmarks and fail when required YOLO names are absent."""
    return load_cad_landmarks(path, required_names=YOLO_LANDMARK_NAMES)


def _landmark_positions_cad(
    cad_landmarks: Mapping[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [np.asarray(cad_landmarks[name], dtype=np.float64).reshape(3) for name in YOLO_LANDMARK_NAMES],
        axis=0,
    )


def _point_inside_image(x: float, y: float, *, image_width: int, image_height: int) -> bool:
    return 0.0 <= x < float(image_width) and 0.0 <= y < float(image_height)


def project_yolo_landmarks_to_image(
    cad_landmarks: Mapping[str, np.ndarray],
    pose: ObjectPose,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_model: MarkerLayout,
    registration: CadRegistration,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray | SampleRejection:
    """Project fixed YOLO landmarks into pixel coordinates."""
    points_cad = _landmark_positions_cad(cad_landmarks)
    if not np.all(np.isfinite(points_cad)):
        return SampleRejection.LANDMARKS

    layout_points = cad_points_to_layout(points_cad, registration)
    camera_points = layout_points_to_camera(layout_points, pose, marker_model)
    if not np.all(np.isfinite(camera_points)):
        return SampleRejection.LANDMARKS
    if np.any(camera_points[:, 2] <= CAMERA_NEAR_M):
        return SampleRejection.LANDMARKS

    image_points = project_camera_points(camera_points, camera_matrix, dist_coeffs)
    if not np.all(np.isfinite(image_points)):
        return SampleRejection.LANDMARKS

    for point in image_points:
        if not _point_inside_image(
            float(point[0]),
            float(point[1]),
            image_width=image_width,
            image_height=image_height,
        ):
            return SampleRejection.LANDMARKS
    return image_points


def build_training_sample(
    *,
    pose: ObjectPose,
    cad_landmarks: Mapping[str, np.ndarray],
    cad_model: CadModel,
    registration: CadRegistration,
    marker_model: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
) -> FrameLabel | SampleRejection:
    """Build one accepted Training Sample label from fused pose and CAD geometry."""
    keypoints = project_yolo_landmarks_to_image(
        cad_landmarks,
        pose,
        camera_matrix,
        dist_coeffs,
        marker_model,
        registration,
        image_width=image_width,
        image_height=image_height,
    )
    if isinstance(keypoints, SampleRejection):
        return keypoints

    bounds = project_cad_mesh_silhouette_bounds(
        pose,
        camera_matrix,
        dist_coeffs,
        marker_model,
        cad_model,
        registration,
        image_width=image_width,
        image_height=image_height,
    )
    if bounds is None:
        return SampleRejection.BBOX
    return FrameLabel(bbox_xyxy=bounds, keypoints_xy=keypoints)


def normalize_yolo_bbox(
    bbox_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel bbox corners to YOLO normalized center and size."""
    x_min, y_min, x_max, y_max = (float(value) for value in bbox_xyxy)
    width = float(image_width)
    height = float(image_height)
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    return x_center, y_center, box_width, box_height


def draw_yolo_pose_label(
    frame: np.ndarray,
    label: FrameLabel,
    *,
    landmark_names: Sequence[str] = YOLO_LANDMARK_NAMES,
) -> np.ndarray:
    """Draw bbox and named keypoints for one accepted Training Sample."""
    output = frame.copy()
    x_min, y_min, x_max, y_max = (int(round(value)) for value in label.bbox_xyxy)
    cv2.rectangle(
        output,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2,
        lineType=cv2.LINE_AA,
    )
    for index, point in enumerate(label.keypoints_xy):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        cv2.circle(output, (x, y), 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(output, (x, y), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        name = landmark_names[index]
        label_origin = (x + 6, y - 6)
        cv2.putText(
            output,
            name,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            name,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    return output


def format_yolo_pose_label(
    label: FrameLabel,
    *,
    image_width: int,
    image_height: int,
) -> str:
    """Serialize one YOLO pose label row with exactly 56 fields."""
    x_center, y_center, box_width, box_height = normalize_yolo_bbox(
        label.bbox_xyxy,
        image_width=image_width,
        image_height=image_height,
    )
    width = float(image_width)
    height = float(image_height)
    fields: list[str] = [
        str(YOLO_CLASS_ID),
        f"{x_center:.6f}",
        f"{y_center:.6f}",
        f"{box_width:.6f}",
        f"{box_height:.6f}",
    ]
    for point in label.keypoints_xy:
        fields.extend(
            [
                f"{float(point[0]) / width:.6f}",
                f"{float(point[1]) / height:.6f}",
                str(YOLO_KEYPOINT_VISIBILITY),
            ]
        )
    if len(fields) != YOLO_FIELD_COUNT:
        raise ValueError(f"YOLO label must contain {YOLO_FIELD_COUNT} fields, got {len(fields)}.")
    return " ".join(fields)


def yolo_label_field_count(label_line: str) -> int:
    """Return the number of whitespace-separated YOLO label fields."""
    stripped = label_line.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _validate_data_yaml(path: Path) -> None:
    if not path.exists():
        return
    existing = path.read_text(encoding="utf-8")
    if existing != DATA_YAML_TEXT:
        raise RuntimeError(
            f"Existing data.yaml at {path} does not match the required training dataset schema."
        )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _make_temp_artifact(directory: Path, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(fd)
    return Path(tmp_path)


def _cleanup_temp_artifacts(*paths: Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink()
        except OSError:
            pass


def ensure_dataset_layout(output_dir: Path, *, split: str, run_name: str) -> None:
    """Create dataset directories and refuse run or schema collisions."""
    if split not in DATASET_SPLITS:
        raise ValueError(f"Dataset split must be one of {sorted(DATASET_SPLITS)}, got {split!r}.")

    run_report_path = output_dir / "runs" / f"{run_name}.json"
    if run_report_path.exists():
        raise RuntimeError(f"Dataset Generation Run already exists: {run_report_path}")

    for directory in (
        output_dir / "images" / split,
        output_dir / "labels" / split,
        output_dir / "labeled-images" / split,
    ):
        if any(directory.glob(f"{run_name}_*")):
            raise RuntimeError(
                f"Dataset sample files already exist for run {run_name!r} under {directory}."
            )

    _validate_data_yaml(output_dir / "data.yaml")
    for path in (
        output_dir / "images" / split,
        output_dir / "labels" / split,
        output_dir / "labeled-images" / split,
        output_dir / "runs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(output_dir / "data.yaml", DATA_YAML_TEXT)


def sample_stem(run_name: str, sample_index: int) -> str:
    """Return the shared image/label filename stem for one Training Sample."""
    return f"{run_name}_{sample_index}"


def write_training_sample(
    output_dir: Path,
    *,
    split: str,
    run_name: str,
    sample_index: int,
    frame: np.ndarray,
    label_line: str,
) -> tuple[str, str]:
    """Write one raw JPEG and matching YOLO label as a recoverable pair."""
    stem = sample_stem(run_name, sample_index)
    image_path = output_dir / "images" / split / f"{stem}.jpg"
    label_path = output_dir / "labels" / split / f"{stem}.txt"
    if image_path.exists() or label_path.exists():
        raise RuntimeError(f"Training Sample collision for {stem} under {output_dir}.")

    encoded, encoded_image = cv2.imencode(".jpg", frame, jpeg_encode_params())
    if not encoded:
        raise RuntimeError(f"Failed to encode JPEG for {image_path}.")

    image_tmp = _make_temp_artifact(image_path.parent, ".jpg.tmp")
    label_tmp = _make_temp_artifact(label_path.parent, ".txt.tmp")
    image_published = False
    try:
        image_tmp.write_bytes(encoded_image.tobytes())
        label_tmp.write_text(label_line + "\n", encoding="utf-8")
        os.replace(image_tmp, image_path)
        image_tmp = None
        image_published = True
        os.replace(label_tmp, label_path)
        label_tmp = None
    except Exception:
        if image_published and image_path.exists():
            try:
                image_path.unlink()
            except OSError:
                pass
        raise
    finally:
        _cleanup_temp_artifacts(image_tmp, label_tmp)

    return (
        str(image_path.relative_to(output_dir)),
        str(label_path.relative_to(output_dir)),
    )


def write_labeled_training_image(
    output_dir: Path,
    *,
    split: str,
    run_name: str,
    sample_index: int,
    frame: np.ndarray,
    label: FrameLabel,
) -> str:
    """Write one annotated JPEG preview for a saved Training Sample."""
    stem = sample_stem(run_name, sample_index)
    image_path = output_dir / "labeled-images" / split / f"{stem}.jpg"
    if image_path.exists():
        raise RuntimeError(f"Labeled Training Sample collision for {stem} under {output_dir}.")

    annotated = draw_yolo_pose_label(frame, label)
    encoded, encoded_image = cv2.imencode(".jpg", annotated, jpeg_encode_params())
    if not encoded:
        raise RuntimeError(f"Failed to encode labeled JPEG for {image_path}.")
    _write_bytes_atomic(image_path, encoded_image.tobytes())
    return str(image_path.relative_to(output_dir))


def validate_run_report(document: Mapping[str, Any]) -> None:
    """Validate a persisted Dataset Generation Run report."""
    if document.get("schema_version") != RUN_REPORT_SCHEMA_VERSION:
        raise ValueError("Run report schema_version is missing or unsupported.")
    for field_name in ("run_name", "split", "sample_rate_hz", "inputs", "status"):
        if field_name not in document:
            raise ValueError(f"Run report is missing required field {field_name!r}.")
    status = document["status"]
    if status not in ("completed", "failed"):
        raise ValueError("Run report status must be completed or failed.")
    if status == "failed":
        error = document.get("error")
        if not isinstance(error, str) or not error:
            raise ValueError("Run report error must be a non-empty string when status is failed.")
    if document["split"] not in DATASET_SPLITS:
        raise ValueError("Run report split must be train or val.")
    counts = document.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("Run report counts must be an object.")
    for field_name in (
        "frames_processed",
        "samples_saved",
        "rejected_no_pose",
        "rejected_landmarks",
        "rejected_bbox",
    ):
        if field_name not in counts:
            raise ValueError(f"Run report counts missing {field_name!r}.")


def serialize_run_report(report: DatasetGenerationReport) -> dict[str, Any]:
    """Convert a run report dataclass to a JSON-serializable document."""
    document: dict[str, Any] = {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "run_name": report.run_name,
        "split": report.split,
        "sample_rate_hz": report.sample_rate_hz,
        "inputs": report.inputs,
        "status": report.status,
        "counts": {
            "frames_processed": report.frames_processed,
            "samples_saved": report.samples_saved,
            "rejected_no_pose": report.rejections.no_pose,
            "rejected_landmarks": report.rejections.landmarks,
            "rejected_bbox": report.rejections.bbox,
        },
        "samples": list(report.samples),
    }
    if report.error is not None:
        document["error"] = report.error
    return document


def finalize_run_report(output_dir: Path, report: DatasetGenerationReport) -> Path:
    """Write ``runs/<run-name>.json`` after schema validation."""
    document = serialize_run_report(report)
    validate_run_report(document)
    path = output_dir / "runs" / f"{report.run_name}.json"
    if path.exists():
        raise RuntimeError(f"Dataset Generation Run report already exists: {path}")
    _write_text_atomic(path, json.dumps(document, indent=2, allow_nan=False) + "\n")
    return path


def write_run_report(output_dir: Path, report: DatasetGenerationReport) -> Path:
    """Write a completed run report."""
    if report.status != "completed":
        raise ValueError("write_run_report expects a completed Dataset Generation Run.")
    return finalize_run_report(output_dir, report)


def _draw_preview_hud(
    frame: np.ndarray,
    *,
    split: str,
    samples_saved: int,
    due_for_sample: bool,
) -> None:
    status = "due" if due_for_sample else "waiting"
    cv2.putText(
        frame,
        f"split={split} saved={samples_saved} sample={status} | q quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _video_media_time_seconds(
    capture: cv2.VideoCapture,
    frame_index: int,
    reported_fps: float,
) -> float:
    pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if math.isfinite(pos_msec) and pos_msec >= 0.0:
        return pos_msec / 1000.0
    if not math.isfinite(reported_fps) or reported_fps <= 0.0:
        raise RuntimeError(
            "Video reported FPS must be finite and positive when frame media timestamps "
            f"are unavailable; got {reported_fps!r}."
        )
    return frame_index / reported_fps


def _process_due_frame(
    *,
    frame: np.ndarray,
    report: DatasetGenerationReport,
    detector: ObjectDetector,
    cad_landmarks: Mapping[str, np.ndarray],
    cad_model: CadModel,
    registration: CadRegistration,
    marker_model: MarkerLayout,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
    output_dir: Path,
    split: str,
    run_name: str,
    sample_index: int,
    due_for_sample: bool,
    current_time: float,
    labeled_images_limit: int | None = None,
) -> bool:
    if not due_for_sample:
        return False

    pose = detector.fuse(detector.find_markers(frame))
    if pose is None:
        report.rejections.no_pose += 1
        return False

    sample = build_training_sample(
        pose=pose,
        cad_landmarks=cad_landmarks,
        cad_model=cad_model,
        registration=registration,
        marker_model=marker_model,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_width,
        image_height=image_height,
    )
    if isinstance(sample, SampleRejection):
        if sample is SampleRejection.LANDMARKS:
            report.rejections.landmarks += 1
        else:
            report.rejections.bbox += 1
        return False

    label_line = format_yolo_pose_label(
        sample,
        image_width=image_width,
        image_height=image_height,
    )
    image_rel, label_rel = write_training_sample(
        output_dir,
        split=split,
        run_name=run_name,
        sample_index=sample_index,
        frame=frame,
        label_line=label_line,
    )
    sample_record: dict[str, str] = {
        "index": str(sample_index),
        "image": image_rel,
        "label": label_rel,
        "time_s": f"{current_time:.6f}",
    }
    if labeled_images_limit is not None and sample_index < labeled_images_limit:
        sample_record["labeled_image"] = write_labeled_training_image(
            output_dir,
            split=split,
            run_name=run_name,
            sample_index=sample_index,
            frame=frame,
            label=sample,
        )
    report.samples_saved += 1
    report.samples.append(sample_record)
    return True


def _run_sampling_loop(
    *,
    capture: cv2.VideoCapture,
    source: FrameSource,
    report: DatasetGenerationReport,
    detector: ObjectDetector,
    cad_landmarks: Mapping[str, np.ndarray],
    cad_model: CadModel,
    registration: CadRegistration,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
    output_dir: Path,
    split: str,
    run_name: str,
    sample_rate_hz: float,
    show_preview: bool,
    loop_on_eof: bool,
    stop_on_eof: bool,
    current_time_fn: Callable[[int], float],
    preview_should_stop: Callable[[], bool] | None = None,
    labeled_images_limit: int | None = None,
) -> None:
    marker_model = detector.marker_model
    sample_interval = 1.0 / sample_rate_hz
    next_sample_time = 0.0
    sample_index = 0
    frame_index = 0

    while True:
        ok, frame = read_frame(capture, source, loop_on_eof=loop_on_eof)
        if not ok or frame is None:
            if stop_on_eof:
                break
            raise RuntimeFrameReadError(format_frame_source(source))

        current_time = current_time_fn(frame_index)
        frame_index += 1
        report.frames_processed += 1
        due_for_sample = current_time >= next_sample_time
        saved = _process_due_frame(
            frame=frame,
            report=report,
            detector=detector,
            cad_landmarks=cad_landmarks,
            cad_model=cad_model,
            registration=registration,
            marker_model=marker_model,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=image_width,
            image_height=image_height,
            output_dir=output_dir,
            split=split,
            run_name=run_name,
            sample_index=sample_index,
            due_for_sample=due_for_sample,
            current_time=current_time,
            labeled_images_limit=labeled_images_limit,
        )
        if saved:
            sample_index += 1
            next_sample_time = current_time + sample_interval

        if show_preview:
            preview = frame.copy()
            _draw_preview_hud(
                preview,
                split=split,
                samples_saved=report.samples_saved,
                due_for_sample=due_for_sample,
            )
            cv2.imshow("YOLO Dataset Generator", preview)
            if preview_should_stop is not None and preview_should_stop():
                break


class RuntimeFrameReadError(RuntimeError):
    """Raised when a live camera frame cannot be read."""


def generate_dataset_from_source(
    *,
    source: FrameSource,
    output_dir: Path,
    split: str,
    run_name: str,
    sample_rate_hz: float,
    detector: ObjectDetector,
    cad_landmarks: Mapping[str, np.ndarray],
    cad_model: CadModel,
    registration: CadRegistration,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
    inputs: Mapping[str, Any],
    show_preview: bool,
    labeled_images_limit: int | None = None,
) -> DatasetGenerationReport:
    """Process a frame source and write accepted Training Samples."""
    require_positive_sample_rate(sample_rate_hz)

    report = DatasetGenerationReport(
        run_name=run_name,
        split=split,
        sample_rate_hz=sample_rate_hz,
        inputs=dict(inputs),
    )
    generation_started = False
    capture: cv2.VideoCapture | None = None
    run_error: BaseException | None = None

    try:
        ensure_dataset_layout(output_dir, split=split, run_name=run_name)
        generation_started = True
        capture = open_frame_source(source, width=image_width, height=image_height)
        if is_camera_source(source):
            clock_start = time.monotonic()

            def camera_time(_frame_index: int) -> float:
                return time.monotonic() - clock_start

            def preview_should_stop() -> bool:
                return (cv2.waitKey(1) & 0xFF) == ord("q")

            _run_sampling_loop(
                capture=capture,
                source=source,
                report=report,
                detector=detector,
                cad_landmarks=cad_landmarks,
                cad_model=cad_model,
                registration=registration,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=image_width,
                image_height=image_height,
                output_dir=output_dir,
                split=split,
                run_name=run_name,
                sample_rate_hz=sample_rate_hz,
                show_preview=show_preview,
                loop_on_eof=False,
                stop_on_eof=False,
                current_time_fn=camera_time,
                preview_should_stop=preview_should_stop if show_preview else None,
                labeled_images_limit=labeled_images_limit,
            )
        else:
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))

            def video_time(frame_index: int) -> float:
                return _video_media_time_seconds(capture, frame_index, reported_fps)

            _run_sampling_loop(
                capture=capture,
                source=source,
                report=report,
                detector=detector,
                cad_landmarks=cad_landmarks,
                cad_model=cad_model,
                registration=registration,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=image_width,
                image_height=image_height,
                output_dir=output_dir,
                split=split,
                run_name=run_name,
                sample_rate_hz=sample_rate_hz,
                show_preview=False,
                loop_on_eof=False,
                stop_on_eof=True,
                current_time_fn=video_time,
                labeled_images_limit=labeled_images_limit,
            )
    except BaseException as error:
        run_error = error
        if generation_started:
            report.status = "failed"
            report.error = str(error)
    finally:
        if capture is not None:
            capture.release()
        if show_preview:
            cv2.destroyAllWindows()
        if generation_started:
            finalize_run_report(output_dir, report)

    if run_error is not None:
        raise run_error
    return report