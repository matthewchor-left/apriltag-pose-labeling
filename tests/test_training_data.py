"""Tests for YOLO training-data generation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from object_apriltag.cad import CadRegistration, load_cad_model, load_cad_registration
from object_apriltag.cad_self_occlusion import (
    YOLO_KEYPOINT_VISIBLE,
    build_cad_self_occlusion_context,
)
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import load_marker_model
from object_apriltag.training_data import (
    DATA_YAML_TEXT,
    DatasetGenerationReport,
    FrameLabel,
    JPEG_QUALITY,
    RejectionCounts,
    RuntimeFrameReadError,
    SampleRejection,
    YOLO_FIELD_COUNT,
    YOLO_KEYPOINT_VISIBILITY,
    YOLO_LANDMARK_NAMES,
    build_training_sample,
    draw_yolo_pose_label,
    ensure_dataset_layout,
    format_yolo_pose_label,
    generate_dataset_from_source,
    jpeg_encode_params,
    load_required_yolo_landmarks,
    project_yolo_landmarks_to_image,
    require_positive_sample_rate,
    require_yolo_landmarks,
    serialize_run_report,
    validate_run_report,
    write_labeled_training_image,
    write_training_sample,
)
from object_apriltag.viz.cad_overlay import project_cad_mesh_silhouette_bounds
from tests.test_cad_overlay import build_triangle_glb, identity_registration_payload, synthetic_camera

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MARKER_MODEL = REPO_ROOT / "config/Model/remote/marker_model.json"


def load_registration_from_payload(payload: dict[str, object]) -> CadRegistration:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        return load_cad_registration(handle.name)


def synthetic_yolo_landmarks(*, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
    vertex_positions = [
        (center[0] - 0.08, center[1] - 0.08, center[2]),
        (center[0] + 0.08, center[1] - 0.08, center[2]),
        (center[0] - 0.08, center[1] + 0.08, center[2]),
        (center[0] + 0.08, center[1] + 0.08, center[2]),
    ]
    landmarks: dict[str, np.ndarray] = {}
    for index, name in enumerate(YOLO_LANDMARK_NAMES):
        landmarks[name] = np.array(vertex_positions[index % len(vertex_positions)], dtype=np.float64)
    return landmarks


def load_triangle_cad_model(*, vertices: np.ndarray, indices: np.ndarray):
    glb = build_triangle_glb(vertices=vertices, indices=indices)
    with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
        handle.write(glb)
        handle.flush()
        return load_cad_model(handle.name)


def default_keypoint_visibility(count: int = 17) -> np.ndarray:
    return np.full(count, YOLO_KEYPOINT_VISIBLE, dtype=np.int32)


def synthetic_occlusion_context(cad_model, landmarks):
    return build_cad_self_occlusion_context(cad_model, landmarks, YOLO_LANDMARK_NAMES)


def synthetic_triangle_model():
    vertices = np.array(
        [
            [-0.08, -0.08, 0.0],
            [0.08, -0.08, 0.0],
            [-0.08, 0.08, 0.0],
            [0.08, 0.08, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint16)
    return load_triangle_cad_model(vertices=vertices, indices=indices)



class TrainingDataContractTests(unittest.TestCase):
    def test_landmark_order_is_fixed(self) -> None:
        self.assertEqual(YOLO_LANDMARK_NAMES[0], "back-center")
        self.assertEqual(len(YOLO_LANDMARK_NAMES), 17)

    def test_sample_rate_must_be_finite_and_positive(self) -> None:
        for sample_rate_hz in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(sample_rate_hz=sample_rate_hz):
                with self.assertRaisesRegex(RuntimeError, "--sample-rate-hz"):
                    require_positive_sample_rate(sample_rate_hz)

    def test_missing_required_landmark_names_fail_at_startup(self) -> None:
        landmarks = synthetic_yolo_landmarks()
        del landmarks["top-right-center"]
        with self.assertRaisesRegex(ValueError, "missing required CAD landmarks"):
            require_yolo_landmarks(landmarks)

    def test_extra_cad_landmark_names_are_ignored(self) -> None:
        landmarks = synthetic_yolo_landmarks()
        landmarks["extra-node"] = np.array([1.0, 2.0, 3.0])
        require_yolo_landmarks(landmarks)

    def test_jpeg_encode_params_use_quality_95(self) -> None:
        self.assertEqual(jpeg_encode_params(), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])


class YoloLabelFormattingTests(unittest.TestCase):
    def test_format_emits_exactly_56_fields_with_constant_visibility(self) -> None:
        label = FrameLabel(
            bbox_xyxy=(100.0, 50.0, 300.0, 250.0),
            keypoints_xy=np.array([[160.0, 120.0], [180.0, 140.0]] + [[170.0, 130.0]] * 15, dtype=np.float64),
            keypoint_visibility=default_keypoint_visibility(),
        )
        row = format_yolo_pose_label(label, image_width=640, image_height=480)
        fields = row.split()
        self.assertEqual(len(fields), YOLO_FIELD_COUNT)
        self.assertEqual(fields[0], "0")
        visibilities = [fields[index] for index in range(7, len(fields), 3)]
        self.assertEqual(visibilities, [str(YOLO_KEYPOINT_VISIBILITY)] * 17)

    def test_keypoint_order_matches_landmark_names(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        camera_matrix, dist_coeffs = synthetic_camera()
        registration = load_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(origin=np.array([0.0, 0.0, 0.8]), rotation=np.eye(3))
        landmarks = synthetic_yolo_landmarks()
        cad_model = synthetic_triangle_model()
        occlusion_context = synthetic_occlusion_context(cad_model, landmarks)
        sample = build_training_sample(
            pose=pose,
            cad_landmarks=landmarks,
            cad_model=cad_model,
            registration=registration,
            marker_model=marker_model,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_width=640,
            image_height=480,
            occlusion_context=occlusion_context,
        )
        assert isinstance(sample, FrameLabel)
        projected = project_yolo_landmarks_to_image(
            landmarks,
            pose,
            camera_matrix,
            dist_coeffs,
            marker_model,
            registration,
            image_width=640,
            image_height=480,
        )
        assert isinstance(projected, np.ndarray)
        np.testing.assert_allclose(sample.keypoints_xy, projected)


class ProjectionRejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        self.marker_model = load_marker_model(TEST_MARKER_MODEL)
        self.camera_matrix, self.dist_coeffs = synthetic_camera()
        self.registration = load_registration_from_payload(identity_registration_payload())
        self.pose = ObjectPose(origin=np.array([0.0, 0.0, 0.8]), rotation=np.eye(3))
        self.cad_model = synthetic_triangle_model()
        self.landmarks = synthetic_yolo_landmarks()
        self.occlusion_context = synthetic_occlusion_context(self.cad_model, self.landmarks)

    def test_rejects_landmark_outside_image(self) -> None:
        landmarks = synthetic_yolo_landmarks(center=(2.0, 2.0, 0.0))
        result = build_training_sample(
            pose=self.pose,
            cad_landmarks=landmarks,
            cad_model=self.cad_model,
            registration=self.registration,
            marker_model=self.marker_model,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            image_width=640,
            image_height=480,
            occlusion_context=self.occlusion_context,
        )
        self.assertIs(result, SampleRejection.LANDMARKS)

    def test_rejects_non_finite_keypoints(self) -> None:
        landmarks = synthetic_yolo_landmarks()
        landmarks["front-center"] = np.array([float("nan"), 0.0, 0.0])
        result = project_yolo_landmarks_to_image(
            landmarks,
            self.pose,
            self.camera_matrix,
            self.dist_coeffs,
            self.marker_model,
            self.registration,
            image_width=640,
            image_height=480,
        )
        self.assertIs(result, SampleRejection.LANDMARKS)

    def test_rejects_bbox_when_mesh_silhouette_is_not_visible(self) -> None:
        behind_pose = ObjectPose(origin=np.array([0.0, 0.0, -0.5]), rotation=np.eye(3))
        bounds = project_cad_mesh_silhouette_bounds(
            behind_pose,
            self.camera_matrix,
            self.dist_coeffs,
            self.marker_model,
            self.cad_model,
            self.registration,
            image_width=640,
            image_height=480,
        )
        self.assertIsNone(bounds)

        landmarks = synthetic_yolo_landmarks()
        with mock.patch(
            "object_apriltag.training_data.project_cad_mesh_silhouette_bounds",
            return_value=None,
        ):
            result = build_training_sample(
                pose=self.pose,
                cad_landmarks=landmarks,
                cad_model=self.cad_model,
                registration=self.registration,
                marker_model=self.marker_model,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                image_width=640,
                image_height=480,
                occlusion_context=self.occlusion_context,
            )
        self.assertIs(result, SampleRejection.BBOX)


class DatasetOutputTests(unittest.TestCase):
    def test_writes_split_paths_and_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ensure_dataset_layout(output_dir, split="train", run_name="run_a")
            frame = np.full((120, 160, 3), 127, dtype=np.uint8)
            label = "0 " + " ".join(["0.5"] * 55)
            image_rel, label_rel = write_training_sample(
                output_dir,
                split="train",
                run_name="run_a",
                sample_index=0,
                frame=frame,
                label_line=label,
            )
            self.assertEqual(image_rel, "images/train/run_a_0.jpg")
            self.assertEqual(label_rel, "labels/train/run_a_0.txt")
            self.assertTrue((output_dir / "data.yaml").exists())
            self.assertEqual((output_dir / "data.yaml").read_text(encoding="utf-8"), DATA_YAML_TEXT)
            encoded = cv2.imread(str(output_dir / "images" / "train" / "run_a_0.jpg"))
            self.assertEqual(encoded.shape, frame.shape)

    def test_refuses_mismatched_existing_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "data.yaml").write_text("train: wrong\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "data.yaml"):
                ensure_dataset_layout(output_dir, split="train", run_name="run_a")

    def test_refuses_existing_run_report_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runs_dir = output_dir / "runs"
            runs_dir.mkdir(parents=True)
            (runs_dir / "run_b.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Dataset Generation Run already exists"):
                ensure_dataset_layout(output_dir, split="val", run_name="run_b")

    def test_refuses_sample_file_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ensure_dataset_layout(output_dir, split="val", run_name="run_b")
            frame = np.zeros((40, 40, 3), dtype=np.uint8)
            write_training_sample(
                output_dir,
                split="val",
                run_name="run_b",
                sample_index=0,
                frame=frame,
                label_line="0 " + " ".join(["0.1"] * 55),
            )
            with self.assertRaisesRegex(RuntimeError, "sample files already exist"):
                ensure_dataset_layout(output_dir, split="val", run_name="run_b")

    def test_pair_write_rolls_back_image_when_label_publish_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ensure_dataset_layout(output_dir, split="train", run_name="pair_run")
            frame = np.full((20, 20, 3), 42, dtype=np.uint8)
            real_replace = os.replace
            calls = {"count": 0}

            def flaky_replace(src, dst):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise OSError("injected label publish failure")
                real_replace(src, dst)

            with (
                mock.patch("object_apriltag.training_data.os.replace", side_effect=flaky_replace),
                self.assertRaises(OSError),
            ):
                write_training_sample(
                    output_dir,
                    split="train",
                    run_name="pair_run",
                    sample_index=0,
                    frame=frame,
                    label_line="0 " + " ".join(["0.2"] * 55),
                )

            image_path = output_dir / "images" / "train" / "pair_run_0.jpg"
            label_path = output_dir / "labels" / "train" / "pair_run_0.txt"
            self.assertFalse(image_path.exists())
            self.assertFalse(label_path.exists())

    def test_run_report_schema_validation(self) -> None:
        document = serialize_run_report(
            DatasetGenerationReport(
                run_name="run_c",
                split="train",
                sample_rate_hz=2.0,
                inputs={"source": "clip.mov"},
                frames_processed=10,
                samples_saved=1,
                rejections=RejectionCounts(no_pose=3, landmarks=2, bbox=1),
                samples=[
                    {
                        "index": "0",
                        "image": "images/train/run_c_0.jpg",
                        "label": "labels/train/run_c_0.txt",
                    }
                ],
            )
        )
        validate_run_report(document)
        failed = serialize_run_report(
            DatasetGenerationReport(
                run_name="run_failed",
                split="train",
                sample_rate_hz=2.0,
                inputs={"source": "0"},
                status="failed",
                error="camera read failed",
            )
        )
        validate_run_report(failed)
        with self.assertRaises(ValueError):
            validate_run_report({"schema_version": 99})


class LabeledImageTests(unittest.TestCase):
    def test_draw_yolo_pose_label_annotates_frame(self) -> None:
        frame = np.full((80, 80, 3), 127, dtype=np.uint8)
        label = FrameLabel(
            bbox_xyxy=(10.0, 10.0, 60.0, 60.0),
            keypoints_xy=np.array([[20.0, 20.0], [40.0, 40.0]] + [[30.0, 30.0]] * 15),
            keypoint_visibility=default_keypoint_visibility(),
        )
        annotated = draw_yolo_pose_label(frame, label)
        self.assertFalse(np.array_equal(annotated, frame))
        self.assertEqual(tuple(annotated[10, 10]), (0, 255, 0))

    def test_write_labeled_training_image_writes_under_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            ensure_dataset_layout(output_dir, split="val", run_name="preview_run")
            frame = np.full((40, 40, 3), 200, dtype=np.uint8)
            label = FrameLabel(
                bbox_xyxy=(5.0, 5.0, 30.0, 30.0),
                keypoints_xy=np.full((17, 2), 15.0, dtype=np.float64),
                keypoint_visibility=default_keypoint_visibility(),
            )
            rel_path = write_labeled_training_image(
                output_dir,
                split="val",
                run_name="preview_run",
                sample_index=0,
                frame=frame,
                label=label,
            )
            self.assertEqual(rel_path, "labeled-images/val/preview_run_0.jpg")
            self.assertTrue((output_dir / rel_path).exists())

    def test_labeled_images_limit_writes_only_first_n_saved_samples(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        camera_matrix, dist_coeffs = synthetic_camera()
        registration = load_registration_from_payload(identity_registration_payload())
        cad_model = synthetic_triangle_model()
        landmarks = synthetic_yolo_landmarks()
        pose = ObjectPose(origin=np.array([0.0, 0.0, 0.8]), rotation=np.eye(3))
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        fake_capture = mock.Mock()
        fake_capture.get.side_effect = [10.0, 0.0, 1000.0, 2000.0, 3000.0]
        read_state = {"index": 0}

        def fake_read(_capture, _source, *, loop_on_eof: bool = True):
            if read_state["index"] >= 3:
                return False, None
            read_state["index"] += 1
            return True, frame

        detector = mock.Mock()
        detector.marker_model = marker_model
        detector.find_markers.return_value = []
        detector.fuse.return_value = pose
        accepted_label = FrameLabel(
            bbox_xyxy=(10.0, 10.0, 100.0, 100.0),
            keypoints_xy=np.zeros((17, 2), dtype=np.float64),
            keypoint_visibility=default_keypoint_visibility(),
        )

        with (
            mock.patch(
                "object_apriltag.training_data.open_frame_source",
                return_value=fake_capture,
            ),
            mock.patch("object_apriltag.training_data.read_frame", side_effect=fake_read),
            mock.patch(
                "object_apriltag.training_data.build_training_sample",
                return_value=accepted_label,
            ),
            mock.patch(
                "object_apriltag.training_data.format_yolo_pose_label",
                return_value="0 " + " ".join(["0.5"] * 55),
            ),
            tempfile.TemporaryDirectory() as tmp,
        ):
            output_dir = Path(tmp)
            report = generate_dataset_from_source(
                source=Path("clip.mov"),
                output_dir=output_dir,
                split="train",
                run_name="labeled_limit_run",
                sample_rate_hz=1.0,
                detector=detector,
                cad_landmarks=landmarks,
                cad_model=cad_model,
                registration=registration,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=640,
                image_height=480,
                inputs={"source": "clip.mov"},
                show_preview=False,
                labeled_images_limit=2,
            )

            self.assertEqual(report.samples_saved, 3)
            labeled_dir = output_dir / "labeled-images" / "train"
            self.assertTrue((labeled_dir / "labeled_limit_run_0.jpg").exists())
            self.assertTrue((labeled_dir / "labeled_limit_run_1.jpg").exists())
            self.assertFalse((labeled_dir / "labeled_limit_run_2.jpg").exists())
            self.assertIn("labeled_image", report.samples[0])
            self.assertIn("labeled_image", report.samples[1])
            self.assertNotIn("labeled_image", report.samples[2])


class SamplingBehaviorTests(unittest.TestCase):
    def _generation_fixtures(self):
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        camera_matrix, dist_coeffs = synthetic_camera()
        registration = load_registration_from_payload(identity_registration_payload())
        cad_model = synthetic_triangle_model()
        landmarks = synthetic_yolo_landmarks()
        pose = ObjectPose(origin=np.array([0.0, 0.0, 0.8]), rotation=np.eye(3))
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        return (
            marker_model,
            camera_matrix,
            dist_coeffs,
            registration,
            cad_model,
            landmarks,
            pose,
            frame,
        )

    def test_video_saves_first_accepted_frame_after_interval_without_consuming_on_reject(self) -> None:
        (
            marker_model,
            camera_matrix,
            dist_coeffs,
            registration,
            cad_model,
            landmarks,
            pose,
            frame,
        ) = self._generation_fixtures()
        frames = [frame] * 6
        fake_capture = mock.Mock()
        fake_capture.get.side_effect = [10.0, 0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
        read_state = {"index": 0}

        def fake_read(_capture, _source, *, loop_on_eof: bool = True):
            self.assertFalse(loop_on_eof)
            index = read_state["index"]
            if index >= len(frames):
                return False, None
            read_state["index"] = index + 1
            return True, frames[index]

        detector = mock.Mock()
        detector.marker_model = marker_model
        detector.find_markers.return_value = []
        detector.fuse.side_effect = [None, None, pose, pose, pose, pose]

        with (
            mock.patch(
                "object_apriltag.training_data.open_frame_source",
                return_value=fake_capture,
            ),
            mock.patch("object_apriltag.training_data.read_frame", side_effect=fake_read),
            mock.patch(
                "object_apriltag.training_data.build_training_sample",
                side_effect=[
                    SampleRejection.LANDMARKS,
                    SampleRejection.LANDMARKS,
                    FrameLabel(
                        bbox_xyxy=(10.0, 10.0, 100.0, 100.0),
                        keypoints_xy=np.zeros((17, 2), dtype=np.float64),
                        keypoint_visibility=default_keypoint_visibility(),
                    ),
                ],
            ) as build_mock,
            mock.patch(
                "object_apriltag.training_data.format_yolo_pose_label",
                return_value="0 " + " ".join(["0.5"] * 55),
            ),
            tempfile.TemporaryDirectory() as tmp,
        ):
            output_dir = Path(tmp)
            report = generate_dataset_from_source(
                source=Path("clip.mov"),
                output_dir=output_dir,
                split="train",
                run_name="video_run",
                sample_rate_hz=2.0,
                detector=detector,
                cad_landmarks=landmarks,
                cad_model=cad_model,
                registration=registration,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=640,
                image_height=480,
                inputs={"source": "clip.mov"},
                show_preview=False,
            )

            self.assertEqual(report.status, "completed")
            self.assertEqual(report.frames_processed, 6)
            self.assertEqual(report.samples_saved, 1)
            self.assertEqual(report.rejections.no_pose, 2)
            self.assertEqual(report.rejections.landmarks, 2)
            self.assertEqual(build_mock.call_count, 3)
            self.assertTrue((output_dir / "runs" / "video_run.json").exists())

    def test_video_writes_accepted_sample_without_mocking_build_training_sample(self) -> None:
        (
            marker_model,
            camera_matrix,
            dist_coeffs,
            registration,
            cad_model,
            landmarks,
            pose,
            frame,
        ) = self._generation_fixtures()
        fake_capture = mock.Mock()
        fake_capture.get.side_effect = [10.0, 0.0, 100.0]
        read_state = {"index": 0}

        def fake_read(_capture, _source, *, loop_on_eof: bool = True):
            if read_state["index"] >= 1:
                return False, None
            read_state["index"] += 1
            return True, frame

        detector = mock.Mock()
        detector.marker_model = marker_model
        detector.find_markers.return_value = []
        detector.fuse.return_value = pose

        with (
            mock.patch(
                "object_apriltag.training_data.open_frame_source",
                return_value=fake_capture,
            ),
            mock.patch("object_apriltag.training_data.read_frame", side_effect=fake_read),
            tempfile.TemporaryDirectory() as tmp,
        ):
            output_dir = Path(tmp)
            report = generate_dataset_from_source(
                source=Path("clip.mov"),
                output_dir=output_dir,
                split="train",
                run_name="real_sample_run",
                sample_rate_hz=10.0,
                detector=detector,
                cad_landmarks=landmarks,
                cad_model=cad_model,
                registration=registration,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=640,
                image_height=480,
                inputs={"source": "clip.mov"},
                show_preview=False,
            )

            self.assertEqual(report.status, "completed")
            self.assertEqual(report.samples_saved, 1)
            label_path = output_dir / "labels" / "train" / "real_sample_run_0.txt"
            self.assertTrue(label_path.exists())
            self.assertEqual(len(label_path.read_text(encoding="utf-8").split()), YOLO_FIELD_COUNT)

    def test_camera_preview_stops_on_q(self) -> None:
        (
            marker_model,
            camera_matrix,
            dist_coeffs,
            registration,
            cad_model,
            landmarks,
            _pose,
            frame,
        ) = self._generation_fixtures()
        fake_capture = mock.Mock()

        detector = mock.Mock()
        detector.marker_model = marker_model
        detector.find_markers.return_value = []
        detector.fuse.return_value = None

        with (
            mock.patch(
                "object_apriltag.training_data.open_frame_source",
                return_value=fake_capture,
            ),
            mock.patch("object_apriltag.training_data.read_frame", return_value=(True, frame)),
            mock.patch("object_apriltag.training_data.cv2.imshow") as imshow,
            mock.patch("object_apriltag.training_data.cv2.waitKey", return_value=ord("q")),
            mock.patch("object_apriltag.training_data.cv2.destroyAllWindows") as destroy,
            tempfile.TemporaryDirectory() as tmp,
        ):
            report = generate_dataset_from_source(
                source=0,
                output_dir=Path(tmp),
                split="val",
                run_name="camera_run",
                sample_rate_hz=5.0,
                detector=detector,
                cad_landmarks=landmarks,
                cad_model=cad_model,
                registration=registration,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                image_width=640,
                image_height=480,
                inputs={"source": "0"},
                show_preview=True,
            )

        self.assertEqual(report.status, "completed")
        imshow.assert_called()
        destroy.assert_called_once()

    def test_camera_read_failure_writes_failed_report(self) -> None:
        (
            marker_model,
            camera_matrix,
            dist_coeffs,
            registration,
            cad_model,
            landmarks,
            _pose,
            frame,
        ) = self._generation_fixtures()
        fake_capture = mock.Mock()

        detector = mock.Mock()
        detector.marker_model = marker_model

        with (
            mock.patch(
                "object_apriltag.training_data.open_frame_source",
                return_value=fake_capture,
            ),
            mock.patch("object_apriltag.training_data.read_frame", return_value=(False, None)),
            tempfile.TemporaryDirectory() as tmp,
        ):
            output_dir = Path(tmp)
            with self.assertRaises(RuntimeFrameReadError):
                generate_dataset_from_source(
                    source=0,
                    output_dir=output_dir,
                    split="train",
                    run_name="failed_camera_run",
                    sample_rate_hz=2.0,
                    detector=detector,
                    cad_landmarks=landmarks,
                    cad_model=cad_model,
                    registration=registration,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    image_width=640,
                    image_height=480,
                    inputs={"source": "0"},
                    show_preview=False,
                )

            report_path = output_dir / "runs" / "failed_camera_run.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertIn("camera", report["error"])


class RequiredLandmarkLoaderTests(unittest.TestCase):
    def test_load_required_yolo_landmarks_reads_glb_nodes(self) -> None:
        nodes = []
        for index, name in enumerate(YOLO_LANDMARK_NAMES):
            nodes.append({"name": name, "translation": [0.01 * index, 0.0, 0.0]})
        gltf = {
            "asset": {"version": "2.0"},
            "nodes": nodes,
            "scenes": [{"nodes": list(range(len(nodes)))}],
        }
        json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        json_padding = (4 - (len(json_chunk) % 4)) % 4
        json_chunk += b" " * json_padding
        header = b"glTF" + (2).to_bytes(4, "little") + (12 + 8 + len(json_chunk) + 8).to_bytes(4, "little")
        json_header = len(json_chunk).to_bytes(4, "little") + b"JSON"
        bin_header = (0).to_bytes(4, "little") + b"BIN\x00"
        glb = header + json_header + json_chunk + bin_header
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            loaded = load_required_yolo_landmarks(handle.name)
        self.assertEqual(set(loaded.landmarks), set(YOLO_LANDMARK_NAMES))


if __name__ == "__main__":
    unittest.main()
