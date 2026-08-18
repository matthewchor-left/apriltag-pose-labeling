"""Tests for CAD mesh self-occlusion visibility."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_apriltag.cad import CadMeshPart, CadModel, load_cad_model
from object_apriltag.cad_self_occlusion import (
    LANDMARK_VERTEX_MATCH_TOLERANCE_M,
    RAY_SEGMENT_TARGET_VISIBILITY_TOLERANCE_M,
    YOLO_KEYPOINT_OCCLUDED,
    YOLO_KEYPOINT_VISIBLE,
    build_cad_self_occlusion_context,
    camera_origin_in_cad,
    classify_cad_keypoint_visibility,
)
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import (
    load_marker_model,
    object_reference_origin,
    object_reference_orientation,
)
from object_apriltag.training_data import (
    YOLO_FIELD_COUNT,
    YOLO_LANDMARK_NAMES,
    FrameLabel,
    build_training_sample,
    draw_yolo_pose_label,
    format_yolo_pose_label,
)
from tests.test_cad_overlay import identity_registration_payload, synthetic_camera
from tests.test_training_data import load_registration_from_payload

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MARKER_MODEL = REPO_ROOT / "config/Model/remote/marker_model.json"


def _pack_glb(gltf: dict[str, object], bin_chunk: bytes = b"") -> bytes:
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_padding = (4 - (len(json_chunk) % 4)) % 4
    json_chunk += b" " * json_padding
    bin_padding = (4 - (len(bin_chunk) % 4)) % 4
    bin_chunk_padded = bin_chunk + b"\x00" * bin_padding
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk_padded)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<I4s", len(json_chunk), b"JSON")
    bin_header = struct.pack("<I4s", len(bin_chunk_padded), b"BIN\x00")
    return header + json_header + json_chunk + bin_header + bin_chunk_padded


def build_mesh_with_landmarks_glb(
    *,
    mesh_vertices: np.ndarray,
    mesh_indices: np.ndarray,
    landmarks: dict[str, np.ndarray],
    part_name: str = "part",
) -> bytes:
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float32).reshape(-1, 3)
    mesh_indices = np.asarray(mesh_indices, dtype=np.uint16).reshape(-1)
    vertex_bytes = mesh_vertices.tobytes()
    index_bytes = mesh_indices.tobytes()
    bin_chunk = vertex_bytes + index_bytes

    nodes: list[dict[str, object]] = [
        {"mesh": 0, "name": part_name},
    ]
    scene_nodes = [0]
    for index, (name, position) in enumerate(landmarks.items(), start=1):
        nodes.append(
            {
                "name": name,
                "translation": np.asarray(position, dtype=np.float64).reshape(3).tolist(),
            }
        )
        scene_nodes.append(index)

    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_chunk)}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(vertex_bytes),
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": len(vertex_bytes),
                "byteLength": len(index_bytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(mesh_vertices),
                "type": "VEC3",
                "max": mesh_vertices.max(axis=0).tolist(),
                "min": mesh_vertices.min(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(mesh_indices),
                "type": "SCALAR",
            },
        ],
        "meshes": [
            {
                "name": part_name,
                "primitives": [{"attributes": {"POSITION": 0}, "indices": 1}],
            }
        ],
        "nodes": nodes,
        "scenes": [{"nodes": scene_nodes}],
    }
    return _pack_glb(gltf, bin_chunk)


def load_model_from_glb_bytes(glb: bytes):
    with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
        handle.write(glb)
        handle.flush()
        return load_cad_model(handle.name)


def landmarks_on_vertices(vertex_positions: list[tuple[float, float, float]]) -> dict[str, np.ndarray]:
    landmarks: dict[str, np.ndarray] = {}
    for index, name in enumerate(YOLO_LANDMARK_NAMES):
        position = vertex_positions[index % len(vertex_positions)]
        landmarks[name] = np.array(position, dtype=np.float64)
    return landmarks


def golden_camera_origin_cad(
    pose: ObjectPose,
    marker_model,
    registration_transform_4x4: np.ndarray,
) -> np.ndarray:
    """Oracle camera origin in CAD from explicit layout/pose math (no production helpers)."""
    ref_origin = object_reference_origin(marker_model)
    ref_orientation = object_reference_orientation(marker_model)
    point_object = -(pose.rotation.T @ pose.origin)
    point_layout = ref_origin + ref_orientation @ point_object
    cad_homogeneous = np.linalg.inv(registration_transform_4x4) @ np.append(point_layout, 1.0)
    return cad_homogeneous[:3]


def blocking_two_part_cad_model() -> tuple[CadModel, dict[str, np.ndarray]]:
    """Part A hosts a back landmark; part B is a non-coincident blocker wall."""
    back_vertices = np.array(
        [
            [-0.05, -0.05, -0.5],
            [0.05, -0.05, -0.5],
            [-0.05, 0.05, -0.5],
        ],
        dtype=np.float64,
    )
    blocker_vertices = np.array(
        [
            [-0.2, -0.2, 0.5],
            [0.2, -0.2, 0.5],
            [-0.2, 0.2, 0.5],
            [0.2, 0.2, 0.5],
        ],
        dtype=np.float64,
    )
    cad_model = CadModel(
        parts=(
            CadMeshPart(
                name="part_a",
                vertices=back_vertices,
                triangles=np.array([[0, 1, 2]], dtype=np.int64),
            ),
            CadMeshPart(
                name="part_b",
                vertices=blocker_vertices,
                triangles=np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
            ),
        )
    )
    landmarks = {
        "blocked": back_vertices[0].copy(),
        "clear": blocker_vertices[1].copy(),
    }
    return cad_model, landmarks


def pose_with_layout_camera_at_z(marker_model, layout_z: float) -> ObjectPose:
    """Build a fused pose whose camera origin maps to ``(0, 0, layout_z)`` in layout/CAD."""
    ref_origin = object_reference_origin(marker_model)
    ref_orientation = object_reference_orientation(marker_model)
    desired_layout = np.array([0.0, 0.0, layout_z], dtype=np.float64)
    point_object = ref_orientation.T @ (desired_layout - ref_origin)
    return ObjectPose(origin=-point_object, rotation=np.eye(3))


def yolo_landmarks_for_blocking_scene() -> dict[str, np.ndarray]:
    """Place YOLO names on back (occluded) and front-blocker (visible) vertices."""
    back_vertices = [
        (-0.05, -0.05, -0.5),
        (0.05, -0.05, -0.5),
        (-0.05, 0.05, -0.5),
    ]
    front_vertices = [
        (-0.2, -0.2, 0.5),
        (0.2, -0.2, 0.5),
        (-0.2, 0.2, 0.5),
        (0.2, 0.2, 0.5),
    ]
    back_names = ("back-center", "back-left-center", "back-right-center")
    front_names = (
        "front-center",
        "front-left-center",
        "front-right-center",
        "top-front-center",
        "top-front-left",
        "top-front-right",
    )
    landmarks: dict[str, np.ndarray] = {}
    for index, name in enumerate(back_names):
        landmarks[name] = np.array(back_vertices[index], dtype=np.float64)
    for index, name in enumerate(front_names):
        landmarks[name] = np.array(front_vertices[index % len(front_vertices)], dtype=np.float64)
    remaining = [name for name in YOLO_LANDMARK_NAMES if name not in landmarks]
    for index, name in enumerate(remaining):
        landmarks[name] = np.array(front_vertices[index % len(front_vertices)], dtype=np.float64)
    return landmarks


def near_target_blocker_cad_model(
    distance_before_target_m: float,
) -> tuple[CadModel, dict[str, np.ndarray]]:
    """Camera-to-landmark segment along z: origin (0,0,1) -> target (0,0,0)."""
    landmark = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    incident_vertices = np.array(
        [
            landmark,
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    blocker_z = distance_before_target_m
    blocker_vertices = np.array(
        [
            [-0.5, -0.5, blocker_z],
            [0.5, -0.5, blocker_z],
            [-0.5, 0.5, blocker_z],
        ],
        dtype=np.float64,
    )
    cad_model = CadModel(
        parts=(
            CadMeshPart(
                name="incident",
                vertices=incident_vertices,
                triangles=np.array([[0, 1, 2]], dtype=np.int64),
            ),
            CadMeshPart(
                name="blocker",
                vertices=blocker_vertices,
                triangles=np.array([[0, 1, 2]], dtype=np.int64),
            ),
        )
    )
    return cad_model, {"target": landmark.copy()}


class TargetVisibilityToleranceTests(unittest.TestCase):
    def test_blocker_within_tolerance_is_ignored(self) -> None:
        camera_origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for distance_before_target_m in (
            RAY_SEGMENT_TARGET_VISIBILITY_TOLERANCE_M - 0.001,
            RAY_SEGMENT_TARGET_VISIBILITY_TOLERANCE_M,
        ):
            cad_model, landmarks = near_target_blocker_cad_model(distance_before_target_m)
            context = build_cad_self_occlusion_context(cad_model, landmarks, ("target",))
            visibility = classify_cad_keypoint_visibility(
                context,
                np.array([landmarks["target"]]),
                camera_origin,
            )
            self.assertEqual(
                int(visibility[0]),
                YOLO_KEYPOINT_VISIBLE,
                f"expected visible for blocker {distance_before_target_m:g} m before target",
            )

    def test_blocker_beyond_tolerance_occludes(self) -> None:
        distance_before_target_m = RAY_SEGMENT_TARGET_VISIBILITY_TOLERANCE_M + 0.001
        cad_model, landmarks = near_target_blocker_cad_model(distance_before_target_m)
        context = build_cad_self_occlusion_context(cad_model, landmarks, ("target",))
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["target"]]),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_OCCLUDED)


class LandmarkAssociationTests(unittest.TestCase):
    def test_associates_each_required_landmark_with_nearest_vertex(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.uint16)
        landmarks = {
            "only": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "near": np.array([0.2, 0.0, 1e-4], dtype=np.float64),
        }
        glb = build_mesh_with_landmarks_glb(
            mesh_vertices=vertices,
            mesh_indices=indices,
            landmarks=landmarks,
        )
        model = load_model_from_glb_bytes(glb)
        context = build_cad_self_occlusion_context(model, landmarks, ("only", "near"))
        self.assertEqual(int(context.landmark_vertex_indices[0]), 0)
        self.assertEqual(int(context.landmark_vertex_indices[1]), 1)

    def test_mismatch_beyond_tolerance_fails_at_startup(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint16)
        landmarks = {
            "far": np.array([0.0, 0.0, LANDMARK_VERTEX_MATCH_TOLERANCE_M * 2.0], dtype=np.float64),
        }
        model = load_model_from_glb_bytes(
            build_mesh_with_landmarks_glb(
                mesh_vertices=vertices,
                mesh_indices=indices,
                landmarks=landmarks,
            )
        )
        with self.assertRaisesRegex(ValueError, "no mesh vertex within"):
            build_cad_self_occlusion_context(model, landmarks, ("far",))


class VisibilityClassificationTests(unittest.TestCase):
    def test_unobstructed_vertex_is_visible(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.uint16)
        landmarks = {"target": vertices[0].astype(np.float64)}
        model = load_model_from_glb_bytes(
            build_mesh_with_landmarks_glb(
                mesh_vertices=vertices,
                mesh_indices=indices,
                landmarks=landmarks,
            )
        )
        context = build_cad_self_occlusion_context(model, landmarks, ("target",))
        camera_origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["target"]]),
            camera_origin,
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_VISIBLE)

    def test_backside_vertex_blocked_by_intervening_triangle(self) -> None:
        front = np.array(
            [
                [-0.1, -0.1, 0.5],
                [0.1, -0.1, 0.5],
                [-0.1, 0.1, 0.5],
                [0.1, 0.1, 0.5],
            ],
            dtype=np.float32,
        )
        back = np.array(
            [
                [-0.1, -0.1, -0.5],
                [0.1, -0.1, -0.5],
                [-0.1, 0.1, -0.5],
                [0.1, 0.1, -0.5],
            ],
            dtype=np.float32,
        )
        vertices = np.vstack([front, back])
        indices = np.array(
            [0, 1, 2, 1, 3, 2, 4, 6, 5, 5, 6, 7],
            dtype=np.uint16,
        )
        landmarks = {"back-center": back[0].astype(np.float64)}
        model = load_model_from_glb_bytes(
            build_mesh_with_landmarks_glb(
                mesh_vertices=vertices,
                mesh_indices=indices,
                landmarks=landmarks,
            )
        )
        context = build_cad_self_occlusion_context(model, landmarks, ("back-center",))
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["back-center"]]),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_OCCLUDED)

    def test_incident_triangle_does_not_false_occlude(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.uint16)
        landmarks = {"corner": vertices[0].astype(np.float64)}
        model = load_model_from_glb_bytes(
            build_mesh_with_landmarks_glb(
                mesh_vertices=vertices,
                mesh_indices=indices,
                landmarks=landmarks,
            )
        )
        context = build_cad_self_occlusion_context(model, landmarks, ("corner",))
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["corner"]]),
            np.array([0.05, 0.05, 0.5], dtype=np.float64),
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_VISIBLE)

    def test_part_b_triangle_blocks_part_a_landmark_while_other_stays_visible(self) -> None:
        cad_model, landmarks = blocking_two_part_cad_model()
        context = build_cad_self_occlusion_context(
            cad_model,
            landmarks,
            ("blocked", "clear"),
        )
        camera_origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["blocked"], landmarks["clear"]]),
            camera_origin,
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_OCCLUDED)
        self.assertEqual(int(visibility[1]), YOLO_KEYPOINT_VISIBLE)

    def test_multiple_parts_and_duplicate_geometry_are_handled(self) -> None:
        from object_apriltag.cad import CadMeshPart, CadModel

        triangle = (
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float64),
            np.array([[0, 1, 2]], dtype=np.int64),
        )
        combined_model = CadModel(
            parts=(
                CadMeshPart(name="part_a", vertices=triangle[0], triangles=triangle[1]),
                CadMeshPart(name="part_b", vertices=triangle[0].copy(), triangles=triangle[1].copy()),
            )
        )
        landmarks = {"shared": np.zeros(3, dtype=np.float64)}
        context = build_cad_self_occlusion_context(combined_model, landmarks, ("shared",))
        self.assertEqual(len(context.vertices), 6)
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["shared"]]),
            np.array([0.02, 0.02, 0.5], dtype=np.float64),
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_VISIBLE)


class CameraOriginTransformTests(unittest.TestCase):
    def test_camera_origin_matches_independent_transform_chain(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        registration = load_registration_from_payload(
            {
                "units": "meters",
                "source_frame": "cad",
                "target_frame": "marker_model",
                "transform_4x4": [
                    [0.0, -1.0, 0.0, 0.05],
                    [1.0, 0.0, 0.0, -0.02],
                    [0.0, 0.0, 1.0, 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
        pose = ObjectPose(
            origin=np.array([0.1, -0.2, 0.9], dtype=np.float64),
            rotation=np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        expected = golden_camera_origin_cad(pose, marker_model, registration.transform_4x4)
        computed = camera_origin_in_cad(pose, marker_model, registration)
        np.testing.assert_allclose(computed, expected, rtol=0.0, atol=1e-12)

    def test_camera_origin_from_pose_yields_expected_occlusion(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        registration = load_registration_from_payload(identity_registration_payload())
        pose = pose_with_layout_camera_at_z(marker_model, layout_z=2.0)
        cad_model, landmarks = blocking_two_part_cad_model()
        context = build_cad_self_occlusion_context(
            cad_model,
            landmarks,
            ("blocked", "clear"),
        )
        origin_cad = camera_origin_in_cad(pose, marker_model, registration)
        np.testing.assert_allclose(origin_cad, np.array([0.0, 0.0, 2.0]), atol=1e-12)
        visibility = classify_cad_keypoint_visibility(
            context,
            np.array([landmarks["blocked"], landmarks["clear"]]),
            origin_cad,
        )
        self.assertEqual(int(visibility[0]), YOLO_KEYPOINT_OCCLUDED)
        self.assertEqual(int(visibility[1]), YOLO_KEYPOINT_VISIBLE)

    def test_forward_registration_origin_diverges_from_inverse_golden(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        registration = load_registration_from_payload(
            {
                "units": "meters",
                "source_frame": "cad",
                "target_frame": "marker_model",
                "transform_4x4": [
                    [0.0, -1.0, 0.0, 0.05],
                    [1.0, 0.0, 0.0, -0.02],
                    [0.0, 0.0, 1.0, 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
        pose = ObjectPose(
            origin=np.array([0.1, -0.2, 0.9], dtype=np.float64),
            rotation=np.array(
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        golden = golden_camera_origin_cad(pose, marker_model, registration.transform_4x4)
        ref_origin = object_reference_origin(marker_model)
        ref_orientation = object_reference_orientation(marker_model)
        point_object = -(pose.rotation.T @ pose.origin)
        point_layout = ref_origin + ref_orientation @ point_object
        forward_origin = (registration.transform_4x4 @ np.append(point_layout, 1.0))[:3]
        self.assertFalse(np.allclose(forward_origin, golden, atol=1e-9))

    def test_blocked_landmark_visibility_flips_with_camera_origin_side(self) -> None:
        cad_model, landmarks = blocking_two_part_cad_model()
        context = build_cad_self_occlusion_context(
            cad_model,
            landmarks,
            ("blocked", "clear"),
        )
        points = np.array([landmarks["blocked"], landmarks["clear"]])
        from_positive_z = classify_cad_keypoint_visibility(
            context,
            points,
            np.array([0.0, 0.0, 2.0], dtype=np.float64),
        )
        from_negative_z = classify_cad_keypoint_visibility(
            context,
            points,
            np.array([0.0, 0.0, -2.0], dtype=np.float64),
        )
        self.assertEqual(int(from_positive_z[0]), YOLO_KEYPOINT_OCCLUDED)
        self.assertEqual(int(from_positive_z[1]), YOLO_KEYPOINT_VISIBLE)
        self.assertEqual(int(from_negative_z[0]), YOLO_KEYPOINT_VISIBLE)
        self.assertNotEqual(from_positive_z.tolist(), from_negative_z.tolist())


class TrainingDataIntegrationTests(unittest.TestCase):
    def test_mixed_visibility_serializes_56_fields(self) -> None:
        label = FrameLabel(
            bbox_xyxy=(100.0, 50.0, 300.0, 250.0),
            keypoints_xy=np.full((17, 2), 170.0, dtype=np.float64),
            keypoint_visibility=np.array(
                [YOLO_KEYPOINT_VISIBLE, YOLO_KEYPOINT_OCCLUDED] + [YOLO_KEYPOINT_VISIBLE] * 15,
                dtype=np.int32,
            ),
        )
        row = format_yolo_pose_label(label, image_width=640, image_height=480)
        fields = row.split()
        self.assertEqual(len(fields), YOLO_FIELD_COUNT)
        visibilities = [fields[index] for index in range(7, len(fields), 3)]
        self.assertEqual(visibilities[0], "2")
        self.assertEqual(visibilities[1], "1")
        self.assertEqual(visibilities[2:], ["2"] * 15)

    def test_preview_uses_distinct_colors_for_visible_and_occluded(self) -> None:
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        label = FrameLabel(
            bbox_xyxy=(10.0, 10.0, 100.0, 100.0),
            keypoints_xy=np.array([[20.0, 20.0], [40.0, 40.0]] + [[30.0, 30.0]] * 15),
            keypoint_visibility=np.array(
                [YOLO_KEYPOINT_VISIBLE, YOLO_KEYPOINT_OCCLUDED] + [YOLO_KEYPOINT_VISIBLE] * 15,
                dtype=np.int32,
            ),
        )
        annotated = draw_yolo_pose_label(frame, label)
        self.assertEqual(tuple(annotated[20, 20]), (0, 165, 255))
        self.assertEqual(tuple(annotated[40, 40]), (0, 0, 255))

    def test_build_training_sample_assigns_expected_visibility_values(self) -> None:
        if not TEST_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(TEST_MARKER_MODEL)
        camera_matrix, dist_coeffs = synthetic_camera()
        registration = load_registration_from_payload(identity_registration_payload())
        pose = pose_with_layout_camera_at_z(marker_model, layout_z=2.0)
        cad_model, _ = blocking_two_part_cad_model()
        landmarks = yolo_landmarks_for_blocking_scene()
        occlusion_context = build_cad_self_occlusion_context(
            cad_model,
            landmarks,
            YOLO_LANDMARK_NAMES,
        )
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
        self.assertIsInstance(sample, FrameLabel)

        visibility_by_name = {
            name: int(sample.keypoint_visibility[index])
            for index, name in enumerate(YOLO_LANDMARK_NAMES)
        }
        self.assertEqual(visibility_by_name["back-center"], YOLO_KEYPOINT_OCCLUDED)
        self.assertEqual(visibility_by_name["back-left-center"], YOLO_KEYPOINT_OCCLUDED)
        self.assertEqual(visibility_by_name["front-center"], YOLO_KEYPOINT_VISIBLE)
        self.assertEqual(visibility_by_name["top-front-center"], YOLO_KEYPOINT_VISIBLE)

        row = format_yolo_pose_label(sample, image_width=640, image_height=480)
        fields = row.split()
        visibilities = [int(fields[index]) for index in range(7, len(fields), 3)]
        self.assertEqual(visibilities[0], YOLO_KEYPOINT_OCCLUDED)
        self.assertEqual(visibilities[3], YOLO_KEYPOINT_VISIBLE)
        self.assertIn(YOLO_KEYPOINT_OCCLUDED, visibilities)
        self.assertIn(YOLO_KEYPOINT_VISIBLE, visibilities)


if __name__ == "__main__":
    unittest.main()
