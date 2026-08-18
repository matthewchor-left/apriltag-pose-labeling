"""Tests for CAD GLB loading and silhouette overlays."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag.cad import (
    CadLandmarks,
    load_cad_model,
    load_cad_registration,
)
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import load_marker_model
from object_apriltag.viz.cad_overlay import (
    _CAD_ONLY_LANDMARK_COLOR_BGR,
    _collect_visible_parts,
    cad_only_landmark_names,
    cad_points_to_layout,
    draw_cad_model_overlay,
    draw_cad_only_landmarks,
    layout_points_to_camera,
    object_model_landmark_names,
    part_color_bgr,
    project_cad_landmarks_to_image,
    project_camera_points,
    render_cad_model_view,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NEXPLAYGROUND_GLB = REPO_ROOT / "config/Model/CAD/nexplayground_sim.glb"
REMOTE_STATIC_MARKER_MODEL = (
    REPO_ROOT / "config/Model/remote_static_1_tag/marker_model.json"
)
REMOTE_MARKER_MODEL = REPO_ROOT / "config/Model/remote/marker_model.json"


def synthetic_camera() -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def identity_registration_payload() -> dict[str, object]:
    return {
        "units": "meters",
        "source_frame": "cad",
        "target_frame": "marker_model",
        "transform_4x4": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


def build_triangle_glb(
    *,
    vertices: np.ndarray,
    indices: np.ndarray,
    part_name: str = "triangle_part",
    materials: list[dict[str, object]] | None = None,
    primitive_material: int | None = None,
) -> bytes:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    indices = np.asarray(indices, dtype=np.uint16).reshape(-1)
    vertex_bytes = vertices.tobytes()
    index_bytes = indices.tobytes()
    bin_chunk = vertex_bytes + index_bytes

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
                "count": len(vertices),
                "type": "VEC3",
                "max": vertices.max(axis=0).tolist(),
                "min": vertices.min(axis=0).tolist(),
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": len(indices),
                "type": "SCALAR",
            },
        ],
        "meshes": [
            {
                "name": part_name,
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        **(
                            {"material": primitive_material}
                            if primitive_material is not None
                            else {}
                        ),
                    }
                ],
            }
        ],
        "nodes": [{"mesh": 0, "name": part_name}],
        "scenes": [{"nodes": [0]}],
    }
    if materials is not None:
        gltf["materials"] = materials

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


class CadGlbLoaderTests(unittest.TestCase):
    def test_loads_synthetic_triangle_glb(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.5],
                [0.0, 0.1, 0.5],
            ],
            dtype=np.float32,
        )
        glb = build_triangle_glb(vertices=vertices, indices=np.array([0, 1, 2], dtype=np.uint16))
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            model = load_cad_model(handle.name)

        self.assertEqual(len(model.parts), 1)
        part = model.parts[0]
        self.assertEqual(part.name, "triangle_part")
        np.testing.assert_allclose(part.vertices, vertices.astype(np.float64))
        np.testing.assert_array_equal(part.triangles, np.array([[0, 1, 2]], dtype=np.int32))

    def test_rejects_missing_bin_chunk(self) -> None:
        glb = build_triangle_glb(
            vertices=np.zeros((3, 3), dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.uint16),
        )
        json_end = glb.index(b"BIN\x00")
        truncated = glb[:json_end]
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(truncated)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "BIN chunk|total length"):
                load_cad_model(handle.name)

    def test_loads_nexplayground_sim_glb_with_named_parts(self) -> None:
        if not NEXPLAYGROUND_GLB.exists():
            self.skipTest("nexplayground_sim.glb fixture is not available")

        model = load_cad_model(NEXPLAYGROUND_GLB)
        names = {part.name for part in model.parts}
        self.assertIn("00_QUBE-TOP-CASE", names)
        self.assertIn("QUBE-FEET", names)
        total_triangles = sum(len(part.triangles) for part in model.parts)
        self.assertGreater(total_triangles, 10_000)
        for part in model.parts:
            self.assertEqual(part.vertices.shape[1], 3)
            self.assertTrue(np.all(part.triangles >= 0))
            self.assertTrue(np.all(part.triangles < len(part.vertices)))

    def test_synthetic_parts_default_to_white_material_color(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.5],
                [0.0, 0.1, 0.5],
            ],
            dtype=np.float32,
        )
        glb = build_triangle_glb(vertices=vertices, indices=np.array([0, 1, 2], dtype=np.uint16))
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            model = load_cad_model(handle.name)

        self.assertEqual(model.parts[0].material_color_bgr, (255, 255, 255))

    def test_loads_primitive_material_base_color_factor(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.5],
                [0.0, 0.1, 0.5],
            ],
            dtype=np.float32,
        )
        materials = [
            {
                "name": "red_mat",
                "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.0, 0.0, 1.0]},
            }
        ]
        glb = build_triangle_glb(
            vertices=vertices,
            indices=np.array([0, 1, 2], dtype=np.uint16),
            materials=materials,
            primitive_material=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            model = load_cad_model(handle.name)

        self.assertEqual(model.parts[0].material_color_bgr, (0, 0, 255))

    def test_converts_linear_material_color_to_display_srgb(self) -> None:
        vertices = np.zeros((3, 3), dtype=np.float32)
        glb = build_triangle_glb(
            vertices=vertices,
            indices=np.array([0, 1, 2], dtype=np.uint16),
            materials=[
                {
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [0.5, 0.5, 0.5, 1.0]
                    }
                }
            ],
            primitive_material=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            model = load_cad_model(handle.name)

        self.assertEqual(model.parts[0].material_color_bgr, (188, 188, 188))

    def test_rejects_invalid_material_index(self) -> None:
        vertices = np.zeros((3, 3), dtype=np.float32)
        glb = build_triangle_glb(
            vertices=vertices,
            indices=np.array([0, 1, 2], dtype=np.uint16),
            materials=[{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}],
            primitive_material=3,
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "invalid material index"):
                load_cad_model(handle.name)

    def test_rejects_malformed_base_color_factor(self) -> None:
        vertices = np.zeros((3, 3), dtype=np.float32)
        glb = build_triangle_glb(
            vertices=vertices,
            indices=np.array([0, 1, 2], dtype=np.uint16),
            materials=[{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.5]}}],
            primitive_material=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "baseColorFactor"):
                load_cad_model(handle.name)

    def test_loads_nexplayground_material_colors(self) -> None:
        if not NEXPLAYGROUND_GLB.exists():
            self.skipTest("nexplayground_sim.glb fixture is not available")

        model = load_cad_model(NEXPLAYGROUND_GLB)
        colors = {part.material_color_bgr for part in model.parts}
        expected = {
            (185, 204, 94),
            (123, 236, 255),
            (0, 0, 0),
            (255, 242, 234),
            (128, 121, 114),
        }
        self.assertTrue(expected.issubset(colors))
        self.assertGreater(len(colors), 1)


class CadRegistrationTests(unittest.TestCase):
    def test_loads_valid_registration(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(identity_registration_payload(), handle)
            handle.flush()
            registration = load_cad_registration(handle.name)

        self.assertEqual(registration.units, "meters")
        self.assertEqual(registration.source_frame, "cad")
        self.assertEqual(registration.target_frame, "marker_model")
        np.testing.assert_allclose(registration.transform_4x4, np.eye(4))

    def test_rejects_wrong_target_frame(self) -> None:
        payload = identity_registration_payload()
        payload["target_frame"] = "camera"
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "target_frame"):
                load_cad_registration(handle.name)

    def test_rejects_non_finite_transform(self) -> None:
        payload = identity_registration_payload()
        payload["transform_4x4"][1][1] = float("nan")
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "finite"):
                load_cad_registration(handle.name)


class CadCoordinateTransformTests(unittest.TestCase):
    def test_registration_and_layout_projection_preserve_known_point(self) -> None:
        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        cad_point = np.array([[0.01, -0.02, 0.03]], dtype=np.float64)
        layout_point = cad_points_to_layout(cad_point, registration)
        camera_point = layout_points_to_camera(layout_point, pose, marker_model)

        camera_matrix, dist_coeffs = synthetic_camera()
        projected = project_camera_points(camera_point, camera_matrix, dist_coeffs)
        self.assertTrue(np.all(np.isfinite(projected)))
        self.assertGreater(projected[0, 1], 0.0)


def load_cad_registration_from_payload(payload: dict[str, object]):
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        return load_cad_registration(handle.name)


class CadOverlayRenderTests(unittest.TestCase):
    def test_overlay_draws_without_darkening_background(self) -> None:
        vertices = np.array(
            [
                [-0.05, -0.05, 0.8],
                [0.05, -0.05, 0.8],
                [0.0, 0.05, 0.8],
            ],
            dtype=np.float32,
        )
        glb = build_triangle_glb(vertices=vertices, indices=np.array([0, 1, 2], dtype=np.uint16))
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            cad_model = load_cad_model(handle.name)

        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()

        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        untouched = frame[0, 0].copy()
        draw_cad_model_overlay(
            frame,
            pose,
            camera_matrix,
            dist_coeffs,
            marker_model,
            cad_model,
            registration,
            alpha=0.35,
        )
        np.testing.assert_array_equal(frame[0, 0], untouched)
        self.assertLess(int(frame[240, 320].sum()), 3 * 255)

    def test_overlay_skips_triangles_behind_camera(self) -> None:
        vertices = np.array(
            [
                [-0.05, -0.05, 0.8],
                [0.05, -0.05, 0.8],
                [0.0, 0.05, 0.8],
            ],
            dtype=np.float32,
        )
        glb = build_triangle_glb(vertices=vertices, indices=np.array([0, 1, 2], dtype=np.uint16))
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            cad_model = load_cad_model(handle.name)

        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        behind_camera = np.array(
            [
                [0.0, 0.0, -0.2],
                [0.1, 0.0, -0.2],
                [0.0, 0.1, -0.2],
            ],
            dtype=np.float64,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with mock.patch(
            "object_apriltag.viz.cad_overlay.layout_points_to_camera",
            return_value=behind_camera,
        ):
            draw_cad_model_overlay(
                frame,
                pose,
                camera_matrix,
                dist_coeffs,
                marker_model,
                cad_model,
                registration,
            )
        self.assertEqual(int(frame.sum()), 0)

    def test_collect_visible_triangles_skips_straddling_camera(self) -> None:
        vertices = np.array(
            [
                [0.0, 0.0, 0.8],
                [0.1, 0.0, 0.8],
                [0.0, 0.1, 0.8],
            ],
            dtype=np.float32,
        )
        glb = build_triangle_glb(vertices=vertices, indices=np.array([0, 1, 2], dtype=np.uint16))
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            cad_model = load_cad_model(handle.name)

        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        straddling = np.array(
            [
                [0.0, 0.0, 0.2],
                [0.1, 0.0, -0.2],
                [0.0, 0.1, 0.2],
            ],
            dtype=np.float64,
        )
        with mock.patch(
            "object_apriltag.viz.cad_overlay.layout_points_to_camera",
            return_value=straddling,
        ):
            parts = _collect_visible_parts(
                pose=pose,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                marker_model=marker_model,
                cad_model=cad_model,
                registration=registration,
            )
        self.assertEqual(parts, [])

    def test_coplanar_triangles_blend_part_only_once(self) -> None:
        vertices = np.array(
            [
                [-0.05, -0.05, 0.8],
                [0.05, -0.05, 0.8],
                [0.05, 0.05, 0.8],
                [-0.05, 0.05, 0.8],
            ],
            dtype=np.float32,
        )
        glb = build_triangle_glb(
            vertices=vertices,
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint16),
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            cad_model = load_cad_model(handle.name)

        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        alpha = 0.5

        draw_cad_model_overlay(
            frame,
            pose,
            camera_matrix,
            dist_coeffs,
            marker_model,
            cad_model,
            registration,
            alpha=alpha,
        )

        color = np.asarray(part_color_bgr("triangle_part"))
        expected = ((1.0 - alpha) * 100 + alpha * color).astype(np.uint8)
        np.testing.assert_allclose(frame[240, 320], expected, atol=1)

    def test_part_colors_are_distinct_and_stable(self) -> None:
        self.assertEqual(part_color_bgr("A"), part_color_bgr("A"))
        self.assertNotEqual(part_color_bgr("A"), part_color_bgr("B"))


class CadOnlyLandmarkOverlayTests(unittest.TestCase):
    def test_object_model_landmark_names_unions_keypoint_sources_and_keypoints(self) -> None:
        document = {
            "keypoint_sources": {"shared": {}, "source_only": {}},
            "keypoints": {"shared": [0.0, 0.0, 0.0], "keypoint_only": [1.0, 0.0, 0.0]},
        }
        self.assertEqual(
            object_model_landmark_names(document),
            frozenset({"shared", "source_only", "keypoint_only"}),
        )

    def test_cad_only_landmark_names_excludes_shared_names(self) -> None:
        cad_landmarks = CadLandmarks(
            landmarks={
                "shared": np.array([0.0, 0.0, 0.0]),
                "cad_only_a": np.array([0.1, 0.0, 0.0]),
                "cad_only_b": np.array([0.0, 0.1, 0.0]),
            }
        )
        self.assertEqual(
            cad_only_landmark_names(cad_landmarks, frozenset({"shared"})),
            ("cad_only_a", "cad_only_b"),
        )

    def test_project_cad_landmarks_skips_points_behind_camera(self) -> None:
        if not REMOTE_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(REMOTE_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        landmarks = {
            "front": np.array([0.0, 0.0, 0.8], dtype=np.float64),
            "behind": np.array([0.0, 0.0, -0.2], dtype=np.float64),
        }
        with mock.patch(
            "object_apriltag.viz.cad_overlay.layout_points_to_camera",
            return_value=np.array(
                [
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, -0.2],
                ],
                dtype=np.float64,
            ),
        ):
            projected = project_cad_landmarks_to_image(
                landmarks,
                ("front", "behind"),
                pose,
                camera_matrix,
                dist_coeffs,
                marker_model,
                registration,
            )

        self.assertIsNotNone(projected["front"])
        self.assertIsNone(projected["behind"])

    def test_project_cad_landmarks_returns_image_point_when_in_front_of_camera(self) -> None:
        if not REMOTE_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(REMOTE_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        landmarks = {"visible": np.array([0.0, 0.0, 0.8], dtype=np.float64)}
        with mock.patch(
            "object_apriltag.viz.cad_overlay.layout_points_to_camera",
            return_value=np.array([[0.0, 0.0, 0.5]], dtype=np.float64),
        ):
            projected = project_cad_landmarks_to_image(
                landmarks,
                ("visible",),
                pose,
                camera_matrix,
                dist_coeffs,
                marker_model,
                registration,
            )

        self.assertEqual(projected["visible"], (320, 240))

    def test_draw_cad_only_landmarks_uses_orange_for_cad_only_names(self) -> None:
        if not REMOTE_MARKER_MODEL.exists():
            self.skipTest("remote marker model fixture is not available")
        marker_model = load_marker_model(REMOTE_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        cad_landmarks = CadLandmarks(
            landmarks={
                "shared": np.array([0.0, 0.0, 0.8], dtype=np.float64),
                "cad_only": np.array([0.02, 0.0, 0.8], dtype=np.float64),
            }
        )
        cad_only_xy = (320, 240)

        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        untouched = frame[0, 0].copy()
        with mock.patch(
            "object_apriltag.viz.cad_overlay.project_cad_landmarks_to_image",
            return_value={"cad_only": cad_only_xy},
        ) as project_mock:
            draw_cad_only_landmarks(
                frame,
                pose,
                camera_matrix,
                dist_coeffs,
                marker_model,
                cad_landmarks,
                registration,
                frozenset({"shared"}),
            )

        project_mock.assert_called_once()
        projected_names = project_mock.call_args[0][1]
        self.assertEqual(projected_names, ("cad_only",))
        np.testing.assert_array_equal(frame[0, 0], untouched)
        np.testing.assert_array_equal(frame[cad_only_xy[1], cad_only_xy[0]], _CAD_ONLY_LANDMARK_COLOR_BGR)


class CadModelViewRenderTests(unittest.TestCase):
    def _triangle_scene(
        self,
    ) -> tuple[object, object, object, object, np.ndarray, np.ndarray]:
        vertices = np.array(
            [
                [-0.05, -0.05, 0.8],
                [0.05, -0.05, 0.8],
                [0.0, 0.05, 0.8],
            ],
            dtype=np.float32,
        )
        materials = [
            {
                "name": "green_mat",
                "pbrMetallicRoughness": {"baseColorFactor": [0.0, 1.0, 0.0, 1.0]},
            }
        ]
        glb = build_triangle_glb(
            vertices=vertices,
            indices=np.array([0, 1, 2], dtype=np.uint16),
            materials=materials,
            primitive_material=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            cad_model = load_cad_model(handle.name)

        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()
        return cad_model, marker_model, registration, pose, camera_matrix, dist_coeffs

    def test_render_returns_black_background_with_material_color_pixels(self) -> None:
        cad_model, marker_model, registration, pose, camera_matrix, dist_coeffs = (
            self._triangle_scene()
        )
        view = render_cad_model_view(
            (480, 640),
            pose,
            camera_matrix,
            dist_coeffs,
            marker_model,
            cad_model,
            registration,
        )

        self.assertEqual(view.shape, (480, 640, 3))
        self.assertEqual(view.dtype, np.uint8)
        self.assertEqual(int(view[0, 0].sum()), 0)
        center = view[240, 320]
        self.assertGreater(int(center.sum()), 0)
        np.testing.assert_array_equal(center, np.array([0, 255, 0], dtype=np.uint8))

    def test_render_accepts_camera_frame_shape(self) -> None:
        cad_model, marker_model, registration, pose, camera_matrix, dist_coeffs = (
            self._triangle_scene()
        )
        view = render_cad_model_view(
            (240, 320, 3),
            pose,
            camera_matrix,
            dist_coeffs,
            marker_model,
            cad_model,
            registration,
        )
        self.assertEqual(view.shape, (240, 320, 3))

    def test_render_stays_black_when_pose_is_behind_camera(self) -> None:
        cad_model, marker_model, registration, pose, camera_matrix, dist_coeffs = (
            self._triangle_scene()
        )
        behind_camera = np.array(
            [
                [0.0, 0.0, -0.2],
                [0.1, 0.0, -0.2],
                [0.0, 0.1, -0.2],
            ],
            dtype=np.float64,
        )
        with mock.patch(
            "object_apriltag.viz.cad_overlay.layout_points_to_camera",
            return_value=behind_camera,
        ):
            view = render_cad_model_view(
                (480, 640),
                pose,
                camera_matrix,
                dist_coeffs,
                marker_model,
                cad_model,
                registration,
            )
        self.assertEqual(int(view.sum()), 0)

    def test_render_performance_sanity_on_nexplayground(self) -> None:
        if not NEXPLAYGROUND_GLB.exists():
            self.skipTest("nexplayground_sim.glb fixture is not available")

        cad_model = load_cad_model(NEXPLAYGROUND_GLB)
        marker_model = load_marker_model(REMOTE_STATIC_MARKER_MODEL)
        registration = load_cad_registration_from_payload(identity_registration_payload())
        pose = ObjectPose(
            origin=np.array([0.0, 0.0, 0.8], dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        camera_matrix, dist_coeffs = synthetic_camera()

        import time

        start = time.perf_counter()
        view = render_cad_model_view(
            (480, 640),
            pose,
            camera_matrix,
            dist_coeffs,
            marker_model,
            cad_model,
            registration,
        )
        elapsed = time.perf_counter() - start
        self.assertEqual(view.shape, (480, 640, 3))
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
