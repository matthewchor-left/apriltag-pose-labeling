"""Tests for CAD landmark extraction from GLB files."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from object_apriltag.cad import load_cad_landmarks, load_cad_model

REPO_ROOT = Path(__file__).resolve().parents[1]
NEXPLAYGROUND_GLB = REPO_ROOT / "config/Model/CAD/nexplayground_sim.glb"
CAD_REGISTRATION_DIAGNOSTICS = REPO_ROOT / "config/Model/CAD/cad_registration_diagnostics.json"

NEXPLAYGROUND_LANDMARK_NAMES = (
    "back-center",
    "front-center",
    "left-center",
    "right-center",
    "top-back-left",
    "top-back-right",
    "top-front-left",
    "top-front-right",
)


def build_triangle_glb(
    *,
    vertices: np.ndarray,
    indices: np.ndarray,
    part_name: str = "triangle_part",
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
                    }
                ],
            }
        ],
        "nodes": [{"mesh": 0, "name": part_name}],
        "scenes": [{"nodes": [0]}],
    }

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


def build_landmark_glb(nodes: list[dict[str, object]], scene_roots: list[int]) -> bytes:
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 0}],
        "nodes": nodes,
        "scenes": [{"nodes": scene_roots}],
    }
    json_chunk = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_padding = (4 - (len(json_chunk) % 4)) % 4
    json_chunk += b" " * json_padding
    bin_chunk = b""
    bin_padding = (4 - (len(bin_chunk) % 4)) % 4
    bin_chunk_padded = bin_chunk + b"\x00" * bin_padding
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk_padded)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<I4s", len(json_chunk), b"JSON")
    bin_header = struct.pack("<I4s", len(bin_chunk_padded), b"BIN\x00")
    return header + json_header + json_chunk + bin_header + bin_chunk_padded


class CadLandmarkLoaderTests(unittest.TestCase):
    def test_meshless_named_nodes_become_landmarks(self) -> None:
        glb = build_landmark_glb(
            nodes=[
                {"name": "lm_a", "translation": [0.1, 0.2, 0.3]},
                {"name": "lm_b", "translation": [-0.4, 0.0, 0.5]},
            ],
            scene_roots=[0, 1],
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            landmarks = load_cad_landmarks(handle.name)

        np.testing.assert_allclose(landmarks.landmarks["lm_a"], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(landmarks.landmarks["lm_b"], [-0.4, 0.0, 0.5])

    def test_nested_parent_transform_applies_to_landmark(self) -> None:
        glb = build_landmark_glb(
            nodes=[
                {
                    "name": "parent",
                    "translation": [0.0, 1.0, 0.0],
                    "children": [1],
                },
                {"name": "nested_lm", "translation": [1.0, 0.0, 0.0]},
            ],
            scene_roots=[0],
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            landmarks = load_cad_landmarks(handle.name)

        np.testing.assert_allclose(landmarks.landmarks["nested_lm"], [1.0, 1.0, 0.0])

    def test_nested_rotation_and_scale_apply_to_landmark(self) -> None:
        angle = np.pi / 2.0
        glb = build_landmark_glb(
            nodes=[
                {
                    "name": "parent",
                    "rotation": [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)],
                    "scale": [2.0, 2.0, 2.0],
                    "children": [1],
                },
                {"name": "rotated_lm", "translation": [1.0, 0.0, 0.0]},
            ],
            scene_roots=[0],
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            landmarks = load_cad_landmarks(handle.name)

        np.testing.assert_allclose(landmarks.landmarks["rotated_lm"], [0.0, 2.0, 0.0], atol=1e-9)

    def test_rejects_duplicate_landmark_names(self) -> None:
        glb = build_landmark_glb(
            nodes=[
                {"name": "dup", "translation": [0.0, 0.0, 0.0]},
                {"name": "dup", "translation": [1.0, 0.0, 0.0]},
            ],
            scene_roots=[0, 1],
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "Duplicate CAD landmark"):
                load_cad_landmarks(handle.name)

    def test_rejects_non_finite_landmark_position(self) -> None:
        glb = build_landmark_glb(
            nodes=[{"name": "bad", "translation": [float("nan"), 0.0, 0.0]}],
            scene_roots=[0],
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_cad_landmarks(handle.name)

    def test_required_names_reports_missing_landmarks(self) -> None:
        glb = build_landmark_glb(
            nodes=[{"name": "only_one", "translation": [0.0, 0.0, 0.0]}],
            scene_roots=[0],
        )
        with tempfile.NamedTemporaryFile(suffix=".glb") as handle:
            handle.write(glb)
            handle.flush()
            with self.assertRaisesRegex(ValueError, "missing required CAD landmarks"):
                load_cad_landmarks(handle.name, required_names=["only_one", "missing"])

    def test_mesh_nodes_are_not_landmarks(self) -> None:
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
            landmarks = load_cad_landmarks(handle.name)
            model = load_cad_model(handle.name)

        self.assertEqual(landmarks.landmarks, {})
        self.assertEqual(len(model.parts), 1)

    def test_loads_nexplayground_landmark_names_and_positions(self) -> None:
        if not NEXPLAYGROUND_GLB.exists():
            self.skipTest("nexplayground_sim.glb fixture is not available")
        if not CAD_REGISTRATION_DIAGNOSTICS.exists():
            self.skipTest("cad_registration_diagnostics.json fixture is not available")

        diagnostics = json.loads(CAD_REGISTRATION_DIAGNOSTICS.read_text(encoding="utf-8"))
        expected = {
            name: np.asarray(payload["cad_world_m"], dtype=np.float64)
            for name, payload in diagnostics["per_landmark"].items()
        }
        landmarks = load_cad_landmarks(NEXPLAYGROUND_GLB, required_names=NEXPLAYGROUND_LANDMARK_NAMES)

        self.assertEqual(set(landmarks.landmarks), set(NEXPLAYGROUND_LANDMARK_NAMES))
        for name in NEXPLAYGROUND_LANDMARK_NAMES:
            np.testing.assert_allclose(landmarks.landmarks[name], expected[name], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
