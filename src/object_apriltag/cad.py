"""Minimal glTF 2.0 GLB loader for CAD silhouette overlays."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_GLTF_MAGIC = b"glTF"
_GLTF_VERSION = 2
_CHUNK_JSON = b"JSON"
_CHUNK_BIN = b"BIN\x00"

_COMPONENT_FLOAT = 5126
_COMPONENT_UNSIGNED_SHORT = 5123
_COMPONENT_UNSIGNED_INT = 5125

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}

_COMPONENT_DTYPE = {
    _COMPONENT_UNSIGNED_SHORT: np.dtype("<u2"),
    _COMPONENT_UNSIGNED_INT: np.dtype("<u4"),
    _COMPONENT_FLOAT: np.dtype("<f4"),
}

_MODE_TRIANGLES = 4


_DEFAULT_BASE_COLOR_FACTOR = (1.0, 1.0, 1.0, 1.0)
_DEFAULT_MATERIAL_COLOR_BGR = (255, 255, 255)


@dataclass(frozen=True)
class CadMeshPart:
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    material_color_bgr: tuple[int, int, int] = _DEFAULT_MATERIAL_COLOR_BGR


@dataclass(frozen=True)
class CadModel:
    parts: tuple[CadMeshPart, ...]


@dataclass(frozen=True)
class CadLandmarks:
    landmarks: dict[str, np.ndarray]


@dataclass(frozen=True)
class CadRegistration:
    units: str
    source_frame: str
    target_frame: str
    transform_4x4: np.ndarray


def load_cad_model(path: Path | str) -> CadModel:
    """Load mesh parts from a binary glTF 2.0 (.glb) file."""
    data = Path(path).read_bytes()
    gltf, bin_chunk = _parse_glb(data)
    return _build_cad_model(gltf, bin_chunk)


def load_cad_landmarks(
    path: Path | str,
    required_names: Sequence[str] | None = None,
) -> CadLandmarks:
    """Load named meshless scene nodes as world-space CAD landmark positions."""
    data = Path(path).read_bytes()
    gltf, _bin_chunk = _parse_glb(data)
    landmarks = _build_cad_landmarks(gltf)
    if required_names is not None:
        missing = sorted(set(required_names) - set(landmarks))
        if missing:
            raise ValueError(
                f"GLB file is missing required CAD landmarks: {missing}."
            )
    return CadLandmarks(landmarks=landmarks)


def load_cad_registration(path: Path | str) -> CadRegistration:
    """Load a CAD-to-marker_model registration JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CAD registration JSON must be an object.")

    units = _require_string(payload, "units")
    source_frame = _require_string(payload, "source_frame")
    target_frame = _require_string(payload, "target_frame")
    if units != "meters":
        raise ValueError(f"CAD registration units must be 'meters', got {units!r}.")
    if source_frame != "cad":
        raise ValueError(f"CAD registration source_frame must be 'cad', got {source_frame!r}.")
    if target_frame != "marker_model":
        raise ValueError(
            f"CAD registration target_frame must be 'marker_model', got {target_frame!r}."
        )

    transform = np.asarray(payload.get("transform_4x4"), dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("CAD registration transform_4x4 must be a 4x4 matrix.")
    if not np.all(np.isfinite(transform)):
        raise ValueError("CAD registration transform_4x4 must contain only finite values.")

    return CadRegistration(
        units=units,
        source_frame=source_frame,
        target_frame=target_frame,
        transform_4x4=transform,
    )


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"CAD registration {field_name} must be a non-empty string.")
    return value


def _parse_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    if len(data) < 12:
        raise ValueError("GLB file is too small to contain a header.")

    magic, version, total_length = struct.unpack_from("<4sII", data, 0)
    if magic != _GLTF_MAGIC:
        raise ValueError("GLB file must start with the glTF magic bytes.")
    if version != _GLTF_VERSION:
        raise ValueError(f"Unsupported glTF version {version}; only version 2 is supported.")
    if total_length != len(data):
        raise ValueError("GLB total length does not match file size.")

    offset = 12
    json_chunk: bytes | None = None
    bin_chunk: bytes | None = None
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("GLB chunk header is truncated.")
        chunk_length, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_data = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == _CHUNK_JSON:
            if json_chunk is not None:
                raise ValueError("GLB file contains multiple JSON chunks.")
            json_chunk = chunk_data
        elif chunk_type == _CHUNK_BIN:
            if bin_chunk is not None:
                raise ValueError("GLB file contains multiple BIN chunks.")
            bin_chunk = chunk_data
        else:
            raise ValueError(f"Unsupported GLB chunk type {chunk_type!r}.")

    if json_chunk is None:
        raise ValueError("GLB file is missing a JSON chunk.")
    if bin_chunk is None:
        raise ValueError("GLB file is missing a BIN chunk.")

    try:
        gltf = json.loads(json_chunk)
    except json.JSONDecodeError as exc:
        raise ValueError("GLB JSON chunk is not valid JSON.") from exc
    if not isinstance(gltf, dict):
        raise ValueError("GLB JSON chunk must decode to an object.")
    return gltf, bin_chunk


def _build_cad_model(gltf: dict[str, Any], bin_chunk: bytes) -> CadModel:
    buffers = gltf.get("buffers")
    if not isinstance(buffers, list) or len(buffers) != 1:
        raise ValueError("Only single-buffer embedded GLB files are supported.")
    if buffers[0].get("uri") is not None:
        raise ValueError("External buffer URIs are not supported.")

    scenes = gltf.get("scenes")
    nodes = gltf.get("nodes")
    meshes = gltf.get("meshes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("GLB file must define at least one scene.")
    if not isinstance(nodes, list):
        raise ValueError("GLB file must define a nodes array.")
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("GLB file must define at least one mesh.")

    scene_index = int(gltf.get("scene", 0))
    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError(f"GLB scene index {scene_index} is out of range.")
    scene = scenes[scene_index]
    root_nodes = scene.get("nodes")
    if not isinstance(root_nodes, list):
        raise ValueError("GLB scene must define a nodes array.")

    parts: list[CadMeshPart] = []
    for node_index in root_nodes:
        _collect_node_parts(
            node_index=int(node_index),
            nodes=nodes,
            meshes=meshes,
            gltf=gltf,
            bin_chunk=bin_chunk,
            parent_transform=np.eye(4, dtype=np.float64),
            parts=parts,
        )

    if not parts:
        raise ValueError("GLB file does not contain any renderable mesh parts.")
    return CadModel(parts=tuple(parts))


def _build_cad_landmarks(gltf: dict[str, Any]) -> dict[str, np.ndarray]:
    scenes = gltf.get("scenes")
    nodes = gltf.get("nodes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("GLB file must define at least one scene.")
    if not isinstance(nodes, list):
        raise ValueError("GLB file must define a nodes array.")

    scene_index = int(gltf.get("scene", 0))
    if scene_index < 0 or scene_index >= len(scenes):
        raise ValueError(f"GLB scene index {scene_index} is out of range.")
    scene = scenes[scene_index]
    root_nodes = scene.get("nodes")
    if not isinstance(root_nodes, list):
        raise ValueError("GLB scene must define a nodes array.")

    landmarks: dict[str, np.ndarray] = {}
    for node_index in root_nodes:
        _collect_node_landmarks(
            node_index=int(node_index),
            nodes=nodes,
            parent_transform=np.eye(4, dtype=np.float64),
            landmarks=landmarks,
        )
    return landmarks


def _collect_node_landmarks(
    *,
    node_index: int,
    nodes: list[dict[str, Any]],
    parent_transform: np.ndarray,
    landmarks: dict[str, np.ndarray],
) -> None:
    if node_index < 0 or node_index >= len(nodes):
        raise ValueError(f"GLB node index {node_index} is out of range.")

    node = nodes[node_index]
    if not isinstance(node, dict):
        raise ValueError(f"GLB node {node_index} must be an object.")

    world_transform = parent_transform @ _node_local_transform(node)
    mesh_index = node.get("mesh")
    if mesh_index is None:
        name = node.get("name")
        if isinstance(name, str) and name:
            if name in landmarks:
                raise ValueError(f"Duplicate CAD landmark name {name!r}.")
            position = world_transform[:3, 3].copy()
            if not np.all(np.isfinite(position)):
                raise ValueError(f"CAD landmark {name!r} has non-finite position.")
            landmarks[name] = position

    children = node.get("children")
    if children is not None:
        if not isinstance(children, list):
            raise ValueError(f"GLB node {node_index} children must be an array.")
        for child_index in children:
            _collect_node_landmarks(
                node_index=int(child_index),
                nodes=nodes,
                parent_transform=world_transform,
                landmarks=landmarks,
            )


def _collect_node_parts(
    *,
    node_index: int,
    nodes: list[dict[str, Any]],
    meshes: list[dict[str, Any]],
    gltf: dict[str, Any],
    bin_chunk: bytes,
    parent_transform: np.ndarray,
    parts: list[CadMeshPart],
) -> None:
    if node_index < 0 or node_index >= len(nodes):
        raise ValueError(f"GLB node index {node_index} is out of range.")

    node = nodes[node_index]
    if not isinstance(node, dict):
        raise ValueError(f"GLB node {node_index} must be an object.")

    world_transform = parent_transform @ _node_local_transform(node)
    mesh_index = node.get("mesh")
    if mesh_index is not None:
        if not isinstance(mesh_index, int) or mesh_index < 0 or mesh_index >= len(meshes):
            raise ValueError(f"GLB node {node_index} references an invalid mesh index.")
        mesh = meshes[mesh_index]
        if not isinstance(mesh, dict):
            raise ValueError(f"GLB mesh {mesh_index} must be an object.")
        part_name = str(node.get("name") or mesh.get("name") or f"mesh_{mesh_index}")
        parts.extend(
            _mesh_primitives_as_parts(
                mesh=mesh,
                mesh_index=mesh_index,
                part_name=part_name,
                world_transform=world_transform,
                gltf=gltf,
                bin_chunk=bin_chunk,
            )
        )

    children = node.get("children")
    if children is not None:
        if not isinstance(children, list):
            raise ValueError(f"GLB node {node_index} children must be an array.")
        for child_index in children:
            _collect_node_parts(
                node_index=int(child_index),
                nodes=nodes,
                meshes=meshes,
                gltf=gltf,
                bin_chunk=bin_chunk,
                parent_transform=world_transform,
                parts=parts,
            )


def _mesh_primitives_as_parts(
    *,
    mesh: dict[str, Any],
    mesh_index: int,
    part_name: str,
    world_transform: np.ndarray,
    gltf: dict[str, Any],
    bin_chunk: bytes,
) -> list[CadMeshPart]:
    primitives = mesh.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise ValueError(f"GLB mesh {mesh_index} must define at least one primitive.")

    parts: list[CadMeshPart] = []
    for primitive_index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            raise ValueError(f"GLB mesh {mesh_index} primitive {primitive_index} must be an object.")
        mode = int(primitive.get("mode", _MODE_TRIANGLES))
        if mode != _MODE_TRIANGLES:
            raise ValueError(
                f"GLB mesh {mesh_index} primitive {primitive_index} uses unsupported mode {mode}; "
                "only triangles are supported."
            )

        attributes = primitive.get("attributes")
        if not isinstance(attributes, dict) or "POSITION" not in attributes:
            raise ValueError(
                f"GLB mesh {mesh_index} primitive {primitive_index} must define POSITION attributes."
            )
        position_index = int(attributes["POSITION"])
        positions = _read_accessor_vec3(gltf, bin_chunk, position_index)
        indices_accessor = primitive.get("indices")
        if indices_accessor is None:
            triangle_indices = np.arange(len(positions), dtype=np.int32).reshape(-1, 3)
        else:
            triangle_indices = _read_accessor_indices(gltf, bin_chunk, int(indices_accessor))

        transformed = _transform_points(positions, world_transform)
        name = part_name if len(primitives) == 1 else f"{part_name}:{primitive_index}"
        material_color_bgr = _primitive_material_color_bgr(
            gltf=gltf,
            primitive=primitive,
            mesh_index=mesh_index,
            primitive_index=primitive_index,
        )
        parts.append(
            CadMeshPart(
                name=name,
                vertices=transformed,
                triangles=triangle_indices,
                material_color_bgr=material_color_bgr,
            )
        )
    return parts


def _primitive_material_color_bgr(
    *,
    gltf: dict[str, Any],
    primitive: dict[str, Any],
    mesh_index: int,
    primitive_index: int,
) -> tuple[int, int, int]:
    material_index = primitive.get("material")
    if material_index is None:
        return _DEFAULT_MATERIAL_COLOR_BGR

    if not isinstance(material_index, int):
        raise ValueError(
            f"GLB mesh {mesh_index} primitive {primitive_index} material index must be an integer."
        )

    materials = gltf.get("materials")
    if not isinstance(materials, list):
        raise ValueError("GLB file must define a materials array when primitives reference materials.")
    if material_index < 0 or material_index >= len(materials):
        raise ValueError(
            f"GLB mesh {mesh_index} primitive {primitive_index} references invalid material index "
            f"{material_index}."
        )

    material = materials[material_index]
    if not isinstance(material, dict):
        raise ValueError(f"GLB material {material_index} must be an object.")

    pbr = material.get("pbrMetallicRoughness")
    if pbr is None:
        return _base_color_factor_to_bgr(_DEFAULT_BASE_COLOR_FACTOR)
    if not isinstance(pbr, dict):
        raise ValueError(f"GLB material {material_index} pbrMetallicRoughness must be an object.")

    base_color_factor = pbr.get("baseColorFactor", _DEFAULT_BASE_COLOR_FACTOR)
    return _base_color_factor_to_bgr(base_color_factor, material_index=material_index)


def _base_color_factor_to_bgr(
    base_color_factor: Any,
    *,
    material_index: int | None = None,
) -> tuple[int, int, int]:
    prefix = (
        f"GLB material {material_index} "
        if material_index is not None
        else "GLB baseColorFactor "
    )
    if not isinstance(base_color_factor, (list, tuple)) or len(base_color_factor) != 4:
        raise ValueError(f"{prefix}baseColorFactor must be an array of four numbers.")

    try:
        rgba = [float(value) for value in base_color_factor]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}baseColorFactor must be an array of four numbers.") from exc
    if not all(np.isfinite(value) for value in rgba):
        raise ValueError(f"{prefix}baseColorFactor must contain only finite values.")

    def linear_to_srgb(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        if value <= 0.0031308:
            return 12.92 * value
        return 1.055 * value ** (1.0 / 2.4) - 0.055

    red, green, blue = (linear_to_srgb(value) for value in rgba[:3])
    bgr = (
        int(np.clip(round(blue * 255.0), 0, 255)),
        int(np.clip(round(green * 255.0), 0, 255)),
        int(np.clip(round(red * 255.0), 0, 255)),
    )
    return bgr


def _node_local_transform(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        matrix = np.asarray(node["matrix"], dtype=np.float64)
        if matrix.shape != (16,):
            raise ValueError("GLB node matrix must contain 16 values.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("GLB node matrix must contain only finite values.")
        return matrix.reshape(4, 4, order="F")

    transform = np.eye(4, dtype=np.float64)
    translation = node.get("translation")
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    rotation = node.get("rotation")
    if rotation is not None:
        transform[:3, :3] = _quaternion_to_matrix(np.asarray(rotation, dtype=np.float64).reshape(4))
    scale = node.get("scale")
    if scale is not None:
        scale_vector = np.asarray(scale, dtype=np.float64).reshape(3)
        transform[:3, :3] = transform[:3, :3] @ np.diag(scale_vector)
    return transform


def _quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    if not np.all(np.isfinite(quaternion_xyzw)):
        raise ValueError("GLB node rotation quaternion must contain only finite values.")
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm == 0.0:
        raise ValueError("GLB node rotation quaternion must be non-zero.")
    x, y, z, w = quaternion_xyzw / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    transformed = (transform @ homogeneous.T).T[:, :3]
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Transformed CAD vertex contains non-finite values.")
    return transformed


def _read_accessor_vec3(gltf: dict[str, Any], bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    accessor = _accessor(gltf, accessor_index)
    if accessor["type"] != "VEC3":
        raise ValueError(f"Accessor {accessor_index} must have type VEC3.")
    if accessor["componentType"] != _COMPONENT_FLOAT:
        raise ValueError(f"Accessor {accessor_index} positions must use FLOAT component type.")
    raw = _read_accessor_raw(gltf, bin_chunk, accessor)
    points = raw.reshape(-1, 3).astype(np.float64, copy=False)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Accessor {accessor_index} contains non-finite position values.")
    return points


def _read_accessor_indices(gltf: dict[str, Any], bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    accessor = _accessor(gltf, accessor_index)
    if accessor["type"] != "SCALAR":
        raise ValueError(f"Accessor {accessor_index} must have type SCALAR.")
    component_type = int(accessor["componentType"])
    if component_type not in (_COMPONENT_UNSIGNED_SHORT, _COMPONENT_UNSIGNED_INT):
        raise ValueError(
            f"Accessor {accessor_index} indices must use UNSIGNED_SHORT or UNSIGNED_INT."
        )
    raw = _read_accessor_raw(gltf, bin_chunk, accessor)
    indices = raw.reshape(-1).astype(np.int32, copy=False)
    if indices.size % 3 != 0:
        raise ValueError(f"Accessor {accessor_index} index count must be divisible by 3.")
    if np.any(indices < 0):
        raise ValueError(f"Accessor {accessor_index} contains negative indices.")
    return indices.reshape(-1, 3)


def _accessor(gltf: dict[str, Any], accessor_index: int) -> dict[str, Any]:
    accessors = gltf.get("accessors")
    if not isinstance(accessors, list):
        raise ValueError("GLB file must define an accessors array.")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise ValueError(f"Accessor index {accessor_index} is out of range.")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, dict):
        raise ValueError(f"Accessor {accessor_index} must be an object.")
    return accessor


def _read_accessor_raw(
    gltf: dict[str, Any],
    bin_chunk: bytes,
    accessor: dict[str, Any],
) -> np.ndarray:
    buffer_views = gltf.get("bufferViews")
    if not isinstance(buffer_views, list):
        raise ValueError("GLB file must define a bufferViews array.")

    buffer_view_index = accessor.get("bufferView")
    if not isinstance(buffer_view_index, int):
        raise ValueError("Accessor must reference a bufferView.")
    if buffer_view_index < 0 or buffer_view_index >= len(buffer_views):
        raise ValueError(f"Buffer view index {buffer_view_index} is out of range.")

    buffer_view = buffer_views[buffer_view_index]
    if not isinstance(buffer_view, dict):
        raise ValueError(f"Buffer view {buffer_view_index} must be an object.")
    if buffer_view.get("buffer", 0) != 0:
        raise ValueError("Only embedded buffer 0 is supported.")

    component_type = int(accessor["componentType"])
    if component_type not in _COMPONENT_DTYPE:
        raise ValueError(f"Unsupported accessor component type {component_type}.")

    accessor_type = accessor["type"]
    if accessor_type not in _TYPE_COMPONENTS:
        raise ValueError(f"Unsupported accessor type {accessor_type!r}.")
    components_per_element = _TYPE_COMPONENTS[accessor_type]
    count = int(accessor["count"])
    dtype = _COMPONENT_DTYPE[component_type]
    element_size = dtype.itemsize * components_per_element

    byte_offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_stride = int(buffer_view.get("byteStride", 0))
    if byte_stride not in (0, element_size):
        raise ValueError(
            f"Accessor byteStride {byte_stride} is not supported for tightly packed elements."
        )

    byte_length = count * element_size
    if byte_offset < 0 or byte_offset + byte_length > len(bin_chunk):
        raise ValueError("Accessor reads outside the GLB BIN chunk.")

    raw = np.frombuffer(bin_chunk, dtype=dtype, count=count * components_per_element, offset=byte_offset)
    return raw
