"""Tests for interactive Object Model keypoint editing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from object_apriltag.board_pose import BoardPoseEstimate, camera_point_to_board
from object_apriltag.detector import ObjectPose
from object_apriltag.layout import (
    CORNER_NAMES,
    build_marker_layout,
    camera_point_to_layout_point,
    footprint_corner_with_padding,
    footprint_from_dict,
    layout_point_to_camera,
    load_marker_model,
)
from object_apriltag.object_model_edit import (
    ObjectModelEditSession,
    apply_keypoint_sources_from_layout,
    board_coordinate_mm_to_layout_point,
    load_object_model_document,
    object_model_for_render,
    object_model_with_keypoint,
    ordered_keypoint_names,
    parse_keypoint_edit_line,
    parse_keypoint_sources,
    save_object_model_keypoints,
)
from object_apriltag.viz.skeleton import load_object_model

REMOTE1_MARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/marker_model.json"
)
REMOTE1_OBJECT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config/Model/remote1/object_model.json"
)


def _square_corners(half: float, z: float = 0.0) -> dict[str, list[float]]:
    return {
        "top_left": [-half, -half, z],
        "top_right": [half, -half, z],
        "bottom_right": [half, half, z],
        "bottom_left": [-half, half, z],
    }


def _synthetic_layout() -> object:
    marker_size = 0.04
    half = marker_size / 2.0
    footprints = {
        0: footprint_from_dict(0, _square_corners(half)),
        1: footprint_from_dict(
            1,
            {
                "top_left": [0.06, -half, 0.0],
                "top_right": [0.06 + marker_size, -half, 0.0],
                "bottom_right": [0.06 + marker_size, half, 0.0],
                "bottom_left": [0.06, half, 0.0],
            },
        ),
    }
    return build_marker_layout(0, marker_size, footprints)


def _object_model_document_with_sources() -> dict:
    return {
        "units": "meters",
        "coordinate_frame": "marker_model",
        "note": "keep-me",
        "keypoints": {
            "top": [0.0, 0.0, 0.0],
            "bottom": [0.0, 0.1, 0.0],
            "manual": [0.2, 0.2, 0.2],
        },
        "skeleton": [["top", "bottom"]],
        "keypoint_sources": {
            "top": {"marker_id": 1, "corner": "top_left"},
            "bottom": {"marker_id": "01", "corner": "bottom_right"},
        },
    }


def load_object_model_document_from_document(document: dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "object_model.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return load_object_model_document(path)


def synthetic_board_pose() -> BoardPoseEstimate:
    return BoardPoseEstimate(
        rotation=np.array(
            [[0.9, 0.0, 0.436], [0.0, 1.0, 0.0], [-0.436, 0.0, 0.9]],
            dtype=np.float64,
        ),
        origin=np.array([0.1, -0.05, 1.2], dtype=np.float64),
        reprojection_rms_px=0.5,
        detected_intersections=12,
        total_intersections=40,
    )


def synthetic_object_pose() -> ObjectPose:
    return ObjectPose(
        origin=np.array([0.0, 0.02, 0.9], dtype=np.float64),
        rotation=np.array(
            [[0.95, 0.0, 0.312], [0.0, 1.0, 0.0], [-0.312, 0.0, 0.95]],
            dtype=np.float64,
        ),
    )


class LayoutPointInverseTests(unittest.TestCase):
    def test_camera_point_to_layout_point_inverts_layout_point_to_camera(self) -> None:
        marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        object_pose = synthetic_object_pose()
        layout_point = np.array([0.02, -0.04, -0.18], dtype=np.float64)
        camera_point = layout_point_to_camera(
            layout_point,
            object_pose.rotation,
            object_pose.origin,
            marker_model,
        )
        recovered = camera_point_to_layout_point(
            camera_point,
            object_pose.rotation,
            object_pose.origin,
            marker_model,
        )
        np.testing.assert_allclose(recovered, layout_point, atol=1e-4)


class BoardCoordinateConversionTests(unittest.TestCase):
    def test_board_mm_roundtrip_through_layout_point(self) -> None:
        marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
        board_pose = synthetic_board_pose()
        object_pose = synthetic_object_pose()
        layout_point = np.array([0.01, -0.055, -0.25], dtype=np.float64)
        camera_point = layout_point_to_camera(
            layout_point,
            object_pose.rotation,
            object_pose.origin,
            marker_model,
        )
        board_point = camera_point_to_board(camera_point, board_pose)
        board_mm = board_point * 1000.0
        recovered = board_coordinate_mm_to_layout_point(
            float(board_mm[0]),
            float(board_mm[1]),
            float(board_mm[2]),
            board_pose,
            object_pose,
            marker_model,
        )
        np.testing.assert_allclose(recovered, layout_point, atol=1e-4)


class KeypointEditParseTests(unittest.TestCase):
    def test_parse_valid_line(self) -> None:
        self.assertEqual(parse_keypoint_edit_line("tip 12.3 -4.5 6.7"), ("tip", 12.3, -4.5, 6.7))

    def test_parse_rejects_bad_token_count(self) -> None:
        with self.assertRaises(ValueError):
            parse_keypoint_edit_line("only-three args here")

    def test_parse_rejects_non_numeric_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            parse_keypoint_edit_line("tip 1.0 foo 3.0")


class ObjectModelMutationTests(unittest.TestCase):
    def test_update_existing_keypoint_preserves_skeleton(self) -> None:
        model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)
        updated = object_model_with_keypoint(model, "center", np.array([0.1, 0.2, 0.3]))
        self.assertEqual(updated.skeleton_edges, model.skeleton_edges)
        np.testing.assert_allclose(updated.keypoints["center"], [0.1, 0.2, 0.3], atol=1e-9)

    def test_append_keypoint_preserves_existing_order(self) -> None:
        model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)
        updated = object_model_with_keypoint(model, "tip", np.array([0.0, 0.0, 0.1]))
        self.assertEqual(updated.keypoint_names, model.keypoint_names + ("tip",))
        self.assertNotIn(("center", "tip"), updated.skeleton_edges)


class SaveObjectModelTests(unittest.TestCase):
    def test_save_preserves_unknown_fields_and_skeleton(self) -> None:
        payload = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "note": "keep-me",
            "keypoints": {"a": [0.0, 0.0, 0.0], "b": [0.1, 0.0, 0.0]},
            "skeleton": [["a", "b"]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            model, document = load_object_model_document(path)
            updated = object_model_with_keypoint(model, "a", np.array([0.2, 0.3, 0.4]))
            save_object_model_keypoints(path, updated, document)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["note"], "keep-me")
            self.assertEqual(saved["skeleton"], [["a", "b"]])
            self.assertEqual(saved["coordinate_frame"], "marker_model")
            np.testing.assert_allclose(saved["keypoints"]["a"], [0.2, 0.3, 0.4], atol=1e-9)

    def test_save_appends_new_keypoint_after_existing(self) -> None:
        payload = {
            "units": "meters",
            "keypoints": {"a": [0.0, 0.0, 0.0], "b": [0.1, 0.0, 0.0]},
            "skeleton": [["a", "b"]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            model, document = load_object_model_document(path)
            updated = object_model_with_keypoint(model, "c", np.array([0.0, 0.1, 0.0]))
            save_object_model_keypoints(path, updated, document)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(list(saved["keypoints"]), ["a", "b", "c"])

    def test_save_failure_leaves_original_file_intact(self) -> None:
        payload = {
            "units": "meters",
            "keypoints": {"a": [0.0, 0.0, 0.0], "b": [0.1, 0.0, 0.0]},
            "skeleton": [["a", "b"]],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            original_text = json.dumps(payload, indent=2) + "\n"
            path.write_text(original_text, encoding="utf-8")
            model, document = load_object_model_document(path)
            updated = object_model_with_keypoint(model, "a", np.array([9.0, 9.0, 9.0]))
            with mock.patch(
                "object_apriltag.object_model_edit.os.replace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    save_object_model_keypoints(path, updated, document)
            self.assertEqual(path.read_text(encoding="utf-8"), original_text)


class ObjectModelEditSessionTests(unittest.TestCase):
    def test_invalid_edit_does_not_mark_dirty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            session = ObjectModelEditSession.from_path(path)
            marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
            before = session.working_model.keypoints["center"].copy()
            ok = session.apply_keypoint_edit("bad input", synthetic_object_pose(), synthetic_board_pose(), marker_model)
            self.assertFalse(ok)
            self.assertFalse(session.dirty)
            np.testing.assert_allclose(session.working_model.keypoints["center"], before, atol=1e-9)

    def test_missing_pose_leaves_model_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            session = ObjectModelEditSession.from_path(path)
            marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
            before = dict(session.working_model.keypoints)
            self.assertFalse(
                session.apply_keypoint_edit(
                    "tip 10.0 20.0 30.0",
                    None,
                    synthetic_board_pose(),
                    marker_model,
                )
            )
            self.assertEqual(session.working_model.keypoints, before)

    def test_successful_edit_marks_dirty_and_save_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            session = ObjectModelEditSession.from_path(path)
            marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
            self.assertTrue(
                session.apply_keypoint_edit(
                    "center 100.0 200.0 300.0",
                    synthetic_object_pose(),
                    synthetic_board_pose(),
                    marker_model,
                )
            )
            self.assertTrue(session.dirty)
            self.assertEqual(session.preview_keypoint_id, "center")
            np.testing.assert_allclose(
                session.preview_board_point_m,
                [0.1, 0.2, 0.3],
                atol=1e-9,
            )
            self.assertTrue(session.save())
            self.assertFalse(session.dirty)
            reloaded, _ = load_object_model_document(path)
            np.testing.assert_allclose(
                reloaded.keypoints["center"],
                session.working_model.keypoints["center"],
                atol=1e-9,
            )


class OrderedKeypointNamesTests(unittest.TestCase):
    def test_new_ids_append_after_document_order(self) -> None:
        document = {"keypoints": {"b": [0, 0, 0], "a": [0, 0, 0]}}
        model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)
        model = object_model_with_keypoint(model, "z", np.zeros(3))
        self.assertEqual(ordered_keypoint_names(document, model)[-1], "z")


class ObjectModelForRenderTests(unittest.TestCase):
    def test_returns_passthrough_when_no_edit_session(self) -> None:
        model = load_object_model(REMOTE1_OBJECT_MODEL_PATH)
        self.assertIs(object_model_for_render(model, None), model)

    def test_reads_latest_working_model_after_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            session = ObjectModelEditSession.from_path(path)
            marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
            before = object_model_for_render(None, session)
            session.apply_keypoint_edit(
                "center 100.0 200.0 300.0",
                synthetic_object_pose(),
                synthetic_board_pose(),
                marker_model,
            )
            after = object_model_for_render(None, session)
            self.assertIs(after, session.working_model)
            self.assertFalse(
                np.allclose(before.keypoints["center"], after.keypoints["center"], atol=1e-9)
            )


class LoadObjectModelDocumentTests(unittest.TestCase):
    def test_reads_file_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            original_read_text = Path.read_text
            read_count = 0

            def counting_read_text(self, *args, **kwargs):
                nonlocal read_count
                read_count += 1
                return original_read_text(self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", counting_read_text):
                load_object_model_document(path)
            self.assertEqual(read_count, 1)


class ParseKeypointSourcesTests(unittest.TestCase):
    def test_parses_marker_id_integer_strings(self) -> None:
        document = {
            "keypoint_sources": {
                "top": {"marker_id": "01", "corner": "top_left"},
            }
        }
        self.assertEqual(parse_keypoint_sources(document), {"top": (1, "top_left", 0.0)})

    def test_parses_omitted_and_zero_padding_mm(self) -> None:
        omitted = {
            "keypoint_sources": {
                "top": {"marker_id": 1, "corner": "top_left"},
            }
        }
        zero = {
            "keypoint_sources": {
                "top": {"marker_id": 1, "corner": "top_left", "padding_mm": 0},
            }
        }
        self.assertEqual(parse_keypoint_sources(omitted)["top"], (1, "top_left", 0.0))
        self.assertEqual(parse_keypoint_sources(zero)["top"], (1, "top_left", 0.0))

    def test_parses_positive_padding_mm(self) -> None:
        document = {
            "keypoint_sources": {
                "tip": {"marker_id": 2, "corner": "bottom_right", "padding_mm": 3.0},
            }
        }
        self.assertEqual(parse_keypoint_sources(document)["tip"], (2, "bottom_right", 0.003))

    def test_rejects_invalid_padding_mm(self) -> None:
        invalid_values = {
            "bool": True,
            "string": "3.0",
            "null": None,
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -1.0,
        }
        for label, padding_mm in invalid_values.items():
            document = {
                "keypoint_sources": {
                    "top": {"marker_id": 1, "corner": "top_left", "padding_mm": padding_mm},
                }
            }
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as ctx:
                    parse_keypoint_sources(document)
                self.assertIn("padding_mm", str(ctx.exception))

    def test_rejects_missing_or_empty_keypoint_sources(self) -> None:
        for document in ({}, {"keypoint_sources": {}}, {"keypoint_sources": []}):
            with self.subTest(document=document):
                with self.assertRaises(ValueError) as ctx:
                    parse_keypoint_sources(document)
                self.assertIn("keypoint_sources", str(ctx.exception))

    def test_rejects_invalid_corner(self) -> None:
        document = {
            "keypoint_sources": {
                "top": {"marker_id": 1, "corner": "center"},
            }
        }
        with self.assertRaises(ValueError) as ctx:
            parse_keypoint_sources(document)
        self.assertIn("corner", str(ctx.exception))
        for corner_name in CORNER_NAMES:
            self.assertIn(corner_name, str(ctx.exception))

    def test_rejects_missing_marker_id_or_corner(self) -> None:
        with self.assertRaises(ValueError) as missing_id:
            parse_keypoint_sources({"keypoint_sources": {"top": {"corner": "top_left"}}})
        self.assertIn("marker_id", str(missing_id.exception))

        with self.assertRaises(ValueError) as missing_corner:
            parse_keypoint_sources({"keypoint_sources": {"top": {"marker_id": 1}}})
        self.assertIn("corner", str(missing_corner.exception))

    def test_rejects_non_integer_marker_id(self) -> None:
        document = {
            "keypoint_sources": {
                "top": {"marker_id": "abc", "corner": "top_left"},
            }
        }
        with self.assertRaises(ValueError) as ctx:
            parse_keypoint_sources(document)
        self.assertIn("marker_id", str(ctx.exception))


class ApplyKeypointSourcesFromLayoutTests(unittest.TestCase):
    def test_copies_calibrated_corners_verbatim(self) -> None:
        layout = _synthetic_layout()
        document = _object_model_document_with_sources()
        model, _ = load_object_model_document_from_document(document)

        updated = apply_keypoint_sources_from_layout(model, document, layout)
        np.testing.assert_allclose(
            updated.keypoints["top"],
            layout.footprints[1].top_left,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            updated.keypoints["bottom"],
            layout.footprints[1].bottom_right,
            atol=1e-12,
        )

    def test_preserves_unmapped_keypoints_skeleton_and_metadata_on_save(self) -> None:
        layout = _synthetic_layout()
        document = _object_model_document_with_sources()
        model, loaded_document = load_object_model_document_from_document(document)
        manual_before = model.keypoints["manual"].copy()

        updated = apply_keypoint_sources_from_layout(model, document, layout)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            save_object_model_keypoints(path, updated, loaded_document)
            saved = json.loads(path.read_text(encoding="utf-8"))

        np.testing.assert_allclose(saved["keypoints"]["manual"], manual_before, atol=1e-12)
        self.assertEqual(saved["skeleton"], document["skeleton"])
        self.assertEqual(saved["note"], "keep-me")
        self.assertEqual(saved["keypoint_sources"], document["keypoint_sources"])

    def test_rejects_unknown_keypoint_or_missing_marker(self) -> None:
        layout = _synthetic_layout()
        model, document = load_object_model_document_from_document(_object_model_document_with_sources())

        unknown_keypoint = dict(document)
        unknown_keypoint["keypoint_sources"] = {
            "missing": {"marker_id": 1, "corner": "top_left"},
        }
        with self.assertRaises(ValueError) as ctx:
            apply_keypoint_sources_from_layout(model, unknown_keypoint, layout)
        self.assertIn("unknown keypoint", str(ctx.exception))

        missing_marker = dict(document)
        missing_marker["keypoint_sources"] = {
            "top": {"marker_id": 9, "corner": "top_left"},
        }
        with self.assertRaises(ValueError) as ctx:
            apply_keypoint_sources_from_layout(model, missing_marker, layout)
        self.assertIn("marker 9", str(ctx.exception))
        self.assertIn("solved marker layout", str(ctx.exception))

    def test_applies_padding_mm_to_mapped_keypoints(self) -> None:
        layout = _synthetic_layout()
        padding_mm = 3.0
        padding_m = padding_mm / 1000.0
        document = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "keypoints": {"tip": [0.0, 0.0, 0.0], "anchor": [0.0, 0.0, 0.0]},
            "skeleton": [["tip", "anchor"]],
            "keypoint_sources": {
                "tip": {"marker_id": 1, "corner": "top_left", "padding_mm": padding_mm},
            },
        }
        model, _ = load_object_model_document_from_document(document)
        footprint = layout.footprints[1]
        expected = footprint_corner_with_padding(footprint, "top_left", padding_m)
        updated = apply_keypoint_sources_from_layout(model, document, layout)
        np.testing.assert_allclose(updated.keypoints["tip"], expected, atol=1e-12)
        self.assertAlmostEqual(
            float(np.linalg.norm(updated.keypoints["tip"] - footprint.top_left)),
            padding_m * np.sqrt(2.0),
            places=12,
        )
        self.assertFalse(
            np.allclose(updated.keypoints["tip"], footprint.top_left, atol=1e-12)
        )

    def test_apply_padding_mm_for_all_square_corners_matches_layout_helper(self) -> None:
        layout = _synthetic_layout()
        padding_mm = 2.5
        padding_m = padding_mm / 1000.0
        footprint = layout.footprints[0]
        keypoint_sources = {
            corner_name: {
                "marker_id": 0,
                "corner": corner_name,
                "padding_mm": padding_mm,
            }
            for corner_name in CORNER_NAMES
        }
        document = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "keypoints": {corner_name: [0.0, 0.0, 0.0] for corner_name in CORNER_NAMES},
            "skeleton": [["top_left", "top_right"]],
            "keypoint_sources": keypoint_sources,
        }
        model, _ = load_object_model_document_from_document(document)
        updated = apply_keypoint_sources_from_layout(model, document, layout)
        for corner_name in CORNER_NAMES:
            expected = footprint_corner_with_padding(footprint, corner_name, padding_m)
            np.testing.assert_allclose(updated.keypoints[corner_name], expected, atol=1e-12)
            self.assertAlmostEqual(
                float(np.linalg.norm(updated.keypoints[corner_name] - footprint.corners_by_name()[corner_name])),
                padding_m * np.sqrt(2.0),
                places=12,
            )

    def test_save_preserves_padding_mm_in_keypoint_sources(self) -> None:
        layout = _synthetic_layout()
        padding_mm = 2.5
        padding_m = padding_mm / 1000.0
        document = {
            "units": "meters",
            "coordinate_frame": "marker_model",
            "note": "keep-me",
            "keypoints": {"tip": [0.0, 0.0, 0.0], "anchor": [0.0, 0.0, 0.0]},
            "skeleton": [["tip", "anchor"]],
            "keypoint_sources": {
                "tip": {"marker_id": 1, "corner": "top_left", "padding_mm": padding_mm},
            },
        }
        model, loaded_document = load_object_model_document_from_document(document)
        updated = apply_keypoint_sources_from_layout(model, document, layout)
        expected = footprint_corner_with_padding(layout.footprints[1], "top_left", padding_m)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            save_object_model_keypoints(path, updated, loaded_document)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["keypoint_sources"], document["keypoint_sources"])
        self.assertEqual(saved["note"], "keep-me")
        np.testing.assert_allclose(saved["keypoints"]["tip"], expected.tolist(), atol=1e-12)


class ObjectModelEditSessionSaveFailureTests(unittest.TestCase):
    def test_save_failure_keeps_dirty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "object_model.json"
            path.write_text(REMOTE1_OBJECT_MODEL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            session = ObjectModelEditSession.from_path(path)
            marker_model = load_marker_model(REMOTE1_MARKER_MODEL_PATH)
            self.assertTrue(
                session.apply_keypoint_edit(
                    "center 100.0 200.0 300.0",
                    synthetic_object_pose(),
                    synthetic_board_pose(),
                    marker_model,
                )
            )
            with mock.patch(
                "object_apriltag.object_model_edit.save_object_model_keypoints",
                side_effect=OSError("disk full"),
            ):
                self.assertFalse(session.save())
            self.assertTrue(session.dirty)
            self.assertIn("save failed", session.status_message)


if __name__ == "__main__":
    unittest.main()
