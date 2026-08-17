"""Tests for Calibration Recipe parsing and validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from object_apriltag.marker_layout_calibration.recipe import (
    BENCHMARK_FRAME_SELECTION_SHARPEST,
    CALIBRATION_RECIPE_VERSION,
    BenchmarkExecution,
    InteractiveExecution,
    load_calibration_recipe,
)

PLAYGROUND_SETUP1 = (
    Path(__file__).resolve().parent.parent / "config/Model/playground/setup1"
)


def _write_intrinsics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "image_size": [640, 480],
                "camera_matrix": [[900.0, 0.0, 320.0], [0.0, 900.0, 240.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )


def _base_recipe_payload(**overrides: object) -> dict:
    payload = {
        "config_version": CALIBRATION_RECIPE_VERSION,
        "inputs": {"source": "clip.mov", "intrinsics": "intrinsics.json"},
        "detector": {"dictionary": "36h11", "sensitivity": "relaxed"},
        "markers": {
            "reference_marker_id": 0,
            "anchor_marker_ids": [0, 1],
            "groups": [
                {"ids": [0, 1], "size_m": 0.07},
                {"ids": ["2-3"], "size_m": 0.05},
            ],
        },
        "execution": {
            "mode": "benchmark",
            "sample_rate_hz": 10.0,
            "frame_selection": BENCHMARK_FRAME_SELECTION_SHARPEST,
        },
        "solver": {
            "policy": "strict",
            "anchor_stop_after_expansion": False,
            "partial_output": False,
            "min_inliers_per_edge": 20,
            "reprojection_rms_gate_px": 2.0,
            "pair_translation_rms_gate_ratio": 0.1,
            "pair_rotation_rms_gate_deg": 5.0,
            "huber_delta_px": 1.0,
            "corner_outlier_px": 3.0,
            "max_ba_iterations": 50,
        },
        "object_model": {
            "keypoint_sources": {
                "a": {"marker_id": 0, "corner": "top_left"},
                "b": {"marker_id": 1, "corner": "top_right", "padding_mm": 2.0},
            },
            "skeleton": [["a", "b"]],
        },
    }
    payload.update(overrides)
    return payload


class CalibrationRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp_dir.name)
        _write_intrinsics(self.workspace / "intrinsics.json")
        (self.workspace / "clip.mov").write_text("", encoding="utf-8")
        nested = self.workspace / "nested"
        nested.mkdir()
        (nested / "clip.mov").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _write_config(self, payload: dict) -> Path:
        config_path = self.workspace / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def test_loads_valid_recipe_with_relative_paths_and_groups(self) -> None:
        payload = _base_recipe_payload()
        payload["inputs"] = {
            "source": "nested/clip.mov",
            "intrinsics": "intrinsics.json",
        }
        recipe = load_calibration_recipe(self._write_config(payload))
        self.assertEqual(recipe.expected_marker_ids, (0, 1, 2, 3))
        self.assertEqual(recipe.marker_sizes_m[2], 0.05)
        self.assertEqual(recipe.default_marker_size_m, 0.07)
        self.assertEqual(recipe.intrinsics_path, (self.workspace / "intrinsics.json").resolve())
        self.assertEqual(recipe.source, (self.workspace / "nested" / "clip.mov").resolve())
        self.assertEqual(recipe.paths.marker_model_path.name, "marker_model.json")
        self.assertEqual(recipe.settings.huber_delta_px, 1.0)
        self.assertEqual(recipe.settings.max_ba_iterations, 50)

    def test_rejects_unknown_top_level_fields(self) -> None:
        payload = _base_recipe_payload(extra_field=True)
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("unknown fields", str(ctx.exception))

    def test_rejects_overlapping_marker_groups(self) -> None:
        payload = _base_recipe_payload()
        payload["markers"]["groups"] = [
            {"ids": [0, 1], "size_m": 0.07},
            {"ids": [1, 2], "size_m": 0.05},
        ]
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("overlap", str(ctx.exception))

    def test_rejects_duplicate_ids_within_marker_group(self) -> None:
        payload = _base_recipe_payload()
        payload["markers"]["groups"][0]["ids"] = [0, 1, 1]
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("duplicate marker IDs", str(ctx.exception))

    def test_rejects_missing_solver_fields(self) -> None:
        payload = _base_recipe_payload()
        del payload["solver"]["huber_delta_px"]
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("missing required fields", str(ctx.exception))

    def test_rejects_partial_output_without_best_effort(self) -> None:
        payload = _base_recipe_payload()
        payload["solver"]["partial_output"] = True
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("partial_output", str(ctx.exception))

    def test_rejects_infinite_solver_setting(self) -> None:
        payload = _base_recipe_payload()
        payload["solver"]["huber_delta_px"] = float("inf")
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("finite and positive", str(ctx.exception))

    def test_rejects_skeleton_keypoint_without_source(self) -> None:
        payload = _base_recipe_payload()
        payload["object_model"]["skeleton"] = [["a", "b"], ["a", "missing"]]
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("keypoint_sources entries", str(ctx.exception))

    def test_allows_keypoint_sources_without_skeleton_edges(self) -> None:
        payload = _base_recipe_payload()
        payload["object_model"]["keypoint_sources"]["c"] = {
            "marker_id": 2,
            "corner": "bottom_left",
        }
        recipe = load_calibration_recipe(self._write_config(payload))
        self.assertIn("c", recipe.keypoint_sources)
        self.assertEqual(recipe.skeleton, (("a", "b"),))

    def test_rejects_unknown_keypoint_source_field(self) -> None:
        payload = _base_recipe_payload()
        payload["object_model"]["keypoint_sources"]["a"]["padding_m"] = 0.004
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("unknown fields", str(ctx.exception))

    def test_rejects_keypoint_source_outside_inventory(self) -> None:
        payload = _base_recipe_payload()
        payload["object_model"]["keypoint_sources"]["c"] = {
            "marker_id": 99,
            "corner": "top_left",
        }
        payload["object_model"]["skeleton"] = [["a", "b"], ["b", "c"]]
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("outside markers.groups", str(ctx.exception))

    def test_parses_interactive_auto_execution(self) -> None:
        payload = _base_recipe_payload()
        payload["execution"] = {
            "mode": "interactive",
            "capture": "auto",
            "sample_rate_hz": 5.0,
            "preview": "none",
        }
        recipe = load_calibration_recipe(self._write_config(payload))
        self.assertIsInstance(recipe.execution, InteractiveExecution)
        assert isinstance(recipe.execution, InteractiveExecution)
        self.assertEqual(recipe.execution.capture, "auto")
        self.assertEqual(recipe.execution.sample_rate_hz, 5.0)

    def test_loads_real_playground_setup1_workspace_configs(self) -> None:
        for workspace_name in ("calibration_01", "calibration_02", "calibration_03"):
            config_path = PLAYGROUND_SETUP1 / workspace_name / "config.json"
            recipe = load_calibration_recipe(config_path)
            self.assertEqual(recipe.reference_marker_id, 19)
            self.assertEqual(recipe.expected_marker_ids, (0, 2, 3, 4, 19, 22, 23, 24, 25, 26, 28, 29))
            self.assertIsInstance(recipe.execution, BenchmarkExecution)
            assert isinstance(recipe.execution, BenchmarkExecution)
            self.assertEqual(recipe.execution.frame_selection, BENCHMARK_FRAME_SELECTION_SHARPEST)
            self.assertEqual(recipe.execution.sample_rate_hz, 10.0)
            self.assertEqual(recipe.policy, "best_effort")
            self.assertTrue(recipe.partial_output)
            self.assertEqual(recipe.default_marker_size_m, 0.07)
            self.assertIsNone(recipe.anchor_marker_ids)

    def test_rejects_interactive_manual_with_sample_rate(self) -> None:
        payload = _base_recipe_payload()
        payload["execution"] = {
            "mode": "interactive",
            "capture": "manual",
            "preview": "none",
            "sample_rate_hz": 5.0,
        }
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("unknown fields", str(ctx.exception))

    def test_defaults_discrete_method_to_pair_consensus(self) -> None:
        recipe = load_calibration_recipe(self._write_config(_base_recipe_payload()))
        self.assertEqual(recipe.settings.discrete_method, "pair_consensus")

    def test_parses_rotation_consistent_discrete_method(self) -> None:
        payload = _base_recipe_payload()
        payload["solver"]["discrete_method"] = "rotation_consistent"
        recipe = load_calibration_recipe(self._write_config(payload))
        self.assertEqual(recipe.settings.discrete_method, "rotation_consistent")

    def test_rejects_unknown_discrete_method(self) -> None:
        payload = _base_recipe_payload()
        payload["solver"]["discrete_method"] = "clique_pcm"
        with self.assertRaises(ValueError) as ctx:
            load_calibration_recipe(self._write_config(payload))
        self.assertIn("discrete_method", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
