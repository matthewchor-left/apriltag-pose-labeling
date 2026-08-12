"""CLI smoke tests for board-frame options."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
REMOTE1_MARKER_MODEL = REPO_ROOT / "config/Model/remote1/marker_model.json"
REMOTE1_ERASER_MODEL = REPO_ROOT / "config/Model/remote1/eraser_model.json"
CALIBRATION = REPO_ROOT / "config/Camera/nexplaygroundcam/intrinsics.json"


def _run_cli_help(command: str) -> str:
    result = subprocess.run(
        ["uv", "run", command, "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


class CliBoardFrameHelpTests(unittest.TestCase):
    def test_object_detect_help_lists_board_options(self) -> None:
        help_text = _run_cli_help("object-detect")
        self.assertIn("--board-frame", help_text)
        self.assertIn("--board-model", help_text)
        self.assertIn("--camera-motion", help_text)
        self.assertIn("static", help_text)
        self.assertIn("dynamic", help_text)

    def test_annotation_tool_help_lists_board_options(self) -> None:
        help_text = _run_cli_help("annotation-tool")
        self.assertIn("--board-frame", help_text)
        self.assertIn("--board-model", help_text)
        self.assertIn("--camera-motion", help_text)

    def test_object_detect_help_default_camera_motion_is_static(self) -> None:
        help_text = _run_cli_help("object-detect")
        self.assertRegex(help_text, r"--camera-motion.*static")


class CliBoardFrameContractTests(unittest.TestCase):
    def test_object_detect_requires_board_model_when_board_frame_enabled(self) -> None:
        from object_apriltag.cli.detect import main as detect_main

        argv = [
            "object-detect",
            "--camera",
            "0",
            "--calibration",
            str(CALIBRATION),
            "--marker-model",
            str(REMOTE1_MARKER_MODEL),
            "--dictionary",
            "36h11",
            "--detection-sensitivity",
            "relaxed",
            "--preview",
            "--board-frame",
        ]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(RuntimeError) as ctx:
                detect_main()
        self.assertIn("--board-model", str(ctx.exception))

    def test_annotation_tool_requires_board_model_when_board_frame_enabled(self) -> None:
        from object_apriltag.cli.annotation_tool import main as annotation_main

        argv = [
            "annotation-tool",
            "--camera",
            "0",
            "--calibration",
            str(CALIBRATION),
            "--marker-model",
            str(REMOTE1_MARKER_MODEL),
            "--eraser-model",
            str(REMOTE1_ERASER_MODEL),
            "--dictionary",
            "36h11",
            "--detection-sensitivity",
            "relaxed",
            "--board-frame",
        ]
        with mock.patch("sys.argv", argv):
            with self.assertRaises(RuntimeError) as ctx:
                annotation_main()
        self.assertIn("--board-model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
