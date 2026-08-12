"""Regression tests for object-detect edit wiring."""

from __future__ import annotations

import unittest


class DetectEditImportTests(unittest.TestCase):
    def test_detect_module_exports_object_detector(self) -> None:
        import object_apriltag.cli.detect as detect_module
        from object_apriltag.detector import ObjectDetector

        self.assertIs(detect_module.ObjectDetector, ObjectDetector)

    def test_detect_main_is_importable(self) -> None:
        from object_apriltag.cli.detect import main

        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
