"""CLI for manifest-driven marker-model evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from object_apriltag.evaluation.report import (
    format_marker_model_evaluation_console_summary,
    save_marker_model_evaluation_report,
)
from object_apriltag.evaluation.runner import evaluate_marker_models_from_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for manifest-driven marker-model evaluation.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]`` when omitted.

    Returns:
        Parsed argument namespace with manifest and output paths.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare marker-model candidates against CAD landmark geometry and "
            "held-out moving-video detection consistency."
        ),
        epilog=(
            "Reads a versioned evaluation manifest, decodes each held-out video once, "
            "freezes AprilTag detections, and scores every candidate with separate "
            "CAD-disagreement and detection-consistency rankings. CAD disagreement is "
            "not calibration error; held-out status is user-declared and unverifiable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Versioned evaluation manifest JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Versioned evaluation report JSON path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate marker-model candidates from a manifest and write a report.

    Args:
        argv: Optional argument vector passed to :func:`parse_args`.

    Returns:
        Process exit code ``0`` on success.
    """
    args = parse_args(argv)
    report = evaluate_marker_models_from_manifest(args.manifest)
    save_marker_model_evaluation_report(args.output, report)
    print(format_marker_model_evaluation_console_summary(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
