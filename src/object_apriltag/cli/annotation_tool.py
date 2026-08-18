"""Generate YOLO pose Training Samples from live or recorded captures."""

from __future__ import annotations

import argparse
from pathlib import Path

from object_apriltag.calibration import load_intrinsics, require_calibration_image_size
from object_apriltag.cad import load_cad_model, load_cad_registration
from object_apriltag.detector import ObjectDetector
from object_apriltag.frame_source import format_frame_source, is_camera_source, parse_frame_source
from object_apriltag.object_model_edit import load_object_model_document
from object_apriltag.training_data import (
    DATASET_SPLITS,
    LABELED_IMAGES_ALL_SAMPLES,
    generate_dataset_from_source,
    load_required_yolo_landmarks,
    require_positive_sample_rate,
)


def main() -> None:
    """Run the YOLO pose dataset generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate YOLO pose training samples from fused object pose and CAD geometry.",
        epilog=(
            "Each Dataset Generation Run belongs wholly to one Dataset Split (train or val).\n"
            "Video sources run headlessly to EOF; camera sources show a preview and stop on q.\n"
            "Saved images are raw full-resolution JPEGs; preview rendering never affects outputs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=parse_frame_source,
        required=True,
        help="Frame source: camera device index (e.g. 0) or path to a video file.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="Camera intrinsics JSON. Image width and height are read from this file.",
    )
    parser.add_argument(
        "--marker-model",
        type=Path,
        required=True,
        help="Marker model JSON used for fused object pose.",
    )
    parser.add_argument(
        "--cad-model",
        type=Path,
        required=True,
        help="CAD GLB path supplying mesh silhouette and fixed YOLO landmarks.",
    )
    parser.add_argument(
        "--object-model",
        type=Path,
        help=(
            "Object model JSON. Required when sibling cad_registration.json is absent "
            "so registration can be fitted in memory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Training Dataset root directory (images/, labels/, data.yaml, runs/).",
    )
    parser.add_argument(
        "--split",
        choices=sorted(DATASET_SPLITS),
        required=True,
        help="Dataset Split for this run: train or val.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique Dataset Generation Run name used in sample filenames and runs/<run-name>.json.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        required=True,
        help="Save the first Accepted Frame once this interval has elapsed since the previous save.",
    )
    parser.add_argument(
        "--dictionary",
        required=True,
        help="AprilTag dictionary name (e.g. 36h11, 25h9).",
    )
    parser.add_argument(
        "--detection-sensitivity",
        choices=("default", "relaxed", "aggressive"),
        required=True,
        help="AprilTag detector preset: default, relaxed, or aggressive.",
    )
    parser.add_argument(
        "--labeled-images",
        nargs="?",
        type=int,
        const=LABELED_IMAGES_ALL_SAMPLES,
        help=(
            "Write annotated JPEG previews under labeled-images/<split>/. "
            "Omit the numeric argument to label every saved sample; pass N to label "
            "only the first N saved samples."
        ),
    )
    args = parser.parse_args()

    require_positive_sample_rate(args.sample_rate_hz)
    if (
        args.labeled_images is not None
        and args.labeled_images != LABELED_IMAGES_ALL_SAMPLES
        and args.labeled_images <= 0
    ):
        raise RuntimeError("--labeled-images must be a positive integer when a limit is given.")
    if not args.calibration.exists():
        raise RuntimeError(f"Calibration file not found: {args.calibration}")
    if not args.marker_model.exists():
        raise RuntimeError(f"Marker model file not found: {args.marker_model}")
    if not args.cad_model.exists():
        raise RuntimeError(f"CAD model file not found: {args.cad_model}")
    if args.object_model is not None and not args.object_model.exists():
        raise RuntimeError(f"Object model file not found: {args.object_model}")

    cad_registration_path = args.cad_model.with_name("cad_registration.json")
    if not cad_registration_path.exists() and args.object_model is None:
        raise RuntimeError(
            f"CAD registration file not found: {cad_registration_path}. "
            "Pass --object-model to fit registration in memory from matching named landmarks."
        )

    camera_matrix, dist_coeffs, image_width, image_height, calibration_source = load_intrinsics(
        args.calibration
    )
    width, height = require_calibration_image_size(image_width, image_height, args.calibration)
    detector = ObjectDetector(
        camera_matrix,
        dist_coeffs,
        marker_model=args.marker_model,
        dictionary=args.dictionary,
        sensitivity=args.detection_sensitivity,
    )
    cad_model = load_cad_model(args.cad_model)
    cad_landmarks = load_required_yolo_landmarks(args.cad_model)
    if cad_registration_path.exists():
        cad_registration = load_cad_registration(cad_registration_path)
    else:
        from object_apriltag.evaluation.cad_geometry import fit_cad_registration

        _, object_model_document = load_object_model_document(args.object_model)
        try:
            cad_registration = fit_cad_registration(
                cad_landmarks,
                object_model_document,
                detector.marker_model,
            )
        except ValueError as error:
            raise RuntimeError(
                f"Cannot generate CAD registration from {args.cad_model} and "
                f"{args.object_model}: {error}"
            ) from error

    print(f"Using marker model: {args.marker_model} ({len(detector.marker_model.marker_ids)} markers)")
    print(f"Using CAD model: {args.cad_model}")
    if cad_registration_path.exists():
        print(f"Using CAD registration: {cad_registration_path}")
    else:
        print(f"Fitted CAD registration in memory from {args.object_model}")
    print(f"Using camera calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")
    print(f"Dataset output: {args.output}")
    print(f"Split: {args.split}")
    print(f"Run name: {args.run_name}")
    print(f"Sample rate: {args.sample_rate_hz:g} Hz")
    if args.labeled_images is not None:
        if args.labeled_images == LABELED_IMAGES_ALL_SAMPLES:
            print("Labeled previews: all saved samples")
        else:
            print(f"Labeled previews: first {args.labeled_images} saved samples")
    print(format_frame_source(args.source))
    if is_camera_source(args.source):
        print(f"Camera preview: target {width}x{height}; press q to stop.")
    else:
        print("Video capture: headless, runs once to EOF.")

    report = generate_dataset_from_source(
        source=args.source,
        output_dir=args.output,
        split=args.split,
        run_name=args.run_name,
        sample_rate_hz=args.sample_rate_hz,
        detector=detector,
        cad_landmarks=cad_landmarks.landmarks,
        cad_model=cad_model,
        registration=cad_registration,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=width,
        image_height=height,
        inputs={
            "source": str(args.source),
            "calibration": str(args.calibration),
            "marker_model": str(args.marker_model),
            "cad_model": str(args.cad_model),
            "cad_registration": str(cad_registration_path) if cad_registration_path.exists() else None,
            "object_model": str(args.object_model) if args.object_model is not None else None,
            "dictionary": args.dictionary,
            "detection_sensitivity": args.detection_sensitivity,
            "labeled_images": args.labeled_images,
        },
        show_preview=is_camera_source(args.source),
        labeled_images_limit=args.labeled_images,
    )

    print(
        "Run complete: "
        f"frames={report.frames_processed} "
        f"saved={report.samples_saved} "
        f"rejected_no_pose={report.rejections.no_pose} "
        f"rejected_landmarks={report.rejections.landmarks} "
        f"rejected_bbox={report.rejections.bbox}"
    )
    print(f"Run report: {args.output / 'runs' / f'{args.run_name}.json'}")


if __name__ == "__main__":
    main()
