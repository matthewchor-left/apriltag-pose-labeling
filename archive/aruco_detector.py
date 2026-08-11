"""ARCHIVED: legacy ArUco object detector. Use apriltag_detector.py instead."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration_io import load_intrinsics
from marker_origins import marker_origin_image_coords, marker_origin_image_point
from object_marker import KEYPOINT_NAMES, OBJECT_POINTS, estimate_extrinsics
from plot_utils import LiveHud, draw_live_hud, make_side_by_side, render_pose_plots

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
}

SKELETON_EDGES = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("bottom", "handle"),
    ("bottom", "top"),
    ("left", "right"),
)

KEYPOINT_COLORS_BGR = {
    "top": (128, 0, 128),
    "left": (255, 255, 0),
    "bottom": (0, 255, 0),
    "handle": (203, 192, 255),
    "right": (165, 42, 42),
}

# Object frame (+Y toward handle) -> marker frame (+Y up on the sticker).
RACKET_TO_MARKER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

FRONT_MARKER_ID = 0
BACK_MARKER_ID = 1
DEFAULT_AXIS_LIMITS = (-0.5, 0.5, -0.5, 0.5, 0.0, 2.0)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration" / "camera_calibration.json"


def object_points_for_marker(marker_id: int, back_marker_id: int = BACK_MARKER_ID) -> np.ndarray:
    points = OBJECT_POINTS.copy()
    if marker_id == back_marker_id:
        points[:, 0] *= -1.0
    return points


def estimate_world_points_from_keypoints(
    image_points: np.ndarray,
    marker_id: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    back_marker_id: int = BACK_MARKER_ID,
) -> tuple[dict[str, list[float]], float] | None:
    try:
        _, _, world_points, reproj_error, _ = estimate_extrinsics(
            image_points,
            camera_matrix,
            dist_coeffs,
            object_points=object_points_for_marker(marker_id, back_marker_id),
        )
    except (RuntimeError, ValueError):
        return None

    return {
        name: point.tolist()
        for name, point in zip(KEYPOINT_NAMES, world_points, strict=True)
    }, reproj_error


def build_detector_parameters(sensitivity: str = "relaxed") -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()

    if sensitivity == "default":
        return params

    if sensitivity == "relaxed":
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshConstant = 5.0
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.02
        params.minDistanceToBorder = 1
        params.minMarkerDistanceRate = 0.02
        params.minOtsuStdDev = 1.0
        params.errorCorrectionRate = 0.75
        params.maxErroneousBitsInBorderRate = 0.45
        params.detectInvertedMarker = True
        return params

    if sensitivity == "aggressive":
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 3.0
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.08
        params.minCornerDistanceRate = 0.01
        params.minDistanceToBorder = 0
        params.minMarkerDistanceRate = 0.01
        params.minOtsuStdDev = 0.5
        params.errorCorrectionRate = 0.85
        params.maxErroneousBitsInBorderRate = 0.55
        params.detectInvertedMarker = True
        params.useAruco3Detection = True
        params.minSideLengthCanonicalImg = 15
        params.minMarkerLengthRatioOriginalImg = 0.0
        return params

    raise ValueError(f"Unknown detection sensitivity {sensitivity!r}. Use default, relaxed, or aggressive.")


def marker_plane_corners(marker_size_m: float) -> np.ndarray:
    half = marker_size_m / 2.0
    return np.array(
        [
            [-half, marker_size_m],
            [half, marker_size_m],
            [half, 0.0],
            [-half, 0.0],
        ],
        dtype=np.float32,
    )


def racket_points_in_marker_frame(marker_id: int, back_marker_id: int = BACK_MARKER_ID) -> np.ndarray:
    points = OBJECT_POINTS.copy()
    if marker_id == back_marker_id:
        points[:, 0] *= -1.0
    return (RACKET_TO_MARKER @ points.T).T


def estimate_racket_keypoints(
    corners: np.ndarray,
    marker_size_m: float,
    marker_id: int = FRONT_MARKER_ID,
    back_marker_id: int = BACK_MARKER_ID,
) -> np.ndarray | None:
    image_corners = corners.reshape(4, 2).astype(np.float32)
    plane_corners = marker_plane_corners(marker_size_m)
    homography = cv2.getPerspectiveTransform(plane_corners, image_corners)
    racket_plane = (
        racket_points_in_marker_frame(marker_id, back_marker_id)[:, :2]
        .astype(np.float32)
        .reshape(-1, 1, 2)
    )
    image_points = cv2.perspectiveTransform(racket_plane, homography)
    return image_points.reshape(-1, 2)


def keypoints_are_valid(
    keypoints: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    if keypoints.shape != (len(KEYPOINT_NAMES), 2):
        return False
    if not np.all(np.isfinite(keypoints)):
        return False
    return bool(
        np.all((keypoints[:, 0] >= 0.0) & (keypoints[:, 0] < image_width))
        and np.all((keypoints[:, 1] >= 0.0) & (keypoints[:, 1] < image_height))
    )


def draw_racket_keypoints(frame: np.ndarray, image_points: np.ndarray) -> None:
    points_by_name = {name: image_points[index] for index, name in enumerate(KEYPOINT_NAMES)}

    for start_name, end_name in SKELETON_EDGES:
        start = points_by_name[start_name]
        end = points_by_name[end_name]
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
            continue
        cv2.line(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    for index, name in enumerate(KEYPOINT_NAMES):
        x, y = image_points[index]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        point = (int(round(x)), int(round(y)))
        color = KEYPOINT_COLORS_BGR[name]
        cv2.circle(frame, point, 7, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, point, 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            name,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            name,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def marker_bottom_center(corners: np.ndarray) -> tuple[int, int]:
    """Bottom-center for marker id 0; kept for compatibility."""
    pts = corners.reshape(4, 2)
    bottom_mid = (pts[2] + pts[3]) / 2.0
    return int(round(bottom_mid[0])), int(round(bottom_mid[1]))


def marker_origin_point(
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float,
) -> np.ndarray:
    return marker_origin_image_point(corners, marker_id, marker_size_m)


def marker_origin(
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float,
) -> tuple[int, int]:
    return marker_origin_image_coords(corners, marker_id, marker_size_m)


def draw_marker_annotations(
    frame: np.ndarray,
    corners: np.ndarray,
    marker_id: int,
    marker_size_m: float = 0.04,
    back_marker_id: int = BACK_MARKER_ID,
) -> np.ndarray | None:
    pts = corners.reshape(4, 2).astype(np.int32)
    ox, oy = marker_origin(corners, marker_id, marker_size_m)

    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

    top_mid = ((pts[0] + pts[1]) // 2).astype(int)
    x_arrow_target = tuple(pts[3]) if marker_id == back_marker_id else tuple(pts[2])
    cv2.arrowedLine(frame, (ox, oy), tuple(top_mid), (0, 0, 255), 2, tipLength=0.25)
    cv2.arrowedLine(frame, (ox, oy), x_arrow_target, (0, 255, 0), 2, tipLength=0.25)

    cv2.circle(frame, (ox, oy), 5, (255, 255, 255), -1)
    cv2.circle(frame, (ox, oy), 5, (0, 0, 0), 1)

    side_label = "back" if marker_id == back_marker_id else "front"
    label = f"id={marker_id} ({side_label})"
    cv2.putText(frame, label, (ox + 8, oy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (ox + 8, oy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    racket_image_points = estimate_racket_keypoints(
        corners,
        marker_size_m,
        marker_id,
        back_marker_id,
    )
    if racket_image_points is not None:
        draw_racket_keypoints(frame, racket_image_points)

    return racket_image_points


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect ArUco marker on an object, estimate 3D pose, and show live plots.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Camera calibration JSON from charuco_calibrate.py.",
    )
    parser.add_argument(
        "--detection-sensitivity",
        choices=("default", "relaxed", "aggressive"),
        default="relaxed",
    )
    parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTS), default="4x4_50")
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.1,
        help="Physical marker side length in meters.",
    )
    parser.add_argument(
        "--plot-width",
        type=int,
        default=540,
        help="Width in pixels for the rendered plot panel.",
    )
    parser.add_argument(
        "--back-marker-id",
        type=int,
        default=BACK_MARKER_ID,
        help=f"ArUco id on the object back (default: {BACK_MARKER_ID}).",
    )
    parser.add_argument(
        "--marker-id",
        type=int,
        default=None,
        help="Only use this ArUco marker id (default: prefer front marker id 0).",
    )
    args = parser.parse_args()

    if not args.calibration.exists():
        raise RuntimeError(
            f"Calibration file not found: {args.calibration}\n"
            "Run charuco_calibrate.py first to create camera_calibration.json."
        )

    camera_matrix, dist_coeffs, _, _, calibration_source = load_intrinsics(args.calibration)
    print(f"Using camera calibration: {args.calibration}")
    if calibration_source:
        print(f"Calibration source: {calibration_source}")

    plot_figsize = (args.plot_width / 50.0, args.plot_width / 100.0)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dictionary])
    detector = cv2.aruco.ArucoDetector(aruco_dict, build_detector_parameters(args.detection_sensitivity))
    print(f"ArUco detection sensitivity: {args.detection_sensitivity}")
    print("Press q to quit.")

    hud = LiveHud()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_height, frame_width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        marker_keypoints: dict[int, np.ndarray] = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
                marker_id = int(marker_id)
                if args.marker_id is not None and marker_id != args.marker_id:
                    continue

                racket_image_points = draw_marker_annotations(
                    frame,
                    marker_corners,
                    marker_id,
                    args.marker_size,
                    args.back_marker_id,
                )
                if (
                    racket_image_points is not None
                    and keypoints_are_valid(racket_image_points, frame_width, frame_height)
                ):
                    marker_keypoints[marker_id] = racket_image_points

                if args.marker_id is not None:
                    break

        active_keypoints = None
        active_marker_id = None
        if marker_keypoints:
            if args.marker_id is not None:
                active_marker_id = args.marker_id
                active_keypoints = marker_keypoints.get(args.marker_id)
            elif FRONT_MARKER_ID in marker_keypoints:
                active_marker_id = FRONT_MARKER_ID
                active_keypoints = marker_keypoints[FRONT_MARKER_ID]
            else:
                active_marker_id = next(iter(marker_keypoints.keys()))
                active_keypoints = marker_keypoints[active_marker_id]

        world_points: dict[str, list[float]] = {}
        reproj_error: float | None = None
        if active_keypoints is not None and active_marker_id is not None:
            pose_result = estimate_world_points_from_keypoints(
                active_keypoints,
                active_marker_id,
                camera_matrix,
                dist_coeffs,
                args.back_marker_id,
            )
            if pose_result is not None:
                world_points, reproj_error = pose_result

        fps, avg_reproj_error = hud.tick(reproj_error)
        draw_live_hud(frame, fps, avg_reproj_error)

        plot_bgr = render_pose_plots(world_points, DEFAULT_AXIS_LIMITS, figsize=plot_figsize)
        display_frame = make_side_by_side(frame, plot_bgr, frame_height)
        cv2.imshow("Object ArUco Detector", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
