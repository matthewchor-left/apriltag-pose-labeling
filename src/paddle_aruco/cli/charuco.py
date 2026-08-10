"""ChArUco board camera calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from paddle_aruco.calibration import DEFAULT_CALIBRATION_PATH, save_calibration_json

ARUCO_DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
}


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}.")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return capture


def build_charuco_board(
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_size: float,
    dictionary_name: str,
) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.CharucoDetector, np.ndarray]:
    if dictionary_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown dictionary {dictionary_name!r}.")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    chessboard_corners_3d = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return board, detector, chessboard_corners_3d


def charuco_object_image_points(
    charuco_corners: np.ndarray,
    charuco_ids: np.ndarray,
    chessboard_corners_3d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corner_ids = charuco_ids.flatten()
    obj_pts = chessboard_corners_3d[corner_ids].reshape(-1, 1, 3)
    img_pts = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    return obj_pts, img_pts


def normalize_charuco_detection(
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if charuco_corners is None or charuco_ids is None:
        return None
    corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    if corners.size == 0 or ids.size == 0 or corners.shape[0] != ids.shape[0]:
        return None
    return corners, ids


def draw_charuco_detection(
    frame: np.ndarray,
    charuco_corners: np.ndarray | None,
    charuco_ids: np.ndarray | None,
    marker_corners: list[np.ndarray] | None,
    marker_ids: np.ndarray | None,
) -> None:
    if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    detection = normalize_charuco_detection(charuco_corners, charuco_ids)
    if detection is None:
        return

    corners, ids = detection
    try:
        cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
    except cv2.error:
        for corner, corner_id in zip(corners, ids, strict=True):
            point = (int(round(corner[0])), int(round(corner[1])))
            cv2.circle(frame, point, 4, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                frame, f"id={int(corner_id)}", (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate camera with a ChArUco board.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--square-size", type=float, default=0.024)
    parser.add_argument("--marker-size", type=float, default=None)
    parser.add_argument("--squares-x", type=int, default=10)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTS), default="4x4_50")
    parser.add_argument("--min-corners", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--save-board", type=Path, default=None)
    args = parser.parse_args()

    marker_size = args.marker_size if args.marker_size is not None else args.square_size * 0.75
    board, charuco_detector, chessboard_corners_3d = build_charuco_board(
        args.squares_x, args.squares_y, args.square_size, marker_size, args.dictionary,
    )
    inner_corners = board.getChessboardSize()
    print(
        f"ChArUco board: {args.squares_x}x{args.squares_y} squares, "
        f"{inner_corners[0]}x{inner_corners[1]} inner corners, "
        f"square={args.square_size:.4f} m, marker={marker_size:.4f} m"
    )

    if args.save_board is not None:
        board_image = board.generateImage((args.squares_x * 200, args.squares_y * 200))
        args.save_board.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_board), board_image)
        print(f"Saved printable board image to {args.save_board}")

    cap = open_camera(args.camera, args.width, args.height)
    print(f"Camera {args.camera}: {args.width}x{args.height}")
    print("Show the ChArUco board from varied angles. Space: capture, q: finish.")

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size = None
    captured_frames = 0
    capture_flash_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
        detection = normalize_charuco_detection(charuco_corners, charuco_ids)
        enough_corners = detection is not None and detection[1].size >= args.min_corners

        draw_charuco_detection(frame, charuco_corners, charuco_ids, marker_corners, marker_ids)
        status = f"Captured frames: {captured_frames}"
        if detection is not None:
            status += f" | corners: {detection[1].size}"
        status += " | Space: capture"
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        if capture_flash_frames > 0:
            cv2.putText(frame, "Captured!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
            capture_flash_frames -= 1
        cv2.imshow("ChArUco Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if enough_corners and detection is not None:
                corners, ids = detection
                obj_pts, img_pts = charuco_object_image_points(corners, ids, chessboard_corners_3d)
                objpoints.append(obj_pts)
                imgpoints.append(img_pts)
                captured_frames += 1
                capture_flash_frames = 15
            else:
                corner_count = 0 if detection is None else int(detection[1].size)
                print(f"Skipped capture: need at least {args.min_corners} corners, got {corner_count}.")

    cap.release()
    cv2.destroyAllWindows()

    if not objpoints:
        raise RuntimeError("No ChArUco detections collected.")

    reproj_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
    )
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print(f"Mean reprojection error: {reproj_error:.4f} px")
    print(f"Frames used: {len(objpoints)}")

    save_calibration_json(
        args.output,
        {
            "calibration_source": "charuco",
            "image_size": [image_size[0], image_size[1]],
            "image_width": image_size[0],
            "image_height": image_size[1],
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
            "mean_reprojection_error_px": float(reproj_error),
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "inner_corners_x": inner_corners[0],
            "inner_corners_y": inner_corners[1],
            "square_size": args.square_size,
            "marker_size": marker_size,
            "dictionary": args.dictionary,
            "captured_frames": len(objpoints),
        },
    )
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
