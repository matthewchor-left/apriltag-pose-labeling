# Paddle AprilTag Tracker

AprilTag-based table tennis paddle pose estimation.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python version is pinned in `.python-version`.

```bash
uv sync --extra viz
```

This creates `.venv`, installs locked dependencies from `uv.lock`, and registers the CLI commands.

Core only (no matplotlib):

```bash
uv sync
```

## 1. Calibrate camera (chessboard pattern)

```bash
uv run paddle-calibrate-camera --save-board calibration/calibration_board.png
```

- **Space** — capture a frame when enough corners are detected
- **q** — finish and save `calibration/camera_calibration.json`

## 2. Detect paddle pose

```bash
uv run paddle-detect --camera 2 --calibration calibration/camera_calibration.json --visualize
```

Press **q** to quit.

## Core API (no visualization)

```python
import cv2
from paddle_apriltag import PaddleDetector
from paddle_apriltag.calibration import load_intrinsics

camera_matrix, dist_coeffs, _, _, _ = load_intrinsics("calibration/camera_calibration.json")
detector = PaddleDetector(
    camera_matrix,
    dist_coeffs,
    marker_layout="calibration/marker_layout.json",
)

cap = cv2.VideoCapture(2)
ok, frame = cap.read()
pose = detector.detect(frame)  # PaddlePose(origin, rotation) or None
```

`PaddlePose.origin` and `PaddlePose.rotation` are in the **camera frame**.

## Coordinate frames

### `marker_layout.json` (camera-aligned when marker 0 faces the camera)

Corner positions in `calibration/marker_layout.json` use a paddle-fixed frame that **matches the OpenCV camera axes when marker 0 (front rubber) faces the camera**:

| Axis | Direction |
|------|-----------|
| **+X** | Right in the image |
| **+Y** | Down in the image |
| **+Z** | Into the scene (away from the camera) |

Marker 0 (front rubber) sits on the **z = 0** plane. The back marker and edge markers use **positive Z** because they are farther from the camera along the optical axis.

```
        camera
          │
          ▼
    ═════════════  z = 0   (marker 0, rubber)
    │   paddle  │
    ═════════════  z ≈ +0.07  (marker 1, back)
```

When the paddle rotates away from this view, layout coordinates no longer line up with the live camera frame — only this reference pose is aligned for easier debugging.

### Detector output (camera frame)

`PaddleDetector` returns pose in the **camera frame** (same OpenCV convention as above). Layout coordinates are transformed into this frame at runtime via the detected marker poses and layout transforms.

## Optional visualization

The `viz` extra adds overlays, skeleton keypoints, and matplotlib plots:

```bash
uv sync --extra viz
uv run paddle-detect --visualize --plot-graph
uv run paddle-calibrate-layout --visualize
```

## Development

```bash
uv sync --extra viz
uv run python -m unittest discover -s tests
```

After changing dependencies in `pyproject.toml`, run `uv lock` to refresh `uv.lock`.

## Project layout

```
src/paddle_apriltag/
  detector.py          # PaddleDetector — frame in, pose out
  pose.py              # marker PnP + multi-marker fusion
  layout.py            # marker layout JSON + transforms
  calibration.py       # camera intrinsics loader
  viz/                 # optional overlays and plots
  cli/                 # paddle-detect, paddle-calibrate-camera, paddle-calibrate-layout
calibration/           # single source of truth for all JSON config
  camera_calibration.json
  marker_layout.json
  paddle_model.json
pyproject.toml         # project metadata and dependencies
uv.lock                # locked dependency versions (commit this)
```

## Common options

| Command | Flag | Description |
|---------|------|-------------|
| `paddle-calibrate-camera` | `--square-size` | Chess square size in meters (default 0.024) |
| `paddle-calibrate-camera` | `--output` | Calibration JSON path |
| `paddle-detect` | `--calibration` | Path to camera calibration JSON |
| `paddle-detect` | `--marker-layout` | Marker layout JSON path |
| `paddle-detect` | `--marker-id` | Use a specific marker id only |
| `paddle-detect` | `--no-visualize` | Camera preview without overlays |
