# Paddle ArUco Tracker

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

## 1. Calibrate camera (ChArUco board)

```bash
uv run paddle-charuco --save-board calibration/charuco_board.png
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

`PaddlePose.rotation` maps paddle-frame vectors to the camera frame using:

- **+X** `[1, 0, 0]` — left → right  
- **+Y** `[0, -1, 0]` — handle → tip (the matrix’s second column points tip → handle)  
- **+Z** `[0, 0, 1]` — out of the rubber  

Example: `pose.rotation @ np.array([0.0, -1.0, 0.0])` is the handle→tip direction in the camera frame.

## Marker layout coordinates

`calibration/marker_layout.json` stores sticker corner positions in a **paddle-fixed layout frame**, not in camera coordinates. The frame stays attached to the paddle regardless of where the camera is.

| Axis | Direction |
|------|-----------|
| **+X** | Left → right across the paddle |
| **+Y** | Handle → blade tip |
| **+Z** | Out of the rubber (normal to the striking surface) |

Marker 0 (front rubber) sits at **z = 0** on that reference plane. The back marker (id 1) is at **z ≈ −0.073** because it lies on the opposite side of the rubber: you move **into** the paddle, opposite **+Z**. Side and edge markers use negative **z** where they wrap around the blade.

When the **rubber faces the camera**, **+Z** points toward the camera and **−Z** points through the blade toward the back — so negative **z** in the layout file corresponds to points that are farther from the camera. That is expected; it does not mean layout **z** is defined as camera depth.

```
        camera
          │
          ▼
    ═════════════  z = 0   (marker 0, rubber)
    │   paddle    │
    ═════════════  z ≈ −0.073   (marker 1, back)
```

If the paddle is viewed from the back or from the side, the mapping between layout **±Z** and closer/farther relative to the camera changes because the paddle rotated — the layout frame itself does not change.

`PaddleDetector` fuses marker poses into **camera-frame** `origin` and `rotation`. Layout coordinates are converted to the camera only through the detected pose and per-marker transforms in `layout.py`.

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
  cli/                 # paddle-detect, paddle-charuco, paddle-calibrate-layout
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
| `paddle-charuco` | `--square-size` | Chess square size in meters (default 0.024) |
| `paddle-charuco` | `--output` | Calibration JSON path |
| `paddle-detect` | `--calibration` | Path to camera calibration JSON |
| `paddle-detect` | `--marker-layout` | Marker layout JSON path |
| `paddle-detect` | `--marker-id` | Use a specific marker id only |
| `paddle-detect` | `--no-visualize` | Camera preview without overlays |
