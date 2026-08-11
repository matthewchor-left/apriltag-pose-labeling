# Object ArUco Tracker

AprilTag-based table tennis object pose estimation.

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

## 1. Calibrate camera (ChArUco or checkerboard)

```bash
uv run object-charuco \
  --camera 0 \
  --layout 7 10 \
  --marker-size 0.018 \
  --output config/Camera/nexplaygroundcam/intrinsics.json
```

ChArUco is the default board type (`--board-type charuco_board`). For a plain checkerboard:

```bash
uv run object-charuco \
  --board-type checkerboard \
  --camera 0 \
  --layout 6 9 \
  --output config/Camera/nexplaygroundcam/intrinsics.json
```

Save a printable ChArUco pattern (true scale on A4, 300 DPI):

```bash
uv run object-charuco \
  --layout 6 9 \
  --marker-size 0.02 \
  --square-size 0.025 \
  --save-board config/charuco_6x9_25mm.png
```

Print the PNG at **100% scale** (not “fit to page”) so each square matches `--square-size`.

- **Space** — capture a frame when the board is fully detected
- **q** — finish and save to `--output`

`--layout` is height × width (rows × columns). For ChArUco that is chess **square** count; for checkerboard it is **inner corner** count.

## 2. Detect object pose

```bash
uv run object-detect \
  --camera 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --marker-model config/Model/object_01/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed
```

Press **q** to quit.

## Core API (no visualization)

```python
import cv2
from object_apriltag import ObjectDetector
from object_apriltag.calibration import load_intrinsics

camera_matrix, dist_coeffs, _, _, _ = load_intrinsics("config/Camera/nexplaygroundcam/intrinsics.json")
detector = ObjectDetector(
    camera_matrix,
    dist_coeffs,
    marker_model="config/Model/object_01/marker_model.json",
)

cap = cv2.VideoCapture(2)
ok, frame = cap.read()
pose = detector.detect(frame)  # ObjectPose(origin, rotation) or None
```

`ObjectPose.origin` and `ObjectPose.rotation` are in the **camera frame**.

`ObjectPose.rotation` maps object-frame vectors to the camera frame using:

- **+X** `[1, 0, 0]` — left → right  
- **+Y** `[0, -1, 0]` — handle → tip (the matrix’s second column points tip → handle)  
- **+Z** `[0, 0, 1]` — out of the rubber  

Example: `pose.rotation @ np.array([0.0, -1.0, 0.0])` is the handle→tip direction in the camera frame.

## Marker model coordinates

`config/Model/<object>/marker_model.json` stores sticker corner positions in a **object-fixed model frame**, not in camera coordinates. The frame stays attached to the object regardless of where the camera is.

| Axis | Direction |
|------|-----------|
| **+X** | Left → right across the object |
| **+Y** | Handle → blade tip |
| **+Z** | Out of the rubber (normal to the striking surface) |

Marker 0 (front rubber) sits at **z = 0** on that reference plane. The back marker (id 1) is at **z ≈ −0.073** because it lies on the opposite side of the rubber: you move **into** the object, opposite **+Z**. Side and edge markers use negative **z** where they wrap around the blade.

When the **rubber faces the camera**, **+Z** points toward the camera and **−Z** points through the blade toward the back — so negative **z** in the model file corresponds to points that are farther from the camera. That is expected; it does not mean model **z** is defined as camera depth.

```
        camera
          │
          ▼
    ═════════════  z = 0   (marker 0, rubber)
    │   object    │
    ═════════════  z ≈ −0.073   (marker 1, back)
```

If the object is viewed from the back or from the side, the mapping between model **±Z** and closer/farther relative to the camera changes because the object rotated — the model frame itself does not change.

`ObjectDetector` fuses marker poses into **camera-frame** `origin` and `rotation`. Model coordinates are converted to the camera only through the detected pose and per-marker transforms in `layout.py`.

## Optional visualization

The `viz` extra adds overlays, skeleton keypoints, and matplotlib plots:

```bash
uv sync --extra viz
uv run object-detect \
  --camera 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --marker-model config/Model/object_01/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --plot-graph \
  --object-model config/Model/object_01/object_model.json
uv run object-calibrate-marker-model --marker-model config/Model/object_01/marker_model.json --visualize
```

## Development

```bash
uv sync --extra viz
uv run python -m unittest discover -s tests
```

After changing dependencies in `pyproject.toml`, run `uv lock` to refresh `uv.lock`.

## Project layout

```
src/object_apriltag/
  detector.py          # ObjectDetector — frame in, pose out
  pose.py              # marker PnP + multi-marker fusion
  layout.py            # marker model JSON + transforms
  calibration.py       # intrinsics loader + config profile paths
  viz/                 # optional overlays and plots
  cli/                 # object-detect, object-charuco, object-calibrate-marker-model
config/
  Camera/
    nexplaygroundcam/  # intrinsics.json, uvcc.json, device.json
    cam1/ cam2/ webcam/  # empty profiles (.gitkeep)
  Model/
    object_01/         # marker_model.json, eraser_model.json, object_model.json
    object_02/         # annotation object marker + eraser
pyproject.toml         # project metadata and dependencies
uv.lock                # locked dependency versions (commit this)
```

## Common options

| Command | Flag | Description |
|---------|------|-------------|
| `object-charuco` | `--board-type` | `charuco_board` or `checkerboard` |
| `object-charuco` | `--layout` | Board height × width (rows × columns) |
| `object-charuco` | `--marker-size` | ArUco marker size in meters (ChArUco only) |
| `object-charuco` | `--camera` | Camera device index |
| `object-charuco` | `--output` | Intrinsics JSON path |
| `object-detect` | `--calibration` | Path to camera intrinsics JSON |
| `object-detect` | `--marker-model` | Marker model JSON path |
| `object-detect` | `--marker-id` | Use a specific marker id only |
| `object-detect` | `--no-visualize` | Camera preview without overlays |
