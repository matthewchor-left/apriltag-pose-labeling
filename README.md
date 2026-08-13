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

Or load geometry from a Board Model profile (`height` × `width` is **h6 × w9** — rows × columns):

```bash
uv run object-charuco \
  --board-model config/Board/charuco_h6_w9_25mm_4x4_50/board_model.json \
  --save-board config/charuco_6x9_25mm.png
```

`--board-model` cannot be combined with `--layout`, `--marker-size`, `--square-size`, or `--dictionary`.

Print the PNG at **100% scale** (not “fit to page”) so each square matches `--square-size`.

- **Space** — capture a frame when the board is fully detected
- **q** — finish and save to `--output`

`--layout` is height × width (rows × columns). For ChArUco that is chess **square** count; for checkerboard it is **inner corner** count.

## Board Model and Board Reference Frame

Board geometry and the canonical **Board Reference Frame** live in `config/Board/<profile>/board_model.json`. The default profile `charuco_h6_w9_25mm_4x4_50` is a 6-row × 9-column ChArUco board with 25 mm squares and 20 mm markers (`4x4_50` dictionary). Profile names use **h6/w9** so height and width are not confused.

`object-visualize-board-frame` detects ChArUco intersections, estimates a **Board Pose Estimate** in the camera frame, and overlays the Board Reference Frame (axes and extended grid). It does not yet implement object detection error analysis.

Still image:

```bash
uv run object-visualize-board-frame \
  --calibration config/Camera/webcam/intrinsics.json \
  --board-model config/Board/charuco_h6_w9_25mm_4x4_50/board_model.json \
  --image path/to/frame.png \
  --output path/to/overlay.png
```

Live camera:

```bash
uv run object-visualize-board-frame \
  --calibration config/Camera/webcam/intrinsics.json \
  --board-model config/Board/charuco_h6_w9_25mm_4x4_50/board_model.json \
  --camera 0 \
  --output path/to/saved_frame.png
```

- **q** — quit live mode
- **S** — save current rendered frame when `--output` is set (live mode only)
- Frame resolution must match the calibration `image_size`; intrinsics are not scaled

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

### Board coordinate overlays

Both `object-detect` and `annotation-tool` can track a printed ChArUco board and annotate the active pose-projection overlay with **Board Coordinates**:

```bash
uv run object-detect \
  --camera 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --marker-model config/Model/remote1/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --overlay-object-model \
  --object-model config/Model/remote1/object_model.json \
  --board-frame \
  --board-model config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json \
  --camera-motion static
```

```bash
uv run annotation-tool \
  --camera 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --marker-model config/Model/remote1/marker_model.json \
  --eraser-model config/Model/remote1/eraser_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --overlay-object-model \
  --object-model config/Model/remote1/object_model.json \
  --board-frame \
  --board-model config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json
```

| Flag | Description |
|------|-------------|
| `--board-frame` | Enable board tracking, XZ grid, XYZ axes, and Board Coordinate labels |
| `--board-model` | ChArUco board model JSON (required with `--board-frame`) |
| `--camera-motion` | `static` (default) keeps the last valid **Board Pose Estimate** when the board drops out; `dynamic` clears grid and labels until the board is visible again |

Board overlays use the same XZ grid and XYZ axes as `object-visualize-board-frame` (two-square margin and axis length by default). **Board Coordinate** labels appear only when both `--board-frame` and a pose-projection overlay are active (`--overlay-object-model`, `--overlay-marker-model`, or `--overlay-eraser-model` on `object-detect`; `--overlay-object-model` on `annotation-tool`). Labels use millimeters with one decimal place, e.g. `(12.3, 45.6, 78.9) mm`.

With `--preview`, `--board-frame`, and `--overlay-object-model` together, `object-detect` also enables **Interactive Object Model Capture**: press **e** to enter a Board Coordinate keypoint in the terminal (`keypoint-id x_mm y_mm z_mm`) and preview it as a magenta target, **s** to save to `--object-model`, **q** to quit when saved, and **x** to discard unsaved edits and quit.

On `object-detect`, `--no-visualize` disables detection and pose-projection overlays, including the board grid and labels; board pose is still solved each frame. During Interactive Object Model Capture, only the editing controls and status remain visible.

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

## Marker model calibration (live)

`object-calibrate-marker-model` estimates sticker footprint positions from live camera views. No tape measure or manual corner entry — move the object so expected markers co-appear while the tool samples at **2 Hz** (override with `--sample-rate-hz`).

```bash
uv run object-calibrate-marker-model \
  --camera 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --marker-size 0.07 \
  --marker-ids 0 1 2 3-10 11 \
  --reference-marker-id 0 \
  --output config/Model/remote1/marker_model.json
```

For layouts with many markers, pass `--anchor-marker-ids` to bootstrap from a small spatially diverse core (must include `--reference-marker-id`). The solver exhaustively resolves IPPE ambiguity only on anchors (`2^k` per frame), expands the remaining markers in strict hierarchical rounds, then runs the same corner bundle adjustment and quality gates. Omit the flag to keep full-set exhaustive assignment. Add `--anchor-stop-after-expansion` to write the expansion-only layout (skips full IPPE reassignment, bundle adjustment, and quality gates) for debugging.

```bash
  --anchor-marker-ids 0 1 4-7 \
  --anchor-stop-after-expansion \
```

- **S** — solve from captured samples; writes `--output` only when quality gates pass
- **Q** — quit without writing

A frame is recorded only when **at least two** expected marker IDs are visible at a sample tick. During solve, frames that cannot be assigned a consistent marker interpretation are rejected automatically; each marker pair used in the layout still needs at least **20** accepted co-visible frames (`--min-pair-inliers`) after rejection. Keep moving the object for diverse views — you do not need to capture pairs in isolated batches. The HUD shows expected/visible IDs, captured sample count, live pair readiness (raw co-visibility and pass/weak status), graph connectivity from the reference marker, and last-solve frame acceptance when you press **S**.

**Scale caveat:** metric layout depends on the physical `--marker-size` and calibrated intrinsics. Wrong marker size or scaled intrinsics will bias the solved geometry.

**Quality gates** (defaults; override on the CLI):

| Gate | Default |
|------|---------|
| Global reprojection RMS | 2 px (`--reprojection-rms-gate-px`) |
| Per-marker reprojection RMS | 2 px (same gate as global) |
| Pair translation RMS | 10% of marker size (`--pair-translation-rms-gate-ratio`) |
| Pair rotation RMS | 5° (`--pair-rotation-rms-gate-deg`) |

Refused solves print diagnostics and resume capture; nothing is written until a solve passes. Use `--force` to overwrite an existing `--output`. Frame resolution must match the calibration `image_size`; intrinsics are not scaled.

Inspect an existing model (terminal or static diagram):

```bash
uv run object-inspect-marker-model --marker-model config/Model/remote1/marker_model.json --visualize
```

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
  cli/                 # object-detect, object-charuco, object-calibrate-marker-model, object-inspect-marker-model
config/
  Board/
    charuco_h6_w9_25mm_4x4_50/  # board_model.json
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
| `object-charuco` | `--board-model` | ChArUco board model JSON (exclusive with geometry flags) |
| `object-charuco` | `--board-type` | `charuco_board` or `checkerboard` |
| `object-charuco` | `--layout` | Board height × width (rows × columns) |
| `object-charuco` | `--marker-size` | ArUco marker size in meters (ChArUco only) |
| `object-charuco` | `--camera` | Camera device index |
| `object-charuco` | `--output` | Intrinsics JSON path |
| `object-visualize-board-frame` | `--calibration` | Path to camera intrinsics JSON |
| `object-visualize-board-frame` | `--board-model` | ChArUco board model JSON path |
| `object-visualize-board-frame` | `--camera` / `--image` | Live camera index or still image |
| `object-visualize-board-frame` | `--grid-margin` | Grid extension in square counts (default 2) |
| `object-detect` | `--calibration` | Path to camera intrinsics JSON |
| `object-detect` | `--marker-model` | Marker model JSON path |
| `object-detect` | `--marker-id` | Use a specific marker id only |
| `object-detect` | `--no-visualize` | Camera preview without overlays |
| `object-detect` / `annotation-tool` | `--board-frame` | Board Reference Frame grid, axes, and coordinate labels |
| `object-detect` / `annotation-tool` | `--board-model` | ChArUco board model JSON (required with `--board-frame`) |
| `object-detect` / `annotation-tool` | `--camera-motion` | `static` (default) or `dynamic` board pose retention |
| `object-calibrate-marker-model` | `--camera` / `--calibration` | Live camera index and intrinsics JSON |
| `object-calibrate-marker-model` | `--marker-ids` / `--reference-marker-id` | Expected unique IDs and layout reference |
| `object-calibrate-marker-model` | `--anchor-marker-ids` | Optional bootstrap core for large layouts |
| `object-calibrate-marker-model` | `--anchor-stop-after-expansion` | Debug: write expansion-only layout (no BA/gates) |
| `object-calibrate-marker-model` | `--marker-size` | Physical tag edge length in meters |
| `object-calibrate-marker-model` | `--sample-rate-hz` | Co-visibility sample rate (default 10 Hz) |
| `object-calibrate-marker-model` | `--min-pair-inliers` | Minimum co-visible frames per pair (default 20) |
| `object-calibrate-marker-model` | `--output` / `--force` | Marker model JSON; refuse overwrite unless `--force` |
| `object-inspect-marker-model` | `--marker-model` / `--visualize` | Print or diagram an existing marker model |
