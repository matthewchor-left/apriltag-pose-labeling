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
  --source 0 \
  --layout 7 10 \
  --marker-size 0.018 \
  --output config/Camera/nexplaygroundcam/intrinsics.json
```

ChArUco is the default board type (`--board-type charuco_board`). For a plain checkerboard:

```bash
uv run object-charuco \
  --board-type checkerboard \
  --source 0 \
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

Live camera (`--source 0`):

- **Space** — capture a frame when the board is fully detected
- **q** — finish and save to `--output`

Video file `--source` captures automatically at `--sample-rate-hz` (default 10 Hz) on video time whenever the board is detected. There is no preview window; calibration runs at end of file.

```bash
uv run object-charuco \
  --source path/to/calibration.mov \
  --layout 7 10 \
  --marker-size 0.018 \
  --sample-rate-hz 10 \
  --output config/Camera/nexplaygroundcam/intrinsics.json
```

`--layout` is height × width (rows × columns). For ChArUco that is chess **square** count; for checkerboard it is **inner corner** count.

## Board Model

Board geometry and the **Board Reference Frame** convention live in `config/Board/<profile>/board_model.json`. The default profile `charuco_h6_w9_25mm_4x4_50` is a 6-row × 9-column ChArUco board with 25 mm squares and 20 mm markers (`4x4_50` dictionary). Profile names use **h6/w9** so height and width are not confused. Pass `--board-model` to `object-charuco` instead of individual geometry flags.

## 2. Detect object pose

```bash
uv run object-detect \
  --source 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --marker-model config/Model/object_01/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed
```

Press **q** to quit.

### CAD overlay

`object-detect` can render a GLB CAD mesh when fused pose is available. If
`cad_registration.json` exists next to `--cad-model`, it is loaded. Otherwise,
registration is fitted in memory from matching named, meshless GLB landmarks and
`--object-model` `keypoint_sources`, using the current `--marker-model`.

- `--overlay-cad-model` draws a semi-transparent mesh on the camera frame (requires `--visualize`). Additive with pose projection overlays.
- `--side2side-cad-model` opens a side-by-side preview: camera on the left, opaque CAD-only rendering on the right (black when pose is unavailable). Requires `--preview`. Still renders the CAD pane under `--no-visualize` (left pane stays raw).
- With `--plot-graph`, layout order is **camera | CAD | skeleton chart**.

```bash
uv run object-detect \
  --source 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --marker-model config/Model/remote1/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --overlay-object-model \
  --object-model config/Model/remote1/object_model.json \
  --overlay-cad-model \
  --side2side-cad-model \
  --cad-model config/Model/CAD/nexplayground_sim.glb
```

| Flag | Description |
|------|-------------|
| `--overlay-cad-model` | Semi-transparent `--cad-model` GLB on camera frame (requires `--visualize`) |
| `--side2side-cad-model` | Side-by-side opaque CAD pane (requires `--preview` and `--cad-model`) |
| `--cad-model` | GLB path; required with either CAD flag. Uses sibling `cad_registration.json`, or auto-fits from `--object-model` when absent |

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

`ObjectPose.rotation` maps **reference-marker-centered** model coordinates into the camera frame. Those axes match the model frame below when the reference marker faces the camera: **+X** right, **+Y** down, **+Z** into the scene. There is no separate object-pose axis flip — `marker_model.json`, `object_model.json`, and runtime pose all use the same convention.

## Marker model coordinates

`config/Model/<object>/marker_model.json` stores sticker corner positions in the **model frame** (also `coordinate_frame: "marker_model"` in `object_model.json`). The frame stays attached to the object regardless of where the camera is.

| Axis | Direction |
|------|-----------|
| **+X** | Right in the image when the reference marker faces the camera |
| **+Y** | Down in the image |
| **+Z** | Into the scene (away from the camera) |

Corner names (`top_left`, `top_right`, …) follow **OpenCV AprilTag detection order** (image-relative), not a physical left/right label on the object. When the sticker is viewed from another angle, the same corner name still refers to the same sticker corner.

When the **reference marker faces the camera**, **+Z** points away from the camera into the object. Markers on the far side of the object therefore have larger **+Z** values in the model file. That is expected; model **Z** is object-fixed depth, not camera depth.

```
        camera
          │
          ▼
    ═════════════  reference marker (near side, smaller +Z)
    │   object    │
    ═════════════  back markers (farther into the scene, larger +Z)
```

If the object is viewed from the back or from the side, the mapping between model **±Z** and closer/farther relative to the camera changes because the object rotated — the model frame itself does not change.

`ObjectDetector` estimates one **camera-frame** `origin` and `rotation` from all visible marker-model corners with a layout-wide RANSAC PnP solve (requires at least two markers). This rejects inconsistent corner detections instead of averaging independent marker poses.

Marker model JSON uses top-level `marker_size_m` as the default physical edge length (meters). Each `markers.<id>` entry may optionally include `size_m` when that sticker differs from the default; omitted `size_m` means the default. Calibration writes `size_m` only for non-default markers. Saved models record `anchor_marker_ids`: every marker in the layout when `--anchor-marker-ids` is omitted, or the explicit bootstrap core when it is provided. Footprint validation checks each marker against its resolved size.

## Marker model calibration (live)

`object-calibrate-marker-model` estimates sticker footprint positions from live camera views. No tape measure or manual corner entry — move the object so expected markers co-appear, inspect each view for sharp detections, and press **C** to capture it.

```bash
uv run object-calibrate-marker-model \
  --source 0 \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --marker-size 0.07 \
  --marker-size-for 4:0.03 10-12:0.025 \
  --marker-ids 0 1 2 3-10 11 \
  --reference-marker-id 0 \
  --output config/Model/remote1/marker_model.json
```

For layouts with many markers, pass `--anchor-marker-ids` to bootstrap from a small spatially diverse core (must include `--reference-marker-id`). The solver exhaustively resolves IPPE ambiguity only on anchors (`2^k` per frame), expands the remaining markers in strict hierarchical rounds, then runs the same corner bundle adjustment and quality gates. Omit the flag to keep full-set exhaustive assignment. Add `--anchor-stop-after-expansion` to write the expansion-only layout (skips full IPPE reassignment, bundle adjustment, and quality gates) for debugging.

```bash
  --anchor-marker-ids 0 1 4-7 \
  --anchor-stop-after-expansion \
```

- **C** — capture the current sharp frame when at least two expected markers are visible (default manual mode)
- **S** — solve from captured samples; writes `--output` only when quality gates pass
- **Q** — quit without writing

By default, frames are recorded only when you press **C** with **at least two** expected marker IDs visible. Pass **`--auto`** to capture periodically at **`--sample-rate-hz`** (default 10 Hz) under the same two-marker rule; **C** still captures an extra frame in automatic mode. Capture sharp, diverse viewpoints rather than repeated stationary views. During solve, frames that cannot be assigned a consistent marker interpretation are rejected automatically; each marker pair used in the layout still needs at least **20** accepted co-visible frames (`--min-pair-inliers`) after rejection. You do not need to capture pairs in isolated batches. The HUD shows expected/visible IDs, captured sample count, live pair readiness (raw co-visibility and pass/weak status), graph connectivity from the reference marker, and last-solve frame acceptance when you press **S**.

### Benchmark mode (video file)

Pass **`--benchmark`** with a video-file **`--source`** to measure offline calibration throughput without preview or HUD. The tool **decodes every frame** once to EOF; **`--benchmark-frame-selection`** (default **`uniform`**) controls which decoded frames run AprilTag detection:

- **`uniform`** — detect on scheduled video-time samples at **`--sample-rate-hz`** (default 10 Hz), using the video's reported frame rate and the same two-marker visibility rule as **`--auto`**.
- **`sharpest`** — still decode every frame to score relative sharpness (downsampled grayscale Laplacian variance), group frames into half-open windows of `1/--sample-rate-hz` seconds, and detect only the sharpest frame per window (earliest frame on ties); the final partial window is flushed at EOF.

**`--auto`** is unnecessary — benchmark mode implies automatic capture. **`--diagnostics-output`** is required; timings, frame counts, and environment metadata are written under a top-level **`benchmark`** object while existing quality diagnostics remain unchanged. Benchmark counts distinguish decoded frames, detector invocations, and skipped frames; marker visibility counts (`frames_with_expected_markers`, `covisible_frames`) apply only to detector-invoked frames. Sharpest mode adds `frame_selection` and `timing_seconds.sharpness_scoring` to the benchmark payload.

```bash
uv run object-calibrate-marker-model \
  --benchmark \
  --source data/calibration.mov \
  --calibration config/Camera/nexplaygroundcam/intrinsics.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --marker-size 0.07 \
  --marker-size-for 4:0.03 25-26:0.02 \
  --marker-ids 0 2 3 4 19 22-26 28 29 \
  --reference-marker-id 19 \
  --sample-rate-hz 10 \
  --output config/Model/playground_static_4_tag/marker_model.json \
  --diagnostics-output config/Model/playground_static_4_tag/marker_model_diagnostics.json \
  --force
```

**Benchmark video:** match calibration **`image_size`** exactly; use a constant frame rate; prefer all-intra or lossless encoding; lock exposure, focus, and white balance; record diverse viewpoints with multiple expected markers co-visible per frame.

Benchmark metrics report offline processing throughput (frames read, captures accepted, wall-clock time), not camera or source real-time FPS.

**Scale caveat:** metric layout depends on the physical `--marker-size` (default edge length) and calibrated intrinsics. Use repeatable `--marker-size-for ID_OR_RANGE:SIZE` overrides when markers differ (e.g. `4:0.03 10-12:0.025`). Pair translation gates scale with `ratio * min(size_a, size_b)` per edge. Wrong sizes or scaled intrinsics will bias the solved geometry.

**Quality gates** (defaults; override on the CLI):

| Gate | Default |
|------|---------|
| Global reprojection RMS | 2 px (`--reprojection-rms-gate-px`) |
| Per-marker reprojection RMS | 2 px (same gate as global) |
| Pair translation RMS | 10% of the smaller marker size in each pair (`--pair-translation-rms-gate-ratio`) |
| Pair rotation RMS | 5° (`--pair-rotation-rms-gate-deg`) |

Refused solves print diagnostics and resume capture; nothing is written until a solve passes. Use `--force` to overwrite an existing `--output`. Frame resolution must match the calibration `image_size`; intrinsics are not scaled.

### Optional object-model keypoint update

Pass **`--object-model PATH`** to copy calibrated marker footprint corners into object-model keypoints after a successful solve (same moment as `--output` is written). The object-model JSON must include a non-empty top-level **`keypoint_sources`** map from existing keypoint names to marker corners:

```json
{
  "units": "meters",
  "coordinate_frame": "marker_model",
  "keypoints": {
    "top": [0.0, 0.0, 0.0],
    "bottom": [0.0, 0.1, 0.0]
  },
  "skeleton": [["top", "bottom"]],
  "keypoint_sources": {
    "top": {"marker_id": 1, "corner": "top_left"},
    "bottom": {"marker_id": "01", "corner": "bottom_right"}
  }
}
```

Supported corners match marker footprints: `top_left`, `top_right`, `bottom_right`, `bottom_left`. `marker_id` may be an integer or decimal string (e.g. `"01"` resolves to `1`). Only keypoints listed in `keypoint_sources` are updated; other keypoints, `skeleton`, and other metadata are preserved. `keypoint_sources` itself is left unchanged in the saved object model.

Validation runs at startup when `--object-model` is set: the file must exist, `keypoint_sources` must be non-empty with valid specs, and every source marker must appear in `--marker-ids`. Refused or no-layout solves do not modify the object model. If a source marker is missing from the solved layout (including partial solves that omit it), the marker model is still saved but the command fails clearly without changing the object model.

## Marker model evaluation (offline)

`object-evaluate-marker-model` compares one or more marker-model candidates against CAD landmark geometry and held-out moving-video detection consistency. It emits separate rankings for CAD disagreement and detection consistency; there is no overall winner, pass/fail gate, or absolute accuracy claim.

```bash
uv run object-evaluate-marker-model \
  --manifest config/evaluation/playground_static_4_tag/manifest.json \
  --output /tmp/playground_static_4_tag_evaluation.json
```

**Manifest (`manifest_version: 1`):** declares shared `cad_model`, `object_model`, `intrinsics`, `detector`, `held_out_videos` (each must set `held_out: true`), and `candidates` with `{name, marker_model, capture_session, solver_variant, calibration_source}`. Relative paths resolve from the repository root (parent of `config/`). Comparable candidates must share the same marker IDs and object-model landmark coverage.

**Held-out declaration:** the manifest states which videos were excluded from calibration. The tool preserves that declaration in the report but cannot independently verify it.

**Primary metrics:**
- **CAD disagreement:** leave-one-marker-out CAD prediction RMSE in millimeters (lower is better). CAD is a nominal reference; disagreement combines installation, CAD/export, padding, and vision-calibration effects without an installation survey.
- **Detection consistency:** P95 held-out corner error in pixels (lower is better), using frozen detections decoded once per held-out video.

**Repeatability grouping:** same-session solver comparisons and cross-session same-variant groups are reported only when at least two candidates qualify; otherwise the report notes insufficient candidates. No statistical confidence is fabricated.

**Output:** versioned JSON with normalized input paths and SHA-256 hashes, correspondence diagnostics, per-candidate metrics, separate rankings, grouping notes, and normalization counters (unknown IDs, duplicate IDs, malformed detections). A concise console summary is printed from the same report object.

```bash
uv run object-calibrate-marker-model \
  ... \
  --output config/Model/remote1/marker_model.json \
  --object-model config/Model/remote1/object_model.json
```

Inspect an existing model (terminal or static diagram):

```bash
uv run object-inspect-marker-model --marker-model config/Model/remote1/marker_model.json --visualize
```

## Optional visualization

The `viz` extra adds overlays, skeleton keypoints, and matplotlib plots:

```bash
uv sync --extra viz
uv run object-detect \
  --source 0 \
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
  cli/                 # object-detect, object-charuco, object-calibrate-marker-model, object-inspect-marker-model, object-evaluate-marker-model
config/
  evaluation/          # versioned marker-model evaluation manifests
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

Live camera and video file CLIs use `--source`: pass a camera device index (e.g. `0`) or a path to a video file. Video files loop on end-of-file in interactive tools. `object-charuco` with a video file does not loop; it samples to EOF and then calibrates.

## Common options

| Command | Flag | Description |
|---------|------|-------------|
| `object-charuco` | `--board-model` | ChArUco board model JSON (exclusive with geometry flags) |
| `object-charuco` | `--board-type` | `charuco_board` or `checkerboard` |
| `object-charuco` | `--layout` | Board height × width (rows × columns) |
| `object-charuco` | `--marker-size` | ArUco marker size in meters (ChArUco only) |
| `object-charuco` | `--source` | Camera device index (e.g. `0`) or path to a video file |
| `object-charuco` | `--output` | Intrinsics JSON path |
| `object-charuco` | `--sample-rate-hz` | Video-file automatic capture rate (default 10 Hz); ignored for live camera |
| `object-detect` | `--calibration` | Path to camera intrinsics JSON |
| `object-detect` | `--marker-model` | Marker model JSON path |
| `object-detect` | `--marker-id` | Use a specific marker id only |
| `object-detect` | `--no-visualize` | Camera preview without overlays |
| `object-detect` | `--overlay-cad-model` | Semi-transparent GLB CAD mesh overlay on camera frame |
| `object-detect` | `--side2side-cad-model` | Side-by-side opaque CAD pane (`--preview` required) |
| `object-detect` | `--cad-model` | GLB path; loads sibling `cad_registration.json` or auto-fits from `--object-model` |
| `object-detect` / `annotation-tool` / `object-calibrate-marker-model` | `--source` | Camera device index (e.g. `0`) or path to a video file |
| `object-calibrate-marker-model` | `--calibration` | Camera intrinsics JSON |
| `object-calibrate-marker-model` | `--marker-ids` / `--reference-marker-id` | Expected unique IDs and layout reference |
| `object-calibrate-marker-model` | `--anchor-marker-ids` | Optional bootstrap core for large layouts |
| `object-calibrate-marker-model` | `--anchor-stop-after-expansion` | Debug: write expansion-only layout (no BA/gates) |
| `object-calibrate-marker-model` | `--marker-size` | Default physical tag edge length in meters |
| `object-calibrate-marker-model` | `--marker-size-for` | Per-marker overrides (`ID_OR_RANGE:SIZE`, repeatable) |
| `object-calibrate-marker-model` | `--min-pair-inliers` | Minimum co-visible frames per pair (default 20) |
| `object-calibrate-marker-model` | `--benchmark` | Video-file throughput mode: decode all frames, detect on selected frames, single solve |
| `object-calibrate-marker-model` | `--benchmark-frame-selection` | Benchmark-only: `uniform` (default) or `sharpest` per-window selection; requires `--benchmark` |
| `object-calibrate-marker-model` | `--diagnostics-output` | Diagnostics JSON (required with `--benchmark`); adds top-level `benchmark` timings/counts/environment |
| `object-calibrate-marker-model` | `--auto` / `--sample-rate-hz` | Periodic capture at sample rate (default 10 Hz); `--auto` not needed with `--benchmark` |
| `object-calibrate-marker-model` | `--output` / `--force` | Marker model JSON; refuse overwrite unless `--force` |
| `object-calibrate-marker-model` | `--object-model` | Optional object model JSON; update mapped keypoints from solved footprints after a successful save |
| `object-inspect-marker-model` | `--marker-model` / `--visualize` | Print or diagram an existing marker model |
| `object-evaluate-marker-model` | `--manifest` | Versioned evaluation manifest JSON |
| `object-evaluate-marker-model` | `--output` | Versioned evaluation report JSON |
