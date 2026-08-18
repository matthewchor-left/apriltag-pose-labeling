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
  --board-model config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json \
  --output config/Camera/logi_hd_1080p/intrinsic.json
```

ChArUco is the default board type (`--board-type charuco_board`). For a plain checkerboard:

```bash
uv run object-charuco \
  --board-type checkerboard \
  --source 0 \
  --layout 11 8 \
  --output config/Camera/logi_hd_1080p/intrinsic.json
```

Save a printable ChArUco pattern (true scale on A4, 300 DPI):

```bash
uv run object-charuco \
  --board-model config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json \
  --save-board config/Board/charuco_h11_w8_25mm_4x4_50/a4_borderless.png
```

The current Board Model profile is **h11 × w8** (height × width, rows × columns), with 25 mm squares and 20 mm markers:

```bash
uv run object-charuco \
  --board-model config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json \
  --save-board config/Board/charuco_h11_w8_25mm_4x4_50/a4_borderless.png
```

`--board-model` cannot be combined with `--layout`, `--marker-size`, `--square-size`, or `--dictionary`.

Print the PNG at **100% scale** (not “fit to page”) so each square matches `--square-size`.

Live camera (`--source 0`):

- **Space** — capture a frame when the board is fully detected
- **q** — finish and save to `--output`

Video file `--source` captures automatically at `--sample-rate-hz` (default 10 Hz) on video time whenever the board is detected. There is no preview window; calibration runs at end of file.

```bash
uv run object-charuco \
  --source data/camera_calibration.mov \
  --board-model config/Board/charuco_h11_w8_25mm_4x4_50/board_model.json \
  --sample-rate-hz 10 \
  --output config/Camera/logi_hd_1080p/intrinsic.json
```

`--layout` is height × width (rows × columns). For ChArUco that is chess **square** count; for checkerboard it is **inner corner** count.

## 2. Detect object pose

```bash
uv run object-detect \
  --source data/playground/setup3/test_01.mov \
  --calibration config/Camera/logi_hd_1080p/intrinsic.json \
  --marker-model config/Model/playground/setup3/calibration_01/marker_model.json \
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
  --source data/playground/setup3/test_01.mov \
  --calibration config/Camera/logi_hd_1080p/intrinsic.json \
  --marker-model config/Model/playground/setup3/calibration_01/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --overlay-object-model \
  --object-model config/Model/playground/setup3/calibration_01/object_model.json \
  --overlay-cad-model \
  --side2side-cad-model \
  --cad-model config/Model/CAD/nexplayground_sim.glb
```


| Flag                    | Description                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--overlay-cad-model`   | Semi-transparent `--cad-model` GLB on camera frame (requires `--visualize`)                                                   |
| `--side2side-cad-model` | Side-by-side opaque CAD pane (requires `--preview` and `--cad-model`)                                                         |
| `--cad-model`           | GLB path; required with either CAD flag. Uses sibling `cad_registration.json`, or auto-fits from `--object-model` when absent |

### Training-data label contract

The fixed 17-keypoint GLB landmark order, YOLO pose row layout, and `data.yaml`
configuration for generating `nexplayground` training data are documented in
[`docs/training-data.md`](docs/training-data.md).

## 3. Generate YOLO pose training data

`annotation-tool` replaces the legacy Background Plate eraser workflow. It writes
Training Samples for one Dataset Split per run: raw JPEG/label pairs, root
`data.yaml`, and `runs/<run-name>.json` provenance.

```bash
uv run annotation-tool \
  --source data/playground/setup3/test_01.mov \
  --calibration config/Camera/logi_hd_1080p/intrinsic.json \
  --marker-model config/Model/playground/setup3/calibration_01/marker_model.json \
  --cad-model config/Model/CAD/nexplayground_sim.glb \
  --output data/yolo/nexplayground \
  --split train \
  --run-name setup3_test01 \
  --sample-rate-hz 2 \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --labeled-images 5
```

Video sources run headlessly to EOF. Live camera shows a preview and stops on **q**.
Add `--labeled-images N` to write annotated previews for the first `N` saved samples
under `labeled-images/<split>/`. See [`docs/training-data.md`](docs/training-data.md)
for acceptance rules, output layout, and the fixed 17-landmark contract.

## Core API

```python
import cv2
from object_apriltag import ObjectDetector
from object_apriltag.calibration import load_intrinsics

camera_matrix, dist_coeffs, _, _, _ = load_intrinsics("config/Camera/logi_hd_1080p/intrinsic.json")
detector = ObjectDetector(
    camera_matrix,
    dist_coeffs,
    marker_model="config/Model/remote/marker_model.json",
)

cap = cv2.VideoCapture(2)
ok, frame = cap.read()
pose = detector.detect(frame)  # ObjectPose(origin, rotation) or None
```

`ObjectPose.origin` and `ObjectPose.rotation` are in the **camera frame**.

`ObjectPose.rotation` maps **reference-marker-centered** model coordinates into the camera frame. Those axes match the model frame below when the reference marker faces the camera: **+X** right, **+Y** down, **+Z** into the scene. There is no separate object-pose axis flip — `marker_model.json`, `object_model.json`, and runtime pose all use the same convention.

## Marker model coordinates

`config/Model/<object>/marker_model.json` stores sticker corner positions in the **model frame** (also `coordinate_frame: "marker_model"` in `object_model.json`). The frame stays attached to the object regardless of where the camera is.


| Axis   | Direction                                                     |
| ------ | ------------------------------------------------------------- |
| **+X** | Right in the image when the reference marker faces the camera |
| **+Y** | Down in the image                                             |
| **+Z** | Into the scene (away from the camera)                         |


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

Marker model JSON uses top-level `marker_size_m` as the default physical edge length (meters). Each `markers.<id>` entry may optionally include `size_m` when that sticker differs from the default; omitted `size_m` means the default. Calibration writes `size_m` only for non-default markers. Saved models may include `anchor_marker_ids` when present in the layout metadata. Footprint validation checks each marker against its resolved size.

## Marker model calibration

`object-calibrate-marker-model` estimates sticker footprint positions from co-visible AprilTag detections and publishes paired `marker_model.json` and `object_model.json` artifacts.

### Calibration Workspace

Point the command at a Calibration Workspace `config.json`. Video and camera-intrinsics paths inside the recipe resolve relative to the config file. Outputs are fixed lowercase siblings of the config:


| Artifact     | Path                            |
| ------------ | ------------------------------- |
| Marker model | `<workspace>/marker_model.json` |
| Object model | `<workspace>/object_model.json` |
| Diagnostics  | `<workspace>/diagnostics.json`  |


```bash
uv run object-calibrate-marker-model \
  --config config/Model/playground/setup3/calibration_01/config.json \
  --force
```

`--force` overwrites existing sibling outputs. Without it, the command refuses when any generated artifact already exists.

**Calibration Recipe (**`config_version: 1`**)** — strict JSON; unknown or missing fields are rejected:

- `inputs` — `source` (video path) and `intrinsics` (camera JSON path), both resolved from the config directory.
- `detector` — `dictionary` and `sensitivity` (`default`, `relaxed`, or `aggressive`).
- `markers` — `reference_marker_id` and `anchor_marker_ids` must be `null` (automatic reference selection); non-overlapping `groups` of `{ids, size_m}` (IDs may be integers or range strings like `"22-26"`). The first group's size becomes the generated Marker Model's default size; later groups are stored as per-marker overrides.
- `execution` — `mode` must be `benchmark` with `sample_rate_hz` and `frame_selection` set to `sharpest` (decode every frame, detect on the sharpest frame per time window).
- `solver` — all fields required. Strategy keys (`policy`, `discrete_method`, `anchor_stop_after_expansion`, `partial_output`) must match the fixed values below (validated at load time; the runtime always uses best-effort partial calibration with rotation-consistent assignment). Quality thresholds: `min_inliers_per_edge`, `reprojection_rms_gate_px`, `pair_translation_rms_gate_ratio`, `pair_rotation_rms_gate_deg`, `huber_delta_px`, `corner_outlier_px`, `max_ba_iterations`.
- `object_model` — `keypoint_sources` and `skeleton`; keypoint positions are generated from solved footprints at publication time (persisted `keypoints` are not recipe inputs).

On refusal (quality gates, missing markers, invalid assignment), diagnostics are still written to `diagnostics.json` while model outputs are not updated. Successful publication writes marker and object models together as a pair when every `keypoint_sources` marker is present in the solved layout.

Example benchmark recipe excerpt:

```json
{
  "config_version": 1,
  "inputs": {
    "source": "../../../../../data/playground/setup1/calibration_01.mov",
    "intrinsics": "../../../../../config/Camera/logi_hd_1080p/intrinsic.json"
  },
  "detector": {"dictionary": "36h11", "sensitivity": "relaxed"},
  "markers": {
    "reference_marker_id": null,
    "anchor_marker_ids": null,
    "groups": [
      {"ids": [0, 2, 3, 4], "size_m": 0.07},
      {"ids": [19, "22-26", "28-29"], "size_m": 0.02}
    ]
  },
  "execution": {
    "mode": "benchmark",
    "sample_rate_hz": 30.0,
    "frame_selection": "sharpest"
  },
  "solver": {
    "policy": "best_effort",
    "discrete_method": "rotation_consistent",
    "anchor_stop_after_expansion": false,
    "partial_output": true,
    "min_inliers_per_edge": 5,
    "reprojection_rms_gate_px": 2.0,
    "pair_translation_rms_gate_ratio": 0.1,
    "pair_rotation_rms_gate_deg": 5.0,
    "huber_delta_px": 1.0,
    "corner_outlier_px": 3.0,
    "max_ba_iterations": 50
  },
  "object_model": {
    "keypoint_sources": {
      "left-center": {"marker_id": 19, "corner": "top_right", "padding_mm": 4.0},
      "front-center": {"marker_id": 22, "corner": "top_right", "padding_mm": 4.0}
    },
    "skeleton": [["left-center", "front-center"]]
  }
}
```

**Benchmark video:** match intrinsics `image_size` exactly; use a constant frame rate; prefer all-intra or lossless encoding; lock exposure, focus, and white balance; record diverse viewpoints with multiple expected markers co-visible per frame.

Benchmark metrics report offline processing throughput (frames read, captures accepted, wall-clock time), not camera or source real-time FPS.

**Scale caveat:** metric layout depends on physical marker sizes in `markers.groups` and calibrated intrinsics. Pair translation gates scale with `pair_translation_rms_gate_ratio * min(size_a, size_b)` per edge. Wrong sizes or scaled intrinsics will bias the solved geometry.

**Quality gates** (defaults in recipe `solver`; override per workspace):


| Gate                        | Default                                                                         |
| --------------------------- | ------------------------------------------------------------------------------- |
| Global reprojection RMS     | 2 px (`reprojection_rms_gate_px`)                                               |
| Per-marker reprojection RMS | 2 px (same gate as global)                                                      |
| Pair translation RMS        | 10% of the smaller marker size in each pair (`pair_translation_rms_gate_ratio`) |
| Pair rotation RMS           | 5° (`pair_rotation_rms_gate_deg`)                                               |


Refused solves print diagnostics and exit after writing `diagnostics.json`; model outputs are not updated until a solve passes publication rules. Use `--force` to overwrite existing outputs. Frame resolution must match the calibration `image_size`; intrinsics are not scaled.

## Marker model evaluation (offline)

`object-evaluate-marker-model` compares one or more marker-model candidates against CAD landmark geometry and held-out moving-video detection consistency. It emits separate rankings for CAD disagreement and detection consistency; there is no overall winner, pass/fail gate, or absolute accuracy claim.

```bash
for setup in setup1 setup3; do
  uv run object-evaluate-marker-model \
    --manifest "config/evaluation/playground/${setup}/manifest.json" \
    --output "config/evaluation/playground/${setup}/report.json"
done
```

**Manifest (**`manifest_version: 1`**):** declares shared `cad_model`, `object_model`, `intrinsics`, `detector`, `held_out_videos` (each must set `held_out: true`), and `candidates` with `{name, marker_model, capture_session, solver_variant, calibration_source}`. Relative paths resolve from the repository root (parent of `config/`). Comparable candidates must share the same marker IDs and object-model landmark coverage.

**Held-out declaration:** the manifest states which videos were excluded from calibration. The tool preserves that declaration in the report but cannot independently verify it.

**Primary metrics:**

- **CAD disagreement:** leave-one-marker-out CAD prediction RMSE in millimeters (lower is better). CAD is a nominal reference; disagreement combines installation, CAD/export, padding, and vision-calibration effects without an installation survey.
- **Detection consistency:** P95 held-out corner error in pixels (lower is better), using frozen detections decoded once per held-out video.

**Repeatability grouping:** same-session solver comparisons and cross-session same-variant groups are reported only when at least two candidates qualify; otherwise the report notes insufficient candidates. No statistical confidence is fabricated.

**Output:** versioned JSON with normalized input paths and SHA-256 hashes, correspondence diagnostics, per-candidate metrics, separate rankings, grouping notes, and normalization counters (unknown IDs, duplicate IDs, malformed detections). A concise console summary is printed from the same report object.

Inspect an existing model (terminal or static diagram):

```bash
uv run object-inspect-marker-model \
  --marker-model config/Model/remote/marker_model.json \
  --visualize
```



## Optional visualization

The `viz` extra adds overlays, skeleton keypoints, and matplotlib plots:

```bash
uv sync --extra viz
uv run object-detect \
  --source 0 \
  --calibration config/Camera/logi_hd_1080p/intrinsic.json \
  --marker-model config/Model/remote/marker_model.json \
  --dictionary 36h11 \
  --detection-sensitivity relaxed \
  --plot-graph \
  --object-model config/Model/remote/object_model.json
```



## Development

```bash
uv sync --extra viz
uv run python -m unittest \
  tests.test_calibration_recipe \
  tests.test_cli_calibrate_marker_model \
  tests.test_marker_layout_calibration
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
  cli/                 # object-detect, annotation-tool, object-charuco, object-calibrate-marker-model, object-inspect-marker-model, object-evaluate-marker-model
config/
  evaluation/
    playground/
      setup1/ setup3/  # per-setup manifest.json and generated report.json
  Board/
    charuco_h11_w8_25mm_4x4_50/  # board_model.json and printable board PNG
  Camera/
    logi_hd_1080p/     # intrinsic.json and camera_setting.json
  Model/
    playground/
      setup1/ setup2/ setup3/  # calibration workspaces
    remote/            # config, marker model, object model, diagnostics
    CAD/               # GLB models and registration diagnostics
data/
  camera_calibration.mov
  playground/
    setup1/ setup2/ setup3/    # calibration and held-out videos
  remote/              # calibration.mov and test.mov
pyproject.toml         # project metadata and dependencies
uv.lock                # locked dependency versions (commit this)
```

Live camera and video file CLIs use `--source`: pass a camera device index (e.g. `0`) or a path to a video file. Video files loop on end-of-file in interactive tools. `object-charuco` with a video file does not loop; it samples to EOF and then calibrates.

## Common options


| Command                             | Flag                             | Description                                                                                   |
| ----------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------- |
| `object-charuco`                    | `--board-model`                  | ChArUco board model JSON (exclusive with geometry flags)                                      |
| `object-charuco`                    | `--board-type`                   | `charuco_board` or `checkerboard`                                                             |
| `object-charuco`                    | `--layout`                       | Board height × width (rows × columns)                                                         |
| `object-charuco`                    | `--marker-size`                  | ArUco marker size in meters (ChArUco only)                                                    |
| `object-charuco`                    | `--source`                       | Camera device index (e.g. `0`) or path to a video file                                        |
| `object-charuco`                    | `--output`                       | Intrinsics JSON path                                                                          |
| `object-charuco`                    | `--sample-rate-hz`               | Video-file automatic capture rate (default 10 Hz); ignored for live camera                    |
| `object-detect`                     | `--calibration`                  | Path to camera intrinsics JSON                                                                |
| `object-detect`                     | `--marker-model`                 | Marker model JSON path                                                                        |
| `object-detect`                     | `--marker-id`                    | Use a specific marker id only                                                                 |
| `object-detect`                     | `--no-visualize`                 | Camera preview without overlays                                                               |
| `object-detect`                     | `--overlay-cad-model`            | Semi-transparent GLB CAD mesh overlay on camera frame                                         |
| `object-detect`                     | `--side2side-cad-model`          | Side-by-side opaque CAD pane (`--preview` required)                                           |
| `object-detect`                     | `--cad-model`                    | GLB path; loads sibling `cad_registration.json` or auto-fits from `--object-model`            |
| `object-detect` / `annotation-tool` | `--source`                       | Camera device index (e.g. `0`) or path to a video file                                        |
| `annotation-tool`                   | `--output`                       | Training Dataset root (`images/`, `labels/`, `data.yaml`, `runs/`)                            |
| `annotation-tool`                   | `--split`                        | Dataset Split for the whole run: `train` or `val`                                             |
| `annotation-tool`                   | `--run-name`                     | Unique Dataset Generation Run name                                                            |
| `annotation-tool`                   | `--sample-rate-hz`               | Save the first Accepted Frame after this interval since the previous save                     |
| `annotation-tool`                   | `--labeled-images`               | Annotated JPEG previews for the first N saved samples under `labeled-images/<split>/`       |
| `annotation-tool`                   | `--cad-model`                    | GLB path; loads sibling `cad_registration.json` or auto-fits from `--object-model`            |
| `object-calibrate-marker-model`     | `--config`                       | Calibration Workspace `config.json`                                                           |
| `object-calibrate-marker-model`     | `--force`                        | Overwrite existing workspace `marker_model.json`, `object_model.json`, and `diagnostics.json` |
| `object-inspect-marker-model`       | `--marker-model` / `--visualize` | Print or diagram an existing marker model                                                     |
| `object-evaluate-marker-model`      | `--manifest`                     | Versioned evaluation manifest JSON                                                            |
| `object-evaluate-marker-model`      | `--output`                       | Versioned evaluation report JSON                                                              |


