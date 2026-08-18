# YOLO Pose Training Data Contract

This branch targets a single-class YOLO pose dataset for `nexplayground`.
`object-detect` provides the prerequisites for generating labels: calibrated
camera intrinsics, a fused object pose, a CAD-to-marker-model registration, and
named landmark positions from meshless GLB EMPTY nodes.

The GLB landmark names below are the label source of truth. Their order is fixed
and must not depend on JSON object ordering or on which landmarks are present in
`object_model.json`.

## CLI

Generate Training Samples with `annotation-tool`:

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

Required flags: `--source`, `--calibration`, `--marker-model`, `--cad-model`,
`--output`, `--split {train,val}`, `--run-name`, `--sample-rate-hz`,
`--dictionary`, and `--detection-sensitivity`. Optional `--labeled-images N`
writes annotated previews for the first `N` saved Training Samples.

`--object-model` is required only when sibling `cad_registration.json` next to
`--cad-model` is absent; registration is then fitted in memory from matching
named GLB landmarks and `keypoint_sources`.

Each Dataset Generation Run belongs wholly to one Dataset Split. There is no
random per-frame split.

## Sampling

Every source frame is processed. Once the interval since the previous saved
Training Sample has elapsed, the generator saves the first Accepted Frame.
Invalid frames do not consume sampling slots.

- Video files: headless, one pass to EOF, timed on video media timestamps.
- Live camera: preview window, stop on `q`, timed on monotonic clock.

Video files must expose usable media timestamps via `CAP_PROP_POS_MSEC`. When
that metadata is unavailable, the generator falls back to `frame_index / FPS`
and requires the video container to report a finite, positive FPS.

A frame is an Accepted Frame when:

1. `ObjectDetector.fuse()` returns a Fused Object Pose (no extra marker-count or
   reprojection gate).
2. All 17 fixed CAD landmarks project finitely, in front of the camera, and
   inside the image bounds.
3. The clipped axis-aligned bounds of the projected CAD mesh silhouette are
   valid.

Cropping, inpainting, and eraser masking are out of scope for this stage.

## Output tree

```text
<output>/
  data.yaml
  images/<split>/<run-name>_<sample-index>.jpg
  labels/<split>/<run-name>_<sample-index>.txt
  labeled-images/<split>/<run-name>_<sample-index>.jpg   # optional, first N saved samples
  runs/<run-name>.json
```

Saved images are raw full-resolution JPEG quality 95. Preview rendering never
contaminates saved files. When `--labeled-images N` is set, the first `N` saved
Training Samples also write annotated JPEG previews under `labeled-images/<split>/`
with the generated bounding box and 17 named keypoints drawn on a copy of the raw
frame.

`data.yaml` is created on first use and must match the schema below on later
appends. The generator refuses run-name or sample-file collisions and validates
the run-report schema before writing `runs/<run-name>.json`.

## Run report

Each Dataset Generation Run writes `runs/<run-name>.json` when generation has
started, even when the run ends early.

| Field | Meaning |
|-------|---------|
| `status: completed` | Normal termination: video reached EOF, or the camera operator pressed `q`. |
| `status: failed` | An error interrupted the run after generation started (frame read failure, JPEG/label write failure, preview error, etc.). |
| `error` | Present only when `status` is `failed`; short message describing the failure. |
| `counts` | Frames processed, samples saved, and rejection counters accumulated before termination. |

Any existing run report reserves that `--run-name`, including a failed run.
The generator refuses to start when `runs/<run-name>.json` already exists or
when sample files matching `<run-name>_*.jpg` / `<run-name>_*.txt` are present.
To retry, choose a new `--run-name` or deliberately remove the prior run report
and any partial sample files for that name.

## Label row

Each image has one YOLO pose label row:

```text
<class> <bbox_x_center> <bbox_y_center> <bbox_width> <bbox_height> <kpt0_x> <kpt0_y> <kpt0_v> ... <kpt16_x> <kpt16_y> <kpt16_v>
```

This is 56 whitespace-separated fields: one class ID, four bounding-box values,
and 17 keypoint triplets. Coordinates are normalized to the image width and
height. The generator always sets class `0`, `v=2` for all 17 keypoints, and does
not encode occlusion or missing-keypoint states.

| Index | GLB EMPTY name | Label triplet position (1-based field #) |
|------:|----------------|------------------------------------------|
| 0 | `back-center` | fields 6–8 (`x`, `y`, `v`) |
| 1 | `back-left-center` | fields 9–11 |
| 2 | `back-right-center` | fields 12–14 |
| 3 | `front-center` | fields 15–17 |
| 4 | `front-left-center` | fields 18–20 |
| 5 | `front-right-center` | fields 21–23 |
| 6 | `left-center` | fields 24–26 |
| 7 | `right-center` | fields 27–29 |
| 8 | `top-back-center` | fields 30–32 |
| 9 | `top-back-left` | fields 33–35 |
| 10 | `top-back-right` | fields 36–38 |
| 11 | `top-center` | fields 39–41 |
| 12 | `top-front-center` | fields 42–44 |
| 13 | `top-front-left` | fields 45–47 |
| 14 | `top-front-right` | fields 48–50 |
| 15 | `top-left-center` | fields 51–53 |
| 16 | `top-right-center` | fields 54–56 |

## Dataset configuration

```yaml
train: images/train
val: images/val
names:
  0: nexplayground
kpt_shape: [17, 3]
```

The label generator must preserve this exact keypoint order and emit one
triplet for every index, including landmarks that are not defined in the object
model.
