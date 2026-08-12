# 2. Corner bundle adjustment for marker calibration

Date: 2026-08-12

## Status

Accepted

## Context

Marker Model Calibration must recover all expected sticker footprints from live co-visible AprilTag detections without manual measurement. Pairwise relative poses from IPPE solves are noisy and can disagree when markers are viewed from few angles or with ambiguous depth.

## Decision

- After robust pair consensus and pose-graph initialization, refine marker layout with sparse **corner-level bundle adjustment** (SciPy `least_squares`, Huber loss), reference marker pose fixed.
- Require hard quality gates before writing `marker_model.json`: global and per-marker reprojection RMS, per-pair translation/rotation RMS relative to marker size, and full connectivity to the reference marker.
- Refuse output when gates fail; the live CLI continues capture without writing.

## Considered options

- **Pose-graph only** — propagate relative pair transforms without joint corner refinement. Simpler and fewer dependencies, but errors compound along chains and weak views are harder to reject consistently.

## Consequences

- SciPy becomes a core dependency for calibration.
- Calibration quality is enforced at solve time; authors must gather enough diverse co-visible samples rather than accepting a marginal graph fit.
- Runtime detection still loads the validated JSON; failed solves never partially overwrite an existing model.
