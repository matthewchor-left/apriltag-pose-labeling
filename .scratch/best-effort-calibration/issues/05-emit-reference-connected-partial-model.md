# 05 — Emit a reference-connected partial marker model

**What to build:** Add a separate opt-in partial-output policy for best-effort calibration. After attempting weak-edge recovery, emit the complete reference-connected subset when some requested markers remain unobservable or disconnected, and make every omission explicit to users and downstream tooling.

**Blocked by:** 02 — Recover connectivity through weak pair consensus.

**Status:** done

- [x] Partial output requires both best-effort mode and an explicit partial-output option.
- [x] The emitted model contains only the connected component containing the reference marker.
- [x] Marker sizes and footprints are resolved and validated against exactly the emitted marker subset.
- [x] Diagnostics and console output list every requested marker omitted from the model and why it was omitted.
- [x] Strict mode and best-effort mode without partial output continue to refuse incomplete requested layouts.
- [x] A partial model uses the existing marker-model format and loads successfully in object detection and model inspection.
- [x] Calibration still refuses when no non-reference marker can be connected to the reference marker or the retained geometry is invalid.

## Maintainability notes

- Partial-output refusal/success paths in `calibrate_marker_layout` still repeat emitted-footprint and partial-emission tails; a shared helper would shrink the diff but was deferred during the correctness follow-up.
