# 01 — Live Pair Readiness

**What to build:** Show continuously updated pair strength and graph readiness during Marker Model Calibration so the operator can judge capture quality before requesting a solve.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] The capture HUD reports raw support, robust inlier support, translation RMS, rotation RMS, and pass/weak/fail status for observed marker pairs.
- [x] Each expected marker indicates whether passing edges connect it to the Reference Marker.
- [x] Readiness follows the solver's configured support and RMS gates rather than introducing separate thresholds.
- [x] Updating diagnostics does not interrupt camera capture (background worker with coalesced snapshots).
- [x] Deterministic tests cover strong, weak, redundant, and disconnected pair graphs.
