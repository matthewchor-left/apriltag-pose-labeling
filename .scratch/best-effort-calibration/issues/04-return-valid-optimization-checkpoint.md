# 04 — Return the latest valid optimization checkpoint

**What to build:** Preserve complete finite calibration checkpoints so best-effort mode can return the latest valid geometry when pruning or refitting damages an otherwise usable solution. The output must explicitly identify the skipped or failed refinement stage.

**Blocked by:** 01 — Emit a provisional model after completed optimization.

**Status:** done

- [x] Calibration records checkpoints after graph initialization, initial bundle adjustment, and successful post-pruning refit.
- [x] Best-effort mode selects only a checkpoint that covers all expected markers, remains connected to the reference marker, and has finite positive-depth geometry.
- [x] A provisional model returned from an earlier checkpoint includes a warning naming the failed or discarded refinement stage.
- [x] Strict mode continues to refuse the same optimization or pruning failures.
- [x] Initial optimization failure with no valid complete checkpoint still produces a hard refusal.
- [x] Regression coverage verifies pre-pruning recovery and rejection of invalid or incomplete checkpoints.
