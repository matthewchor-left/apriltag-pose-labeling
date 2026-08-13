# 02 — Recover connectivity through weak pair consensus

**What to build:** In best-effort mode, retain the dominant relative-pose consensus for weak marker pairs and restore only the strongest rejected edges needed to connect every expected marker to the reference marker. Quality thresholds remain mode-separation and confidence signals rather than unconditional edge deletion.

**Blocked by:** 01 — Emit a provisional model after completed optimization.

**Status:** done

- [x] Each pair retains at most one dominant IPPE consensus mode with no more than one hypothesis per frame.
- [x] Weak edges are ranked using distinct-frame support, support fraction, rotation disagreement, and translation disagreement.
- [x] Best-effort mode restores only enough ranked weak edges to connect all expected markers to the reference marker.
- [x] Restored weak edges require at least two supporting frames (`_BEST_EFFORT_WEAK_EDGE_MIN_SUPPORT`); ranking, support fraction, and `RestoredPairEdge` diagnostics all use dominant-consensus inlier frames (`_weak_edge_consensus_support`), while dropped-pair records retain the original audit counts.
- [x] Diagnostics identify every restored edge, its original rejection reason, and its confidence measurements.
- [x] Strict mode retains its current minimum-inlier and RMS rejection behavior.
- [x] Observations whose raw pair graph cannot connect all expected markers still produce a hard refusal.
- [x] Regression coverage demonstrates recovery of a weak bridge pair without combining opposite IPPE modes.
