# 02 — Recover connectivity through weak pair consensus

**What to build:** In best-effort mode, retain the dominant relative-pose consensus for weak marker pairs and restore only the strongest rejected edges needed to connect every expected marker to the reference marker. Quality thresholds remain mode-separation and confidence signals rather than unconditional edge deletion.

**Blocked by:** 01 — Emit a provisional model after completed optimization.

**Status:** ready-for-agent

- [ ] Each pair retains at most one dominant IPPE consensus mode with no more than one hypothesis per frame.
- [ ] Weak edges are ranked using distinct-frame support, support fraction, rotation disagreement, and translation disagreement.
- [ ] Best-effort mode restores only enough ranked weak edges to connect all expected markers to the reference marker.
- [ ] Diagnostics identify every restored edge, its original rejection reason, and its confidence measurements.
- [ ] Strict mode retains its current minimum-inlier and RMS rejection behavior.
- [ ] Observations whose raw pair graph cannot connect all expected markers still produce a hard refusal.
- [ ] Regression coverage demonstrates recovery of a weak bridge pair without combining opposite IPPE modes.
