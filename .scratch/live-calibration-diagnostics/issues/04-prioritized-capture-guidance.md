# 04 — Prioritized Capture Guidance

**What to build:** Turn assignment-aware graph health into specific capture instructions so the operator knows which marker combinations should be made co-visible next.

**Blocked by:** 03 — Assignment-Aware Background Diagnostics.

**Status:** ready-for-agent

- [ ] The live view identifies expected markers that are not connected to the Reference Marker through passing edges.
- [ ] Guidance prioritizes feasible bridge pairs that would connect missing markers or components.
- [ ] The HUD lists the weakest accepted edges using support retention and translation/rotation consistency.
- [ ] A failing redundant edge is not presented as blocking when a passing connected path already exists.
- [ ] Guidance updates only from the latest completed diagnostic snapshot and clearly indicates stale or unavailable results.
- [ ] Synthetic graph tests verify bridge prioritization, redundant-edge handling, and fully ready calibration states.
