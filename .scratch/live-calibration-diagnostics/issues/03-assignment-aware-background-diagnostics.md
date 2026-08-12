# 03 — Assignment-Aware Background Diagnostics

**What to build:** Periodically analyze a stable observation snapshot through IPPE assignment and edge filtering, without bundle adjustment, and publish assignment-aware calibration health while capture continues.

**Blocked by:** 01 — Live Pair Readiness; 02 — Explain IPPE Assignment Rejections.

**Status:** ready-for-agent

- [ ] Diagnostics run against an immutable, monotonically identified observation snapshot outside the capture loop.
- [ ] The HUD reports accepted and rejected frame counts, accepted support and retention ratio per pair, and the snapshot frame count or age.
- [ ] Newer snapshots cannot be overwritten by stale worker results.
- [ ] The diagnostic path stops before corner bundle adjustment and never writes a Marker Model.
- [ ] Explicit solve and save remain synchronous and authoritative over the latest complete observation set.
- [ ] Tests cover snapshot consistency, stale-result suppression, worker failure, and continued capture responsiveness.
