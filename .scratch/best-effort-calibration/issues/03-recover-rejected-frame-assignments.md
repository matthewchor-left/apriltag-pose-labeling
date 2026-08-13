# 03 — Recover frames rejected by strict IPPE assignment

**What to build:** When strict frame assignment leaves insufficient usable observations, let best-effort mode select the globally lowest-cost complete IPPE assignment relative to the recovered pair graph, allowing optimization to proceed without silently mixing incompatible pose branches.

**Blocked by:** 02 — Recover connectivity through weak pair consensus.

**Status:** ready-for-agent

- [ ] The existing strict assignment pass always runs before fallback assignment.
- [ ] Fallback assignment scores complete candidate combinations by normalized rotation and translation disagreement against the recovered pair graph.
- [ ] Each recovered frame uses one finite, globally consistent candidate per visible marker.
- [ ] Diagnostics distinguish strictly accepted frames from fallback-assigned frames and report their disagreement costs.
- [ ] Strict mode retains its current assignment rejection behavior.
- [ ] Frames with no finite or geometrically valid complete assignment remain rejected.
- [ ] Regression coverage includes minority IPPE flips and verifies that fallback assignment follows one consistent global branch.
