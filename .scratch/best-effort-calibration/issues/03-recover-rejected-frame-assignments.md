# 03 — Recover frames rejected by strict IPPE assignment

**What to build:** When strict frame assignment leaves insufficient usable observations, let best-effort mode select the globally lowest-cost complete IPPE assignment relative to the recovered pair graph, allowing optimization to proceed without silently mixing incompatible pose branches.

**Blocked by:** 02 — Recover connectivity through weak pair consensus.

**Status:** done

- [x] The existing strict assignment pass always runs before fallback assignment.
- [x] Fallback assignment scores complete candidate combinations by normalized rotation and translation disagreement against the recovered pair graph.
- [x] Each recovered frame uses one finite, globally consistent candidate per visible marker.
- [x] Diagnostics distinguish strictly accepted frames from fallback-assigned frames and report their disagreement costs.
- [x] Strict mode retains its current assignment rejection behavior.
- [x] Frames with no finite or geometrically valid complete assignment remain rejected.
- [x] Regression coverage includes minority IPPE flips and verifies that fallback assignment follows one consistent global branch.
