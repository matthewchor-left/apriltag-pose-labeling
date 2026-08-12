# 02 — Explain IPPE Assignment Rejections

**What to build:** Report why captured frames fail planar-pose candidate assignment so the operator can identify the marker pairs and viewing conditions responsible for rejected data.

**Blocked by:** None — can start immediately.

**Status:** done

- [x] Every rejected frame records a concrete reason, including the worst conflicting marker pair when pair consistency causes rejection.
- [x] Diagnostics distinguish missing constraints from translation-gate and rotation-gate violations.
- [x] The CLI summarizes the most frequent rejection causes and affected pairs.
- [x] Diagnostic reporting does not change which candidate assignment the authoritative solver selects.
- [x] Tests verify explanations for accepted frames, conflicting candidates, and frames without a constrained pair.
