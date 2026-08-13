# 01 — Emit a provisional model after completed optimization

**What to build:** Add an opt-in best-effort calibration policy that writes a complete marker model when optimization succeeds but strict quality gates fail. The command must distinguish accepted output, provisional output, and hard refusal while preserving the existing marker-model format and strict-mode behavior.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `--best-effort` writes a complete provisional marker model when the completed solution fails only strict reprojection, translation, or rotation quality gates.
- [ ] Strict mode continues to withhold the same gate-failing model.
- [ ] Calibration results represent accepted, provisional, and hard-failure outcomes without returning both a layout and a hard failure reason.
- [ ] Best-effort output automatically records its policy, quality status, and all failed gates in calibration diagnostics.
- [ ] The marker-model format remains unchanged and provisional output loads successfully in object detection.
- [ ] The CLI clearly warns when provisional output is written and exits successfully whenever a model is produced.
- [ ] `--best-effort` and expansion-only debugging mode are rejected when supplied together.
