# Kickoff prompt for the next Claude session

Paste the block below into a fresh Claude Code session in the `turboQuantPlayground` repo. It's self-contained — the new session should not need to read the previous transcript.

---

```
Read docs/handoff-2026-04-26.md first — it's the full state-of-the-world from
the previous session. Then we have two open threads from that handoff:

(1) The user asked "lets also do qnn and htp" but that ask is ambiguous —
    the existing cpp/backends/qnn_htp/ scaffold accelerates KV-cache rotate
    + value_dequant only, not SigLIP / mmproj / mtmd. The Live VLM bottleneck
    on Tab S9+ (board "kalama", SD 8 Gen 2) is SigLIP image-encode, which the
    QNN scaffold does NOT touch. Three meaningfully different work plans
    (Options A / B / C) are written up in the handoff doc — pick one *with the
    user* via the brainstorming skill before writing any code. Do not assume
    which one they meant.

(2) Cheap interim experiment that might unblock Tab S9+ Live without
    committing to QNN: try llama.cpp's GGML_VULKAN backend. Adreno's Vulkan
    driver hits a different code path than its OpenCL driver, so there's a
    real chance mtmd_helper_eval_chunks doesn't deadlock on Vulkan. One
    evening of work to wire and test. Only do this if the user agrees it's
    worth it before committing to a multi-week QNN port.

Hard constraints carried from prior sessions:

- The cpp/ core must stay OS-free — no <jni.h>, no Android logging, no
  __ANDROID__ ifdefs in cpp/src/ or cpp/include/. JNI lives only under
  android/app/src/main/cpp/.
- Every benchmark must show baseline vs TurboQuant side-by-side. No solo
  TurboQuant numbers without the comparison.
- The same source must build for QNX (SA8295P) and Linux aarch64 (SA8775P)
  with toolchain swap only. Don't pull in dependencies that break that.
- Don't start Path 2.1c Steps 2–5 (custom ggml attention op) unless the
  user explicitly asks. That's a 3–4 week algorithm port that's not the
  most pressing user-visible item.

Devices the user has access to:
- Galaxy Tab S9+ (R52X1000WHK, board kalama, SD 8 Gen 2 / Adreno 740)
  — VLM falls back to CPU (~165 s/frame) due to the upstream Adreno+mtmd
    hang at our pinned llama.cpp commit
- Galaxy S24 Ultra (board pineapple, SD 8 Gen 3 / Adreno 750)
  — VLM runs on Adreno fine (~1–2 s/frame)

Today's date is 2026-04-26. Branch is `main` at ee2b7b7. No open branches.

Start by:
1. Reading docs/handoff-2026-04-26.md end-to-end.
2. Asking the user: "Did you mean Option A (wire existing QNN scaffold —
   small, doesn't help Live), Option B (port SigLIP to QNN — multi-week,
   unblocks Tab S9+ Live), or Option C (use Qualcomm AI Hub VLM — medium,
   architectural fork)? Or should we try the Vulkan experiment first as a
   cheap might-work bypass?"
3. Wait for their answer before writing any code.
```

---

**Why this prompt is structured the way it is**

- Tells the next session to **read the handoff doc first** — saves a long re-discovery cycle.
- Calls out the **ambiguous ask** explicitly so the next session doesn't pick a default and over-commit.
- Lists the **hard constraints** that govern what's acceptable (OS-free core, paired benchmarks, automotive portability) — these survived the prior compaction and need to carry forward.
- Tells the next session **what NOT to start** (Path 2.1c Steps 2–5) so it doesn't disappear into a multi-week algorithm port without authorization.
- Asks the user a single, well-scoped question rather than firing off speculative work.
