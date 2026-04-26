# Kickoff prompt — paste into the next clean Claude session

```
We're working in /Users/yahavzamari/Projects/GitHub/turboQuantPlayground.

Last session ended with the Adreno OpenCL fix on the branch
fix/adreno-opencl-uses-native-library — verified working on S24 Ultra
(SmolVLM decode at 9 tok/s on Adreno, no more "platform IDs not available"
log). One commit ahead of origin/main, working tree clean.

Read docs/handoff-2026-04-26-evening.md for the full context. The previous
day's handoff (docs/handoff-2026-04-26.md) is the parent context.

Today's first task is to push the fix branch and open the PR. After that,
the open decisions in the handoff are:
  A2 — re-test Tab S9+ chat decode with the fix (separate device, same fix
       should apply since Tab S9+ also has vendor libOpenCL.so).
  B  — resume Path 2.1c (custom ggml attention op for kvType=3 — the
       multi-week milestone that puts TurboQuant in live decode).
  C  — mtmd-on-OpenCL or QNN vision tower (unblocks Live VLM perf; multi-week).

Caveat to remember: the OpenCL fix accelerates the LLM decoder. The mtmd
vision tower (SigLIP) still runs on CPU — that's why Live VLM image-eval
was 227 s on S24 Ultra even with the fix. That path needs separate work.

Don't take any destructive actions (no force-push, no rebase of main).
Confirm before opening any external-facing thing (PR, push to origin/main).
```
