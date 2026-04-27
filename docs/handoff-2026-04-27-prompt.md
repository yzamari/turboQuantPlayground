# Kickoff prompt for the next Claude session

Paste the block below into a fresh Claude Code session in the `turboQuantPlayground` repo. It's self-contained — the new session should not need to read the previous transcript.

---

```
Read docs/handoff-2026-04-27.md first — it's the full state-of-the-world.
Yesterday's session shipped P2.1c α to main (TurboQuant attention through
llama.cpp's hot decode path) and chat now produces coherent replies on
Galaxy S24 Ultra with kvType=3. Three PRs merged: #25 (the integrated
A0/A1/A2/A3 + chat-coherent defaults), #26 (submodule pointer fix), #27
(TurboQuantKVCache::append() impl). Today's date is 2026-04-27.

The seven open follow-ups (F1..F7) are listed in the handoff. In priority
order for this session:

(F2) NEON-vectorize TurboQuantKVCache::attention_scores and ::attend
     buffer-portion inner loops in cpp/src/kv_cache.cpp (lines ~197-214 and
     ~244-256). They're currently scalar C++ with double-precision FMA in
     the inner loop, which is why decode is ~1-2 s/token on chat-length
     replies. Replace with vfmaq_f32 reductions for ARM/Android, keep the
     scalar path for x86 host so ctest still passes everywhere. This is
     the biggest single decode-perf win available without touching the
     IBackend interface — half-day work, ~4× speedup expected.

(F1) Implement TurboQuantKVCache::flush_buffer_(). Currently throws
     std::logic_error. The append() impl that landed in PR #27 grows the
     unquantized buffer indefinitely — fine for chat-length replies but
     unbounded memory for long contexts. flush_buffer_() should quantize
     the oldest buffer slot and graft it into key_q_ / value_q_ at the
     [BH, n_quant, D] insertion offsets. Notes in the function header in
     kv_cache.cpp explain the design.

(F4) Verify mtmd-on-OpenCL Phase 1 on device. The clip.cpp patches are
     already merged into main (via the integrated branch's submodule
     bump): GPU backend at index 0 in scheduler, MTMD_VISION_GPU_DISABLE
     env-var kill switch, Adreno 740 SoC bypass via device-description
     match. Never on-device tested. On S24 Ultra: tap Live tab, switch
     VLM to Qwen2.5-VL in Settings, watch logcat for SigLIP placement.
     Prior CPU baseline was 227 s per 480x640 frame. Target with Phase 1
     on Adreno 750: ~15 s per frame.

Lower priority follow-ups (in handoff under F3, F5, F6, F7):
- F3 — OpenCL prepare_keys() + cl_mem pool (multi-day, the long-term
  Path A2 finish; until done, NEON is the right priority order)
- F5 — Tab S9+ verify + paired bench CSV (gated on device returning)
- F6 — chat coherence on a 3B+ model (current 1B is at the edge)
- F7 — QNN SigLIP scaffold has stub methods; multi-week Linux/Docker
  toolchain dependency

Hard constraints carried from prior sessions:

- The cpp/ core must stay OS-free — no <jni.h>, no Android logging, no
  __ANDROID__ ifdefs in cpp/src/ or cpp/include/. JNI lives only under
  android/app/src/main/cpp/.
- Every benchmark must show baseline vs TurboQuant side-by-side. No solo
  TurboQuant numbers without the comparison.
- The same source must build for QNX (SA8295P) and Linux aarch64 (SA8775P)
  with toolchain swap only. Don't pull in dependencies that break that.
- Don't commit directly to main/master. PR-and-merge via gh.
- Backend priority is currently NEON > OpenCL until prepare_keys() lands.
  See the comment block in cpp/src/backend_factory.cpp::create_best_backend.
  Re-order only when prepare_keys() exists.

Tripwires from yesterday — read docs/handoff-2026-04-27.md "Six on-device
crashes" section before touching the custom op or session cache. Briefly:
kq_mask must be a graph dep (use ggml_custom_4d not ggml_map_custom3),
Q/K/V are F16 (upcast per-element), K cache layout is [D, n_head_kv, n_kv]
(NOT [D, n_kv, n_head_kv]; v_trans=true on our path), session-cached path
must run ith=0-only (otherwise BH-partitioned threads collide on the
shared cache), value_bits supports {2,4,8} only.

Devices the user has access to:
- Galaxy S24 Ultra (R5CX11REJ2X, board pineapple, SD 8 Gen 3 / Adreno 750
  / Hexagon V75) — connected. Llama-3.2-1B + kvType=3 verified working
  end-to-end yesterday.
- Galaxy Tab S9+ (R52X1000WHK, board kalama, SD 8 Gen 2 / Adreno 740 /
  Hexagon V73) — offline since 2026-04-26 evening; F5 blocked on it.

Today's date is 2026-04-27. Branch `main` at HEAD (PR #28 last merged).
No open feature branches that should be merged. PR #24 is stale and should
be closed without merging — its fix landed via cherry-pick into the
p2.1c/integrated branch (commit 6af419f).

Start by:
1. Reading docs/handoff-2026-04-27.md end-to-end.
2. Confirming current device state — `adb devices` should list the S24
   Ultra. Run the verification recipe from the handoff to confirm main
   still builds and the on-device chat still produces tokens.
3. Picking ONE of (F2 / F1 / F4) and brainstorming the approach before
   writing code. F2 is recommended for fastest user-visible win.
4. NEON intrinsics for the buffer attention should reuse the project's
   existing NEON patterns — see cpp/backends/cpu_neon/neon_backend.cpp
   for the convention (vfmaq_f32, 4-wide unrolled). Don't invent a new
   style.

Don't take destructive actions (no force-push, no rebase of main, no
deleting merged branches). Confirm before opening any new external-facing
thing (PR, push to origin/main).
```
