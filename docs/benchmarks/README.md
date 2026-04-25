# Benchmark Methodology

This page documents **how we measure** — not what the numbers say. The
authoritative source for "what the numbers say" is the per-device CSVs
under `cpp/bench/results/`. The authoritative source for "how to read
those CSVs" is this file plus the comments at the top of
`cpp/bench/bench_cli.cpp`.

The implementation plan
(`/Users/yahavzamari/.claude/plans/squishy-jumping-fog.md`, "Benchmark
methodology — A/B 'with vs without TurboQuant'") defines the metrics; this
doc documents how they translate into the actual CLI and CSV.

## 1. The non-negotiable: paired A/B reporting

A bench number for "TurboQuant attention is X ms" by itself is **not
acceptable** in this repo. Every bench run reports two numbers per cell —
**baseline** (the same backend, no TurboQuant) and **TurboQuant** — so
the reviewer immediately sees what compression bought us.

Concretely, for each `(seq_len, backend, bits)` cell, both columns
populate:

```
   attention latency:  baseline_attn_ms     vs     tq_attn_ms
   memory footprint:   baseline_mem_bytes   vs     tq_mem_bytes
   accuracy:           reference (= itself) vs     attn_score_cosine_sim
                                                  attn_output_rel_l2
```

The "baseline" path is implemented in `cpp/bench/baseline_kv_cache.{hpp,cpp}`
as a plain FP16 (configurably FP32) KV cache:

- `prefill(K, V, BH, S)` — copies tensors to internal storage as-is.
- `attention_scores(Q, BH, n_q, scale)` — straight `Q @ K^T`.
- `attend(weights, BH, n_q)` — straight `weights @ V`.

It runs on the **same backend** (NEON / QNN / OpenCL / Vulkan) as the
TurboQuant path — same ARM core, same Adreno, same HTP. Apples-to-apples.

## 2. The CSV layout

Defined by `write_csv()` at the top of `cpp/bench/bench_cli.cpp`. One
header line, one row per `(seq_len, backend, bits)`:

```
device,backend,seq_len,bh,d,bits,
  baseline_attn_ms,tq_attn_ms,attn_speedup,
  baseline_mem_bytes,tq_mem_bytes,compression_ratio,
  encode_ms,
  attn_score_cosine_sim,attn_output_rel_l2
```

Every column explained:

| Column | Type | What it is | Source in `bench_runner.hpp` |
|---|---|---|---|
| `device` | str | Free-text device tag, e.g. `s24-ultra` or `android` | `Row.device` |
| `backend` | str | Backend name from `--backend` | `Row.backend` |
| `seq_len` | int | KV-cache length in tokens | `Row.seq_len` |
| `bh` | int | Batch×Heads (folded). Default `8` | `Row.bh` |
| `d` | int | Head dim. Default `128` | `Row.d` |
| `bits` | int | Key bits. `3` is the headline result | `Row.bits` |
| `baseline_attn_ms` | f64 | Mean per-decode attention latency, FP16 baseline | timed `iters` runs of `BaselineKVCache::attention_scores` |
| `tq_attn_ms` | f64 | Mean per-decode attention latency, TurboQuant | timed `iters` runs of `TurboQuantKVCache::attention_scores` |
| `attn_speedup` | f64 | `baseline_attn_ms / tq_attn_ms` | computed |
| `baseline_mem_bytes` | u64 | KV-cache memory at chosen `--baseline-dtype` | `BaselineKVCache::memory_bytes_fp16/32` |
| `tq_mem_bytes` | u64 | TurboQuant compressed KV-cache memory | `TurboQuantKVCache::memory_bytes()` |
| `compression_ratio` | f64 | `baseline_mem_bytes / tq_mem_bytes` | computed |
| `encode_ms` | f64 | One-shot prefill time for TurboQuant (the "overhead" of compression) | one timed `tq.prefill(...)` |
| `attn_score_cosine_sim` | f64 | `cos(softmax(scores_tq), softmax(scores_baseline))` over the full `[BH, n_q, seq_len]` score tensor | `cosine_sim` |
| `attn_output_rel_l2` | f64 | `‖attend_tq − attend_baseline‖₂ / ‖attend_baseline‖₂` over the full `[BH, n_q, D]` output | `rel_l2` |

`attn_score_cosine_sim` is the headline accuracy number. Per the plan's
pass criterion, it must be `≥ 0.99` for a phase to be considered passing.

## 3. The console table

Same data, formatted by `bench_runner.hpp::format_row` for human eyeball:

```
seq_len  baseline    tq         speedup  fp16_mem    tq_mem      comp     enc_ms   cos    relL2
-------  ---------- ---------- -------  ----------- ----------- -------- -------- ------ ------
   128    0.142 ms   0.118 ms   1.20x     2.0 MB      0.6 MB    3.33x    1.42  0.998  0.082
   512    0.581 ms   0.245 ms   2.37x     8.0 MB      2.3 MB    3.48x    5.71  0.997  0.087
  1024    1.158 ms   0.358 ms   3.23x    16.0 MB      4.5 MB    3.55x   11.42  0.997  0.090
  2048    2.314 ms   0.401 ms   5.77x    32.0 MB      9.0 MB    3.55x   22.81  0.996  0.093
```

These numbers are illustrative, not measured.

## 4. Pass criteria (from the plan)

A run is considered successful for a phase when **all three** are met
at the headline cell `bits=3`:

1. `compression_ratio ≥ 3.0×` at `seq_len = 1024`.
2. `attn_score_cosine_sim ≥ 0.99` (TurboQuant attention scores
   numerically match baseline distribution after softmax).
3. `attn_speedup ≥ 1.0×` at `seq_len ≥ 2048` for any GPU/NPU backend.
   For `cpu_neon` the bar is relaxed to `≥ 0.7×` — NEON's win is in
   memory savings, not compute.

If any of those criteria fails, the phase is not done. Don't go to the
next phase.

## 5. Sweep dimensions

Defaults match the existing Python README table:

```
BH = 8
D  = 128
key_bits ∈ {2, 3, 4}
seq_len ∈ {128, 256, 512, 1024, 2048, 4096}    (extend to 8192/16384 if RAM permits)
```

These are configurable on the CLI:

```
--bh 8 --d 128 --bits 3
--seq-lens 128,256,512,1024,2048,4096
--baseline-dtype fp16     (or fp32 for the older A/B layout)
--warmup 1 --iters 5      (timing)
```

## 6. Modes

```
turboquant_bench --check        --backend <name>
turboquant_bench --bench        --backend <name>  --csv <path>
turboquant_bench --check-cross  --backend <name>      ← P1+ (multi-backend)
```

| Mode | What it does | When to use |
|---|---|---|
| `--check` | Smoke test: encode/dequantize a small random tensor, report cosine similarity. Returns 0 if `cos > 0.85`. | First-run sanity. Fastest. |
| `--bench` | Full A/B sweep across `--seq-lens`. Writes CSV to `--csv`, prints table. | Phase verification. The headline. |
| `--check-cross` | Run every enabled backend on the same input, assert pairwise tolerance. | After enabling a new backend, before merging. |

## 7. Running on the S24 Ultra (the connected device)

The exact incantation from a clean state:

```
# Build for Android arm64 with all backends enabled
cmake --preset android-arm64 \
      -DTQ_WITH_NEON=ON \
      -DTQ_WITH_QNN=ON \
      -DTQ_WITH_OPENCL=ON \
      -DTQ_WITH_VULKAN=ON
cmake --build build-android -j8

# Push the bench binary
adb push build-android/bench/turboquant_bench /data/local/tmp/

# Push QNN runtime libs (only needed for the QNN backend)
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so          /data/local/tmp/qnn/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so       /data/local/tmp/qnn/
adb push $QNN_SDK_ROOT/lib/hexagon-v75/libQnnHtpV75Skel.so       /data/local/tmp/qnn/

# Smoke test each backend
adb shell '/data/local/tmp/turboquant_bench --check --backend cpu_neon'
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=/data/local/tmp/qnn && \
           /data/local/tmp/turboquant_bench --check --backend qnn_htp'
adb shell '/data/local/tmp/turboquant_bench --check --backend opencl'
adb shell '/data/local/tmp/turboquant_bench --check --backend vulkan'

# Full bench on QNN (the headline result)
adb shell 'export LD_LIBRARY_PATH=/data/local/tmp/qnn:$LD_LIBRARY_PATH && \
           export ADSP_LIBRARY_PATH=/data/local/tmp/qnn && \
           /data/local/tmp/turboquant_bench --bench \
              --backend qnn_htp \
              --seq-lens 128,256,512,1024,2048,4096 \
              --bits 3 --bh 8 --d 128 \
              --csv /sdcard/qnn.csv'

# Pull the CSV back to the repo
adb pull /sdcard/qnn.csv cpp/bench/results/s24-ultra-qnn.csv

# Repeat for the other backends
adb shell '/data/local/tmp/turboquant_bench --bench --backend cpu_neon \
           --csv /sdcard/neon.csv'
adb pull /sdcard/neon.csv cpp/bench/results/s24-ultra-neon.csv

adb shell '/data/local/tmp/turboquant_bench --bench --backend opencl \
           --csv /sdcard/opencl.csv'
adb pull /sdcard/opencl.csv cpp/bench/results/s24-ultra-opencl.csv

adb shell '/data/local/tmp/turboquant_bench --bench --backend vulkan \
           --csv /sdcard/vulkan.csv'
adb pull /sdcard/vulkan.csv cpp/bench/results/s24-ultra-vulkan.csv
```

`ADSP_LIBRARY_PATH` is mandatory for the QNN/HTP path — it tells the
FastRPC client where to find the device-side skel library. Forgetting it
yields `RPC error 0x80000406` at the first `graphExecute`.

## 8. Cross-backend equivalence (P1+)

Once at least two backends are enabled, run:

```
adb shell '/data/local/tmp/turboquant_bench --check-cross'
```

It iterates every enabled backend on the same fixed input and asserts:

- packed-byte outputs are **bit-exact** across backends.
- float outputs differ by `< 1e-3` pairwise (tolerance is the looser of
  the two backends' tolerances; an FP16 backend forces `1e-3`).

Use this as a regression gate before merging a backend change.

## 9. What to commit and what NOT to commit

- ✓ Commit: the per-device CSVs at `cpp/bench/results/<device>-<backend>.csv`
  after each phase. They are small and auditable.
- ✗ Don't commit: per-iteration latency traces, GPU shader-disassembly
  dumps, QNN graph dumps. These are big, machine-specific, and obsolete
  the moment the SDK version changes. Run them locally; cite them in the
  PR description if relevant.

## 10. Reproducibility

The bench harness uses a **fixed RNG seed** (`std::mt19937 rng(0xA5A5)`)
for the data tensors. Two runs of the same backend on the same SoC at
the same SDK version should produce numerically equivalent rows
(timings will jitter; accuracy columns should not).

If you see the accuracy columns drift between runs, that's a real bug —
look for use of `time()` in the test harness or a missing fixed seed in
a new code path.

## 11. Limitations and honest caveats

1. **Timer resolution.** `std::chrono::steady_clock` on Android is sub-µs;
   on QNX it can be coarser. For very small workloads (`seq_len = 128`,
   `cpu_neon`) you may see 0.1 ms quantization. Use `--iters 20` to
   average it out.
2. **Thermal state.** As noted in
   [`../qualcomm/hexagon-htp.md`](../qualcomm/hexagon-htp.md) §7, the
   first few iterations after a cold start are 2× faster than steady
   state. The harness's `--warmup 1` is too low for that to dominate;
   for paper-grade numbers, run `--warmup 5 --iters 20`.
3. **`encode_ms` is one-shot.** It's the time for a single
   `tq.prefill()` over the full `seq_len`, not amortized. At
   `seq_len = 4096` this is the dominant TurboQuant overhead — but it
   only happens once per request, so it's not on the per-decode hot path.
4. **`attn_output_rel_l2` is not a great metric for short sequences.**
   At `seq_len = 128` the softmax distribution is concentrated, so a
   small score perturbation can move the output by a noticeable
   relative amount. The cosine-sim of scores is the more stable
   accuracy indicator at short context.
