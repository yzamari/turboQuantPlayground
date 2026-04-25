# Qualcomm Snapdragon — Hardware Overview

This page is the entry point for the Qualcomm-specific docs. It explains
**what is inside a Snapdragon SoC**, **which Snapdragon parts we care about
(mobile + automotive)**, and **how each on-die block maps onto one of the four
compute backends in `cpp/backends/`**. Deeper dives live in the sibling files:

- [`hexagon-htp.md`](hexagon-htp.md) — the NPU
- [`adreno-gpu.md`](adreno-gpu.md) — the GPU
- [`automotive.md`](automotive.md) — automotive lineup, OS, ASIL

The connected dev device that drives every benchmark in this repo is a
**Samsung Galaxy S24 Ultra** (`SM-S928B`) with an `SM8650` SoC (Snapdragon 8
Gen 3 *for Galaxy* — overclocked variant), Adreno 750, Hexagon V75 HTP,
Cortex-X4 + Cortex-A720 cores, Android 16.

## 1. Snapdragon SoC anatomy

A modern Snapdragon ("Mobile Compute Platform" in Qualcomm marketing) is a
heterogeneous SoC. Every block we care about is in one die, sharing system
DRAM via the on-die SoC interconnect. Roughly:

```
                 ┌────────────────────────────────────────────────────┐
                 │                Snapdragon 8 Gen 3 SoC              │
                 │                                                    │
   ┌────────┐    │  ┌────────────┐  ┌──────────────┐  ┌──────────────┐│
   │  DRAM  │◄──►│  │  Kryo CPU  │  │   Adreno     │  │   Hexagon    ││
   │ LPDDR5X│    │  │  cluster   │  │   GPU 750    │  │   HTP V75    ││
   └────────┘    │  │ (X4+A720)  │  │              │  │   (NPU)      ││
                 │  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘│
                 │        │                │                 │        │
                 │  ┌─────┴────────────────┴─────────────────┴──────┐ │
                 │  │           SoC interconnect / cache             │ │
                 │  └─────┬─────────────┬───────────────┬────────────┘ │
                 │        │             │               │              │
                 │   ┌────┴───┐    ┌────┴───┐      ┌────┴───┐          │
                 │   │  ISP   │    │ Modem  │      │ Sensor │          │
                 │   │(Spectra│    │(X75 5G)│      │ Hub    │          │
                 │   │  HDR)  │    └────────┘      └────────┘          │
                 │   └────────┘                                        │
                 └────────────────────────────────────────────────────┘
```

| Block | What it is | Our backend |
|---|---|---|
| **Kryo CPU** (8 Gen 3: 1×X4 prime @ 3.3 GHz + 5×A720 perf + 2×A520 eff) | ARMv9.2-A cluster with `i8mm`, `bf16`, `asimddp`, `asimdfhm`. **No SVE on the Galaxy variant.** | `cpu_scalar` (portable C++17), `cpu_neon` (NEON/SIMD). |
| **Adreno GPU** (750 on 8G3) | Tile-based deferred renderer + general-purpose compute units. OpenCL 3.0 + Vulkan 1.3 ICDs. | `opencl` (primary), `vulkan` (secondary). |
| **Hexagon DSP / HTP** (V75 on 8G3) | VLIW DSP with Hexagon Vector eXtensions (HVX, 1024-bit) + Hexagon Tensor Core (HMX). The NPU portion is called **HTP** — "Hexagon Tensor Processor". | `qnn_htp` via the QNN SDK. |
| **ISP** (Spectra) | Image signal processor for camera. | Not used by us. |
| **Modem** (Snapdragon X75 5G) | Cellular baseband. | Not used. |
| **Sensor Hub** (always-on micro) | Low-power sensor fusion. | Not used. |

For ML / KV-cache compression workloads, the relevant mapping reduces to:

```
   CPU  ──►  cpu_scalar  +  cpu_neon          (ARM NEON intrinsics)
   GPU  ──►  opencl      +  vulkan            (Adreno via Khronos APIs)
   NPU  ──►  qnn_htp                          (Qualcomm QNN SDK)
```

## 2. Mobile lineup we care about

We track three generations because each one has a different Hexagon HTP
revision and a different Adreno revision. This matters: HVX intrinsics that
work on V75 may not be supported on V73, and Adreno 750 supports compute
features that Adreno 740 lacks.

| SoC | Phone (anchor) | CPU prime | Adreno | Hexagon | Process |
|---|---|---|---|---|---|
| **Snapdragon 8 Gen 2** (SM8550) | Galaxy S23 / OnePlus 11 | Cortex-X3 @ 3.2 GHz | 740 | V73 | TSMC N4 |
| **Snapdragon 8 Gen 3** (SM8650) | **Galaxy S24 (our target)** | Cortex-X4 @ 3.3 GHz (3.39 on Galaxy) | 750 | V75 | TSMC N4P |
| **Snapdragon 8 Elite** (SM8750) | Galaxy S25 / Xiaomi 15 | Oryon-class custom @ 4.32 GHz | 830 | V79 (HMX2) | TSMC N3E |

Why we care about three generations and not just our test device:

- **8 Gen 2** is the realistic floor for "shipped today, large install base"
  apps. If our QNN graph requires V75 HMX features it won't run on 8 Gen 2 and
  the user has to fall back to NEON. Our v1 explicitly stays inside the
  V73-compatible op subset.
- **8 Gen 3** is our test device — all numbers in `cpp/bench/results/` are
  measured on this part.
- **8 Elite** is the forward target. The Oryon CPU has SVE2 (still ARMv9-A),
  so a future NEON backend rev can become an SVE2 path with no algorithm
  change. Adreno 830 has materially higher INT8 / FP16 throughput than 750.

> **Note on the "for Galaxy" variant.** Samsung ships a clock-bumped SM8650
> SKU only on Galaxy phones (`X4 @ 3.39 GHz`, slightly faster Adreno 750).
> Functionally identical to the standard SM8650; ISA, NPU, and GPU feature
> set are the same. Bench numbers may run ~3–5 % faster on Galaxy than on a
> non-Galaxy 8 Gen 3 phone like the OnePlus 12.

## 3. Automotive lineup — three generations of Cockpit / Ride

Snapdragon Auto SoCs share most IP with mobile but lag mobile by a generation
or two and add safety / qualification features (ASIL-B, longer support
windows, thermal envelopes for in-vehicle deployments). The three SoCs that
matter for us, **as of 2025**:

| SoC | Generation | CPU | Hexagon | Adreno | Notes |
|---|---|---|---|---|---|
| **SA8155P** | 1st-gen Cockpit (≈ Snapdragon 855 era) | 8× Cortex-A76 | 690 | 640 | The "default" production Cockpit chip 2020-2024. Common in current vehicles. |
| **SA8295P** | 2nd-gen Cockpit (≈ Snapdragon 8 Gen 2 era) | 8× Cortex-A78AE (AE = Auto-Enhanced) | V69 | 690 | "Ride Flex" capable. Shipping in MY24/25 vehicles. |
| **SA8775P** | 3rd-gen Cockpit (Cockpit Plus / Ride; ≈ Snapdragon 8 Gen 3 era) | 8× Oryon-class (Auto-Enhanced) | V73 | 740-class | Snapdragon Digital Chassis flagship; MY26+ vehicles. |

> **Disclaimer (as of 2025):** Automotive SDK versioning is independent from
> mobile. The QNN SDK has separate "auto" build tracks that release on a
> different cadence (typically 1–3 quarters behind mobile). Hexagon NN /
> Hexagon SDK availability for automotive is access-controlled
> (Qualcomm-internal partner program) and not all HVX intrinsics that work on
> mobile V75 are documented as supported on auto V73 in the public SDK. When
> in doubt, treat the auto Hexagon revision as **the floor** of what the
> mobile revision of the same generation supports, not as identical.

We test against mobile and design against automotive. The C++ port never
references either: it sees only `IBackend`. Whether it's running on an S24
or an SA8295P dashboard at 80 °C is a toolchain-and-runtime choice the
backend factory makes.

## 4. Why mobile + automotive in one codebase

The whole point of the strict layering in `cpp/` (see
[`../architecture/system-overview.md`](../architecture/system-overview.md)) is
that **the same algorithm core and the same QNN graph code port unchanged
between mobile and automotive**, with only the toolchain swapped:

```
   ┌──────────────────────┐
   │  cpp/src/  (core)    │  ── pure C++17, no OS deps ──► moves anywhere
   │  cpp/backends/qnn_htp │  ── dlopens libQnnHtp.so  ──►  same .so name
   │  cpp/backends/opencl  │  ── dlopens libOpenCL.so  ──►  Adreno on auto
   └──────────────────────┘

   Toolchain swap:
       toolchain-android-arm64.cmake   (NDK r26+)        — S24 Ultra (mobile)
       toolchain-qnx-aarch64.cmake     (QCC)             — SA8295P / SA8775P
       toolchain-linux-aarch64.cmake   (aarch64 GCC)     — auto Linux
```

Concretely, this gives us:

1. **One algorithm file** (`cpp/src/quantizer.cpp`) for every SKU — mobile and
   automotive — so a bug fix on the S24 ships to the SA8775P with no port.
2. **One QNN graph definition** for `rotate` / `mse_score` / `value_dequant`
   that runs on V75 (mobile 8G3) and on V69/V73 (auto). HTP backends differ
   only in which `.so` is dlopened at runtime.
3. **One OpenCL kernel** per op that runs on Adreno 750 (mobile) and on
   Adreno 690-class auto GPUs. Code restricted to OpenCL 1.2 baseline so
   QNX-side OpenCL ICDs (which historically lag) accept it.
4. **One bench harness** (`cpp/bench/bench_cli.cpp`) that produces the same
   CSV format whether run on Android via `adb` or on a QNX target via `ssh`.

This is the rationale behind the "every backend must be automotive-portable"
hard requirement in the plan. It's also why we **never** add Android-specific
includes (`<android/log.h>`, `<jni.h>`) anywhere except the JNI shim — putting
those in the core would silently kill the QNX build.

## 5. Pointers to Qualcomm's official docs

Mobile / Snapdragon 8 Gen 3 / Adreno 750 / Hexagon V75:

- Snapdragon 8 Gen 3 product page:
  https://www.qualcomm.com/products/mobile/snapdragon/smartphones/snapdragon-8-series-mobile-platforms/snapdragon-8-gen-3-mobile-platform
- Qualcomm Neural Network SDK (QNN, formerly Qualcomm AI Engine Direct):
  https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
- Hexagon SDK landing:
  https://www.qualcomm.com/developer/software/hexagon-dsp-sdk
- Adreno GPU SDK + OpenCL programmer's guide:
  https://www.qualcomm.com/developer/software/adreno-gpu-sdk

Automotive:

- Snapdragon Cockpit / Ride / Digital Chassis overview:
  https://www.qualcomm.com/products/automotive
- Snapdragon Ride flexible compute (SA8295P / SA8775P):
  https://www.qualcomm.com/products/automotive/snapdragon-ride-platform

> **Caveat.** Automotive SDKs (QNN-Auto, Hexagon-Auto, Adreno-Auto) are gated
> behind a Qualcomm partner agreement; the public Developer Network pages
> describe capabilities but do not host SDK downloads. Replace the public-docs
> assumption with internal-portal access when you actually start an
> automotive build.

## 6. Where to go next

- To understand **how we actually use the NPU**, read
  [`hexagon-htp.md`](hexagon-htp.md). It is the longest and most important of
  the Qualcomm docs because the HTP path is the one that lights up the
  Qualcomm-distinctive hardware.
- To understand **how we actually use the GPU**, read
  [`adreno-gpu.md`](adreno-gpu.md).
- To understand **how this codebase ships into a car**, read
  [`automotive.md`](automotive.md).
