# Snapdragon Automotive

This document explains the **automotive** Snapdragon family that the C++ port
is forward-targeted at — the same reason every backend in `cpp/backends/` is
gated through portable `dlopen` and a CMake toolchain file rather than
hard-coded Android NDK paths. We do not have an automotive board on the desk;
the work here is **design-time portability** so that when an SA8295P or
SA8775P arrives, the build is a toolchain swap, not a rewrite.

> **Automotive caveat (as of 2025).** Qualcomm's automotive SDK versioning is
> independent from mobile and is gated behind a partner program. Anything in
> this doc that names specific Hexagon revisions, OS minor versions, or QNN
> SDK versions is best-effort public information; treat it as orientation,
> not contract.

## 1. The three product lines

Qualcomm's automotive offering is bucketed into three product lines that
share a lot of underlying silicon but target different jobs in the car.

```
                 ┌──────────────────────────────────────┐
                 │     Snapdragon Digital Chassis       │
                 │  (the umbrella platform brand)       │
                 └──────────┬───────────────────────────┘
                            │
        ┌───────────────────┼─────────────────────┐
        │                   │                     │
        ▼                   ▼                     ▼
┌──────────────┐   ┌──────────────────┐   ┌────────────────┐
│   Cockpit    │   │      Ride        │   │ Telematics /   │
│  (in-cabin   │   │ (autonomous-     │   │ Connectivity   │
│   IVI, HMI,  │   │  driving / ADAS) │   │ (modems, V2X)  │
│   passenger  │   │                  │   │                │
│   compute)   │   │                  │   │                │
└──────────────┘   └──────────────────┘   └────────────────┘
   SA8155P             SA8540P (Ride 1)        Modems we
   SA8255P             SA8650P (Ride 2)        do not target
   SA8295P
   SA8775P (Cockpit Plus / Ride flexible)
```

For us, **Cockpit** is the realistic target — it is the SoC running
infotainment / digital instrument cluster / passenger displays, and it has
the kind of NPU + GPU resources that make sense for KV-cache compression
work. The Ride family is dominated by safety-of-life ADAS workloads where
running an LLM is far from the primary mission. The Cockpit Plus variants
(SA8775P) blur the line — they have Ride-flex compute available for
in-cabin AI features.

## 2. The three Cockpit generations we track

| SoC | Generation | Approx mobile peer | CPU | NPU | GPU | OS we expect |
|---|---|---|---|---|---|---|
| **SA8155P** | 1st-gen | SD 855 (≈ 2019) | 8× Cortex-A76 | Hexagon 690 (no HMX) | Adreno 640 | QNX 7.x / Android 11 / Linux |
| **SA8295P** | 2nd-gen | SD 8 Gen 2 (≈ 2022) | 8× Cortex-A78AE | Hexagon V69 (HMX gen 1) | Adreno 690 | QNX 7.x / Android 13 / Linux |
| **SA8775P** | 3rd-gen | SD 8 Gen 3 (≈ 2024) | 8× Oryon-class (Auto-Enhanced) | Hexagon V73 (HMX gen 2) | Adreno 740-class | QNX 8.x / Android 14 / Linux |

"Auto-Enhanced" (the `AE` suffix on the Cortex-A78AE) means the CPU has
built-in support for **lockstep / split-lock execution modes** required for
ASIL-B and ASIL-D certification. Functionally for us, AE cores are a strict
superset of the non-AE variants — anything that compiles for A78 compiles for
A78AE, and the safety features are runtime-configurable.

## 3. Which OS does the SoC run

This is the most-asked question after "which Hexagon revision is it" and the
honest answer is "depends on the OEM build". The realistic options:

```
   OS                          Vendor coverage              Our story
   ─────────────────────────   ──────────────────────────   ─────────────────────
   QNX Neutrino 7.x / 8.x      Most common in production    Toolchain stub ready
                               in-vehicle Cockpit today.    (cpp/cmake/toolchain-
                               BlackBerry/QNX licensed.     qnx-aarch64.cmake)
                               POSIX-ish, real-time.
   Android Automotive 13/14    Volvo, Polestar, GM, Ford.   Android NDK builds
                               Same NDK as mobile but        unchanged on AAOS.
                               "automotive" SDK target.     Already verified on
                                                            mobile NDK r26.
   Embedded Linux              Yocto-based; some OEMs use    Linux-aarch64
                               Mercedes' MB.OS, BMW's        toolchain stub
                               IDP, etc. on top of a        committed.
                               Linux base.
   Hypervisor + multiple       SA8295P / SA8775P often       The core .a builds
   guests (QNX + Android)      run a Type-1 hypervisor      for whichever guest
                               (QNX Hypervisor, Hypervisor   has the SDK; QNN
                               for Apps) with QNX as the    runtime libs are
                               cluster guest and Android    OEM-specific so
                               as IVI guest.                we do not bundle.
```

Our **algorithm core** (`cpp/src/`) is `<jni.h>`-free, `<android/log.h>`-free,
allocation-free in the hot path, and uses only C++17 standard headers. This
is what makes "build for any of those OSes with a toolchain swap" actually
true. It is also the single hardest invariant to keep across PRs — see the
plan file's "Automotive portability — design rules" section.

### What changes per OS

| Item | Android | QNX | Linux |
|---|---|---|---|
| Compiler | NDK r26+ Clang | QCC (QNX-tuned Clang) | aarch64-linux-gnu-gcc |
| `libdl.so` for dlopen | ✓ | ✓ (QNX has `dlopen`) | ✓ |
| QNN runtime path | `/data/local/tmp/qnn` | OEM-defined, often `/usr/lib/qnn` | `/opt/qnn/lib` |
| OpenCL ICD | `libOpenCL_adreno.so` | QNX OpenCL package | Mesa or vendor blob |
| Vulkan loader | `libvulkan.so` | QNX Vulkan package | `libvulkan.so.1` |
| Filesystem | restricted (Android) | flexible | flexible |

**Filesystem is the trap.** Android's per-app sandboxing means the QNN
backend cannot read arbitrary paths; this is why our codebooks ship as
embedded byte arrays (no JSON parsing at runtime, no filesystem assumption).
That same constraint is exactly right for QNX, where you may not have a
writable filesystem at all in an automotive partition.

## 4. ASIL implications

We don't claim ASIL certification — that's the integrator's job, not the
library's. But our design choices need to **not preclude** an integrator from
certifying their product.

| Concern | Our response |
|---|---|
| **Determinism (ASIL-B+).** Same input must produce the same output bit-for-bit for integer ops; bounded for floats. | We require bit-exact for INT outputs, `<1e-4` for FP32, `<1e-3` for FP16. Tested in `cpp/tests/parity_test.cpp`. |
| **No dynamic allocation in hot path.** Heap ops are non-deterministic on RTOS scheduling. | All work buffers allocated at backend `init()`, reused. Plan rule #2. |
| **No exceptions in hot path.** Exception unwinding is implementation-defined latency on QNX. | Core is `noexcept` on the `IBackend::*` methods; errors returned via bool/int. |
| **No filesystem at runtime.** Auto partitions may be read-only. | Codebooks embedded as `const unsigned char[]`. Plan rule #4. |
| **FP32 fallback for safety-relevant deployments.** FP16 has insufficient mantissa for some ASIL-B numerical envelopes. | QNN backend has a constructor flag `--fp32-fallback` selecting an FP32 graph. Documented in `qnn_htp/README.md`. |
| **Side-channel of dynamic linking.** Some auto deployments forbid `dlopen` after init. | All `dlopen` calls happen in `IBackend::init()`. Hot path uses pre-resolved function pointers. |
| **Watchdog interaction.** QNX guests run watchdog timers that kill any thread that's quiet for too long. | Per-call latency budget is microseconds for NEON, sub-millisecond for HTP. We don't run inside any single dispatch long enough to upset a watchdog. |
| **Coverage testing.** ASIL-B integrators will MC/DC-cover the code they ship. | Backends are independently selectable at compile time so an integrator can ship only the backends they certify (e.g. CPU + QNN, no GPU). |

We deliberately do **not** ship a "safety mode" runtime flag. The right
abstraction for safety-critical deployments is "compile only the backends
you want, in a build configuration you've certified". The CMake options
`TQ_WITH_NEON / TQ_WITH_QNN / TQ_WITH_OPENCL / TQ_WITH_VULKAN` exist for
exactly this.

## 5. How our design transfers — concretely

The repo already commits the toolchain file stubs:

```
cpp/cmake/toolchain-qnx-aarch64.cmake     # stub for QNX 7.x / 8.x with QCC
cpp/cmake/toolchain-linux-aarch64.cmake   # stub for embedded Linux
```

A QNX build (when QCC is in `$PATH`) is:

```
cmake -S cpp -B build-qnx \
      -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-qnx-aarch64.cmake \
      -DTQ_WITH_NEON=ON \
      -DTQ_WITH_QNN=ON \
      -DTQ_WITH_OPENCL=ON \
      -DTQ_WITH_VULKAN=OFF \         # QNX Vulkan support is uneven
      -DQNN_SDK_ROOT=/opt/qcom/qnn-auto-2.x

cmake --build build-qnx -j8
```

The expectation is **everything builds without per-file changes**. If a
file under `cpp/src/` fails to compile because of an Android-specific
include or header, that's a regression and the PR that introduced it must be
fixed before landing.

Things that legitimately need OS handling are exactly two:

1. **dlopen path resolution** — the *path string* differs per OS. Solved by
   making it a build-time `#define` from CMake (`TQ_QNN_LIB_PATH`) with
   sensible per-OS defaults.
2. **Logging** — Android wants `__android_log_print`, QNX wants `slog2`,
   Linux wants `fprintf(stderr, ...)`. Solved by a function-pointer hook the
   *host* installs on the core; the core never logs by itself.

Everything else is portable C++17 and lives in `cpp/src/`.

## 6. Why we explicitly do NOT design for ADAS / Snapdragon Ride

KV-cache compression is an inference-throughput / memory-footprint
optimization for transformer language models. The Snapdragon Ride workload
is sensor-fusion / perception / planning, where the latency tail matters
more than throughput, and where the compute is dominated by CNN /
transformer-encoder ops with very different shapes from LLM decode. Our
algorithm doesn't help that workload, so we deliberately scope ourselves to
**Cockpit / IVI** deployments where in-cabin AI assistants and
on-device LLMs are the user-visible feature.

If a Ride deployment ever wants the same algorithm (unlikely but possible),
the *core* still works — but `IBackend::*` would need plumbing to whatever
DSPs Ride uses (some Ride parts have separate "Ride NPU" silicon). That's
out of scope for this plan.

## 7. Sources

- Snapdragon Digital Chassis overview:
  https://www.qualcomm.com/products/automotive
- Snapdragon Cockpit (product line):
  https://www.qualcomm.com/products/automotive/snapdragon-cockpit-platforms
- Snapdragon Ride (product line):
  https://www.qualcomm.com/products/automotive/snapdragon-ride-platform
- ISO 26262 / ASIL primer (third-party explainer; Qualcomm doesn't host one):
  https://www.iso.org/standard/68383.html
- QNX SDP product page (the OS most Cockpits run):
  https://blackberry.qnx.com/en/products/foundation-software/qnx-sdp
