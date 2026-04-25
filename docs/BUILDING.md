# Build Instructions — TurboQuant C++ port for Qualcomm Snapdragon

This document covers every supported build path: host CI, Android arm64, Linux
aarch64, QNX (automotive). The C++ core is plain CMake; only target toolchains
differ.

## Prerequisites

| Tool | Where to get it | Required for |
|------|------------------|--------------|
| **CMake ≥ 3.22** | `brew install cmake` (macOS), `apt install cmake` (Ubuntu) | every build |
| **Clang or GCC with C++17** | macOS: Xcode CLT; Linux: `apt install clang` | every build |
| **Android NDK r26+** | Android Studio → SDK Manager → NDK | Android build |
| **glslc** | `brew install glslang` (macOS), included with NDK at `ndk/.../shader-tools/.../glslc` | Vulkan backend |
| **Python 3.11+ venv** | repo root: `python -m venv venv && source venv/bin/activate && pip install -e ".[torch,dev]"` | gen_golden.py |
| **adb** (Platform Tools) | Android Studio → SDK Manager → Platform Tools | on-device test |
| **Qualcomm AI Engine Direct (QAIRT) 2.27.x** | Qualcomm Developer Network (license-walled) | QNN/HTP backend |
| **Gradle 8.7+ or Android Studio** | `brew install gradle` or AS | Android demo APK |

---

## 1. Host build — for CI / parity verification

The fastest path to verify everything builds and parity tests pass:

```bash
cmake -S cpp -B cpp/build-host
cmake --build cpp/build-host -j
ctest --test-dir cpp/build-host --output-on-failure
```

Expected output:
```
Test #1: packing ... Passed
Test #2: smoke   ... Passed
Test #3: parity  ... Passed   (727 / 727 byte-exact vs Python golden)
```

Only `TQ_WITH_CPU_SCALAR` is enabled by default on host. NEON is auto-skipped on
non-arm64 hosts.

---

## 2. Generating the golden corpus (one-time, if changed)

The parity test loads pre-generated golden binaries from `cpp/tests/golden/`.
These are checked into the repo; regenerate only if the algorithm changes:

```bash
source venv/bin/activate
python cpp/tools/gen_golden.py
# Produces 38 .bin files + manifest.json under cpp/tests/golden/
```

The script drives the existing Python `turboquant_mac` library with the
`pytorch` backend (CPU-deterministic) and seeds 42/1042.

---

## 3. Android arm64 build — the headline target

```bash
# Configure
cmake -S cpp -B cpp/build-android \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-android-arm64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTQ_WITH_NEON=ON \
  -DTQ_WITH_OPENCL=ON \
  -DTQ_WITH_VULKAN=ON

# Build
cmake --build cpp/build-android -j
```

Outputs (under `cpp/build-android/`):
- `bench/turboquant_bench` — paired A/B benchmark CLI
- `tests/{tq_packing_test,tq_smoke_test,tq_parity_test}` — test binaries

The toolchain file auto-detects the NDK at `~/Library/Android/sdk/ndk/<ver>/`; set
`ANDROID_NDK` env var to override.

### Push & run on device

```bash
# Binaries
adb push cpp/build-android/bench/turboquant_bench /data/local/tmp/
adb push cpp/build-android/tests/tq_parity_test /data/local/tmp/
# Golden corpus (small, ~200 KB)
adb shell mkdir -p /data/local/tmp/golden
adb push cpp/tests/golden/. /data/local/tmp/golden/

# Verify on-device
adb shell '/data/local/tmp/tq_parity_test /data/local/tmp/golden'
# → 727 / 727 checks passed

# Benchmark
adb shell '/data/local/tmp/turboquant_bench --bench --backend cpu_neon \
           --bits 3 --bh 8 --seq-lens 128,512,2048,4096 \
           --warmup 2 --iters 5 \
           --csv /data/local/tmp/s24-cpu_neon.csv'
adb pull /data/local/tmp/s24-cpu_neon.csv cpp/bench/results/
```

---

## 4. Activating QNN/HTP (Hexagon NPU)

Requires the Qualcomm AI Engine Direct (QAIRT) SDK. Free download from the
[Qualcomm Developer Network](https://www.qualcomm.com/developer/software) (account
+ license accept).

```bash
# After unpacking the SDK to e.g. ~/qairt/2.27.x
export QNN_SDK_ROOT=~/qairt/2.27.x

cmake --fresh -S cpp -B cpp/build-android-qnn \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-android-arm64.cmake \
  -DTQ_WITH_NEON=ON -DTQ_WITH_QNN=ON
cmake --build cpp/build-android-qnn -j

# Push the QNN runtime libs to the device
adb shell mkdir -p /data/local/tmp/qnn_libs
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so       /data/local/tmp/qnn_libs/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so /data/local/tmp/qnn_libs/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so    /data/local/tmp/qnn_libs/
adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/qnn_libs/

# Run with the QNN libs on the loader path
adb shell 'LD_LIBRARY_PATH=/data/local/tmp/qnn_libs \
           ADSP_LIBRARY_PATH=/data/local/tmp/qnn_libs \
           /data/local/tmp/turboquant_bench --bench --backend qnn_htp \
           --bits 3 --bh 8 --seq-lens 1024,4096'
```

See `cpp/backends/qnn_htp/README.md` for SoC-specific Hexagon variant tables (V73
for SD 8 Gen 2, V75 for SD 8 Gen 3, V79 for SD 8 Elite).

---

## 5. Android demo APK

The `android/` directory is a complete Android Studio / Gradle project. The
Gradle wrapper jar is **not** checked in; generate it once:

```bash
# Option A — command line
brew install gradle           # if you don't have it
cd android
gradle wrapper --gradle-version 8.7
./gradlew :app:assembleDebug

# Option B — Android Studio
# File → Open → select android/  → wait for sync → Run ▶
```

The APK lands at `android/app/build/outputs/apk/debug/app-debug.apk`.

### Install + run

```bash
adb install android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.yzamari.turboquant/.MainActivity
```

The app shows a backend selector (filtered to backends actually compiled into
the .so), seq-lens text field, bits chips, and a Run button that invokes the
paired baseline-vs-TurboQuant benchmark and displays the results table.

---

## 6. Linux aarch64 (automotive Linux placeholder)

Stub toolchain file at `cpp/cmake/toolchain-linux-aarch64.cmake`. Activate on a
machine with `aarch64-linux-gnu-gcc`:

```bash
cmake -S cpp -B cpp/build-linux-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-linux-aarch64.cmake \
  -DTQ_WITH_NEON=ON
cmake --build cpp/build-linux-arm64 -j
```

This is the path for SA8775P-class automotive Snapdragon running Linux. The core
+ NEON backend builds with no source changes from the Android arm64 build.

---

## 7. QNX aarch64 (automotive QNX placeholder)

Stub toolchain at `cpp/cmake/toolchain-qnx-aarch64.cmake`. Requires QNX SDP 7.1+:

```bash
source ~/qnx710/qnxsdp-env.sh
cmake -S cpp -B cpp/build-qnx \
  -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-qnx-aarch64.cmake \
  -DTQ_WITH_NEON=ON -DTQ_WITH_QNN=ON
cmake --build cpp/build-qnx -j
```

This is the path for SA8155P / SA8295P automotive Snapdragon running QNX
Neutrino. The core + NEON + QNN backends are designed to compile here unchanged
(the QNN SDK has a QNX target).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `adb: device unauthorized` | USB debugging not approved | tap "Allow" on phone, "Always allow from this computer" |
| `BuildProgram failed: definition of builtin function 'rotate'` | Adreno OpenCL reserves `rotate` | already fixed (kernel renamed `tq_rotate`); pull latest |
| `qnn_htp built as INTERFACE only` | `QNN_SDK_ROOT` not set | follow section 4 |
| Vulkan `VK_ERROR_INCOMPATIBLE_DRIVER` on macOS host | no MoltenVK | normal on host; works on Adreno on-device |
| Parity test loads but fails at d=128 b=4 | golden corpus stale | `python cpp/tools/gen_golden.py` to regenerate |
| `glslc not found` during Vulkan configure | shader compiler missing | `brew install glslang` or use `find_program(GLSLC ...)` from NDK shader-tools |
| `libOpenCL.so` missing on device | very rare; we dlopen from `/vendor/lib64/` | `adb shell ls /vendor/lib64/libOpenCL*` to verify |

---

## What runs in CI

`cmake -S cpp -B build && cmake --build build && ctest` runs in seconds and
covers the scalar backend + 727-check golden parity. NEON / OpenCL / Vulkan /
QNN backends are validated **on-device** via `adb shell <bench> --check-cross`
since CI hosts don't have Adreno or Hexagon hardware.
