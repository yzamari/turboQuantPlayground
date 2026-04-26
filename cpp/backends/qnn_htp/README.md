# Qualcomm QNN / HTP NPU backend

This backend runs the rotation matmul and value-dequant kernels on the
Qualcomm Hexagon Tensor Processor (HTP, a.k.a. NPU) via the Qualcomm AI
Engine Direct (QNN) SDK. Everything else (`mse_encode`, `mse_score`,
`qjl_score`) is forwarded to the composed `cpu_neon` (or `cpu_scalar`)
backend — composition over inheritance.

## Why this directory ships empty by default

The QNN SDK is **not** redistributable. It requires a Qualcomm Developer
Network account and explicit license acceptance. We therefore gate the build
on the `QNN_SDK_ROOT` CMake variable / environment variable. When unset, the
CMake target degrades to an `INTERFACE` library with no sources and the rest
of the project still builds.

## Installing the QNN SDK

1. Sign in at <https://qpm.qualcomm.com/> and grab **Qualcomm AI Engine Direct
   (QAIRT)** — version `2.27.x` is what this scaffold is written against.
2. Unzip into a stable location, e.g. `~/sdk/qairt/2.27.0/`. The directory
   you point `QNN_SDK_ROOT` at must contain:

   ```
   include/QNN/QnnInterface.h
   include/QNN/HTP/QnnHtpDevice.h
   lib/aarch64-android/libQnnHtp.so
   lib/aarch64-android/libQnnSystem.so
   examples/Models/SampleApp/...
   ```

3. macOS hosts: the SDK ships only Linux/Windows host tools. You can still
   *configure* (this repo's CMake will skip the actual build of the .so when
   the host triple isn't supported). For real on-device runs, cross-compile
   from an aarch64 Linux host or via the Android NDK toolchain.

## Building

```sh
export QNN_SDK_ROOT=$HOME/sdk/qairt/2.27.0
cmake -S cpp -B cpp/build-android \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
      -DTQ_WITH_NEON=ON -DTQ_WITH_QNN=ON
cmake --build cpp/build-android -j
```

## Pushing runtime libs to the device

The `.so`s aren't statically linkable; they must be on the device search path
at run time. The standard layout for a non-rooted handset / EVK:

```sh
adb shell mkdir -p /data/local/tmp/qnn_libs
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so          /data/local/tmp/qnn_libs/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so   /data/local/tmp/qnn_libs/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so       /data/local/tmp/qnn_libs/
adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/qnn_libs/
```

Adjust `V75` to the target Hexagon version (`V73` for SA8295P, `V69` for
SA8155P, `V79` for Snapdragon 8 Gen 4).

Then for any `adb shell` invocation:

```sh
adb shell "LD_LIBRARY_PATH=/data/local/tmp/qnn_libs:\$LD_LIBRARY_PATH \
           ADSP_LIBRARY_PATH=/data/local/tmp/qnn_libs;/vendor/lib/rfsa/adsp \
           /data/local/tmp/turboquant_bench --backend qnn_htp ..."
```

## Existing on-device libs we can fall back to

Several Snapdragon devices already ship vendor QNN runtimes — handy when you
can't push the official SDK (e.g. retail handsets):

| Path                                          | Notes                                                          |
|-----------------------------------------------|----------------------------------------------------------------|
| `/vendor/lib64/snap/libQnnHtp.so`             | **Samsung Galaxy S24 Ultra (OneUI 6.1+)** — full QNN drop      |
| `/vendor/lib64/snap/libQnnHtpV75Stub.so`      | Hexagon V75 stub for SD 8 Gen 3                                |
| `/vendor/lib64/snap/libQnnSystem.so`          | Graph-system support lib                                       |
| `/vendor/lib64/rfs/dsp/snap/libQnnHtpV75Skel.so` | Hexagon DSP skel — picked up via FastRPC `ADSP_LIBRARY_PATH`   |
| `/vendor/lib64/libsnap_qnn.so`                | Samsung's wrapped QNN ("snap") — older OneUI builds            |
| `/vendor/lib64/libcdsprpc.so`                 | Hexagon RPC; required, always present (vendor-public)          |
| `/vendor/lib64/libQnnHtp*.so`                 | Newer OEM builds may ship this directly                        |

`qnn_loader.cpp` probes these in priority order and warns to stderr if none resolve.

> **App namespace caveat.** The Samsung-shipped runtimes under `/vendor/lib64/snap/`
> are *not* on `/vendor/etc/public.libraries.txt`, so a normal Android app cannot
> `dlopen()` them from its own linker namespace by default. To use them from
> `com.yzamari.turboquant`, add the relevant entry to `AndroidManifest.xml`:
>
> ```xml
> <uses-native-library android:name="libsnap_qnn.so" android:required="false"/>
> ```
>
> If that turns out to be insufficient (i.e. the Samsung snap libs aren't in the
> vendor-public allow-list at all), fall back to pushing the official QNN SDK
> runtime to `/data/local/tmp/qnn_libs/` per the section above — that path is
> always reachable.

## Building from inside the Android Gradle project

The Android app's `app/build.gradle.kts` automatically threads `QNN_SDK_ROOT`
(env or `gradle.properties` `turboquant.qnnSdkRoot`) to the CMake configure
step. If set, `TQ_WITH_QNN=ON` is forwarded to the libturboquant build and
the QNN backend appears in `TurboQuantNative.listBackends()` at runtime — the
**Bench tab** then shows it as a selectable backend chip.

```sh
# one-time per machine
echo "turboquant.qnnSdkRoot=$HOME/sdk/qairt/2.27.0" >> android/gradle.properties

# rebuild
( cd android && ./gradlew :app:installDebug )
```

## Numerical tolerances

- **FP16 path** (default on HTP): expect `<1e-3` absolute deviation from the
  scalar reference.
- **FP32 fallback** (set `use_fp32_fallback=true`, ASIL deployments): expect
  `<1e-4`.

## TODOs left in code

- `mse_score_graph` is stubbed; it currently delegates to NEON. Building it on
  HTP needs a custom-op for the bit-unpack-then-gather fused kernel — tracked
  in `qnn_graph.hpp`.

## SigLIP scaffold (Phase 1)

The `qnn_siglip.{hpp,cpp}` files land the framework for a SigLIP-base vision
encoder running on Hexagon HTP — the `Path C` plan. **Phase 1 is scaffolding
only**: every method currently throws `std::logic_error("unimplemented")`.
Subsequent commits on `path-c/qnn-siglip-phase1` fill in the seven build
steps (LayerNorm, Q/K/V triple, attention, residual, LN2, MLP+GELU,
residual). See `docs/path-c-qnn-siglip-design.md` for the full design.

### macOS host limitation

The Phase 1 test pipeline is:

1. Run `cpp/tools/siglip_block_ref.py` to load HuggingFace `SigLIP-base-patch16-224`,
   capture FP32 intermediates for block 0, dump them to
   `cpp/tests/golden/siglip_block_d768_h12.bin`.
2. Convert the same block weights to a QNN `.bin` via the QAIRT converter.
3. Build the C++ test with `-DTQ_WITH_QNN=ON -DQNN_SDK_ROOT=...` and run
   `tq_qnn_siglip_block_test` — cosine vs the golden ref must be ≥ 0.99.

Step 2 is a hard blocker on macOS: **the QAIRT 2.27.x converter binaries
ship Linux x86_64 only**. Three viable workarounds:

| Option | Description | Tradeoff |
|---|---|---|
| **A. Docker on macOS** | Wrap the converter in a Linux container. Build the image from a Qualcomm-provided base. | Local iteration; needs Docker Desktop + Qualcomm SDK acceptance inside the image. |
| **B. GitHub Actions Linux runner** | `siglip-convert.yml` workflow auto-converts on push, uploads the `.bin` as an artifact. | Hands-off CI; first install of QAIRT inside the runner is non-trivial (see TODO in the workflow). |
| **C. Linux pair box** | Run `siglip_block_ref.py` + the QAIRT converter on a Linux host, `rsync` the `.bin` back. | Simplest; needs hardware. |

Recommended: **B for CI + A for fast iteration**. See
`.github/workflows/siglip-convert.yml`.

### Running the (currently failing) test

The test executable builds on macOS hosts but *will fail* because
`QnnSigLipBlock::forward()` is a stub. It is NOT registered with `ctest`
to keep CI green — invoke it manually:

```sh
cmake -S cpp -B cpp/build-host -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build-host -j

# Generate the golden ref on a Linux box first (or pull the artifact from
# the siglip-convert workflow), then:
./cpp/build-host/cpp/tests/tq_qnn_siglip_block_test
```

Expected Phase 1 output: a single line `forward() threw (expected at
Phase 1): QnnSigLipBlock::forward unimplemented — Phase 1 scaffold only.`
followed by `1 / 1 checks passed (0 failures)` — that's the harness
catching the `std::logic_error` and treating it as the contract.
