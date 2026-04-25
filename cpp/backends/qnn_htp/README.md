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

| Path                              | Notes                                  |
|-----------------------------------|----------------------------------------|
| `/vendor/lib64/libsnap_qnn.so`    | Samsung's wrapped QNN ("snap")         |
| `/vendor/lib64/libcdsprpc.so`     | Hexagon RPC; required, always present  |
| `/vendor/lib64/libQnnHtp*.so`     | Newer OEM builds may ship this directly|

`qnn_loader.cpp` searches these paths in order and warns to stderr if none
resolve.

## Numerical tolerances

- **FP16 path** (default on HTP): expect `<1e-3` absolute deviation from the
  scalar reference.
- **FP32 fallback** (set `use_fp32_fallback=true`, ASIL deployments): expect
  `<1e-4`.

## TODOs left in code

- `mse_score_graph` is stubbed; it currently delegates to NEON. Building it on
  HTP needs a custom-op for the bit-unpack-then-gather fused kernel — tracked
  in `qnn_graph.hpp`.
