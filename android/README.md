# TurboQuant Bench — Android demo

A single-screen Compose app that runs the TurboQuant C++ benchmark on-device.
Picks a backend, sweeps a few sequence lengths, and prints a paired
baseline-vs-TurboQuant table for each.

## What's in here

```
android/
  build.gradle.kts            # root project (Kotlin DSL)
  settings.gradle.kts
  gradle.properties
  app/
    build.gradle.kts          # :app — Compose + externalNativeBuild
    src/main/
      AndroidManifest.xml
      cpp/
        CMakeLists.txt        # add_subdirectory(<repo>/cpp …) + JNI shim
        turboquant_jni.cpp    # the only Android-specific C++ in the repo
      java/com/yzamari/turboquant/
        App.kt                # Compose UI
        MainActivity.kt
        TurboQuantNative.kt   # external fun bindings
      res/values/
        strings.xml
        themes.xml
```

The portable C++ core lives at `<repo>/cpp/`; this module pulls it in via
`add_subdirectory` and layers a JNI shim on top. The JNI shim re-uses the
bench harness from `cpp/bench/bench_runner.hpp` (header-only) and
`cpp/bench/baseline_kv_cache.cpp` — nothing Android-specific leaks into
the core library.

## Building

The Gradle wrapper jar/scripts are **not** checked in. Generate them once,
then build the APK:

```bash
cd android
gradle wrapper --gradle-version 8.7
./gradlew :app:assembleDebug
```

…or just open `android/` in Android Studio (Hedgehog or newer) and let it
generate the wrapper for you.

The signed-debug APK lands at:

```
android/app/build/outputs/apk/debug/app-debug.apk
```

## Installing & running

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.yzamari.turboquant/.MainActivity
```

In the app:

1. Pick a backend from the dropdown — only backends that compiled in **and**
   successfully `init()` on the device appear.
2. Edit the seq-len list (default `128,512,2048`).
3. Pick the bit width (2 / 3 / 4).
4. Tap **Run benchmark**. Results stream into the scrollable text area
   below.

## Backend matrix

| Backend      | Compiled here? | Notes                                                |
|--------------|---------------:|------------------------------------------------------|
| `cpu_scalar` | yes            | always works                                         |
| `cpu_neon`   | yes            | armv8-a NEON                                         |
| `opencl`     | yes            | needs an OpenCL ICD on the device (Adreno OK)        |
| `vulkan`     | yes            | Vulkan 1.1+                                          |
| `qnn_htp`    | **no**         | requires the QNN SDK; flip `TQ_WITH_QNN=ON` once installed |

## ABIs

Only `arm64-v8a` is built. Adjust `abiFilters` in `app/build.gradle.kts`
if you need 32-bit (you don't — TurboQuant is 64-bit only).
