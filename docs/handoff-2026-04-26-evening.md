# Handoff — 2026-04-26 (evening session)

Continuation of `handoff-2026-04-26.md`. Same project, same goal. This handoff
covers the tail of the day, after PR #23 (Qwen + QNN scaffold) merged.

## What landed this session

### Adreno OpenCL on Android — FIXED (branch `fix/adreno-opencl-uses-native-library`)

**Symptom (post-P2.0).** Both Tab S9+ (kalama) and S24 Ultra (pineapple) were
silently CPU-falling back at chat decode. Logcat showed
`ggml_opencl: platform IDs not available`.

**Root cause.** Our APK was bundling
`android/app/src/main/jniLibs/arm64-v8a/libOpenCL.so` — the Khronos ICD-loader
*stub* used at link time. At runtime the Android linker preferred this
in-APK lib over the device's real vendor loader at
`/vendor/lib64/libOpenCL.so`. Our stub has no ICD descriptors on device, so
`clGetPlatformIDs()` returned 0 platforms. ggml-opencl logged
"platform IDs not available" and let the CPU backend take over.

**Fix (two lines).**

1. `android/app/src/main/AndroidManifest.xml` — opt the app's linker
   namespace into the system-bundled `libOpenCL.so`:

   ```xml
   <uses-native-library android:name="libOpenCL.so" android:required="false" />
   ```

   `required="false"` keeps install OK on devices without the vendor lib
   (we just degrade to CPU there).

2. `android/app/build.gradle.kts` — drop our stub from the APK so the
   runtime resolver picks up `/vendor/lib64/libOpenCL.so`:

   ```kotlin
   packaging.jniLibs.excludes += "**/libOpenCL.so"
   ```

   The stub stays on disk in `jniLibs/` for CMake's *link* step.

**Why the manifest entry is required.** S24 Ultra's
`/vendor/etc/public.libraries.txt` does list `libOpenCL.so` as vendor-public,
but Android still requires apps to opt in via `<uses-native-library>` for
target SDK ≥ 30 (we're on 34). Without it, dlopen of the un-bundled name
fails: `dlopen failed: library "libOpenCL.so" not found in clns-9`. With it,
dlopen finds the vendor loader.

**Verification on S24 Ultra (board=pineapple, Adreno 750).**

- nativeloader log: `Configuring clns-9 ... uses_libraries=libOpenCL.so` ✓
- No `platform IDs not available` in logs ✓
- Adreno log lines stream during model layer load → real OpenCL upload ✓
- SmolVLM-256M decode: **9 tok/s** (Adreno) vs prior CPU-fallback baseline.
- VLM `loadModel` ready time: 13.0 s (mmproj 0.25 s)
- `mtmd_jni: describe: image eval done in 227391.9 ms` — see caveat below.

### Open caveat — mtmd vision tower still CPU

The 227 s for SigLIP-256M on a 480×640 frame is unchanged by this fix. That
is because llama.cpp's current `mtmd` library evaluates the vision tower on
the CPU regardless of `n_gpu_layers`. Path 2 of OpenCL acceleration would
require either:

- Patching upstream `mtmd_helper_eval_chunks` to dispatch the vision graph
  via `ggml-opencl`, OR
- Replacing the SigLIP encoder entirely with a QNN/HTP graph (matches the
  larger Path C plan from earlier today's handoff).

The LLM decoder *is* GPU-accelerated now — that's a real win for chat and
for the second half of any VLM call.

## State of the working tree

- **Branch.** `fix/adreno-opencl-uses-native-library`, 1 commit ahead of
  `origin/main`. Ready to push + PR.
- **Working tree.** Clean.
- **Installed APK on S24 Ultra (R5CX11REJ2X).** Includes the fix; live VLM
  test ran end-to-end (SmolVLM, Adreno + CPU SigLIP).
- **Tab S9+.** Not connected this session. Same fix should apply — Tab S9+
  is also a Snapdragon device with the vendor `libOpenCL.so`. Worth
  re-testing chat decode there separately.

## Open decisions

| ID | Question | Notes |
|---|---|---|
| **A1** | Push the OpenCL fix branch + open PR? | One-line summary: "fix(android): restore Adreno OpenCL via uses-native-library". Should be uncontroversial — verified working, tiny diff. |
| **A2** | Re-test Tab S9+ chat decode with the fix? | Tab S9+ Live VLM is independently broken (mtmd hang on Adreno) and that's already routed to CPU via the SoC heuristic. But chat decode (no images) on Tab S9+ should now use Adreno OpenCL. Wants verification. |
| **B**  | Path 2.1c (custom ggml attention op for kvType=3)? | Still pending. Per the algorithm playbook, this is the multi-week milestone that finally puts TurboQuant in the live decode path instead of the standalone verifier. Nothing here this session. |
| **C**  | mtmd-on-OpenCL or QNN vision tower? | Would unblock Live VLM perf on both devices. Multi-week — best treated as its own plan (brainstorming skill). |

## Suggested kick-off task for the next session

Push and PR-merge the OpenCL fix, re-test Tab S9+, then resume Path 2.1c —
unless something shifts the priority.

## Files touched this session

```
android/app/build.gradle.kts            (+10 lines: jniLibs.excludes)
android/app/src/main/AndroidManifest.xml (+10 lines: uses-native-library)
docs/handoff-2026-04-26-evening.md      (this file)
docs/handoff-2026-04-26-evening-prompt.md (kickoff for next session)
```

## Reproduction recipe (for the next session, in case you want to re-verify)

```sh
# Build + install on whichever device is connected
( cd android && ./gradlew :app:installDebug )

# Force-stop and relaunch
adb shell am force-stop com.yzamari.turboquant
adb shell am start -n com.yzamari.turboquant/.MainActivity

# Watch the loader bind libOpenCL.so to clns-9 (positive signal)
adb logcat -d | grep -E 'nativeloader.*turboquant|ggml_opencl|Adreno'
# Should see: 'uses_libraries=libOpenCL.so' (manifest entry honored)
# Should NOT see: 'ggml_opencl: platform IDs not available'
# DURING model load you'll see 'Adreno : ...' streaming = real GPU upload
```
