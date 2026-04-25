# Adreno GPU — Notes for the OpenCL & Vulkan Backends

The Adreno GPU is our **secondary** accelerator path. The primary is the
Hexagon HTP NPU; Adreno is what we light up when the workload is GEMM-shaped
but small enough that QNN graph-launch overhead would dominate, or when the
HTP path is unavailable (older Snapdragons, restricted automotive
deployments). It is also the easiest port from the existing Apple Metal
kernels — those kernels translate close to line-for-line into OpenCL or GLSL
compute.

## 1. Generations we care about

| GPU | SoC | Compute units (rough) | FP32 peak | FP16 peak | INT8 (dot4) peak |
|---|---|---|---|---|---|
| Adreno 730 | SD 8 Gen 1 / 8+ Gen 1 | 2 SP, 768 ALUs | ~2.0 TFLOPS | ~4.0 TFLOPS | ~8 TOPS |
| Adreno 740 | SD 8 Gen 2 (SM8550) | 3 SP | ~2.7 TFLOPS | ~5.4 TFLOPS | ~10–12 TOPS |
| **Adreno 750** | **SD 8 Gen 3 (SM8650) — our target** | 3–4 SP, ~1.5 GHz | ~3.0 TFLOPS | ~6.0 TFLOPS | ~13–15 TOPS |
| Adreno 830 | SD 8 Elite | 4 SP, ~1.7 GHz | ~4.6 TFLOPS | ~9.3 TFLOPS | ~20+ TOPS |

> Numbers are *order-of-magnitude*; Qualcomm does not publish official peak
> compute for Adreno. Treat them as "what we expect at the top of the
> roofline if we had perfect occupancy", and never as a benchmark target.

The 730/740/750/830 sequence is roughly +30 % per generation on FP16. For
KV-cache compression, where the dominant op is FP16 GEMM (the rotation
matmul), this is a meaningful gen-over-gen win on the GPU path **even though**
HTP is the better target on every modern SoC.

## 2. OpenCL on Adreno

This is the primary GPU-path API. On the S24:

```
$ adb shell ls /vendor/lib64 | grep -i opencl
libOpenCL.so          # Khronos ICD loader — what we dlopen
libOpenCL_adreno.so   # Qualcomm-specific Adreno ICD implementation
```

`libOpenCL.so` is what we dlopen at runtime. It is the standard Khronos ICD
loader and resolves to `libOpenCL_adreno.so` for Adreno GPUs.

### Version support

- **OpenCL 1.2 — fully supported, our baseline.** All four kernels we ship
  (`mse_score.cl`, `mse_encode.cl`, `qjl_score.cl`, `value_dequant.cl`) are
  written to 1.2 because:
  - Automotive Adreno OpenCL ICDs (QNX-side) historically lag mobile by
    one to two minor versions.
  - 1.2 has every feature we actually need (kernels, buffers, work-groups,
    barriers, FP16 via the `cl_khr_fp16` extension).
- **OpenCL 2.x — partially supported on Adreno.** SVM (shared virtual
  memory) and pipes are not reliable. Generic address spaces (`generic`)
  are supported on 730+. We do not use any 2.x feature in our kernels.
- **OpenCL 3.0 — Adreno 740+ advertise 3.0.** This is mostly a relabeling
  of 2.x with optional features; same caveats apply.

### Adreno-specific extensions worth knowing

These are `cl_qcom_*` extensions you'll see querying
`clGetDeviceInfo(CL_DEVICE_EXTENSIONS)`:

| Extension | What it does | Do we use it? |
|---|---|---|
| `cl_khr_fp16` | Half-precision floats in kernels (`half`, `half4`, ...) | **Yes** — FP16 path. |
| `cl_khr_subgroups` | Subgroup ops (warp-level reductions) | Future — useful for reduction kernels we don't have yet. |
| `cl_qcom_perf_hint` | Per-context "I want low-latency / high-throughput / balanced" hint | Considering for the `--bench` runs (low-latency hint for decode steps). |
| `cl_qcom_priority_hint` | Schedule this queue ahead of UI rendering | **Not** for us — we are a userspace bench, not a foreground compositor priority client. |
| `cl_qcom_dot_product8` | Native INT8 dot-product instruction | Future — would make a INT8 `mse_score` variant. |
| `cl_qcom_recordable_queues` | Pre-record dispatches into a replayable queue | Considering for the bench loop — replaces 5 `clEnqueueNDRangeKernel` calls per decode with one `clQcomReplayCommandQueue`. |

For the v1 port we deliberately use **none** of the `cl_qcom_*` extensions —
the kernels are vanilla 1.2 plus `cl_khr_fp16`. This keeps the code
automotive-portable. Once the bench shows a real bottleneck on
`clEnqueueNDRangeKernel` overhead, `cl_qcom_recordable_queues` is the first
thing to try.

### Loader strategy

The Android NDK does not ship OpenCL headers or a stub library. Our backend:

1. Vendors `cl.h` from Khronos under `cpp/backends/opencl/include_khronos/`.
2. dlopens `libOpenCL.so` at runtime via `dlopen("libOpenCL.so", RTLD_NOW)`.
3. Resolves the entry points we use (`clGetPlatformIDs`, `clGetDeviceIDs`,
   `clCreateContext`, `clCreateProgramWithSource`, `clBuildProgram`,
   `clCreateKernel`, `clSetKernelArg`, `clEnqueueNDRangeKernel`,
   `clEnqueueReadBuffer`, `clFinish`, `clCreateBuffer`,
   `clEnqueueWriteBuffer`, `clReleaseMemObject`, `clReleaseKernel`,
   `clReleaseProgram`, `clReleaseCommandQueue`, `clReleaseContext`).

Same `dlopen` works on QNX (with the QNX OpenCL package) and on automotive
Linux. Hence "automotive-portable, no algorithmic change".

## 3. Vulkan compute on Adreno

Our **secondary** GPU path. We add Vulkan because:

- Some Adreno OpenCL ICDs have driver bugs that Vulkan compute does not
  hit (different code path inside the Adreno blob).
- On automotive Linux, Vulkan support is generally healthier than OpenCL.
- Vulkan compute pipelines are more compositor-friendly if we ever embed
  this into an app that's also rendering UI on the same GPU (the WSI
  fences chain naturally).

### Version support

- **Vulkan 1.1 — solid on Adreno 730+.** Our baseline. Our shaders are
  GLSL `#version 450 core` with no compute extensions beyond the 1.1 feature
  set.
- **Vulkan 1.3 advertised on Adreno 750+.** Useful (dynamic rendering,
  push-descriptors, sync2) but not required by us.

### Useful Adreno-Vulkan features for our kernels

- **Subgroup ops** (`VK_VERSION_1_1`'s subgroup operations): wave-level
  reductions for sums (the `mse_score` accumulator). Useful for D=128
  reductions across 32 lanes.
- **`shaderFloat16` + `storageBuffer16BitAccess`**: lets us keep tensors in
  FP16 SSBOs and read them directly into `f16vec4` GLSL variables. The
  rotated keys are a perfect candidate.
- **Specialization constants**: bake `D`, `BITS`, `VALS_PER_BYTE` into the
  shader at pipeline-create time. SPIR-V then constant-folds the inner
  loops. Same trick the Metal kernels use with template parameters; same
  trick the OpenCL kernels use with `-DD=128`.

### Loader strategy

We use `volk` (a header-only Vulkan loader) so the link surface is
`-ldl` only. No `libvulkan.so` link-time dependency, which keeps us
permission-free on Android and trivial to cross-compile.

## 4. Common pitfalls — read this section twice

### 4.1 Per-kernel-launch overhead

Adreno's `clEnqueueNDRangeKernel` (and the Vulkan equivalent
`vkCmdDispatch`) costs roughly **50–100 µs of CPU time** per launch on the
S24. For us, dominant work is small (D=128, BH=8) so this is the killer
overhead, not GPU compute time. Mitigations, in order of effectiveness:

1. **Fuse kernels.** Our `mse_encode` already combines searchsorted +
   bit-pack into one kernel; ditto `value_dequant` (extract + dequant).
   Keep doing this.
2. **Avoid per-call buffer alloc.** Allocate work buffers once at backend
   `init()`. The plan's "no dynamic allocation in hot paths" rule is enforced
   for exactly this reason.
3. **Reuse `cl_program` and `cl_kernel` objects.** Build programs once;
   keep `cl_kernel` handles resident; only `clSetKernelArg` per call.
4. **Submit batched.** Bundle multiple kernels into one
   `clEnqueueNDRangeKernel` if shapes match (rare for us). Otherwise rely
   on the OOO command queue + a single `clFinish` at the end of the decode
   step.

### 4.2 Tiny buffer copies

Anything <4 KiB you `clEnqueueWrite` to the GPU goes through a slow path.
For the centroid table (32 floats × 4 bytes = 128 B) we upload **once at
backend init** and never touch it again — it lives in `__constant` memory
and is referenced by every `mse_score` call.

### 4.3 Don't go through host between kernels

Common bug: do `mse_score` on the GPU, copy result to host, do `qjl_score`
on host, copy back to GPU. We deliberately keep `qjl_score` on **NEON**, not
GPU, so the round trip never happens — the score buffer never leaves the
GPU between MSE and the final softmax in the bench harness.

### 4.4 The `rotate` builtin name — **important**

OpenCL 1.2 reserves `rotate(...)` as a built-in function (bitwise rotate of
integers — see Khronos OpenCL 1.2 spec, section 6.12.3). If you write
a kernel function named `rotate(...)`, your build will silently use the
builtin in some compiler versions. **We hit this** during the OpenCL
backend port — the bug presents as garbage rotation output with no compile
error.

**Convention for this codebase:** the kernel that performs `Q @ Pi^T` is
named `rotate_qkt` in `cpp/backends/opencl/kernels/*.cl`, **never**
`rotate`. The C-side `IBackend::rotate(...)` method is fine because it's a
C++ class member, not an OpenCL identifier.

If you are reviewing a future PR that introduces a kernel called `rotate`,
**block the PR.** Same applies to GLSL compute (Vulkan): GLSL doesn't
reserve `rotate` but the consistency-with-OpenCL convention is to keep the
name `rotate_qkt` everywhere.

### 4.5 Adreno work-group size sweet spot

For our kernel shapes (one thread per `(BH, N)` pair):

- **`mse_score`**: prefer `local_size = (min(N, 256), 1, 1)`. Adreno's wave
  is 64 wide; 256 is a 4-wave work-group, fills both compute pipes per SP.
- **`mse_encode`**: prefer `local_size = (min(PACKED_D, 64), 1, 1)`. The
  per-thread work is small (one byte of output); wider work-groups don't
  help.
- **`value_dequant`**: prefer `local_size = (min(D, 128), 1, 1)`.

The Apple Metal kernels use a `min(.x, 256)` rule which translates fine.
On Adreno you can sometimes do better at 128, sometimes at 256; let the
auto-tuner figure it out (planned, not in v1).

### 4.6 FP16 conversion is not free

`f16vec4` loads cost the same as `vec4` loads on Adreno 730+, but if your
arithmetic is FP32 internally and you only store-as-FP16, the converts add
up. Pattern we use: load FP16, **arithmetic in FP16**, store FP16. The
final reduction is FP32 to keep the score numerically stable, then we
write FP32 back.

## 5. Why not GPU-only?

It's a fair question — Adreno 750 is fast enough that we could in principle
keep everything on the GPU and skip the HTP path. We don't, because:

1. **HTP wins on FP16 GEMM by ~3×.** HMX's matrix throughput at FP16 is
   roughly 3× Adreno's at the same power envelope.
2. **HTP frees the GPU for other work.** A real app probably has the GPU
   busy with UI rendering or video; running our compute on HTP lets that
   coexist.
3. **Power.** HTP at full tilt draws ~1.5 W; Adreno at full tilt is closer
   to 4 W. On battery this matters.

Adreno is the **fallback** for SoCs without HTP (older Snapdragons in
automotive — SA8155P) and the **alternative** when QNN graph-launch
overhead is problematic.

## 6. Sources / further reading

- Adreno GPU SDK landing:
  https://www.qualcomm.com/developer/software/adreno-gpu-sdk
- Adreno OpenCL Programmer's Guide (look for the latest version on the
  Adreno SDK page):
  https://docs.qualcomm.com/bundle/publicresource/topics/80-NB295-11/index.html
- Khronos OpenCL 1.2 specification, section 6.12.3 (built-in functions):
  https://www.khronos.org/registry/OpenCL/specs/opencl-1.2.pdf — the
  authoritative answer for "what names are reserved".
- Vulkan-Hpp / volk (loader we use):
  https://github.com/zeux/volk
