# turboQuantPlayground — Documentation

This folder collects the design / hardware / methodology notes for the
**C++ port of TurboQuant** living under `cpp/`. The original Python reference
implementation (`src/turboquant_mac/`) is the porting spec; the C++ port targets
**Qualcomm Snapdragon** mobile (Galaxy S24 Ultra, SM8650 / SD 8 Gen 3 for
Galaxy) and, by deliberate design, **Snapdragon automotive** SoCs
(SA8155P / SA8295P / SA8775P) without algorithmic rewrites.

The implementation plan itself lives outside this tree at
`/Users/yahavzamari/.claude/plans/squishy-jumping-fog.md`. These docs do
**not** duplicate the plan; they explain the hardware we are building on and
the architecture we chose, and they document benchmark methodology so results
are reproducible.

## Map

### `qualcomm/` — what we're running on

| File | What's inside |
|---|---|
| [`qualcomm/README.md`](qualcomm/README.md) | Snapdragon SoC anatomy. Mobile lineup (8 Gen 2 / Gen 3 / Elite). Automotive lineup (SA8155P / SA8295P / SA8775P). How each SoC block maps to our four backends. |
| [`qualcomm/hexagon-htp.md`](qualcomm/hexagon-htp.md) | Hexagon HTP (V73 / V75) deep dive. HVX vector model. QNN SDK. UDO (custom ops). Why the v1 hybrid HTP+NEON split. |
| [`qualcomm/adreno-gpu.md`](qualcomm/adreno-gpu.md) | Adreno 730 / 740 / 750 / 830. OpenCL vs Vulkan. Adreno-specific extensions. Pitfalls (per-launch overhead, the reserved `rotate` builtin name). |
| [`qualcomm/automotive.md`](qualcomm/automotive.md) | Cockpit / Ride / Digital Chassis. QNX vs Android Automotive vs Linux. ASIL implications for our design. The QNX toolchain stub. |

### `architecture/` — what we're building

| File | What's inside |
|---|---|
| [`architecture/system-overview.md`](architecture/system-overview.md) | Big ASCII diagram: Python reference -> C++ core -> `IBackend` -> 5 backends -> hardware. Data shapes annotated at every layer boundary. |
| [`architecture/data-flow.md`](architecture/data-flow.md) | One ASCII diagram per kernel (`mse_encode`, `mse_score`, `qjl_score`, `value_dequant`). Inputs, transformation, outputs. Mirrors the Metal kernel spec. |
| [`architecture/kv-cache-flow.md`](architecture/kv-cache-flow.md) | Prefill -> decode life cycle. Quantized portion + recent buffer. Buffer-flush mechanism. |
| [`architecture/backend-comparison.md`](architecture/backend-comparison.md) | All five backends side by side (where they run, throughput estimate, launch overhead, automotive availability, current status). When-each-one-wins guidance. |

### `benchmarks/` — how we measure

| File | What's inside |
|---|---|
| [`benchmarks/README.md`](benchmarks/README.md) | The paired baseline-vs-TurboQuant methodology. Every column in the CSV. Pass criteria. The exact `adb push` / `adb shell` commands to reproduce a run on the S24 Ultra. |

## Reading order

If you are new to the project, in this order:

1. `qualcomm/README.md` — what hardware we target.
2. `architecture/system-overview.md` — what the C++ port looks like at 10,000 ft.
3. `architecture/kv-cache-flow.md` — the central data structure.
4. `architecture/data-flow.md` — the four hot kernels.
5. `architecture/backend-comparison.md` — pick a backend.
6. `qualcomm/hexagon-htp.md` or `qualcomm/adreno-gpu.md` — go deep on the
   accelerator you'll touch.
7. `benchmarks/README.md` — measure, don't guess.

## Conventions in these docs

- All ASCII diagrams use Unicode box-drawing characters; they render correctly
  in any monospace font (terminal, GitHub, VS Code, IntelliJ).
- Tensor shapes use the Python-numpy convention: `[BH, N, D]` means batch-heads
  outer, tokens middle, head-dim innermost (row-major).
- "Mobile" means Snapdragon 8-series for handsets. "Automotive" means
  SA8xxxP family for cars. Mobile is verified end-to-end (we have an S24 Ultra
  on the desk); automotive is design-only at plan time.
- Wherever automotive specs are uncertain, the doc says **"as of 2025"** and
  flags that automotive QNN/OpenCL SDK versioning lags mobile by 1–3 quarters.
- Source citations are linked from the Qualcomm Developer Network where
  available; deep links are stable for the SDK landing pages but the per-API
  doc URLs change between SDK releases — always verify against the SDK version
  you actually downloaded.

## Adding to these docs

If you change behavior in the C++ port, the doc that mentions that behavior
must be updated in the same PR. The four files most likely to drift are:

- `architecture/data-flow.md` — if a kernel signature changes.
- `architecture/backend-comparison.md` — if a backend's status moves.
- `benchmarks/README.md` — if the CSV columns change.
- `qualcomm/hexagon-htp.md` — if the HTP/NEON split shifts (e.g. when we move
  `mse_encode` onto an HVX UDO).

The rest of the Qualcomm docs are mostly hardware reference and rarely change.
