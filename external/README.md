# `external/` — vendored upstreams

External dependencies that ship as part of this repo via git submodule or
checked-in source. Each entry is pinned; bumps are explicit commits, not
silent floats.

## `llama.cpp/` — submodule

- **Source of truth:** `https://github.com/yzamari/llama.cpp.git`
  (long-running fork of `ggml-org/llama.cpp`).
- **Branch:** `tq-main` — our patched branch. Initially equals upstream
  release `b8935`. The TurboQuant KV cache patch (Path 2.1) lands on this
  branch.
- **Pinned commit:** `f454bd7eb8944629aabca163ea1c6e67e53fd77e`
  (upstream tag `b8935`, *opencl: add iq4_nl support*, 2026-04-26).
- **Why this pin:** closest stable tag to the commit that produced the
  prebuilt `.so`s previously shipped under
  `android/app/src/main/jniLibs/arm64-v8a/`. Carries the OpenCL +
  Hexagon work needed for our Adreno + QNN paths.
- **Why a fork rather than a clean submodule of upstream:** Path 2 patches
  `llama-kv-cache-unified.{h,cpp}`, `llama-graph.cpp`, and adds a new
  `ggml/src/ggml-custom-turboquant.c`. Long-running patch; rebased onto
  upstream `master` periodically. Eventual upstream PR is the goal.

### Bumping the pin

```bash
cd external/llama.cpp
git fetch origin
# pick a new upstream commit/tag
git rebase <upstream-master-or-tag>   # rebase tq-main onto it
git push --force-with-lease origin tq-main
cd ../..
git add external/llama.cpp
git commit -m "external: bump llama.cpp pin to <new-sha>"
```

### Cloning fresh

```bash
git clone --recurse-submodules https://github.com/yzamari/turboQuantPlayground.git
# or, if you already cloned:
git submodule update --init --recursive
```

## `llama-turboquant-kv-tool/` — checked-in source

Path 1 standalone TurboQuant verifier. Loads a real GGUF, queries layer
geometry from the live llama.cpp context, runs `libturboquant` on
shape-matched K/V. Independent of the chat hot path. Stays in this repo
because it's our code, not vendored.
