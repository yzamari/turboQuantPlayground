# llama-turboquant-kv

Source for the llama.cpp tool that integrates TurboQuant KV-cache
compression with a real on-device LLM.

These files are copied to `/tmp/llama.cpp/tools/turboquant_kv/` in
the llama.cpp checkout for the actual build:

```
cp -r external/llama-turboquant-kv-tool/* /path/to/llama.cpp/tools/turboquant_kv/
```

then add `add_subdirectory(turboquant_kv)` (gated on
`LLAMA_TOOL_TURBOQUANT_KV`) at the bottom of `tools/CMakeLists.txt` and
configure with:

```
cmake -S . -B build-android \
  -DLLAMA_TOOL_TURBOQUANT_KV=ON \
  -DTURBOQUANT_KV_DIR=$REPO/cpp \
  -DGGML_CCACHE=OFF
```

See [`docs/llamacpp-integration.md`](../../docs/llamacpp-integration.md)
for the full design, on-device numbers, and roadmap to a drop-in
`llama_kv_cache_turboquant` class.
