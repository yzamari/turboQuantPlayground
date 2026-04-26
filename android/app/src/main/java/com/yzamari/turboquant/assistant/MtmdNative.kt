package com.yzamari.turboquant.assistant

/**
 * Thin Kotlin facade over the persistent mtmd JNI shim
 * (libturboquant_jni.so). Replaces the previous
 * Runtime.exec("libllama-mtmd-cli.so") fork-per-image pattern in
 * [VlmRunner] — the LLM and mmproj load once at app start, every
 * subsequent describe() call only pays for image-encode + token decode.
 *
 * All methods are blocking and **must** be called from a background
 * dispatcher (e.g. `Dispatchers.IO`). The `describe` method invokes the
 * supplied [GenCallback] from the same thread it is called on, once per
 * generated token piece.
 *
 * Backed by the same `libturboquant_jni.so` as [LlamaNative]. We call
 * `System.loadLibrary` from this object's static init too because the
 * Live tab can land on this class without ever touching [LlamaNative]
 * (user opens Live straight from a fresh launch). Without this init the
 * JNI symbol resolver fails: *No implementation found for long
 * MtmdNative.loadModel(...)*. `System.loadLibrary` is idempotent, so
 * calling it from both objects is safe.
 */
object MtmdNative {

    init {
        System.loadLibrary("turboquant_jni")
    }

    /** Streaming-token callback. Implementations should return quickly —
     *  every token blocks the decoder. */
    fun interface GenCallback {
        fun onToken(piece: String)
    }

    /**
     * Loads a VLM (LLM + mmproj pair) for persistent in-process inference.
     *
     * @param modelPath  GGUF path of the language model (e.g. SmolVLM-256M-Instruct-Q8_0.gguf).
     * @param mmprojPath GGUF path of the multimodal projector (e.g. mmproj-…-Q8_0.gguf).
     * @param contextSize llama_context size in tokens. 4096 is plenty for
     *                    a single image + 200-token description.
     * @param threads    CPU thread count for inference.
     * @param gpuLayers  99 = offload all transformer layers to Adreno via
     *                    ggml-opencl, 0 = CPU only. The mmproj/SigLIP
     *                    encoder follows the same toggle (use_gpu).
     * @param kvType     0 = FP16, 1 = q4_0 (4× compressed), 2 = q8_0 (2×),
     *                    3 = TurboQuant native (Path 2 scaffold —
     *                    currently FP16-equivalent; algorithm pending).
     * @return non-zero handle on success, 0 on failure.
     */
    @JvmStatic external fun loadModel(
        modelPath: String,
        mmprojPath: String,
        contextSize: Int,
        threads: Int,
        gpuLayers: Int,
        kvType: Int,
    ): Long

    /** Frees a VLM session previously returned by [loadModel]. */
    @JvmStatic external fun unloadModel(handle: Long)

    /**
     * Describes [imagePath] given a free-form text [prompt]. Streams the
     * model's reply through [cb] one token piece at a time.
     *
     * The KV cache is reset before every call — each describe() is an
     * independent single-turn conversation. (Multi-turn vision chat is
     * a follow-up; opening a new MtmdNative call before today's resets
     * the cursor.)
     */
    @JvmStatic external fun describe(
        handle: Long,
        imagePath: String,
        prompt: String,
        maxTokens: Int,
        cb: GenCallback,
    )

    /** Returns a JSON string with the most recent describe() call's stats:
     *  load_ms, mmproj_load_ms, image_eval_ms, n_image_tokens, n_prompt,
     *  n_decode, decode_ms, decode_tok_per_sec. */
    @JvmStatic external fun getStats(handle: Long): String
}
