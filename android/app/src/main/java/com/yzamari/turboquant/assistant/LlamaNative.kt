package com.yzamari.turboquant.assistant

/**
 * Thin Kotlin facade over the llama.cpp JNI shim (libturboquant_jni.so).
 *
 * All methods are blocking and **must** be called from a background
 * dispatcher (e.g. `Dispatchers.IO`). The `generate` method invokes
 * the supplied [GenCallback] from the same thread it is called on.
 */
object LlamaNative {

    init {
        // libturboquant_jni links to libllama.so + libggml*.so which are
        // packaged alongside it under jniLibs/arm64-v8a. Android's loader
        // pulls in deps automatically when SONAMEs match.
        System.loadLibrary("turboquant_jni")
    }

    /**
     * Streaming-token callback used by [generate]. Implementations should
     * return quickly — every token blocks the decoder.
     */
    fun interface GenCallback {
        fun onToken(piece: String)
    }

    /** Loads a GGUF model. Returns 0 on failure or a non-zero opaque handle. */
    @JvmStatic external fun loadModel(modelPath: String, contextSize: Int, threads: Int): Long

    /** Frees a model previously returned by [loadModel]. */
    @JvmStatic external fun unloadModel(handle: Long)

    /**
     * Runs a generation pass. Streams pieces through [cb] one at a time.
     * The model continues from its current KV-cache state; call [resetContext]
     * between distinct conversations.
     */
    @JvmStatic external fun generate(handle: Long, prompt: String, maxTokens: Int, cb: GenCallback)

    /** Tokenizes [text] and returns the count (no special tokens added). */
    @JvmStatic external fun tokenCount(handle: Long, text: String): Int

    /** Returns a JSON string with the most recent generation's stats. */
    @JvmStatic external fun getStats(handle: Long): String

    /**
     * Renders a list of messages through the model's built-in chat template.
     * [roles] and [contents] must be the same length.
     */
    @JvmStatic external fun applyChatTemplate(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        addAssistant: Boolean
    ): String

    /** Clears the model's KV-cache (start a fresh conversation). */
    @JvmStatic external fun resetContext(handle: Long)
}
