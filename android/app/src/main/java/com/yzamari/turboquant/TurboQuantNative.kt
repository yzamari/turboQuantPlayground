package com.yzamari.turboquant

/**
 * Thin JNI wrapper around libturboquant_jni.so.
 *
 * The .so bundles:
 *   - libturboquant (portable C++ core) statically linked in
 *   - the JNI shim (turboquant_jni.cpp)
 *   - the bench harness (header-only, included from cpp/bench/)
 */
object TurboQuantNative {

    init {
        System.loadLibrary("turboquant_jni")
    }

    /** Names of backends actually compiled into the .so AND able to init(). */
    external fun listBackends(): Array<String>

    /** Quick smoke check on a single backend. Returns a human-readable report. */
    external fun runCheck(backendName: String): String

    /**
     * Paired baseline-vs-TurboQuant benchmark across [seqLens].
     *
     * @param backendName one of the strings returned by [listBackends]
     * @param seqLens     sequence lengths to sweep
     * @param bits        key-quantization bits (2, 3, or 4)
     * @param bh          batch * heads
     * @param d           head dim
     * @return formatted multi-line table (header + one row per seq_len)
     */
    external fun runBenchmark(
        backendName: String,
        seqLens: IntArray,
        bits: Int,
        bh: Int,
        d: Int
    ): String
}
