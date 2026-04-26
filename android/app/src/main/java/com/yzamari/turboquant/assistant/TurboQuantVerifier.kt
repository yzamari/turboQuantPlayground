package com.yzamari.turboquant.assistant

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Runs the bundled `llama-turboquant-kv` tool against the loaded model and
 * returns the parsed summary. This is what the user sees when they tap
 * "Verify TurboQuant" in Settings — proof that the C++ TurboQuant library
 * (cpu_scalar / cpu_neon / opencl / vulkan backends) is loaded into the app
 * and actively quantizing the real K/V tensors of the active LLM.
 *
 * Architecture note: today the verifier RUNS the same TurboQuantKVCache class
 * that llama.cpp would use in Path 2. The path-2 integration (replacing the
 * llama_kv_cache during inference) is the next milestone — see
 * docs/llamacpp-integration.md.
 */
class TurboQuantVerifier(private val ctx: Context) {

    private val nativeDir = ctx.applicationInfo.nativeLibraryDir
    private val tqkvBin   = "$nativeDir/libllama-turboquant-kv.so"

    fun isAvailable(): Boolean = File(tqkvBin).canExecute()

    suspend fun verify(modelPath: String, prompt: String = "Hello world."): String =
        withContext(Dispatchers.IO) {
            if (!File(tqkvBin).exists()) return@withContext "TurboQuant binary missing at $tqkvBin"
            if (!File(modelPath).exists()) return@withContext "Model not found at $modelPath"

            val cmd = arrayOf(
                tqkvBin,
                "-m", modelPath,
                "-p", prompt,
                "-n", "1",
                "-t", "8",
                "-c", "512",
            )
            val pb = ProcessBuilder(*cmd)
                .redirectErrorStream(true)
            pb.environment()["LD_LIBRARY_PATH"] = nativeDir
            val proc = pb.start()
            val out = proc.inputStream.bufferedReader().readText()
            if (!proc.waitFor(120, TimeUnit.SECONDS)) {
                proc.destroyForcibly()
                return@withContext "(timed out after 120s)"
            }
            // Extract the summary block.
            val sumStart = out.indexOf("=== summary")
            if (sumStart < 0) return@withContext out.takeLast(2000)
            val sumEnd = out.indexOf("\n\n", sumStart).let { if (it < 0) out.length else it }
            out.substring(sumStart, sumEnd)
        }
}
