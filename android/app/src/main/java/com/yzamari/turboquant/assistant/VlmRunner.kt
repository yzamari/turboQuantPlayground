package com.yzamari.turboquant.assistant

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Runs the bundled `llama-mtmd-cli` against an image to get a text description.
 *
 * Architecture:
 *   - llama-mtmd-cli is shipped as `libllama-mtmd-cli.so` in jniLibs/arm64-v8a,
 *     so the OS extracts it to <app>/lib/arm64-v8a/ at install time and we
 *     can Runtime.exec() it from there (Android 28+ allows execution out of
 *     the app's nativeLibraryDir).
 *   - libllama.so + libggml*.so + libmtmd.so live in the same directory and
 *     are on LD_LIBRARY_PATH for the child process.
 *   - The SmolVLM model + mmproj are expected in the app's external files
 *     dir; if absent the runner returns a friendly error string.
 *
 * Limitations:
 *   - First-run is slow (~7 s on Tab S9+) because the SigLIP vision encoder
 *     runs on CPU. Generation after that is ~50 tok/s.
 *   - The full mtmd-cli output is parsed heuristically — we look for the text
 *     between "image decoded" and the perf summary.
 */
class VlmRunner(private val ctx: Context) {

    private val nativeDir = ctx.applicationInfo.nativeLibraryDir
    private val mtmdBin   = "$nativeDir/libllama-mtmd-cli.so"

    fun isAvailable(): Boolean {
        val bin = File(mtmdBin)
        if (!bin.exists() || !bin.canExecute()) return false
        val files = ctx.getExternalFilesDir(null) ?: return false
        return File(files, MODEL_FILE).exists() && File(files, MMPROJ_FILE).exists()
    }

    fun missingFilesMessage(): String {
        val files = ctx.getExternalFilesDir(null)?.absolutePath ?: "<no external files dir>"
        val sb = StringBuilder()
        if (!File(mtmdBin).exists()) sb.appendLine("• vision binary missing at $mtmdBin")
        if (files != "<no external files dir>") {
            if (!File(files, MODEL_FILE).exists())  sb.appendLine("• missing $MODEL_FILE in $files")
            if (!File(files, MMPROJ_FILE).exists()) sb.appendLine("• missing $MMPROJ_FILE in $files")
        }
        sb.appendLine()
        sb.appendLine("Push from host:")
        sb.appendLine("  adb push <path>/$MODEL_FILE $files/")
        sb.appendLine("  adb push <path>/$MMPROJ_FILE $files/")
        return sb.toString().trim()
    }

    /** Result of one VLM run: the model's reply + a one-line stats string. */
    data class Result(val reply: String, val stats: String)

    suspend fun describe(
        imagePath: String,
        prompt: String = "Describe this image in one sentence.",
        maxTokens: Int = 60,
        threads: Int = 8,
    ): Result = withContext(Dispatchers.IO) {
        val filesDir = ctx.getExternalFilesDir(null)
            ?: return@withContext Result("External files dir not accessible.", "")
        val model  = File(filesDir, MODEL_FILE)
        val mmproj = File(filesDir, MMPROJ_FILE)
        if (!model.exists() || !mmproj.exists() || !File(mtmdBin).exists()) {
            return@withContext Result(missingFilesMessage(), "")
        }

        val cmd = arrayOf(
            mtmdBin,
            "-m",       model.absolutePath,
            "--mmproj", mmproj.absolutePath,
            "--image",  imagePath,
            "-p",       prompt,
            "-n",       maxTokens.toString(),
            "-t",       threads.toString(),
            "--no-warmup",
        )
        val env = arrayOf("LD_LIBRARY_PATH=$nativeDir")
        Log.i(TAG, "exec: ${cmd.joinToString(" ")}")
        val proc = Runtime.getRuntime().exec(cmd, env)
        val stdout = proc.inputStream.bufferedReader().readText()
        val stderr = proc.errorStream.bufferedReader().readText()
        proc.waitFor(120, TimeUnit.SECONDS)
        Log.d(TAG, "stdout=${stdout.length} bytes, stderr=${stderr.length} bytes")

        // The model's actual reply is the lines AFTER "image decoded" and
        // BEFORE the perf summary lines. Heuristic but reliable on mtmd-cli.
        val lines = stdout.lines()
        val start = lines.indexOfFirst { it.contains("image decoded") }
        val end   = lines.indexOfFirst { it.startsWith("llama_perf_") }
            .takeIf { it >= 0 } ?: lines.size
        val reply = if (start >= 0 && end > start)
            lines.subList(start + 1, end).joinToString(" ").trim()
        else
            stdout.trim()

        // Parse perf lines for tok/s. mtmd-cli prints e.g.:
        //   llama_perf_context_print: prompt eval time = ... (11.02 tokens per second)
        //   llama_perf_context_print:        eval time = ... (51.24 tokens per second)
        val tokensPerSec = Regex("([0-9.]+)\\s+tokens per second").findAll(stdout).toList()
        val (pp, gen) = if (tokensPerSec.size >= 2) {
            tokensPerSec[0].groupValues[1].toFloatOrNull() to tokensPerSec[1].groupValues[1].toFloatOrNull()
        } else if (tokensPerSec.size == 1) {
            null to tokensPerSec[0].groupValues[1].toFloatOrNull()
        } else null to null
        val stats = buildString {
            if (gen != null) append("VLM gen ${"%.1f".format(gen)} tok/s")
            if (pp  != null) {
                if (isNotEmpty()) append(" · ")
                append("prompt ${"%.1f".format(pp)} tok/s")
            }
        }

        Result(
            reply = reply.ifBlank { "(no reply parsed; stderr tail: ${stderr.takeLast(200)})" },
            stats = stats,
        )
    }

    companion object {
        private const val TAG         = "VlmRunner"
        private const val MODEL_FILE  = "SmolVLM-256M-Instruct-Q8_0.gguf"
        private const val MMPROJ_FILE = "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
    }
}
