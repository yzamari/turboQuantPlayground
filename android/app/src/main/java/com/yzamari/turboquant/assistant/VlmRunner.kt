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

    /** Selectable VLM models. Each entry maps to a (gguf, mmproj) pair sitting
     *  in the app's external files dir. */
    enum class Model(val displayName: String, val modelFile: String, val mmprojFile: String) {
        SMOLVLM_256M(
            "SmolVLM-256M (small / English-leaning)",
            "SmolVLM-256M-Instruct-Q8_0.gguf",
            "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
        ),
        QWEN25_VL_3B(
            "Qwen2.5-VL-3B (multilingual / much better quality)",
            "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        );
    }

    /** Currently-selected VLM model. Mutable so Settings can change it live. */
    @Volatile
    var activeModel: Model = Model.SMOLVLM_256M

    private val nativeDir = ctx.applicationInfo.nativeLibraryDir
    private val mtmdBin   = "$nativeDir/libllama-mtmd-cli.so"

    fun isAvailable(model: Model = activeModel): Boolean {
        val bin = File(mtmdBin)
        if (!bin.exists() || !bin.canExecute()) return false
        val files = ctx.getExternalFilesDir(null) ?: return false
        return File(files, model.modelFile).exists() && File(files, model.mmprojFile).exists()
    }

    /** All models that have their .gguf + mmproj on disk right now. */
    fun availableModels(): List<Model> = Model.values().filter { isAvailable(it) }

    fun missingFilesMessage(model: Model = activeModel): String {
        val files = ctx.getExternalFilesDir(null)?.absolutePath ?: "<no external files dir>"
        val sb = StringBuilder()
        if (!File(mtmdBin).exists()) sb.appendLine("• vision binary missing at $mtmdBin")
        if (files != "<no external files dir>") {
            if (!File(files, model.modelFile).exists())  sb.appendLine("• missing ${model.modelFile} in $files")
            if (!File(files, model.mmprojFile).exists()) sb.appendLine("• missing ${model.mmprojFile} in $files")
        }
        sb.appendLine()
        sb.appendLine("Push from host:")
        sb.appendLine("  adb push <path>/${model.modelFile} $files/")
        sb.appendLine("  adb push <path>/${model.mmprojFile} $files/")
        return sb.toString().trim()
    }

    /** Result of one VLM run: the model's reply + a one-line stats string. */
    data class Result(val reply: String, val stats: String)

    /** Stream the VLM's tokens to a callback as they appear. */
    suspend fun describe(
        imagePath: String,
        prompt: String = "Describe this image in detail.",
        maxTokens: Int = 200,
        threads: Int = 8,
        onToken: ((String) -> Unit)? = null,
    ): Result = withContext(Dispatchers.IO) {
        val startMs = System.currentTimeMillis()
        val filesDir = ctx.getExternalFilesDir(null)
            ?: return@withContext Result("External files dir not accessible.", "")
        val sel    = activeModel
        val model  = File(filesDir, sel.modelFile)
        val mmproj = File(filesDir, sel.mmprojFile)
        if (!model.exists() || !mmproj.exists() || !File(mtmdBin).exists()) {
            return@withContext Result(missingFilesMessage(sel), "")
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
        // Merge stderr → stdout so we see "image decoded" / "llama_perf_*"
        // sentinels (printed on stderr) and the model's actual reply (stdout)
        // in one unified stream.
        val pb = ProcessBuilder(*cmd)
            .redirectErrorStream(true)
        for (e in env) {
            val (k, v) = e.split('=', limit = 2)
            pb.environment()[k] = v
        }
        val proc = pb.start()

        // Read everything synchronously. Periodically emit the parsed reply
        // text so far via onToken so the UI sees the description forming.
        val stdoutBuf = StringBuilder()
        var lastEmittedLen = 0
        val readerThread = Thread {
            try {
                val br = proc.inputStream.bufferedReader()
                val buf = CharArray(512)
                while (true) {
                    val n = br.read(buf)
                    if (n < 0) break
                    stdoutBuf.append(buf, 0, n)
                    // Live-parse the reply window and emit any new portion.
                    val s = stdoutBuf.indexOf("image decoded")
                    if (s >= 0) {
                        val replyStart = stdoutBuf.indexOf('\n', s + 14).let { if (it < 0) -1 else it + 1 }
                        if (replyStart > 0) {
                            val perfIdx = stdoutBuf.indexOf("llama_perf_", replyStart)
                            val end = if (perfIdx >= 0) perfIdx else stdoutBuf.length
                            if (end > replyStart + lastEmittedLen) {
                                val piece = stdoutBuf.substring(replyStart + lastEmittedLen, end)
                                lastEmittedLen = end - replyStart
                                if (piece.isNotEmpty()) onToken?.invoke(piece)
                            }
                        }
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "reader thread: $t")
            }
        }.apply { isDaemon = true; start() }

        // Larger VLMs (Qwen2.5-VL-3B) need more time on CPU than SmolVLM.
        val timeoutSec = when (activeModel) {
            Model.SMOLVLM_256M -> 180L
            Model.QWEN25_VL_3B -> 900L  // 15 min — first vision-encode on CPU is slow
        }
        if (!proc.waitFor(timeoutSec, TimeUnit.SECONDS)) {
            proc.destroyForcibly()
            return@withContext Result(reply = "(VLM timed out after ${timeoutSec}s)", stats = "")
        }
        readerThread.join(3000)
        val stdout = stdoutBuf.toString()
        val stderr = ""  // merged into stdout above
        Log.d(TAG, "merged stdout=${stdout.length} bytes; head=${stdout.take(200)}")
        // Robust reply extraction from the merged stdout. The model's text
        // sits between "image decoded (batch …) in N ms\n\n" and the first
        // "llama_perf_" line.
        val reply = run {
            val s = stdout.indexOf("image decoded")
            if (s < 0) return@run stdout.trim()
            val nl = stdout.indexOf('\n', s + 14)
            if (nl < 0) return@run ""
            val e = stdout.indexOf("llama_perf_", nl + 1)
            val end = if (e >= 0) e else stdout.length
            stdout.substring(nl + 1, end).trim()
        }

        // Parse perf lines for tok/s. mtmd-cli prints e.g.:
        //   llama_perf_context_print: prompt eval time = ... (11.02 tokens per second)
        //   llama_perf_context_print:        eval time = ... (51.24 tokens per second)
        val tokensPerSec = Regex("([0-9.]+)\\s+tokens per second").findAll(stdout).toList()
        val (pp, gen) = if (tokensPerSec.size >= 2) {
            tokensPerSec[0].groupValues[1].toFloatOrNull() to tokensPerSec[1].groupValues[1].toFloatOrNull()
        } else if (tokensPerSec.size == 1) {
            null to tokensPerSec[0].groupValues[1].toFloatOrNull()
        } else null to null
        val elapsedMs = System.currentTimeMillis() - startMs
        val elapsedSec = elapsedMs / 1000.0
        val modelLabel = when (activeModel) {
            Model.SMOLVLM_256M -> "SmolVLM-256M"
            Model.QWEN25_VL_3B -> "Qwen2.5-VL-3B"
        }
        val stats = buildString {
            append("⏱ ${"%.1f".format(elapsedSec)}s")
            if (gen != null) append(" · gen ${"%.1f".format(gen)} tok/s")
            if (pp  != null) append(" · prompt ${"%.1f".format(pp)} tok/s")
            append(" · $modelLabel")
        }

        Result(
            reply = reply.ifBlank { "(no reply parsed; stderr tail: ${stderr.takeLast(200)})" },
            stats = stats,
        )
    }

    companion object {
        private const val TAG = "VlmRunner"
    }
}
