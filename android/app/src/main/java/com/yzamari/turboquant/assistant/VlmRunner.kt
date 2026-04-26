package com.yzamari.turboquant.assistant

import android.content.Context
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Persistent in-process VLM runner.
 *
 * Backed by [MtmdNative] / `mtmd_jni.cpp`. The LLM + mmproj are loaded
 * once on the first [describe] call (or when [Model] changes) and stay
 * resident for the rest of the app's lifetime — every subsequent image
 * pays only for the SigLIP/multi-scale encode + token decode, not for
 * the multi-second model-load tax.
 *
 * Compared to the previous `Runtime.exec(libllama-mtmd-cli.so)` path:
 *
 *   First describe()  : ~3 s on Adreno (SmolVLM-256M, fresh load)
 *   Subsequent calls  : ~0.8–1.2 s (encode + 50 tok/s decode)
 *   Old (fork/image)  : ~3.4 s every call (re-loaded everything)
 *
 * Model switching is handled by unloading the current native session
 * and lazy-loading the new one on next describe().
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

    /** Currently-selected VLM model. Mutable so Settings can change it live;
     *  changing it triggers a reload on the next [describe] call.
     *
     *  Default SmolVLM-256M: ~256 MB on disk, ~1 FPS continuous on Adreno
     *  with the persistent session. Qwen2.5-VL-3B is available as the
     *  multilingual / higher-quality option in Settings (much heavier:
     *  ~2.6 GB on disk, ~0.2 FPS). */
    @Volatile
    var activeModel: Model = Model.SMOLVLM_256M
        set(value) {
            if (value != field) {
                Log.i(TAG, "active model: ${field.displayName} -> ${value.displayName}; unloading native session")
                unload()
                field = value
            }
        }

    @Volatile private var nativeHandle: Long  = 0L
    @Volatile private var loadedModel: Model? = null

    fun isAvailable(model: Model = activeModel): Boolean {
        val files = ctx.getExternalFilesDir(null) ?: return false
        return File(files, model.modelFile).exists() && File(files, model.mmprojFile).exists()
    }

    /** All models that have their .gguf + mmproj on disk right now. */
    fun availableModels(): List<Model> = Model.values().filter { isAvailable(it) }

    fun missingFilesMessage(model: Model = activeModel): String {
        val files = ctx.getExternalFilesDir(null)?.absolutePath ?: "<no external files dir>"
        val sb = StringBuilder()
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

    /** Stream the VLM's tokens to a callback as they appear. Lazy-loads the
     *  native session on first call (or after [activeModel] changed).
     *  `threads=8` matches the big-core count on SD 8 Gen 2 / Gen 3, which
     *  is what we want when the VLM falls back to CPU due to the
     *  Adreno+mtmd hang on Tab S9+. */
    suspend fun describe(
        imagePath: String,
        prompt: String = "Describe this image in detail.",
        maxTokens: Int = 200,
        threads: Int = 8,
        onToken: ((String) -> Unit)? = null,
    ): Result = withContext(Dispatchers.IO) {
        val sel = activeModel
        if (!isAvailable(sel)) return@withContext Result(missingFilesMessage(sel), "")

        // Make sure the native session matches activeModel.
        if (nativeHandle == 0L || loadedModel != sel) {
            if (nativeHandle != 0L) {
                Log.i(TAG, "model changed; unloading previous native session")
                MtmdNative.unloadModel(nativeHandle)
                nativeHandle = 0L
            }
            val files = ctx.getExternalFilesDir(null)
                ?: return@withContext Result("External files dir not accessible.", "")
            val modelPath  = File(files, sel.modelFile).absolutePath
            val mmprojPath = File(files, sel.mmprojFile).absolutePath
            Log.i(TAG, "loading native VLM session: ${sel.displayName}")
            val tLoadStart = System.currentTimeMillis()
            // Pick gpuLayers based on the SoC. The mtmd vision pipeline
            // hangs in mtmd_helper_eval_chunks on Adreno 740 / Snapdragon 8
            // Gen 2 ("kalama" — Galaxy Tab S9+) at our pinned llama.cpp
            // commit; verified by reproducing the same hang with
            // upstream's stock llama-mtmd-cli on the same device. Adreno
            // 750 / Snapdragon 8 Gen 3 ("pineapple" — Galaxy S24 Ultra)
            // works fine. Default conservatively to CPU on anything we
            // haven't allow-listed — slower but correct.
            val gpuOk = isAdrenoVlmKnownGood()
            val gpuLayersForVlm = if (gpuOk) 99 else 0
            Log.i(TAG, "SoC board='${Build.BOARD}' → gpuLayers=$gpuLayersForVlm " +
                       "(vlm-on-Adreno ${if (gpuOk) "OK" else "broken — falling back to CPU"})")
            nativeHandle = MtmdNative.loadModel(
                modelPath  = modelPath,
                mmprojPath = mmprojPath,
                contextSize = 4096,
                threads    = threads,
                gpuLayers  = gpuLayersForVlm,
                kvType     = 0,
            )
            val tLoad = System.currentTimeMillis() - tLoadStart
            if (nativeHandle == 0L) {
                return@withContext Result("Failed to load VLM (${sel.displayName})", "")
            }
            loadedModel = sel
            Log.i(TAG, "VLM loaded in ${tLoad} ms")
        }

        val tStart = System.currentTimeMillis()
        val reply = StringBuilder()
        try {
            MtmdNative.describe(
                handle    = nativeHandle,
                imagePath = imagePath,
                prompt    = prompt,
                maxTokens = maxTokens,
            ) { piece ->
                reply.append(piece)
                onToken?.invoke(piece)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "describe failed: $t")
            return@withContext Result("(VLM error: ${t.message ?: t.javaClass.simpleName})", "")
        }
        val elapsedMs = System.currentTimeMillis() - tStart

        // Pull per-call stats from native (image_eval_ms / decode tok/s / etc).
        val statsJson = runCatching { MtmdNative.getStats(nativeHandle) }.getOrDefault("{}")
        val stats = formatStats(elapsedMs, statsJson, sel)

        Result(
            reply = reply.toString().trim().ifBlank { "(no reply)" },
            stats = stats,
        )
    }

    /** Drop the native session if held. Safe to call repeatedly. */
    fun unload() {
        val h = nativeHandle
        if (h != 0L) {
            nativeHandle = 0L
            loadedModel  = null
            runCatching { MtmdNative.unloadModel(h) }
                .onFailure { Log.w(TAG, "unloadModel: $it") }
        }
    }

    private fun formatStats(elapsedMs: Long, statsJson: String, model: Model): String {
        // Cheap regex parse — the JSON is fixed-shape from mtmd_jni.cpp.
        fun num(key: String): Double? =
            Regex("\"$key\":([0-9.]+)").find(statsJson)?.groupValues?.get(1)?.toDoubleOrNull()
        val imgMs = num("image_eval_ms")
        val tps   = num("decode_tok_per_sec")
        val nDec  = num("n_decode")?.toInt()
        val modelLabel = when (model) {
            Model.SMOLVLM_256M -> "SmolVLM-256M"
            Model.QWEN25_VL_3B -> "Qwen2.5-VL-3B"
        }
        return buildString {
            append("⏱ ${"%.1f".format(elapsedMs / 1000.0)}s")
            if (imgMs != null) append(" · img ${"%.0f".format(imgMs)}ms")
            if (tps != null && nDec != null) append(" · gen ${nDec}t @ ${"%.1f".format(tps)} tok/s")
            append(" · $modelLabel")
        }
    }

    /** SoC allow-list for running mtmd's vision graph on Adreno. Tracks
     *  devices where we've actually seen the pipeline complete a frame
     *  without hanging in `mtmd_helper_eval_chunks`. Anything else falls
     *  back to CPU. */
    private fun isAdrenoVlmKnownGood(): Boolean {
        val board = Build.BOARD.lowercase()
        // Snapdragon 8 Gen 3 (S24 Ultra) — works.
        if (board.startsWith("pineapple")) return true
        // Snapdragon 8 Elite — assumed-good (newer than known-good Gen 3).
        if (board.startsWith("sun") || board.startsWith("8elite")) return true
        // Known-bad: SD 8 Gen 2 (Tab S9+) — board "kalama" — falls through.
        return false
    }

    companion object {
        private const val TAG = "VlmRunner"
    }
}
