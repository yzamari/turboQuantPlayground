package com.yzamari.turboquant.assistant

import android.app.Application
import android.os.Environment
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class AssistantViewModel(app: Application) : AndroidViewModel(app) {

    sealed class ChatEntry {
        data class User(val text: String) : ChatEntry()
        data class AssistantMsg(
            var text: String,
            val streaming: Boolean = false,
            val timing: String? = null,   // e.g. "⏱ 2.4s · 30 tok/s"
            val ctxLeft: String? = null,  // e.g. "ctx 1840 / 2048 (210 left)"
        ) : ChatEntry()
        data class Tool(val name: String, val args: String, var result: String? = null) : ChatEntry()
        data class System(val text: String) : ChatEntry()
    }

    val messages = mutableStateListOf<ChatEntry>()

    var modelStatus by mutableStateOf("No model loaded")
        private set
    var loading    by mutableStateOf(false)
        private set
    var generating by mutableStateOf(false)
        private set
    var ttsEnabled by mutableStateOf(true)
    var threads    by mutableStateOf(4)
    var contextSize by mutableStateOf(2048)
    var statsJson  by mutableStateOf("")
        private set
    /** Maximum context size advertised by the active model (in tokens). */
    var maxContext by mutableStateOf(0)
        private set
    /** Tokens currently in the conversation history (rough estimate). */
    var ctxUsed by mutableStateOf(0)
        private set

    private var handle: Long = 0L
    private var assistant: Assistant? = null
    private val dispatcher = ToolDispatcher(app.applicationContext)
    private val voice = Voice(app.applicationContext)
    private val vlm   = VlmRunner(app.applicationContext)
    private var generationJob: Job? = null

    init {
        voice.initTts()
        messages.add(ChatEntry.System(
            "Welcome to TurboQuant Assistant — a fully on-device personal assistant " +
            "powered by Llama-3.2-1B. Load the model from the Settings tab to begin."
        ))
    }

    /**
     * Search the usual locations for the GGUF model and return the first hit.
     */
    fun findModelPath(): String? {
        val candidates = listOf(
            File(getApplication<Application>().getExternalFilesDir(null),
                "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            File("/sdcard/Download/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
            File("/data/local/tmp/llama/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        )
        return candidates.firstOrNull { it.exists() && it.canRead() }?.absolutePath
    }

    fun loadModel(path: String) {
        if (loading || handle != 0L) return
        loading = true
        modelStatus = "Loading $path…"
        viewModelScope.launch {
            val h = withContext(Dispatchers.IO) {
                runCatching {
                    LlamaNative.loadModel(path, contextSize, threads)
                }.onFailure { Log.e(TAG, "loadModel threw", it) }
                 .getOrDefault(0L)
            }
            if (h == 0L) {
                modelStatus = "Failed to load model. Check the path: $path"
            } else {
                handle    = h
                assistant = Assistant(h, dispatcher,
                    maxToolHops = 3, maxTokensPerTurn = 512)
                maxContext = contextSize
                modelStatus = "Loaded ${File(path).name} (ctx=$contextSize, threads=$threads)"
                messages.add(ChatEntry.System(
                    "Model loaded: ${File(path).name} · max context = $contextSize tokens"
                ))
            }
            loading = false
        }
    }

    fun unloadModel() {
        val h = handle
        if (h == 0L) return
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { LlamaNative.unloadModel(h) }
        }
        handle    = 0L
        assistant = null
        modelStatus = "No model loaded"
        messages.add(ChatEntry.System("Model unloaded."))
    }

    fun isModelReady(): Boolean = handle != 0L && assistant != null

    fun cancelGeneration() {
        generationJob?.cancel()
        generationJob = null
        generating = false
    }

    fun send(userText: String) {
        if (userText.isBlank()) return
        val a = assistant
        if (a == null) {
            messages.add(ChatEntry.System("No model loaded — go to Settings."))
            return
        }
        val startMs = System.currentTimeMillis()
        messages.add(ChatEntry.User(userText))
        val replyEntry = ChatEntry.AssistantMsg("", streaming = true)
        messages.add(replyEntry)
        generating = true
        // Approximate ctx usage = chars/4 (Llama tokenizer ratio); refined later.
        ctxUsed = (messages.sumOf { entry ->
            when (entry) {
                is ChatEntry.User -> entry.text.length
                is ChatEntry.AssistantMsg -> entry.text.length
                else -> 0
            }
        } / 4)

        generationJob = viewModelScope.launch {
            val builder = StringBuilder()
            withContext(Dispatchers.IO) {
                a.respondStreaming(userText).collectLatest { ev ->
                    when (ev) {
                        is Assistant.AssistantEvent.Token -> {
                            builder.append(ev.text)
                            withContext(Dispatchers.Main) {
                                replyEntry.text = builder.toString()
                                // Force list refresh by replacing in place.
                                val idx = messages.indexOf(replyEntry)
                                if (idx >= 0) {
                                    messages[idx] = replyEntry.copy(text = builder.toString(),
                                                                    streaming = true)
                                }
                            }
                        }
                        is Assistant.AssistantEvent.ToolCall -> {
                            withContext(Dispatchers.Main) {
                                messages.add(ChatEntry.Tool(ev.name, ev.argsJson))
                            }
                        }
                        is Assistant.AssistantEvent.ToolResultEvent -> {
                            withContext(Dispatchers.Main) {
                                val last = messages.indexOfLast {
                                    it is ChatEntry.Tool && it.name == ev.name && it.result == null
                                }
                                if (last >= 0) {
                                    val t = messages[last] as ChatEntry.Tool
                                    messages[last] = t.copy(result = ev.resultJson)
                                }
                            }
                        }
                        is Assistant.AssistantEvent.Final -> {
                            val finalText = if (ev.reply.isNotBlank()) ev.reply
                                            else builder.toString()
                            val elapsedMs = System.currentTimeMillis() - startMs
                            // Estimate tokens generated ~ chars/4
                            val genTokens = (finalText.length / 4).coerceAtLeast(1)
                            val tps = genTokens.toDouble() * 1000 / elapsedMs.coerceAtLeast(1)
                            val timing = "⏱ ${"%.1f".format(elapsedMs / 1000.0)}s · " +
                                         "${"%.1f".format(tps)} tok/s"
                            // Update ctx estimate after generation
                            ctxUsed += genTokens
                            val ctxLeft = if (maxContext > 0) {
                                "ctx ${ctxUsed} / ${maxContext} (${(maxContext - ctxUsed).coerceAtLeast(0)} left)"
                            } else null
                            withContext(Dispatchers.Main) {
                                val idx = messages.indexOf(replyEntry)
                                if (idx >= 0) {
                                    messages[idx] = ChatEntry.AssistantMsg(
                                        text = finalText.trim(),
                                        streaming = false,
                                        timing = timing,
                                        ctxLeft = ctxLeft,
                                    )
                                }
                                if (ttsEnabled) voice.speak(stripJson(finalText))
                                statsJson = runCatching {
                                    LlamaNative.getStats(handle)
                                }.getOrDefault("")
                            }
                        }
                        is Assistant.AssistantEvent.Stats -> {
                            withContext(Dispatchers.Main) { statsJson = ev.text }
                        }
                        is Assistant.AssistantEvent.ErrorEvent -> {
                            withContext(Dispatchers.Main) {
                                val idx = messages.indexOf(replyEntry)
                                if (idx >= 0) {
                                    messages[idx] = ChatEntry.AssistantMsg(
                                        text = "(error) ${ev.message}",
                                        streaming = false)
                                }
                            }
                        }
                    }
                }
            }
            generating = false
        }
    }

    /**
     * Describe a captured image via the on-device SmolVLM. The image is read
     * from `imagePath` (a regular filesystem path the app has read access to).
     * The result is appended to the chat as an assistant message.
     */
    fun describeImage(imagePath: String, prompt: String? = null) {
        val userBubble = ChatEntry.User("📷 (sent an image to the assistant)")
        messages.add(userBubble)
        // Add the streaming placeholder; remember its index, not its instance,
        // so we can keep updating the same slot as we replace immutable copies.
        messages.add(ChatEntry.AssistantMsg("Looking at the image…", streaming = true))
        val slotIndex = messages.lastIndex
        generating = true
        val startMs = System.currentTimeMillis()
        val streamBuilder = StringBuilder()
        generationJob = viewModelScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    vlm.describe(
                        imagePath,
                        prompt ?: "Describe this image in detail.",
                    ) { piece ->
                        streamBuilder.append(piece)
                        viewModelScope.launch(Dispatchers.Main) {
                            if (slotIndex < messages.size) {
                                messages[slotIndex] = ChatEntry.AssistantMsg(
                                    text = streamBuilder.toString().trim(),
                                    streaming = true,
                                )
                            }
                        }
                    }
                }.onFailure { Log.e(TAG, "VLM failed", it) }
                 .getOrElse { VlmRunner.Result(reply = "(VLM error: ${it.localizedMessage ?: it})", stats = "") }
            }
            val elapsedMs = System.currentTimeMillis() - startMs
            withContext(Dispatchers.Main) {
                if (slotIndex < messages.size) {
                    messages[slotIndex] = ChatEntry.AssistantMsg(
                        text = result.reply.trim().ifBlank { "(empty reply)" },
                        streaming = false,
                        timing = "⏱ ${"%.1f".format(elapsedMs / 1000.0)}s · SmolVLM-256M",
                        ctxLeft = result.stats.takeIf { it.isNotBlank() },
                    )
                }
                if (result.stats.isNotBlank()) {
                    statsJson = result.stats
                }
                if (ttsEnabled) voice.speak(result.reply)
                generating = false
            }
        }
    }

    fun vlmAvailable(): Boolean = vlm.isAvailable()
    fun vlmDiagnostic(): String = vlm.missingFilesMessage()

    fun resetConversation() {
        cancelGeneration()
        assistant?.reset()
        messages.clear()
        messages.add(ChatEntry.System("New conversation."))
    }

    // ---- Voice ----
    fun startVoice(onText: (String) -> Unit) {
        voice.startListening(object : Voice.Listener {
            override fun onFinal(text: String) { onText(text) }
            override fun onError(message: String) {
                messages.add(ChatEntry.System("Mic: $message"))
            }
        })
    }
    fun stopVoice() = voice.stopListening()
    fun stopTts() = voice.stopSpeaking()

    override fun onCleared() {
        super.onCleared()
        voice.shutdown()
        if (handle != 0L) {
            runCatching { LlamaNative.unloadModel(handle) }
        }
    }

    private fun stripJson(text: String): String {
        // Don't read JSON tool calls aloud.
        return text.replace(Regex("\\{\\s*\"tool\"[\\s\\S]*?\\}\\s*\\}"), "").trim()
    }

    companion object { private const val TAG = "AssistantVM" }
}
