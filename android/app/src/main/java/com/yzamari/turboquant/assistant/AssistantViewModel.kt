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
        data class AssistantMsg(var text: String, val streaming: Boolean = false) : ChatEntry()
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

    private var handle: Long = 0L
    private var assistant: Assistant? = null
    private val dispatcher = ToolDispatcher(app.applicationContext)
    private val voice = Voice(app.applicationContext)
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
                    maxToolHops = 3, maxTokensPerTurn = 256)
                modelStatus = "Loaded ${File(path).name} (ctx=$contextSize, threads=$threads)"
                messages.add(ChatEntry.System("Model loaded: ${File(path).name}"))
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
        messages.add(ChatEntry.User(userText))
        val replyEntry = ChatEntry.AssistantMsg("", streaming = true)
        messages.add(replyEntry)
        generating = true

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
                            withContext(Dispatchers.Main) {
                                val idx = messages.indexOf(replyEntry)
                                if (idx >= 0) {
                                    messages[idx] = ChatEntry.AssistantMsg(
                                        text = finalText.trim(), streaming = false)
                                }
                                if (ttsEnabled) voice.speak(stripJson(finalText))
                                statsJson = runCatching {
                                    LlamaNative.getStats(handle)
                                }.getOrDefault("")
                            }
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
