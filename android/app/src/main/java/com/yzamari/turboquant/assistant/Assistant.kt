package com.yzamari.turboquant.assistant

import android.util.Log
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flow
import org.json.JSONObject
import java.util.regex.Pattern

/**
 * Stateful chat orchestrator that owns:
 *   - the llama handle
 *   - conversation history (system + user + assistant + tool messages)
 *   - tool dispatch
 *
 * One conversation turn -> one [respond] call. The flow streams
 * [AssistantEvent]s for the UI to render.
 */
class Assistant(
    private val handle: Long,
    private val toolDispatcher: ToolDispatcher,
    private val maxToolHops: Int = 3,
    private val maxTokensPerTurn: Int = 256,
) {
    sealed class AssistantEvent {
        data class Token(val text: String) : AssistantEvent()
        data class ToolCall(val name: String, val argsJson: String) : AssistantEvent()
        data class ToolResultEvent(val name: String, val resultJson: String) : AssistantEvent()
        data class Final(val reply: String) : AssistantEvent()
        data class ErrorEvent(val message: String) : AssistantEvent()
    }

    data class Message(val role: String, val content: String)

    val history: MutableList<Message> = mutableListOf(
        Message("system", Tools.systemPrompt())
    )

    fun reset() {
        history.clear()
        history.add(Message("system", Tools.systemPrompt()))
        try { LlamaNative.resetContext(handle) } catch (_: Throwable) {}
    }

    /**
     * Process a single user turn. Yields tokens, tool calls, and a final
     * spoken reply. Multi-step: if the model emits a tool call, the tool
     * is run, its result is fed back, and generation continues.
     */
    fun respond(userText: String): Flow<AssistantEvent> = flow {
        history.add(Message("user", userText))

        var hop = 0
        var finalText = ""
        while (hop < maxToolHops) {
            val prompt = renderPrompt()
            val raw = StringBuilder()

            // Run a single LLM pass.
            try {
                LlamaNative.generate(handle, prompt, maxTokensPerTurn) { piece ->
                    raw.append(piece)
                }
            } catch (t: Throwable) {
                emit(AssistantEvent.ErrorEvent("Generation failed: ${t.message}"))
                return@flow
            }

            // Stream what we got.
            val text = raw.toString().trim()
            // Detect a tool call. The model is told to emit a single JSON
            // line — try to find the first {"tool":...} chunk.
            val call = parseToolCall(text)
            if (call != null) {
                emit(AssistantEvent.ToolCall(call.first, call.second.toString()))
                // Add the assistant's tool-call utterance to history so the
                // model can refer back to it.
                history.add(Message("assistant", text))
                // Dispatch the tool, then add its result as a "user" message
                // (Llama-3.2 has no native "tool" role in the simple template).
                val res = toolDispatcher.dispatch(call.first, call.second)
                emit(AssistantEvent.ToolResultEvent(call.first, res.toJson()))
                history.add(Message("user", "Tool '${call.first}' returned: ${res.toJson()}. " +
                    "Reply naturally to the original request now."))
                hop++
                continue
            }

            // No tool call — this is the model's final answer.
            // Emit the whole thing as one Token event so the UI can show it.
            // (We chose not to stream token-by-token to keep tool detection simple.)
            emit(AssistantEvent.Token(text))
            history.add(Message("assistant", text))
            finalText = text
            break
        }

        if (hop >= maxToolHops) {
            emit(AssistantEvent.ErrorEvent("Stopped after $maxToolHops tool hops."))
        }
        emit(AssistantEvent.Final(finalText))
    }

    /**
     * Streaming variant that emits each token as it is produced. Tool calls
     * are detected by buffering the tail of the stream and looking for a
     * JSON line. This is what the UI actually uses.
     */
    fun respondStreaming(userText: String): Flow<AssistantEvent> = callbackFlow {
        history.add(Message("user", userText))

        try {
            var hop = 0
            var finalText = ""
            while (hop < maxToolHops) {
                val prompt = renderPrompt()
                val collected = StringBuilder()
                var sawJson = false

                LlamaNative.generate(handle, prompt, maxTokensPerTurn) { piece ->
                    collected.append(piece)
                    if (!sawJson) {
                        // Don't stream once we see the start of a tool-call JSON.
                        if (collected.toString().contains("{\"tool\"")) {
                            sawJson = true
                        } else {
                            trySend(AssistantEvent.Token(piece))
                        }
                    }
                }

                val text = collected.toString().trim()
                val call = parseToolCall(text)
                if (call != null) {
                    trySend(AssistantEvent.ToolCall(call.first, call.second.toString()))
                    history.add(Message("assistant", text))
                    val res = toolDispatcher.dispatch(call.first, call.second)
                    trySend(AssistantEvent.ToolResultEvent(call.first, res.toJson()))
                    history.add(Message("user",
                        "Tool '${call.first}' returned: ${res.toJson()}. " +
                            "Now reply briefly to my original request."))
                    hop++
                    continue
                }

                history.add(Message("assistant", text))
                finalText = text
                break
            }
            trySend(AssistantEvent.Final(finalText))
        } catch (t: Throwable) {
            Log.e(TAG, "respondStreaming failed", t)
            trySend(AssistantEvent.ErrorEvent("Generation failed: ${t.message}"))
        }
        close()
        awaitClose { /* nothing */ }
    }

    // -----------------------------------------------------------------

    private fun renderPrompt(): String {
        val roles = history.map { it.role }.toTypedArray()
        val contents = history.map { it.content }.toTypedArray()
        return LlamaNative.applyChatTemplate(handle, roles, contents, true)
    }

    private fun parseToolCall(text: String): Pair<String, JSONObject>? {
        // Look for the first {"tool": ... } JSON object in the text.
        val pattern = Pattern.compile("\\{[^{}]*\"tool\"\\s*:[^{}]*\"arguments\"\\s*:\\s*\\{[^{}]*\\}[^{}]*\\}")
        val m = pattern.matcher(text)
        if (!m.find()) return null
        return try {
            val obj = JSONObject(m.group())
            val name = obj.optString("tool")
            val args = obj.optJSONObject("arguments") ?: JSONObject()
            if (name.isBlank()) null else name to args
        } catch (_: Throwable) {
            null
        }
    }

    companion object {
        private const val TAG = "Assistant"
    }
}
