package com.yzamari.turboquant.assistant

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale

/**
 * Wraps the platform [SpeechRecognizer] and [TextToSpeech] services with
 * a minimal callback surface suitable for a Compose UI.
 */
class Voice(private val context: Context) {

    interface Listener {
        fun onPartial(text: String) {}
        fun onFinal(text: String) {}
        fun onError(message: String) {}
    }

    private var recognizer: SpeechRecognizer? = null
    private var tts: TextToSpeech? = null
    private var ttsReady: Boolean = false
    private var listener: Listener? = null
    private var listening: Boolean = false

    fun isListening(): Boolean = listening

    fun startListening(l: Listener) {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            l.onError("Speech recognition is not available on this device.")
            return
        }
        stopListening()
        listener = l

        val r = SpeechRecognizer.createSpeechRecognizer(context)
        recognizer = r
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() { listening = false }
            override fun onError(error: Int) {
                listening = false
                listener?.onError(speechErrorMessage(error))
            }
            override fun onResults(results: Bundle?) {
                listening = false
                val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = list?.firstOrNull().orEmpty()
                if (text.isNotBlank()) listener?.onFinal(text)
            }
            override fun onPartialResults(partialResults: Bundle?) {
                val list = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                val text = list?.firstOrNull().orEmpty()
                if (text.isNotBlank()) listener?.onPartial(text)
            }
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }
        listening = true
        r.startListening(intent)
    }

    fun stopListening() {
        try { recognizer?.stopListening() } catch (_: Throwable) {}
        try { recognizer?.destroy()       } catch (_: Throwable) {}
        recognizer = null
        listening  = false
    }

    // ---- TTS ----

    fun initTts() {
        if (tts != null) return
        tts = TextToSpeech(context) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) tts?.language = Locale.getDefault()
        }
    }

    fun speak(text: String) {
        if (text.isBlank()) return
        if (!ttsReady) {
            initTts()
            return
        }
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "msg-${System.currentTimeMillis()}")
    }

    fun stopSpeaking() {
        try { tts?.stop() } catch (_: Throwable) {}
    }

    fun shutdown() {
        stopListening()
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (_: Throwable) {}
        tts = null
        ttsReady = false
    }

    private fun speechErrorMessage(code: Int): String = when (code) {
        SpeechRecognizer.ERROR_AUDIO              -> "Audio error"
        SpeechRecognizer.ERROR_CLIENT             -> "Client error"
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission denied"
        SpeechRecognizer.ERROR_NETWORK            -> "Network error"
        SpeechRecognizer.ERROR_NETWORK_TIMEOUT    -> "Network timeout"
        SpeechRecognizer.ERROR_NO_MATCH           -> "No speech recognized"
        SpeechRecognizer.ERROR_RECOGNIZER_BUSY    -> "Recognizer busy"
        SpeechRecognizer.ERROR_SERVER             -> "Recognition server error"
        SpeechRecognizer.ERROR_SPEECH_TIMEOUT     -> "Speech timeout"
        else -> "Speech error ($code)"
    }

    companion object {
        private const val TAG = "Voice"
    }
}
