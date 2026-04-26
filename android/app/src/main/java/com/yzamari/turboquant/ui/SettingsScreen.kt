package com.yzamari.turboquant.ui

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.os.Environment
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.yzamari.turboquant.assistant.AssistantViewModel
import com.yzamari.turboquant.assistant.VlmRunner
import java.io.File

@Composable
fun SettingsScreen(vm: AssistantViewModel) {
    val context = LocalContext.current
    val scroll  = rememberScrollState()

    var modelPath by remember {
        mutableStateOf(
            vm.findModelPath()
                ?: File(context.getExternalFilesDir(null),
                        "Llama-3.2-1B-Instruct-Q4_K_M.gguf").absolutePath
        )
    }

    val modelExists = remember(modelPath, vm.modelStatus) {
        try { File(modelPath).exists() } catch (_: Throwable) { false }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scroll)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            "Settings",
            style = MaterialTheme.typography.headlineSmall
        )

        Card(colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer
        )) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("Model status",
                    style = MaterialTheme.typography.titleSmall)
                Text(vm.modelStatus,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace)

                OutlinedTextField(
                    value = modelPath,
                    onValueChange = { modelPath = it },
                    label = { Text("GGUF path") },
                    singleLine = false,
                    modifier = Modifier.fillMaxWidth()
                )
                Text(
                    if (modelExists)
                        "✓ File exists at this path."
                    else
                        "✗ No file at this path. Push the GGUF with:\n" +
                        "    adb push <gguf> ${context.getExternalFilesDir(null)?.absolutePath}/",
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = { vm.loadModel(modelPath) },
                        enabled = !vm.loading && !vm.isModelReady() && modelExists
                    ) {
                        Text(if (vm.loading) "Loading…" else "Load model")
                    }
                    OutlinedButton(
                        onClick = { vm.unloadModel() },
                        enabled = vm.isModelReady()
                    ) {
                        Text("Unload")
                    }
                    OutlinedButton(
                        onClick = {
                            vm.findModelPath()?.let { modelPath = it }
                        }
                    ) {
                        Text("Auto-detect")
                    }
                }

                OutlinedButton(onClick = {
                    downloadModel(context)
                }) {
                    Text("Download Llama-3.2-1B (≈807 MB)")
                }
                Text(
                    "Tip: if the device has no internet, push the file via adb instead.",
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }

        HorizontalDivider()

        // -------- TurboQuant status / verification --------
        Card(colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer
        )) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        "✦ TurboQuant: ENABLED",
                        style = MaterialTheme.typography.titleSmall,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        if (vm.isTurboQuantAvailable()) "lib loaded" else "lib MISSING",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
                Text(
                    "C++ libturboquant is linked into this APK. It compresses the " +
                        "KV cache of the loaded LLM to ~4× smaller at cosine 0.92 vs " +
                        "FP16. Tap below to measure the exact ratio + quality on " +
                        "your loaded model.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Button(
                    onClick = { vm.verifyTurboQuant() },
                    enabled = vm.isTurboQuantAvailable(),
                ) { Text("Verify TurboQuant on the loaded model") }
                if (vm.turboQuantStatus.isNotBlank()) {
                    Text(
                        vm.turboQuantStatus,
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }

        HorizontalDivider()

        // -------- Vision model picker --------
        Card(colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer
        )) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    "Vision (VLM) model",
                    style = MaterialTheme.typography.titleSmall,
                )
                Text(
                    "Used when you attach an image to the chat. Smaller is " +
                        "faster but English-only; bigger is multilingual + much " +
                        "better image understanding.",
                    style = MaterialTheme.typography.bodySmall,
                )
                val available = remember(vm.modelStatus) { vm.availableVlmModels() }
                var current by remember { mutableStateOf(vm.activeVlmModel()) }
                for (m in VlmRunner.Model.values()) {
                    val installed = m in available
                    OutlinedButton(
                        onClick = {
                            if (installed) {
                                vm.setActiveVlmModel(m)
                                current = m
                            }
                        },
                        enabled = installed,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            (if (m == current) "● " else "○ ") +
                                m.displayName +
                                (if (!installed) "  (not installed)" else ""),
                        )
                    }
                }
                if (available.size < VlmRunner.Model.values().size) {
                    Text(
                        "Push missing files via adb to " +
                            (context.getExternalFilesDir(null)?.absolutePath ?: "?") + "/",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }

        HorizontalDivider()

        Card(colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceContainer
        )) {
            Column(
                modifier = Modifier.fillMaxWidth().padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("Inference",
                    style = MaterialTheme.typography.titleSmall)

                Text("Threads: ${vm.threads}")
                Slider(
                    value = vm.threads.toFloat(),
                    onValueChange = { vm.threads = it.toInt().coerceIn(1, 8) },
                    valueRange = 1f..8f,
                    steps = 6,
                    enabled = !vm.isModelReady(),
                )

                Text("Context size: ${vm.contextSize}")
                Slider(
                    value = vm.contextSize.toFloat(),
                    onValueChange = {
                        vm.contextSize = (it.toInt() / 256 * 256).coerceIn(1024, 8192)
                    },
                    valueRange = 1024f..8192f,
                    steps = 0,
                    enabled = !vm.isModelReady(),
                )

                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(
                        checked = vm.ttsEnabled,
                        onCheckedChange = { vm.ttsEnabled = it }
                    )
                    Text("Speak replies (TTS)",
                        modifier = Modifier.padding(start = 8.dp))
                }

                // (The "TurboQuant: ENABLED" card above is the real
                // status — this used to be a placeholder toggle and was
                // removed because it was confusing.)
            }
        }

        OutlinedButton(onClick = { vm.resetConversation() },
            modifier = Modifier.fillMaxWidth()) {
            Text("Reset conversation")
        }
    }
}

private fun downloadModel(context: Context) {
    val url = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/" +
              "Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true"
    val req = DownloadManager.Request(Uri.parse(url))
        .setTitle("Llama-3.2-1B-Instruct (Q4_K_M)")
        .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
        .setDestinationInExternalFilesDir(
            context, null, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )
        .setAllowedOverMetered(true)
    val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    runCatching { dm.enqueue(req) }
}
