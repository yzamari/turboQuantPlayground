package com.yzamari.turboquant.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.yzamari.turboquant.TurboQuantNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Minimal bench screen — runs the paired baseline-vs-TurboQuant CLI harness via
 * JNI and dumps the resulting CSV table. Kept simple so the chat experience is
 * the focus; the standalone `turboquant_bench` binary is the comprehensive tool.
 */
@Composable
fun BenchScreen() {
    val scope = rememberCoroutineScope()
    var running by remember { mutableStateOf(false) }
    var output by remember { mutableStateOf("Tap Run to compare baseline FP16 vs TurboQuant on this device.") }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.Start,
    ) {
        Text("TurboQuant bench (cpu_neon, BH=8, D=128, 3-bit)", style = MaterialTheme.typography.titleMedium)

        Box(modifier = Modifier.padding(top = 8.dp)) {
            Button(
                enabled = !running,
                onClick = {
                    running = true
                    scope.launch {
                        val result = withContext(Dispatchers.IO) {
                            try {
                                TurboQuantNative.runBenchmark("cpu_neon", intArrayOf(128, 512, 2048), 3, 8, 128)
                            } catch (t: Throwable) {
                                "Bench failed: ${t.message}"
                            }
                        }
                        output = result
                        running = false
                    }
                },
            ) {
                Text(if (running) "Running…" else "Run benchmark")
            }
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(top = 16.dp)
                .verticalScroll(rememberScrollState()),
        ) {
            Text(
                text = output,
                fontFamily = FontFamily.Monospace,
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}
