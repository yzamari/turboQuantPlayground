package com.yzamari.turboquant.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.yzamari.turboquant.TurboQuantNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val DEFAULT_SEQ_LENS = "128,512,2048"
private val BITS_OPTIONS     = listOf(2, 3, 4)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchScreen() {
    val backends = remember {
        runCatching { TurboQuantNative.listBackends().toList() }
            .getOrElse { listOf("cpu_scalar") }
            .ifEmpty { listOf("cpu_scalar") }
    }

    var selectedBackend by remember { mutableStateOf(backends.first()) }
    var backendMenuOpen by remember { mutableStateOf(false) }

    var seqLensText by remember { mutableStateOf(DEFAULT_SEQ_LENS) }
    var selectedBits by remember { mutableStateOf(3) }

    var running by remember { mutableStateOf(false) }
    var output by remember { mutableStateOf("Press Run to benchmark TurboQuant on-device.") }
    val scope = rememberCoroutineScope()

    fun runBench() {
        if (running) return
        running = true
        output = "Running on $selectedBackend ..."
        val seqLens = parseSeqLens(seqLensText)
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching {
                    TurboQuantNative.runBenchmark(
                        backendName = selectedBackend,
                        seqLens     = seqLens,
                        bits        = selectedBits,
                        bh          = 8,
                        d           = 128,
                    )
                }.getOrElse { e -> "Native error: ${e.message ?: e::class.java.simpleName}" }
            }
            output  = result
            running = false
        }
    }

    fun runSmoke() {
        if (running) return
        running = true
        output  = "Checking $selectedBackend ..."
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { TurboQuantNative.runCheck(selectedBackend) }
                    .getOrElse { "Native error: ${it.message}" }
            }
            output  = result
            running = false
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text  = "TurboQuant Bench",
            style = MaterialTheme.typography.headlineSmall
        )
        Text(
            text = "Compare a baseline KV cache against TurboQuant on the available backends.",
            style = MaterialTheme.typography.bodySmall
        )

        ExposedDropdownMenuBox(
            expanded         = backendMenuOpen,
            onExpandedChange = { backendMenuOpen = !backendMenuOpen }
        ) {
            OutlinedTextField(
                value         = selectedBackend,
                onValueChange = { /* read-only */ },
                readOnly      = true,
                label         = { Text("Backend") },
                trailingIcon  = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = backendMenuOpen) },
                modifier      = Modifier.menuAnchor().fillMaxWidth()
            )
            androidx.compose.material3.ExposedDropdownMenu(
                expanded         = backendMenuOpen,
                onDismissRequest = { backendMenuOpen = false }
            ) {
                backends.forEach { b ->
                    DropdownMenuItem(
                        text    = { Text(b) },
                        onClick = {
                            selectedBackend = b
                            backendMenuOpen = false
                        }
                    )
                }
            }
        }

        OutlinedTextField(
            value         = seqLensText,
            onValueChange = { seqLensText = it },
            label         = { Text("Sequence lengths (comma-separated)") },
            singleLine    = true,
            modifier      = Modifier.fillMaxWidth()
        )

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Bits:", modifier = Modifier.padding(top = 8.dp))
            BITS_OPTIONS.forEach { b ->
                FilterChip(
                    selected = (b == selectedBits),
                    onClick  = { selectedBits = b },
                    label    = { Text(b.toString()) }
                )
            }
        }

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Button(enabled = !running, onClick = ::runBench) {
                Text(if (running) "Running..." else "Run benchmark")
            }
            Button(enabled = !running, onClick = ::runSmoke) {
                Text("Smoke check")
            }
            if (running) {
                Spacer(Modifier.weight(1f))
                CircularProgressIndicator(modifier = Modifier.height(24.dp))
            }
        }

        val scroll = rememberScrollState()
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .verticalScroll(scroll)
        ) {
            Text(
                text       = output,
                fontFamily = FontFamily.Monospace,
                fontSize   = 12.sp,
                style      = MaterialTheme.typography.bodySmall
            )
        }
    }
}

private fun parseSeqLens(text: String): IntArray =
    text.split(',', ' ', '\t', '\n')
        .mapNotNull { it.trim().takeIf(String::isNotEmpty)?.toIntOrNull() }
        .filter { it > 0 }
        .toIntArray()
        .ifEmpty { intArrayOf(128, 512, 2048) }
