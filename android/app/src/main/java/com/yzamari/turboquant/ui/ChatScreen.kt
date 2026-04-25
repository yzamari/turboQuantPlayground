package com.yzamari.turboquant.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.Send
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeOff
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import com.yzamari.turboquant.assistant.AssistantViewModel
import com.yzamari.turboquant.assistant.AssistantViewModel.ChatEntry

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: AssistantViewModel) {
    val context = LocalContext.current
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    val micPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            vm.startVoice { text -> input = text }
        }
    }

    fun toggleMic() {
        val granted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            micPermLauncher.launch(Manifest.permission.RECORD_AUDIO)
        } else {
            vm.startVoice { text -> input = text }
        }
    }

    fun sendCurrent() {
        if (input.isBlank() || vm.generating) return
        val text = input.trim()
        input = ""
        vm.send(text)
    }

    LaunchedEffect(vm.messages.size) {
        if (vm.messages.isNotEmpty()) {
            listState.animateScrollToItem(vm.messages.size - 1)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = {
                Column {
                    Text("TurboQuant Assistant",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold)
                    Text(
                        if (vm.isModelReady())
                            "on-device · Llama-3.2-1B"
                        else
                            "no model loaded — open Settings",
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            },
            actions = {
                IconButton(onClick = {
                    vm.ttsEnabled = !vm.ttsEnabled
                    if (!vm.ttsEnabled) vm.stopTts()
                }) {
                    Icon(
                        if (vm.ttsEnabled) Icons.Filled.VolumeUp else Icons.Filled.VolumeOff,
                        contentDescription = "Toggle voice output",
                    )
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.surfaceContainer
            )
        )

        // Stats line (subtle).
        if (vm.statsJson.isNotBlank()) {
            Text(
                vm.statsJson,
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 2.dp)
            )
        }

        LazyColumn(
            state = listState,
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(horizontal = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 8.dp)
        ) {
            items(vm.messages) { entry ->
                ChatBubble(entry)
            }
        }

        if (vm.generating) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(strokeWidth = 2.dp, modifier = Modifier.padding(end = 8.dp))
                Text("Thinking…", style = MaterialTheme.typography.bodySmall)
                Box(modifier = Modifier.weight(1f))
                IconButton(onClick = { vm.cancelGeneration() }) {
                    Icon(Icons.Filled.Stop, contentDescription = "Stop")
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Ask anything…") },
                maxLines = 4,
                keyboardOptions = KeyboardOptions.Default,
                keyboardActions = KeyboardActions(onSend = { sendCurrent() }),
            )
            FilledIconButton(
                onClick = { toggleMic() },
                colors = IconButtonDefaults.filledIconButtonColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Icon(
                    if (vm.generating) Icons.Filled.MicOff else Icons.Filled.Mic,
                    contentDescription = "Mic",
                )
            }
            FilledIconButton(
                onClick = { sendCurrent() },
                enabled = input.isNotBlank() && !vm.generating && vm.isModelReady(),
            ) {
                Icon(Icons.Filled.Send, contentDescription = "Send")
            }
        }
    }
}

@Composable
private fun ChatBubble(entry: ChatEntry) {
    when (entry) {
        is ChatEntry.User -> {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor   = MaterialTheme.colorScheme.onPrimary,
                    ),
                    shape  = RoundedCornerShape(16.dp, 4.dp, 16.dp, 16.dp),
                    modifier = Modifier.widthIn(max = 480.dp).wrapContentWidth(Alignment.End)
                ) {
                    Text(
                        entry.text,
                        modifier = Modifier.padding(12.dp),
                    )
                }
            }
        }
        is ChatEntry.AssistantMsg -> {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant,
                        contentColor   = MaterialTheme.colorScheme.onSurfaceVariant,
                    ),
                    shape  = RoundedCornerShape(4.dp, 16.dp, 16.dp, 16.dp),
                    modifier = Modifier.widthIn(max = 480.dp)
                ) {
                    Text(
                        entry.text.ifBlank { "…" },
                        modifier = Modifier.padding(12.dp),
                    )
                }
            }
        }
        is ChatEntry.Tool -> {
            Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp)) {
                Text(
                    "🔧 ${entry.name}(${entry.args})",
                    style = MaterialTheme.typography.labelMedium,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 12.sp,
                    modifier = Modifier
                        .background(
                            MaterialTheme.colorScheme.tertiaryContainer,
                            RoundedCornerShape(8.dp)
                        )
                        .padding(8.dp),
                )
                if (entry.result != null) {
                    Text(
                        "✅ ${entry.result}",
                        style = MaterialTheme.typography.labelMedium,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 12.sp,
                        modifier = Modifier
                            .padding(top = 4.dp)
                            .background(
                                MaterialTheme.colorScheme.secondaryContainer,
                                RoundedCornerShape(8.dp)
                            )
                            .padding(8.dp),
                    )
                }
            }
        }
        is ChatEntry.System -> {
            Box(modifier = Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text(
                    entry.text,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.padding(horizontal = 16.dp)
                )
            }
        }
    }
}
