package com.yzamari.turboquant

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.yzamari.turboquant.assistant.AssistantViewModel
import com.yzamari.turboquant.ui.BenchScreen
import com.yzamari.turboquant.ui.ChatScreen
import com.yzamari.turboquant.ui.SettingsScreen

private enum class Tab(val label: String) {
    Chat("Assistant"),
    Bench("Bench"),
    Settings("Settings"),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun App() {
    var current by rememberSaveable { mutableStateOf(Tab.Chat) }
    val vm: AssistantViewModel = viewModel()

    Scaffold(
        bottomBar = {
            NavigationBar {
                Tab.values().forEach { tab ->
                    NavigationBarItem(
                        selected = (current == tab),
                        onClick  = { current = tab },
                        icon     = {
                            val icon = when (tab) {
                                Tab.Chat     -> Icons.Filled.Chat
                                Tab.Bench    -> Icons.Filled.Speed
                                Tab.Settings -> Icons.Filled.Settings
                            }
                            Icon(icon, contentDescription = tab.label)
                        },
                        label    = { Text(tab.label) },
                    )
                }
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            when (current) {
                Tab.Chat     -> ChatScreen(vm)
                Tab.Bench    -> BenchScreen()
                Tab.Settings -> SettingsScreen(vm)
            }
        }
    }
}
