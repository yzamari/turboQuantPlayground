package com.yzamari.turboquant.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
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
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.PhotoCamera
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

    // Pending image attachment. When non-null, the next Send routes to VLM.
    var attachedImagePath by remember { mutableStateOf<String?>(null) }
    var pendingCameraPath by remember { mutableStateOf<String?>(null) }
    var showAttachMenu    by remember { mutableStateOf(false) }

    val micPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            vm.startVoice { text -> input = text }
        }
    }

    val galleryLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            attachedImagePath = copyUriToCache(context, uri)
        }
    }

    val cameraLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) {
            // The camera writes a full-resolution JPEG (~12 MP on S24 rear cam).
            // Downscale in place — the VLM vision encoder resizes anyway, and a
            // smaller file means less JPEG decode + cleaner clip.cpp work.
            attachedImagePath = pendingCameraPath?.let { downscaleJpegInPlace(it) ?: it }
        } else {
            pendingCameraPath = null
        }
    }

    val cameraPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) launchCamera(context, cameraLauncher) { p -> pendingCameraPath = p }
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

    fun openCamera() {
        val granted = ContextCompat.checkSelfPermission(
            context, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        if (!granted) cameraPermLauncher.launch(Manifest.permission.CAMERA)
        else launchCamera(context, cameraLauncher) { p -> pendingCameraPath = p }
    }

    fun sendCurrent() {
        if (vm.generating) return
        val text = input.trim()
        val img  = attachedImagePath
        if (img != null) {
            // Route to VLM (Vision-Language Model)
            input = ""
            attachedImagePath = null
            val prompt = if (text.isNotBlank()) text else "Describe this image."
            vm.describeImage(img, prompt)
        } else if (text.isNotBlank()) {
            // Route to LLM
            input = ""
            vm.send(text)
        }
    }

    LaunchedEffect(vm.messages.size) {
        if (vm.messages.isNotEmpty()) {
            listState.animateScrollToItem(vm.messages.size - 1)
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // Gradient header for a more inspiring look.
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(
                    androidx.compose.ui.graphics.Brush.horizontalGradient(
                        colors = listOf(
                            MaterialTheme.colorScheme.primaryContainer,
                            MaterialTheme.colorScheme.tertiaryContainer,
                        )
                    )
                )
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        "✦ TurboQuant Assistant",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                    Text(
                        if (vm.isModelReady())
                            "on-device · Llama-3.2-1B (text) · " +
                                vm.activeVlmModel().displayName.substringBefore(" (") +
                                " (vision)"
                        else
                            "no model loaded — open Settings to load Llama-3.2-1B",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.85f),
                    )
                }
                IconButton(onClick = {
                    vm.ttsEnabled = !vm.ttsEnabled
                    if (!vm.ttsEnabled) vm.stopTts()
                }) {
                    Icon(
                        if (vm.ttsEnabled) Icons.Filled.VolumeUp else Icons.Filled.VolumeOff,
                        contentDescription = "Toggle voice output",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
            }
        }

        // Stats chip — prominent, with timing + tok/s + active model.
        if (vm.statsJson.isNotBlank()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 4.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    vm.statsJson,
                    style = MaterialTheme.typography.labelMedium,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier
                        .background(
                            MaterialTheme.colorScheme.secondaryContainer,
                            androidx.compose.foundation.shape.RoundedCornerShape(12.dp),
                        )
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                )
            }
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

        // If an image is attached, show a small chip above the input.
        attachedImagePath?.let { path ->
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    "📷 image attached → will route to VLM (SmolVLM-256M)",
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.weight(1f),
                )
                IconButton(onClick = { attachedImagePath = null }) {
                    Icon(Icons.Filled.Close, contentDescription = "Remove image")
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
            // Attach: pops a small chooser between Camera and Gallery.
            Box {
                FilledIconButton(
                    onClick = { showAttachMenu = true },
                    colors = IconButtonDefaults.filledIconButtonColors(
                        containerColor = MaterialTheme.colorScheme.tertiaryContainer
                    ),
                ) {
                    Icon(Icons.Filled.AddPhotoAlternate, contentDescription = "Attach image")
                }
                androidx.compose.material3.DropdownMenu(
                    expanded = showAttachMenu,
                    onDismissRequest = { showAttachMenu = false }
                ) {
                    androidx.compose.material3.DropdownMenuItem(
                        text = { Text("Take photo") },
                        leadingIcon = { Icon(Icons.Filled.PhotoCamera, contentDescription = null) },
                        onClick = { showAttachMenu = false; openCamera() }
                    )
                    androidx.compose.material3.DropdownMenuItem(
                        text = { Text("Pick from gallery") },
                        leadingIcon = { Icon(Icons.Filled.AddPhotoAlternate, contentDescription = null) },
                        onClick = { showAttachMenu = false; galleryLauncher.launch("image/*") }
                    )
                }
            }
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.weight(1f),
                placeholder = {
                    Text(
                        if (attachedImagePath != null) "Add a question about the image (optional)…"
                        else "Ask anything…"
                    )
                },
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
                enabled = !vm.generating &&
                          (attachedImagePath != null || (input.isNotBlank() && vm.isModelReady())),
            ) {
                Icon(Icons.Filled.Send, contentDescription = "Send")
            }
        }
    }
}

// ---- Camera & gallery helpers ----------------------------------------------

private fun launchCamera(
    context: Context,
    launcher: androidx.activity.result.ActivityResultLauncher<Uri>,
    onPath: (String) -> Unit,
) {
    val dir = File(context.cacheDir, "images").apply { mkdirs() }
    val file = File(dir, "shot_${System.currentTimeMillis()}.jpg")
    val authority = "${context.packageName}.fileprovider"
    val uri = FileProvider.getUriForFile(context, authority, file)
    onPath(file.absolutePath)
    launcher.launch(uri)
}

// Longest-side cap fed to the VLM. Sweet spot: SmolVLM/SigLIP only consumes
// 384×384, Qwen2.5-VL-3B tiles native resolution but caps tile count, so a
// 1024-on-longest-side input preserves the detail both encoders can use while
// cutting JPEG-decode + internal-resize cost ~10–20× vs a raw 12 MP camera
// shot. JPEG quality 90 keeps the recompression artifact under what the
// vision encoder cares about.
private const val VLM_MAX_DIM   = 1024
private const val VLM_JPEG_Q    = 90
private const val VLM_LOG_TAG   = "VlmDownscale"

/** Decode + downscale [uri] (e.g. gallery pick) into a fresh JPEG in cache. */
private fun copyUriToCache(context: Context, uri: Uri): String? {
    return try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }
        val srcW = bounds.outWidth
        val srcH = bounds.outHeight
        if (srcW <= 0 || srcH <= 0) return null

        val opts = BitmapFactory.Options().apply {
            inSampleSize  = sampleSizeFor(srcW, srcH, VLM_MAX_DIM)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val raw = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        } ?: return null

        val scaled = scaleToMaxDim(raw, VLM_MAX_DIM)
        val outFile = File(File(context.cacheDir, "images").apply { mkdirs() },
                           "pick_${System.currentTimeMillis()}.jpg")
        FileOutputStream(outFile).use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, VLM_JPEG_Q, out)
        }
        if (scaled !== raw) raw.recycle()
        scaled.recycle()
        Log.i(VLM_LOG_TAG,
              "gallery: ${srcW}x${srcH} -> max ${VLM_MAX_DIM} -> ${outFile.length() / 1024} KB")
        outFile.absolutePath
    } catch (t: Throwable) {
        Log.w(VLM_LOG_TAG, "copyUriToCache failed: $t")
        null
    }
}

/** Downscale a JPEG file in-place. Used after [ActivityResultContracts.TakePicture]
 *  writes a full-resolution camera capture into our FileProvider URI. Falls
 *  back to the original path on any error so the chat still gets *some* image. */
private fun downscaleJpegInPlace(srcPath: String): String? {
    return try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(srcPath, bounds)
        val srcW = bounds.outWidth
        val srcH = bounds.outHeight
        if (srcW <= 0 || srcH <= 0) return null

        // If the camera already produced something within budget, skip the
        // re-encode round-trip — it would just add JPEG-of-JPEG artefacts.
        val longestSide = maxOf(srcW, srcH)
        if (longestSide <= VLM_MAX_DIM) {
            Log.i(VLM_LOG_TAG, "camera: ${srcW}x${srcH} already <= $VLM_MAX_DIM, skip")
            return srcPath
        }

        val opts = BitmapFactory.Options().apply {
            inSampleSize  = sampleSizeFor(srcW, srcH, VLM_MAX_DIM)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val raw = BitmapFactory.decodeFile(srcPath, opts) ?: return null
        val scaled = scaleToMaxDim(raw, VLM_MAX_DIM)
        FileOutputStream(srcPath).use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, VLM_JPEG_Q, out)
        }
        if (scaled !== raw) raw.recycle()
        scaled.recycle()
        Log.i(VLM_LOG_TAG,
              "camera: ${srcW}x${srcH} -> max ${VLM_MAX_DIM} -> ${File(srcPath).length() / 1024} KB")
        srcPath
    } catch (t: Throwable) {
        Log.w(VLM_LOG_TAG, "downscaleJpegInPlace failed: $t")
        null
    }
}

/** Pick the largest power-of-two `inSampleSize` whose subsampled output is
 *  still > target on the longest side (BitmapFactory's contract). */
private fun sampleSizeFor(w: Int, h: Int, target: Int): Int {
    if (target <= 0) return 1
    var s = 1
    while ((w / (s * 2)) >= target && (h / (s * 2)) >= target) s *= 2
    return s
}

/** Final exact-fit scale to [maxDim] preserving aspect ratio. Returns the
 *  input bitmap unchanged when it's already within budget. */
private fun scaleToMaxDim(src: Bitmap, maxDim: Int): Bitmap {
    val longest = maxOf(src.width, src.height)
    if (longest <= maxDim) return src
    val r = maxDim.toFloat() / longest
    val w = (src.width  * r).toInt().coerceAtLeast(1)
    val h = (src.height * r).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(src, w, h, /*filter=*/true)
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
                androidx.compose.foundation.layout.Column(
                    modifier = Modifier.widthIn(max = 480.dp)
                ) {
                    if (entry.streaming) {
                        Text(
                            "✦ thinking…",
                            style = MaterialTheme.typography.labelSmall,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.tertiary,
                            modifier = Modifier.padding(start = 4.dp, bottom = 2.dp),
                        )
                    }
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant,
                            contentColor   = MaterialTheme.colorScheme.onSurfaceVariant,
                        ),
                        shape  = RoundedCornerShape(4.dp, 16.dp, 16.dp, 16.dp),
                    ) {
                        Text(
                            entry.text.ifBlank { "…" },
                            modifier = Modifier.padding(12.dp),
                        )
                    }
                    if (!entry.streaming && (entry.timing != null || entry.ctxLeft != null)) {
                        Row(
                            modifier = Modifier.padding(top = 4.dp, start = 4.dp),
                            horizontalArrangement = Arrangement.spacedBy(6.dp),
                        ) {
                            entry.timing?.let { t ->
                                Text(
                                    t,
                                    style = MaterialTheme.typography.labelSmall,
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                                    modifier = Modifier
                                        .background(
                                            MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.6f),
                                            RoundedCornerShape(8.dp),
                                        )
                                        .padding(horizontal = 6.dp, vertical = 2.dp),
                                )
                            }
                            entry.ctxLeft?.let { c ->
                                Text(
                                    c,
                                    style = MaterialTheme.typography.labelSmall,
                                    fontFamily = FontFamily.Monospace,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                                    modifier = Modifier
                                        .background(
                                            MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.6f),
                                            RoundedCornerShape(8.dp),
                                        )
                                        .padding(horizontal = 6.dp, vertical = 2.dp),
                                )
                            }
                        }
                    }
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
