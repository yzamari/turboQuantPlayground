package com.yzamari.turboquant.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
import android.util.Log
import android.util.Size
import android.view.ViewGroup.LayoutParams
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.SwapHoriz
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.compositeOver
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.yzamari.turboquant.assistant.AssistantViewModel
import com.yzamari.turboquant.assistant.VlmRunner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * "Live" tab — point the camera at things, get continuous on-device VLM
 * descriptions. Uses the same persistent `MtmdNative` session that the
 * single-shot chat path now uses (PR #10), so per-frame cost is just the
 * SigLIP encode + token decode (~0.8–1.2 s on Adreno for SmolVLM-256M).
 *
 * Frame pump:
 *   - CameraX `ImageAnalysis` use case at the lowest resolution that still
 *     covers SmolVLM/SigLIP's 384×384 input — 640×480 in 4:3, since the
 *     mtmd encoder will downscale anyway.
 *   - `STRATEGY_KEEP_ONLY_LATEST` so the analyzer never queues frames.
 *   - Single-flight gate: if the previous VLM call hasn't returned, the
 *     analyzer drops the frame. Effective FPS is naturally capped to
 *     1 / per-frame-VLM-latency.
 */
@Composable
fun LiveCameraScreen(vm: AssistantViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED
        )
    }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasCameraPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasCameraPermission) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    // Live state visible to the UI (mutated by frame analyzer & VLM callback).
    var caption        by remember { mutableStateOf("Point the camera at something…") }
    var lastFrameMs    by remember { mutableStateOf(0L) }
    var lastDecodeTps  by remember { mutableStateOf(0f) }
    var framesAnalyzed by remember { mutableStateOf(0) }
    var paused         by remember { mutableStateOf(false) }
    // Default to front camera so the user sees themselves the moment they
    // tap Live — works as an immediate self-test ("describe me") without
    // needing to flip the device. Switch button still works.
    var useFrontCam    by remember { mutableStateOf(true) }

    if (!hasCameraPermission) {
        Box(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                "Camera permission is required for Live VLM.\n" +
                    "Tap to grant.",
                style = MaterialTheme.typography.bodyLarge,
            )
        }
        return
    }

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {

        // ---- PreviewView + analyzer ------------------------------------
        // Tab S9+ (board "kalama") + every other unrecognised SoC routes
        // SigLIP/SmolVLM through CPU because the upstream mtmd helper hangs
        // on Adreno 740 (see VlmRunner.isAdrenoVlmKnownGood). One frame on
        // CPU ≈ 165 s; without on-screen feedback the Live tab looks frozen
        // for the first three minutes. Show an explicit "Encoding…" status
        // the moment a frame is in flight so the user knows the analyzer
        // is grinding rather than dead.
        val cpuFallback = !isAdrenoVlmKnownGoodBoard()
        CameraPreviewView(
            paused        = paused,
            useFrontCam   = useFrontCam,
            vlmRunner     = vm.vlmRunner,
            scope         = scope,
            onFrameStart  = {
                caption = if (framesAnalyzed == 0 && cpuFallback) {
                    "Encoding first frame on CPU — about 2½ min on this SoC. " +
                        "Subsequent frames take a similar amount of time. " +
                        "(Adreno+mtmd hang at our pinned llama.cpp commit; " +
                        "S24 Ultra runs this on Adreno and is ~10× faster.)"
                } else if (cpuFallback) {
                    "Encoding image on CPU…"
                } else {
                    "Encoding image…"
                }
            },
            onAnalyzedFrame = { ms, tps ->
                lastFrameMs    = ms
                lastDecodeTps  = tps
                framesAnalyzed += 1
            },
            onCaption     = { piece, _ ->
                caption = piece
            },
            modifier      = Modifier.fillMaxSize(),
        )

        // ---- Top stats chip -------------------------------------------
        StatsChip(
            paused         = paused,
            modelLabel     = vm.vlmRunner.activeModel.displayName.substringBefore(" ("),
            framesAnalyzed = framesAnalyzed,
            lastFrameMs    = lastFrameMs,
            lastDecodeTps  = lastDecodeTps,
            modifier       = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(top = 16.dp, start = 16.dp, end = 16.dp),
        )

        // ---- Bottom caption card --------------------------------------
        Card(
            colors = CardDefaults.cardColors(
                containerColor = Color.Black.copy(alpha = 0.65f)
                    .compositeOver(Color(0xFF222222)),
            ),
            shape = RoundedCornerShape(16.dp),
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(start = 16.dp, end = 16.dp, bottom = 96.dp),
        ) {
            Text(
                caption,
                color    = Color.White,
                modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                style    = MaterialTheme.typography.titleMedium.copy(
                    fontFamily = FontFamily.Default,
                    fontWeight = FontWeight.Medium,
                ),
            )
        }

        // ---- Floating controls ----------------------------------------
        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(bottom = 24.dp),
            horizontalArrangement = Arrangement.Center,
        ) {
            IconButton(onClick = { useFrontCam = !useFrontCam }) {
                Icon(Icons.Filled.SwapHoriz,
                     contentDescription = "Switch camera",
                     tint = Color.White)
            }
            FloatingActionButton(
                onClick = { paused = !paused },
                modifier = Modifier.padding(horizontal = 24.dp),
            ) {
                if (paused) Icon(Icons.Filled.PlayArrow, "Resume")
                else        Icon(Icons.Filled.Stop,      "Pause")
            }
        }
    }
}

@Composable
private fun StatsChip(
    paused: Boolean,
    modelLabel: String,
    framesAnalyzed: Int,
    lastFrameMs: Long,
    lastDecodeTps: Float,
    modifier: Modifier = Modifier,
) {
    val fps = if (lastFrameMs > 0) 1000f / lastFrameMs else 0f
    Card(
        colors = CardDefaults.cardColors(
            containerColor = Color.Black.copy(alpha = 0.55f),
        ),
        shape = RoundedCornerShape(12.dp),
        modifier = modifier,
    ) {
        Column(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Text(
                if (paused) "⏸ paused · $modelLabel"
                else        "▶ $modelLabel · ${"%.2f".format(fps)} FPS · last ${lastFrameMs}ms" +
                            (if (lastDecodeTps > 0) " · ${"%.0f".format(lastDecodeTps)} tok/s" else ""),
                color = Color.White,
                style = MaterialTheme.typography.labelMedium,
                fontFamily = FontFamily.Monospace,
            )
            if (framesAnalyzed > 0) {
                Text(
                    "$framesAnalyzed frames analyzed",
                    color = Color.White.copy(alpha = 0.7f),
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
    }
}

@Composable
private fun CameraPreviewView(
    paused: Boolean,
    useFrontCam: Boolean,
    vlmRunner: VlmRunner,
    scope: CoroutineScope,
    onFrameStart: () -> Unit,
    onAnalyzedFrame: (frameMs: Long, decodeTps: Float) -> Unit,
    onCaption: (text: String, isFinal: Boolean) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    // Single-flight gate: only one VLM call in flight at a time.
    val inFlight = remember { AtomicBoolean(false) }
    // Background executor for the ImageAnalysis pipeline.
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }
    val captionBuilder = remember { StringBuilder() }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            layoutParams = android.view.ViewGroup.LayoutParams(
                LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        }
    }

    val pausedRef = remember { mutableStateOf(paused) }
    pausedRef.value = paused

    // Bind / re-bind the camera whenever lens facing changes.
    DisposableEffect(useFrontCam) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        val analyzer = ImageAnalysis.Analyzer { imageProxy ->
            try {
                if (pausedRef.value) {
                    imageProxy.close()
                    return@Analyzer
                }
                if (!inFlight.compareAndSet(false, true)) {
                    // Previous frame still in VLM. Drop this frame.
                    imageProxy.close()
                    return@Analyzer
                }

                // Encode the frame to JPEG on this analyzer thread (cheap
                // vs the VLM call). The ResolutionSelector below *requests*
                // 640×480 from CameraX, but on some HALs that's only a hint
                // — actual frames can come back larger. SigLIP downscales
                // to 384×384 internally anyway, so anything above 640×480
                // is wasted JPEG-encode and (more importantly) wasted
                // SigLIP preprocess work. Cap defensively here.
                val tFrameStart = System.currentTimeMillis()
                val bitmap: Bitmap = imageProxy.toBitmap()
                val rotation = imageProxy.imageInfo.rotationDegrees
                imageProxy.close()
                val rotated = if (rotation != 0) rotateBitmap(bitmap, rotation) else bitmap
                val downscaled = downscaleToFit(rotated, MAX_LONG_EDGE_PX)
                val jpegFile = File(context.cacheDir, "live_frame.jpg")
                FileOutputStream(jpegFile).use { fos ->
                    downscaled.compress(Bitmap.CompressFormat.JPEG, 85, fos)
                }
                if (rotated !== bitmap) bitmap.recycle()
                if (downscaled !== rotated) rotated.recycle()
                downscaled.recycle()

                // Hand the JPEG path to the VLM. Returns when generation
                // ends, then we release the gate so the analyzer accepts
                // the next frame.
                scope.launch(Dispatchers.Main) {
                    onFrameStart()
                }
                scope.launch(Dispatchers.IO) {
                    captionBuilder.setLength(0)
                    val result = runCatching {
                        vlmRunner.describe(
                            imagePath = jpegFile.absolutePath,
                            prompt    = "What's in this image? One sentence.",
                            maxTokens = 60,
                            threads   = 4,
                        ) { piece ->
                            captionBuilder.append(piece)
                            scope.launch(Dispatchers.Main) {
                                onCaption(captionBuilder.toString().trim(), false)
                            }
                        }
                    }
                    val frameMs = System.currentTimeMillis() - tFrameStart
                    val statsJson = result.getOrNull()?.stats ?: ""
                    val tps = parseDecodeTps(statsJson)
                    withContext(Dispatchers.Main) {
                        onAnalyzedFrame(frameMs, tps)
                        result.fold(
                            onSuccess = { r -> onCaption(r.reply.ifBlank { "(no reply)" }, true) },
                            onFailure = { onCaption("(error: ${it.message ?: it.javaClass.simpleName})", true) },
                        )
                    }
                    inFlight.set(false)
                }
            } catch (t: Throwable) {
                Log.w("LiveCamera", "analyzer: $t")
                inFlight.set(false)
                runCatching { imageProxy.close() }
            }
        }

        val listener = Runnable {
            try {
                val provider = providerFuture.get()
                provider.unbindAll()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                // Keep frame size small — SigLIP/SmolVLM consumes 384×384 anyway,
                // so 640×480 is plenty and saves a lot of YUV→bitmap work.
                val resolutionSelector = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy(
                        AspectRatio.RATIO_4_3,
                        AspectRatioStrategy.FALLBACK_RULE_AUTO))
                    .setResolutionStrategy(ResolutionStrategy(
                        Size(640, 480),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                    .build()
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setResolutionSelector(resolutionSelector)
                    .build()
                analysis.setAnalyzer(analyzerExecutor, analyzer)

                val selector = if (useFrontCam) CameraSelector.DEFAULT_FRONT_CAMERA
                               else            CameraSelector.DEFAULT_BACK_CAMERA
                provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
                Log.i("LiveCamera", "camera bound (front=$useFrontCam)")
            } catch (t: Throwable) {
                Log.e("LiveCamera", "camera bind failed: $t")
            }
        }
        providerFuture.addListener(listener, ContextCompat.getMainExecutor(context))

        onDispose {
            try {
                ProcessCameraProvider.getInstance(context).get().unbindAll()
            } catch (_: Throwable) { }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            analyzerExecutor.shutdown()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier,
    )
}

private fun rotateBitmap(src: Bitmap, degrees: Int): Bitmap {
    if (degrees == 0) return src
    val matrix = android.graphics.Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, /*filter=*/true)
}

/** SigLIP downsamples to 384×384, so feeding anything beyond ~640 on the long
 *  edge is wasted JPEG-encode + wasted SigLIP preprocess work. Returns [src]
 *  unchanged when it's already small enough. */
private const val MAX_LONG_EDGE_PX = 640
private fun downscaleToFit(src: Bitmap, maxLongEdge: Int): Bitmap {
    val longEdge = maxOf(src.width, src.height)
    if (longEdge <= maxLongEdge) return src
    val scale = maxLongEdge.toFloat() / longEdge
    val w = (src.width  * scale).toInt().coerceAtLeast(1)
    val h = (src.height * scale).toInt().coerceAtLeast(1)
    return Bitmap.createScaledBitmap(src, w, h, /*filter=*/true)
}

private fun parseDecodeTps(statsJson: String): Float {
    if (statsJson.isBlank()) return 0f
    val m = Regex("\"decode_tok_per_sec\":([0-9.]+)").find(statsJson)
    return m?.groupValues?.get(1)?.toFloatOrNull() ?: 0f
}

/** Mirror of VlmRunner.isAdrenoVlmKnownGood — kept private to the UI so the
 *  Live screen can tell the user upfront when this SoC will route VLM
 *  through CPU and a single frame will take ~2½ min. */
private fun isAdrenoVlmKnownGoodBoard(): Boolean {
    val board = Build.BOARD.lowercase()
    if (board.startsWith("pineapple")) return true
    if (board.startsWith("sun") || board.startsWith("8elite")) return true
    return false
}
