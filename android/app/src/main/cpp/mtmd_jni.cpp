// JNI bridge for the mtmd (multimodal text+image) decoder.
//
// Backs com.yzamari.turboquant.assistant.MtmdNative. Replaces the previous
// Runtime.exec("libllama-mtmd-cli.so") fork-per-image pattern in VlmRunner
// with a long-lived in-process session: load the LLM + mmproj once,
// describe N images without paying the per-call model-load tax (~3 s for
// SmolVLM-256M, ~8 s for Qwen2.5-VL-3B on Snapdragon 8 Gen 3).
//
// Public API (all C-linkage, JNI naming):
//   loadModel(modelPath, mmprojPath, ctxSize, threads, gpuLayers, kvType) -> long handle
//   unloadModel(long handle)
//   describe(long handle, imagePath, prompt, maxTokens, GenCallback cb) -> void
//   getStats(long handle) -> String  (JSON, matches LlamaNative.getStats)
//
// Linkage: depends on the public 'mtmd' library target from llama.cpp/tools/mtmd/.
// Backends are loaded lazily on the first call (init_backends_once below);
// llama_jni.cpp does the same thing on its side, the second caller is a no-op.

#include <jni.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <android/log.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "mtmd_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct VlmStats {
    double load_ms        = 0.0;
    double mmproj_load_ms = 0.0;
    double image_eval_ms  = 0.0;
    double prompt_ms      = 0.0;
    double decode_ms      = 0.0;
    int    n_image_tokens = 0;
    int    n_prompt       = 0;
    int    n_decode       = 0;
};

struct VlmSession {
    llama_model   * model       = nullptr;
    llama_context * ctx         = nullptr;
    llama_sampler * sampler     = nullptr;
    mtmd_context  * vision      = nullptr;
    int             n_ctx       = 0;
    int             n_threads   = 4;
    int             n_batch     = 512;
    llama_pos       n_past      = 0;     // running cursor across describe() calls
    VlmStats        last;
    std::mutex      mu;
};

std::atomic<bool> g_backend_inited{false};
std::mutex        g_backend_init_mu;

void init_backends_once() {
    if (g_backend_inited.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_backend_init_mu);
    if (g_backend_inited.load(std::memory_order_relaxed)) return;
    auto bridge = [](enum ggml_log_level level, const char * text, void * /*ud*/) {
        // Bridge every llama/ggml/clip log to logcat at its proper Android
        // level. Was filtered to >= WARN previously, which dropped the
        // mtmd-OpenCL Phase 1 markers (clip.cpp uses LOG_INF for "CLIP
        // using ... backend", "MTMD_VISION_GPU_DISABLE is set", and the
        // Adreno 740 bypass) so F4 verification couldn't see whether the
        // GPU placement actually triggered.
        int prio = ANDROID_LOG_INFO;
        switch (level) {
            case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
            case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
            case GGML_LOG_LEVEL_DEBUG: prio = ANDROID_LOG_DEBUG; break;
            default:                   prio = ANDROID_LOG_INFO;  break;
        }
        __android_log_print(prio, "llama", "%s", text);
    };
    llama_log_set(bridge, nullptr);
    // clip.cpp / mtmd.cpp use a SEPARATE g_logger_state inside the mtmd
    // library (see clip-impl.h:LOG_INF -> clip_log_internal). llama_log_set
    // does not reach it. mtmd_log_set is the public hook for that
    // namespace and routes the same way to logcat.
    mtmd_log_set(bridge, nullptr);
    ggml_backend_load_all();
    LOGI("ggml backends loaded");
    g_backend_inited.store(true, std::memory_order_release);
}

std::string jstring_to_std(JNIEnv * env, jstring js) {
    if (!js) return {};
    const char * c = env->GetStringUTFChars(js, nullptr);
    std::string  s = c ? c : "";
    if (c) env->ReleaseStringUTFChars(js, c);
    return s;
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string token_to_piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, /*special=*/true);
    if (n < 0) return {};
    return std::string(buf, n);
}

// Build a chat-template-formatted prompt with the mtmd image marker placed
// inside the user turn. Uses the model's built-in template via
// `llama_chat_apply_template` (public C API) — hardcoding ChatML the way an
// earlier draft did was the cause of the live-VLM hang on Tab S9+: SmolVLM
// uses a different chat format, so its tokenizer never produced the EOG
// token from a ChatML prompt and the decode loop ran forever.
//
// Fallback to ChatML if the model has no built-in template (rare; mtmd
// requires one anyway, since otherwise the upstream mtmd-cli also exits).
std::string build_vlm_prompt(const llama_model * model, const std::string & user_text) {
    const char * marker = mtmd_default_marker();

    std::string content;
    content.reserve(user_text.size() + 32);
    content += marker ? marker : "<__media__>";
    content += "\n";
    content += user_text;

    llama_chat_message msg{};
    msg.role    = "user";
    msg.content = content.c_str();

    const char * tmpl = llama_model_chat_template(model, /*name=*/nullptr);

    auto chatml_fallback = [&]() {
        std::string p;
        p.reserve(content.size() + 80);
        p += "<|im_start|>user\n";
        p += content;
        p += "<|im_end|>\n<|im_start|>assistant\n";
        return p;
    };

    if (!tmpl) {
        LOGW("build_vlm_prompt: no chat template on model — falling back to ChatML");
        return chatml_fallback();
    }

    // Probe the required buffer size, then format.
    int32_t needed = llama_chat_apply_template(
        tmpl, &msg, /*n_msg=*/1, /*add_ass=*/true, nullptr, 0);
    if (needed <= 0) {
        LOGW("build_vlm_prompt: llama_chat_apply_template probe returned %d "
             "— falling back to ChatML", (int) needed);
        return chatml_fallback();
    }

    std::string out(static_cast<size_t>(needed), '\0');
    int32_t actual = llama_chat_apply_template(
        tmpl, &msg, 1, true, out.data(), static_cast<int32_t>(out.size()));
    if (actual <= 0) {
        LOGW("build_vlm_prompt: llama_chat_apply_template returned %d "
             "— falling back to ChatML", (int) actual);
        return chatml_fallback();
    }
    out.resize(static_cast<size_t>(actual));
    return out;
}

}  // namespace

extern "C" {

// -------------------------------------------------------------------------
// loadModel(modelPath, mmprojPath, ctxSize, threads, gpuLayers, kvType) -> long
JNIEXPORT jlong JNICALL
Java_com_yzamari_turboquant_assistant_MtmdNative_loadModel(
    JNIEnv * env, jclass,
    jstring jModelPath, jstring jMmprojPath,
    jint ctxSize, jint threads, jint gpuLayers, jint kvType) {

    init_backends_once();

    const std::string modelPath  = jstring_to_std(env, jModelPath);
    const std::string mmprojPath = jstring_to_std(env, jMmprojPath);
    if (modelPath.empty() || mmprojPath.empty()) {
        LOGE("loadModel: empty path(s)");
        return 0;
    }
    LOGI("loadModel: model=%s mmproj=%s ctx=%d threads=%d gpu=%d kvType=%d",
         modelPath.c_str(), mmprojPath.c_str(), ctxSize, threads, gpuLayers, kvType);

    const double t0 = now_ms();

    auto sess = std::make_unique<VlmSession>();
    sess->n_ctx     = std::max(512, (int) ctxSize);
    sess->n_threads = std::max(1,   (int) threads);
    sess->n_batch   = sess->n_ctx;

    // ---- 1. Load the LLM ------------------------------------------------
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = (int) gpuLayers;
    sess->model = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!sess->model) {
        LOGE("loadModel: llama_model_load_from_file(%s) failed", modelPath.c_str());
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) sess->n_ctx;
    cparams.n_batch         = (uint32_t) sess->n_batch;
    cparams.n_threads       = sess->n_threads;
    cparams.n_threads_batch = sess->n_threads;

    switch ((int) kvType) {
        case 1:
            cparams.type_k          = GGML_TYPE_Q4_0;
            cparams.type_v          = GGML_TYPE_Q4_0;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            LOGI("loadModel: KV cache = q4_0 (4× compressed, +flash attn)");
            break;
        case 2:
            cparams.type_k          = GGML_TYPE_Q8_0;
            cparams.type_v          = GGML_TYPE_Q8_0;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            LOGI("loadModel: KV cache = q8_0 (2× compressed, +flash attn)");
            break;
        case 3:
            cparams.kv_turboquant   = true;
            LOGI("loadModel: KV cache = TurboQuant (Path 2 scaffold)");
            break;
        default:
            LOGI("loadModel: KV cache = f16 (baseline)");
            break;
    }

    sess->ctx = llama_init_from_model(sess->model, cparams);
    if (!sess->ctx) {
        LOGE("loadModel: llama_init_from_model failed");
        llama_model_free(sess->model);
        return 0;
    }

    // ---- 2. Sampler (greedy is plenty for image description) -----------
    sess->sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sess->sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sess->sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sess->sampler, llama_sampler_init_temp (0.7f));
    llama_sampler_chain_add(sess->sampler, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));

    // ---- 3. Load the mmproj / vision encoder ---------------------------
    const double t_mm0 = now_ms();
    mtmd_context_params vparams = mtmd_context_params_default();
    // Phase 1 (Path C — mtmd-on-OpenCL): SigLIP can now run on Adreno
    // because clip.cpp itself does the SoC bypass for the known-hung
    // Adreno 740 (Tab S9+ kalama). On Adreno 750 (S24 Ultra) and other
    // safe SoCs we get the GPU vision tower (~15× faster than CPU per
    // the design doc). The pre-Phase-1 hardcoded `use_gpu = false` was
    // the app-side mitigation that this Phase 1 work replaces.
    //
    // Escape hatch if a new SoC turns out to hang: set the
    // MTMD_VISION_GPU_DISABLE env var (any non-empty value) via
    //   adb shell setprop debug.com.yzamari.turboquant.mtmd_gpu 0
    // (we'd need to wire that prop -> env var). For now the Adreno 740
    // bypass in clip.cpp is the only known case.
    vparams.use_gpu       = true;
    vparams.print_timings = false;
    vparams.n_threads     = sess->n_threads;
    vparams.warmup        = false;
    sess->vision = mtmd_init_from_file(mmprojPath.c_str(), sess->model, vparams);
    if (!sess->vision) {
        LOGE("loadModel: mtmd_init_from_file(%s) failed", mmprojPath.c_str());
        llama_sampler_free(sess->sampler);
        llama_free(sess->ctx);
        llama_model_free(sess->model);
        return 0;
    }
    sess->last.mmproj_load_ms = now_ms() - t_mm0;
    sess->last.load_ms        = now_ms() - t0;
    LOGI("loadModel: ready in %.1f ms (mmproj %.1f ms)",
         sess->last.load_ms, sess->last.mmproj_load_ms);

    return reinterpret_cast<jlong>(sess.release());
}

// -------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_com_yzamari_turboquant_assistant_MtmdNative_unloadModel(JNIEnv *, jclass, jlong handle) {
    auto * sess = reinterpret_cast<VlmSession *>(handle);
    if (!sess) return;
    if (sess->vision)  mtmd_free(sess->vision);
    if (sess->sampler) llama_sampler_free(sess->sampler);
    if (sess->ctx)     llama_free(sess->ctx);
    if (sess->model)   llama_model_free(sess->model);
    delete sess;
    LOGI("unloadModel: freed");
}

// -------------------------------------------------------------------------
// describe(handle, imagePath, prompt, maxTokens, GenCallback cb) -> void
JNIEXPORT void JNICALL
Java_com_yzamari_turboquant_assistant_MtmdNative_describe(
    JNIEnv * env, jclass,
    jlong handle,
    jstring jImagePath, jstring jPrompt,
    jint maxTokens, jobject jcb) {

    auto * sess = reinterpret_cast<VlmSession *>(handle);
    if (!sess) {
        LOGE("describe: null handle");
        return;
    }
    std::lock_guard<std::mutex> lk(sess->mu);

    const std::string imagePath = jstring_to_std(env, jImagePath);
    const std::string userText  = jstring_to_std(env, jPrompt);

    jclass  cbClass = jcb ? env->GetObjectClass(jcb) : nullptr;
    jmethodID onTok = cbClass ? env->GetMethodID(cbClass, "onToken",
                                                  "(Ljava/lang/String;)V") : nullptr;
    auto emit = [&](const std::string & piece) {
        if (!cbClass || !onTok || piece.empty()) return;
        jstring s = env->NewStringUTF(piece.c_str());
        env->CallVoidMethod(jcb, onTok, s);
        env->DeleteLocalRef(s);
    };

    // Reset the KV cache cursor — each describe() call is an independent
    // single-turn conversation. (Multi-turn vision chat is a follow-up.)
    llama_memory_clear(llama_get_memory(sess->ctx), /*data=*/true);
    sess->n_past = 0;

    // ---- Load the image bitmap from disk -------------------------------
    mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_file(sess->vision, imagePath.c_str());
    if (!bmp) {
        LOGE("describe: failed to load bitmap from %s", imagePath.c_str());
        emit("(failed to load image)");
        return;
    }
    LOGI("describe: image %s -> %ux%u",
         imagePath.c_str(),
         (unsigned) mtmd_bitmap_get_nx(bmp),
         (unsigned) mtmd_bitmap_get_ny(bmp));

    // ---- Tokenize prompt + image into mtmd chunks ----------------------
    const std::string prompt = build_vlm_prompt(sess->model, userText);
    LOGI("describe: prompt (%zu chars): %s",
         prompt.size(), prompt.size() > 200 ? "<truncated>" : prompt.c_str());
    mtmd_input_text text{};
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    LOGI("describe: tokenizing...");
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bmps[1] = { bmp };
    if (mtmd_tokenize(sess->vision, chunks, &text, bmps, 1) != 0) {
        LOGE("describe: mtmd_tokenize failed");
        emit("(tokenize failed)");
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmp);
        return;
    }
    sess->last.n_image_tokens = (int) mtmd_helper_get_n_tokens(chunks);
    sess->last.n_prompt       = (int) mtmd_helper_get_n_pos(chunks);
    LOGI("describe: tokenized — n_image_tokens=%d n_prompt=%d, eval_chunks…",
         sess->last.n_image_tokens, sess->last.n_prompt);

    // ---- Eval all chunks (image + text) into KV ------------------------
    const double t_img0 = now_ms();
    llama_pos new_n_past = 0;
    int rc = mtmd_helper_eval_chunks(
        sess->vision,
        sess->ctx,
        chunks,
        /*n_past=*/0,
        /*seq_id=*/0,
        /*n_batch=*/sess->n_batch,
        /*logits_last=*/true,
        &new_n_past);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bmp);
    if (rc != 0) {
        LOGE("describe: mtmd_helper_eval_chunks failed (%d)", rc);
        emit("(eval failed)");
        return;
    }
    sess->n_past             = new_n_past;
    sess->last.image_eval_ms = now_ms() - t_img0;
    LOGI("describe: image eval done in %.1f ms (new_n_past=%d), starting decode…",
         sess->last.image_eval_ms, (int) new_n_past);

    // ---- Decode loop: sample, emit, advance KV -------------------------
    const llama_vocab * vocab = llama_model_get_vocab(sess->model);
    llama_batch batch = llama_batch_init(1, 0, 1);
    const double t_dec0 = now_ms();
    int n_decoded = 0;
    for (int i = 0; i < (int) maxTokens; ++i) {
        const llama_token tok = llama_sampler_sample(sess->sampler, sess->ctx, -1);
        if (i < 4) LOGI("describe: sampled tok=%d at step=%d", (int) tok, i);
        if (llama_vocab_is_eog(vocab, tok)) break;

        const std::string piece = token_to_piece(vocab, tok);
        emit(piece);
        n_decoded++;

        batch.n_tokens   = 1;
        batch.token[0]   = tok;
        batch.pos[0]     = sess->n_past++;
        batch.n_seq_id[0]= 1;
        batch.seq_id[0][0]= 0;
        batch.logits[0]  = true;
        if (llama_decode(sess->ctx, batch) != 0) {
            LOGE("describe: llama_decode failed at step %d", i);
            break;
        }
    }
    llama_batch_free(batch);
    sess->last.decode_ms = now_ms() - t_dec0;
    sess->last.n_decode  = n_decoded;

    LOGI("describe: image %.1f ms · decode %d tok in %.1f ms (%.1f tok/s)",
         sess->last.image_eval_ms, n_decoded, sess->last.decode_ms,
         n_decoded / std::max(1.0, sess->last.decode_ms / 1000.0));
}

// -------------------------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_com_yzamari_turboquant_assistant_MtmdNative_getStats(JNIEnv * env, jclass, jlong handle) {
    auto * sess = reinterpret_cast<VlmSession *>(handle);
    if (!sess) return env->NewStringUTF("{}");
    char buf[512];
    const auto & s = sess->last;
    const double dec_tps = s.decode_ms > 0 ? s.n_decode / (s.decode_ms / 1000.0) : 0.0;
    std::snprintf(buf, sizeof(buf),
        "{\"load_ms\":%.1f,\"mmproj_load_ms\":%.1f,"
         "\"image_eval_ms\":%.1f,\"n_image_tokens\":%d,"
         "\"n_prompt\":%d,\"n_decode\":%d,\"decode_ms\":%.1f,"
         "\"decode_tok_per_sec\":%.2f}",
        s.load_ms, s.mmproj_load_ms,
        s.image_eval_ms, s.n_image_tokens,
        s.n_prompt, s.n_decode, s.decode_ms, dec_tps);
    return env->NewStringUTF(buf);
}

}  // extern "C"
