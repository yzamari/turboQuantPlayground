// JNI bridge for llama.cpp — backs com.yzamari.turboquant.assistant.LlamaNative.
//
// Public API (all C-linkage, JNI naming):
//   loadModel(String path, int ctxSize, int threads) -> long handle
//   unloadModel(long handle) -> void
//   generate(long handle, String prompt, int maxTokens, GenCallback cb) -> void
//   tokenCount(long handle, String text) -> int
//   getStats(long handle) -> String  (JSON)
//
// The handle is a pointer to a Session struct allocated on the C++ heap.
// Stays portable: no dependencies on libcommon — only llama.h public API.

#include <jni.h>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include "turboquant/api.hpp"

#include <android/log.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "llama_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Stats {
    int      n_prompt   = 0;
    int      n_decode   = 0;
    double   prompt_ms  = 0.0;
    double   decode_ms  = 0.0;
};

struct Session {
    llama_model   * model   = nullptr;
    llama_context * ctx     = nullptr;
    llama_sampler * sampler = nullptr;
    int             n_ctx   = 0;
    int             n_threads = 4;
    Stats           last;
    std::mutex      mu;     // guards generation / stats
};

// Thunk for the TurboQuant attention provider hook (P2.1c Step 4). Bound at
// init_backends_once() time to llama_set_turboquant_attn_fn(). The fork's
// custom ggml op (ggml-custom-turboquant.c) calls this when kvType=3 builds
// a llama_kv_cache_turboquant cache and flash-attn is off.
//
// session is the kv_cache_turboquant pointer (opaque to us); layer_il is
// the transformer layer index. We forward both into libturboquant so it
// can cache prepared K-state per (session, layer) and amortize the rotate +
// quantize work across decode steps (Phase A2).
extern "C" void tq_attn_thunk(
    const void * session, int layer_il,
    const float * q, const float * k, const float * v,
    int BH, int n_q, int n_kv, int D,
    float scale, const float * mask, float * out) {
    const uint64_t session_id = reinterpret_cast<uintptr_t>(session);
    turboquant::attention_turboquant(
        q, k, v, BH, n_q, n_kv, D, scale, mask, out,
        /*key_bits  =*/ 3, /*value_bits=*/ 2,
        /*seed      =*/ 42 + 7 * (uint64_t) (layer_il < 0 ? 0 : layer_il),
        session_id, layer_il);
}

// Forward decl — definition lives further down (helper used by both the
// init_backends_once timing and per-request stats below).
double now_ms();

// Single global flag — the first loadModel call performs ggml backend init.
std::atomic<bool> g_backend_inited{false};
std::mutex        g_backend_init_mu;

void init_backends_once() {
    if (g_backend_inited.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lk(g_backend_init_mu);
    if (g_backend_inited.load(std::memory_order_relaxed)) return;
    llama_log_set([](enum ggml_log_level level, const char * text, void * /*ud*/) {
        if (level >= GGML_LOG_LEVEL_WARN) {
            __android_log_print(ANDROID_LOG_INFO, "llama", "%s", text);
        }
    }, nullptr);
    ggml_backend_load_all();
    llama_set_turboquant_attn_fn(tq_attn_thunk);

    // Eagerly build the libturboquant singleton backend (best available;
    // OpenCL on Adreno when present). Pays the OpenCL program-compilation
    // cost (~500 ms) here at app init instead of on the first decode token.
    // P2.1c Phase A1 — see /Users/yahavzamari/.claude/plans/...
    const double t_be0 = now_ms();
    turboquant::ensure_backends_initialized();
    const double t_be_ms = now_ms() - t_be0;
    LOGI("ggml backends loaded; turboquant attention provider registered "
         "(turboquant backend init %.1f ms)", t_be_ms);
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

// Tokenize using the public llama API (avoids depending on common.h).
std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                   bool add_bos, bool parse_special) {
    int n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                             nullptr, 0, add_bos, parse_special);
    if (n <= 0) return {};
    std::vector<llama_token> out(n);
    int rc = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(),
                             out.data(), (int32_t) out.size(), add_bos, parse_special);
    if (rc < 0) return {};
    return out;
}

std::string token_to_piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, /*special=*/true);
    if (n < 0) return {};
    return std::string(buf, n);
}

}  // namespace

extern "C" {

// -------------------------------------------------------------------------
// loadModel(String path, int ctxSize, int threads, int kvType, int gpuLayers) -> long
//   kvType    : 0 = FP16  (baseline)
//               1 = q4_0  (TurboQuant cousin — 4× compression, ~0.85 cosine, +flash attn)
//               2 = q8_0  (8-bit, 2× compression, ~0.99 cosine)
//   gpuLayers : 99 = offload all to Adreno; 0 = CPU only
// -------------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_loadModel(
    JNIEnv * env, jclass /*clazz*/,
    jstring jPath, jint ctxSize, jint threads, jint kvType, jint gpuLayers)
{
    init_backends_once();

    const std::string path = jstring_to_std(env, jPath);
    if (path.empty()) {
        LOGE("loadModel: empty path");
        return 0;
    }

    LOGI("loadModel: %s ctx=%d threads=%d kvType=%d gpuLayers=%d",
         path.c_str(), (int) ctxSize, (int) threads, (int) kvType, (int) gpuLayers);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = (int) gpuLayers;  // 99 = all on Adreno via ggml-opencl

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (!model) {
        LOGE("loadModel: llama_model_load_from_file failed");
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = (uint32_t) std::max(512, (int) ctxSize);
    cparams.n_batch     = cparams.n_ctx;
    cparams.n_threads   = std::max(1, (int) threads);
    cparams.n_threads_batch = cparams.n_threads;

    // KV-cache compression — the in-the-hot-path TurboQuant equivalent.
    // q4_0 KV requires flash attention.
    switch ((int) kvType) {
        case 1:
            cparams.type_k    = GGML_TYPE_Q4_0;
            cparams.type_v    = GGML_TYPE_Q4_0;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            LOGI("loadModel: KV cache = q4_0 (4× compressed, +flash attn)");
            break;
        case 2:
            cparams.type_k    = GGML_TYPE_Q8_0;
            cparams.type_v    = GGML_TYPE_Q8_0;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            LOGI("loadModel: KV cache = q8_0 (2× compressed, +flash attn)");
            break;
        case 3:
            // Path 2.1c: route the attention triple through libturboquant via
            // the custom ggml op + provider hook. Flash attention must be off
            // — our custom op replaces the standard Q@K.T → softmax → attn@V
            // path, not the fused flash kernel.
            cparams.kv_turboquant   = true;
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            LOGI("loadModel: KV cache = TurboQuant (PolarQuant + 1-bit QJL); "
                 "flash attn disabled, custom ggml op active");
            break;
        default:
            LOGI("loadModel: KV cache = f16 (baseline)");
            break;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOGE("loadModel: llama_init_from_model failed");
        llama_model_free(model);
        return 0;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist (LLAMA_DEFAULT_SEED));

    Session * s   = new Session();
    s->model      = model;
    s->ctx        = ctx;
    s->sampler    = smpl;
    s->n_ctx      = (int) cparams.n_ctx;
    s->n_threads  = (int) cparams.n_threads;

    LOGI("loadModel: success, handle=%p", (void *) s);
    return reinterpret_cast<jlong>(s);
}

// -------------------------------------------------------------------------
// unloadModel(long handle)
// -------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_unloadModel(
    JNIEnv * /*env*/, jclass /*clazz*/, jlong handle)
{
    Session * s = reinterpret_cast<Session *>(handle);
    if (!s) return;
    std::lock_guard<std::mutex> lk(s->mu);
    if (s->sampler) llama_sampler_free(s->sampler);
    if (s->ctx)     llama_free(s->ctx);
    if (s->model)   llama_model_free(s->model);
    delete s;
    LOGI("unloadModel: handle=%p freed", (void *) s);
}

// -------------------------------------------------------------------------
// tokenCount(long handle, String text) -> int
// -------------------------------------------------------------------------
JNIEXPORT jint JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_tokenCount(
    JNIEnv * env, jclass /*clazz*/, jlong handle, jstring jText)
{
    Session * s = reinterpret_cast<Session *>(handle);
    if (!s) return 0;
    const auto txt = jstring_to_std(env, jText);
    if (txt.empty()) return 0;
    const llama_vocab * vocab = llama_model_get_vocab(s->model);
    auto toks = tokenize(vocab, txt, /*add_bos=*/false, /*parse_special=*/true);
    return (jint) toks.size();
}

// -------------------------------------------------------------------------
// generate(long handle, String prompt, int maxTokens, GenCallback cb)
//
// `cb` is a Java object with a single method `onToken(String)` that the
// native side calls for each token piece. This is a synchronous, blocking
// call from the caller's perspective; the caller must already be on a
// background thread.
// -------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_generate(
    JNIEnv * env, jclass /*clazz*/,
    jlong handle, jstring jPrompt, jint maxTokens, jobject cb)
{
    Session * s = reinterpret_cast<Session *>(handle);
    if (!s) {
        LOGE("generate: null handle");
        return;
    }
    std::lock_guard<std::mutex> lk(s->mu);

    const std::string prompt = jstring_to_std(env, jPrompt);
    if (prompt.empty()) return;

    // Resolve the GenCallback.onToken method once.
    jclass    cbClass = nullptr;
    jmethodID onToken = nullptr;
    if (cb) {
        cbClass = env->GetObjectClass(cb);
        onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");
        if (!onToken) {
            LOGE("generate: GenCallback.onToken(String) not found");
            return;
        }
    }

    const llama_vocab * vocab = llama_model_get_vocab(s->model);

    // First-call detection: if the seq is empty, llama needs a BOS.
    const bool is_first =
        (llama_memory_seq_pos_max(llama_get_memory(s->ctx), 0) == -1);

    auto prompt_tokens = tokenize(vocab, prompt, /*add_bos=*/is_first,
                                   /*parse_special=*/true);
    if (prompt_tokens.empty()) {
        LOGE("generate: tokenization produced 0 tokens");
        return;
    }

    Stats st;
    st.n_prompt = (int) prompt_tokens.size();

    const double t0 = now_ms();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                             (int32_t) prompt_tokens.size());

    int produced = 0;
    const int hard_cap = std::max(1, (int) maxTokens);

    bool prompt_done = false;
    double t_prompt_done = t0;

    while (produced < hard_cap) {
        // Make sure we have ctx for this batch.
        const int n_ctx_total = llama_n_ctx(s->ctx);
        const int n_ctx_used  = llama_memory_seq_pos_max(llama_get_memory(s->ctx), 0) + 1;
        if (n_ctx_used + batch.n_tokens > n_ctx_total) {
            LOGW("generate: ctx full (%d/%d), stopping", n_ctx_used, n_ctx_total);
            break;
        }

        const int rc = llama_decode(s->ctx, batch);
        if (rc != 0) {
            LOGE("generate: llama_decode rc=%d", rc);
            break;
        }

        if (!prompt_done) {
            t_prompt_done = now_ms();
            prompt_done   = true;
        }

        const llama_token id = llama_sampler_sample(s->sampler, s->ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            LOGI("generate: EOG after %d tokens", produced);
            break;
        }

        const std::string piece = token_to_piece(vocab, id);
        if (cb && !piece.empty()) {
            jstring jPiece = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(cb, onToken, jPiece);
            env->DeleteLocalRef(jPiece);
            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
                LOGE("generate: Java callback threw — aborting");
                break;
            }
        }

        ++produced;

        // Feed the sampled token back for the next decode. Use a stable
        // (thread-local) buffer because llama_batch_get_one only stores a
        // pointer to the token array.
        static thread_local llama_token next_tok;
        next_tok = id;
        batch    = llama_batch_get_one(&next_tok, 1);
    }

    const double t1 = now_ms();
    st.prompt_ms = t_prompt_done - t0;
    st.decode_ms = t1 - t_prompt_done;
    st.n_decode  = produced;
    s->last      = st;
}

// -------------------------------------------------------------------------
// getStats(long handle) -> String (JSON)
// -------------------------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_getStats(
    JNIEnv * env, jclass /*clazz*/, jlong handle)
{
    Session * s = reinterpret_cast<Session *>(handle);
    std::ostringstream ss;
    ss << "{";
    if (s) {
        const auto & st = s->last;
        const double prompt_tps = st.prompt_ms > 0
            ? (st.n_prompt * 1000.0 / st.prompt_ms) : 0.0;
        const double decode_tps = st.decode_ms > 0
            ? (st.n_decode * 1000.0 / st.decode_ms) : 0.0;
        ss << "\"n_prompt\":"  << st.n_prompt   << ","
           << "\"n_decode\":"  << st.n_decode   << ","
           << "\"prompt_ms\":" << st.prompt_ms  << ","
           << "\"decode_ms\":" << st.decode_ms  << ","
           << "\"prompt_tps\":" << prompt_tps   << ","
           << "\"decode_tps\":" << decode_tps   << ","
           << "\"n_ctx\":"     << s->n_ctx      << ","
           << "\"threads\":"   << s->n_threads;
    } else {
        ss << "\"error\":\"no_handle\"";
    }
    ss << "}";
    return env->NewStringUTF(ss.str().c_str());
}

// -------------------------------------------------------------------------
// applyChatTemplate(long handle, String[] roles, String[] contents,
//                   boolean addAssistant) -> String
//
// Light wrapper around llama_chat_apply_template using the model's built-in
// template. Returns the formatted prompt as a String.
// -------------------------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_applyChatTemplate(
    JNIEnv * env, jclass /*clazz*/,
    jlong handle,
    jobjectArray jRoles, jobjectArray jContents, jboolean addAssistant)
{
    Session * s = reinterpret_cast<Session *>(handle);
    if (!s) return env->NewStringUTF("");
    if (!jRoles || !jContents) return env->NewStringUTF("");

    const jsize n_roles    = env->GetArrayLength(jRoles);
    const jsize n_contents = env->GetArrayLength(jContents);
    const jsize n = std::min(n_roles, n_contents);
    if (n <= 0) return env->NewStringUTF("");

    // Pull strings into stable storage.
    std::vector<std::string>          roles_s(n), contents_s(n);
    std::vector<llama_chat_message>   msgs(n);
    for (jsize i = 0; i < n; ++i) {
        jstring r = (jstring) env->GetObjectArrayElement(jRoles, i);
        jstring c = (jstring) env->GetObjectArrayElement(jContents, i);
        roles_s[i]    = jstring_to_std(env, r);
        contents_s[i] = jstring_to_std(env, c);
        msgs[i].role    = roles_s[i].c_str();
        msgs[i].content = contents_s[i].c_str();
        if (r) env->DeleteLocalRef(r);
        if (c) env->DeleteLocalRef(c);
    }

    const char * tmpl = llama_model_chat_template(s->model, /*name=*/nullptr);

    // Try once with a 4 KiB buffer; resize if needed.
    std::vector<char> buf(4096);
    int new_len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                             addAssistant != 0,
                                             buf.data(), (int32_t) buf.size());
    if (new_len > (int) buf.size()) {
        buf.resize(new_len);
        new_len = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                             addAssistant != 0,
                                             buf.data(), (int32_t) buf.size());
    }
    if (new_len < 0) {
        LOGE("applyChatTemplate: failed");
        return env->NewStringUTF("");
    }
    return env->NewStringUTF(std::string(buf.data(), new_len).c_str());
}

// -------------------------------------------------------------------------
// resetContext(long handle) -> void
//
// Clears the KV cache so a new conversation can start from scratch.
// -------------------------------------------------------------------------
JNIEXPORT void JNICALL
Java_com_yzamari_turboquant_assistant_LlamaNative_resetContext(
    JNIEnv * /*env*/, jclass /*clazz*/, jlong handle)
{
    Session * s = reinterpret_cast<Session *>(handle);
    if (!s) return;
    std::lock_guard<std::mutex> lk(s->mu);
    auto * mem = llama_get_memory(s->ctx);
    if (mem) {
        llama_memory_clear(mem, /*data=*/true);
    }
}

}  // extern "C"
