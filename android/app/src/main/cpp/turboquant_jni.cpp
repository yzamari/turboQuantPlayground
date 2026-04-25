// JNI bridge — the ONLY Android-specific C++ in this repo.
//
// The bench harness (bench_runner.hpp) and the FP32 baseline reference
// (baseline_kv_cache.cpp) live under cpp/bench/ and are pulled in by the
// app's CMakeLists.txt. The core libturboquant stays portable.

#include <jni.h>

#include "bench_runner.hpp"            // turboquant_bench::run_one + formatting
#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <android/log.h>

#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define LOG_TAG "turboquant_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace turboquant;
using turboquant_bench::Row;
using turboquant_bench::RunConfig;

namespace {

std::string jstring_to_std(JNIEnv* env, jstring js) {
    if (!js) return {};
    const char* c = env->GetStringUTFChars(js, nullptr);
    std::string s = c ? c : "";
    if (c) env->ReleaseStringUTFChars(js, c);
    return s;
}

// All backend kinds the core knows about. We probe each at runtime via
// create_backend(); only those that return non-null get exposed to Kotlin.
struct KnownBackend { BackendKind kind; const char* name; };
constexpr KnownBackend kKnownBackends[] = {
    { BackendKind::CpuScalar, "cpu_scalar" },
    { BackendKind::CpuNeon,   "cpu_neon"   },
    { BackendKind::OpenCL,    "opencl"     },
    { BackendKind::Vulkan,    "vulkan"     },
    { BackendKind::QnnHtp,    "qnn_htp"    },
};

}  // namespace

extern "C" {

JNIEXPORT jobjectArray JNICALL
Java_com_yzamari_turboquant_TurboQuantNative_listBackends(JNIEnv* env, jclass /*clazz*/) {
    std::vector<std::string> available;
    for (const auto& kb : kKnownBackends) {
        auto b = create_backend(kb.kind);
        if (b) available.emplace_back(kb.name);
    }
    jclass strCls = env->FindClass("java/lang/String");
    jobjectArray arr = env->NewObjectArray(
        static_cast<jsize>(available.size()), strCls, nullptr);
    for (size_t i = 0; i < available.size(); ++i) {
        jstring s = env->NewStringUTF(available[i].c_str());
        env->SetObjectArrayElement(arr, static_cast<jsize>(i), s);
        env->DeleteLocalRef(s);
    }
    return arr;
}

JNIEXPORT jstring JNICALL
Java_com_yzamari_turboquant_TurboQuantNative_runCheck(JNIEnv* env, jclass /*clazz*/,
                                                       jstring backendName) {
    const std::string name = jstring_to_std(env, backendName);
    auto kind    = backend_kind_from_name(name.c_str());
    auto backend = create_backend(kind);
    std::ostringstream out;
    if (!backend) {
        out << "Backend '" << name << "' not available in this build.\n";
        return env->NewStringUTF(out.str().c_str());
    }

    constexpr int D = 128;
    constexpr int N = 8;
    constexpr int bits = 3;

    auto Pi = generate_pi_qr(D, 42);
    auto S  = generate_qjl_S(D, 1042);
    TurboQuantProd prod(D, bits, backend.get(), Pi, S);

    std::vector<float> x(static_cast<size_t>(N) * D);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    auto q = prod.quantize(x.data(), N);
    std::vector<float> y(static_cast<size_t>(N) * D);
    prod.dequantize(q, y.data());

    double cos = turboquant_bench::cosine_sim(x.data(), y.data(), N * D);

    out << "Backend  : " << backend->name() << "\n"
        << "Version  : " << version_string() << "\n"
        << "D        : " << D << "\n"
        << "N        : " << N << "\n"
        << "bits     : " << bits << "\n"
        << "round-trip cosine = " << cos
        << "  (" << (cos > 0.85 ? "PASS" : "FAIL") << ", expect > 0.85)\n";
    return env->NewStringUTF(out.str().c_str());
}

JNIEXPORT jstring JNICALL
Java_com_yzamari_turboquant_TurboQuantNative_runBenchmark(JNIEnv* env, jclass /*clazz*/,
                                                           jstring backendName,
                                                           jintArray seqLensJ,
                                                           jint bits,
                                                           jint bh,
                                                           jint d) {
    const std::string name = jstring_to_std(env, backendName);
    auto kind    = backend_kind_from_name(name.c_str());
    auto backend = create_backend(kind);
    std::ostringstream out;
    if (!backend) {
        out << "Backend '" << name << "' not available in this build.\n";
        return env->NewStringUTF(out.str().c_str());
    }

    // Marshal the seq_lens int[].
    std::vector<int> seq_lens;
    if (seqLensJ) {
        jsize n = env->GetArrayLength(seqLensJ);
        seq_lens.resize(static_cast<size_t>(n));
        env->GetIntArrayRegion(seqLensJ, 0, n, seq_lens.data());
    }
    if (seq_lens.empty()) seq_lens = {128, 512, 2048};

    RunConfig cfg;
    cfg.backend = name;
    cfg.bits    = bits;
    cfg.bh      = bh;
    cfg.d       = d;
    cfg.n_q     = 1;
    cfg.warmup  = 1;
    cfg.iters   = 3;          // mobile is slow; keep iter count modest
    cfg.baseline_dtype = "fp16";

    out << "Backend  : " << backend->name()
        << "    " << version_string() << "\n"
        << "BH=" << bh << "  D=" << d << "  bits=" << bits
        << "  iters=" << cfg.iters << "\n\n";
    out << turboquant_bench::format_header();

    // We compile with -fno-exceptions, so failures inside run_one terminate.
    // Any guard would have to be at the API level (return-codes); for the
    // demo path, all sizes here are small and well-tested.
    for (int s : seq_lens) {
        Row r = turboquant_bench::run_one(cfg, backend.get(), s);
        out << turboquant_bench::format_row(r);
    }

    return env->NewStringUTF(out.str().c_str());
}

}  // extern "C"
