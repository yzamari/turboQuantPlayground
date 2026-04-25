// Reusable bench harness — shared by the CLI (bench_cli.cpp) and the Android
// JNI shim (turboquant_jni.cpp). Lives under cpp/bench/ so it stays out of
// the core library; the JNI translation unit just `#include`s this header.
//
// All functions are header-inline (no separate .cpp) so the JNI build doesn't
// need to depend on the bench static library.

#pragma once

#include "baseline_kv_cache.hpp"

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace turboquant_bench {

struct RunConfig {
    std::string backend = "cpu_scalar";
    int  bits    = 3;
    int  bh      = 8;
    int  d       = 128;
    int  n_q     = 1;
    int  warmup  = 1;
    int  iters   = 5;
    std::string baseline_dtype = "fp16";  // affects reported memory only
};

struct Row {
    std::string device;
    std::string backend;
    int seq_len;
    int bh;
    int d;
    int bits;

    double baseline_attn_ms;
    double tq_attn_ms;
    double attn_speedup;

    std::size_t baseline_mem_bytes;
    std::size_t tq_mem_bytes;
    double compression_ratio;

    double encode_ms;

    double attn_score_cosine_sim;
    double attn_output_rel_l2;
};

inline double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na * nb) + 1e-12);
}

inline double rel_l2(const float* tq, const float* ref, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        double d = static_cast<double>(tq[i]) - ref[i];
        num += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / (den + 1e-12));
}

inline void softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
    float inv = static_cast<float>(1.0 / (sum + 1e-12));
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

inline Row run_one(const RunConfig& a, turboquant::IBackend* backend, int seq_len) {
    using clk = std::chrono::steady_clock;
    using namespace turboquant;

    Row row;
    row.device  = "android";
    row.backend = a.backend;
    row.seq_len = seq_len;
    row.bh      = a.bh;
    row.d       = a.d;
    row.bits    = a.bits;

    const int BH  = a.bh;
    const int D   = a.d;
    const int n_q = a.n_q;

    // Prepare data
    std::mt19937 rng(0xA5A5);
    std::normal_distribution<float> nd(0.f, 1.f);
    const std::size_t n_kv  = static_cast<std::size_t>(BH) * seq_len * D;
    const std::size_t n_q_total = static_cast<std::size_t>(BH) * n_q * D;
    std::vector<float> keys (n_kv);
    std::vector<float> values(n_kv);
    std::vector<float> query(n_q_total);
    for (auto& v : keys ) v = nd(rng);
    for (auto& v : values) v = nd(rng);
    for (auto& v : query) v = nd(rng);

    // Baseline
    BaselineKVCache base(D);
    base.prefill(keys.data(), values.data(), BH, seq_len);
    std::vector<float> base_scores (static_cast<std::size_t>(BH) * n_q * seq_len);
    std::vector<float> base_weights(static_cast<std::size_t>(BH) * n_q * seq_len);
    std::vector<float> base_out    (static_cast<std::size_t>(BH) * n_q * D);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (int w = 0; w < a.warmup; ++w) {
        base.attention_scores(query.data(), BH, n_q, base_scores.data(), scale);
    }
    auto t0 = clk::now();
    for (int i = 0; i < a.iters; ++i) {
        base.attention_scores(query.data(), BH, n_q, base_scores.data(), scale);
    }
    auto t1 = clk::now();
    row.baseline_attn_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / a.iters;

    base_weights = base_scores;
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            float* row_ = base_weights.data() + (b * n_q + t) * seq_len;
            softmax_inplace(row_, seq_len);
        }
    }
    base.attend(base_weights.data(), BH, n_q, base_out.data());

    row.baseline_mem_bytes =
        (a.baseline_dtype == "fp32")
            ? base.memory_bytes()
            : base.memory_bytes_fp16();

    // TurboQuant
    auto Pi = generate_pi_qr(D, /*seed=*/42);
    auto S  = generate_qjl_S(D, /*seed=*/1042);
    TurboQuantKVCache::Config cfg;
    cfg.head_dim         = D;
    cfg.key_bits         = a.bits;
    cfg.value_bits       = 2;
    cfg.value_group_size = 32;
    cfg.buffer_size      = 0;
    cfg.layer_idx        = 0;
    TurboQuantKVCache tq(cfg, backend, Pi, S);

    auto te0 = clk::now();
    tq.prefill(keys.data(), values.data(), BH, seq_len);
    auto te1 = clk::now();
    row.encode_ms = std::chrono::duration<double, std::milli>(te1 - te0).count();

    std::vector<float> tq_scores (static_cast<std::size_t>(BH) * n_q * seq_len);
    std::vector<float> tq_weights(static_cast<std::size_t>(BH) * n_q * seq_len);
    std::vector<float> tq_out    (static_cast<std::size_t>(BH) * n_q * D);
    for (int w = 0; w < a.warmup; ++w) {
        tq.attention_scores(query.data(), BH, n_q, tq_scores.data(), scale);
    }
    auto u0 = clk::now();
    for (int i = 0; i < a.iters; ++i) {
        tq.attention_scores(query.data(), BH, n_q, tq_scores.data(), scale);
    }
    auto u1 = clk::now();
    row.tq_attn_ms =
        std::chrono::duration<double, std::milli>(u1 - u0).count() / a.iters;

    tq_weights = tq_scores;
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            float* row_ = tq_weights.data() + (b * n_q + t) * seq_len;
            softmax_inplace(row_, seq_len);
        }
    }
    tq.attend(tq_weights.data(), BH, n_q, tq_out.data());

    row.tq_mem_bytes      = tq.memory_bytes();
    row.attn_speedup      = (row.tq_attn_ms > 0)
                          ? row.baseline_attn_ms / row.tq_attn_ms
                          : 0.0;
    row.compression_ratio = (row.tq_mem_bytes > 0)
                          ? static_cast<double>(row.baseline_mem_bytes) / row.tq_mem_bytes
                          : 0.0;
    row.attn_score_cosine_sim =
        cosine_sim(tq_weights.data(), base_weights.data(),
                   static_cast<int>(tq_weights.size()));
    row.attn_output_rel_l2 =
        rel_l2(tq_out.data(), base_out.data(),
               static_cast<int>(tq_out.size()));
    return row;
}

inline std::string format_header() {
    return "seq_len  baseline    tq         speedup  fp16_mem    tq_mem      comp     enc_ms   cos    relL2\n"
           "-------  ---------- ---------- -------  ----------- ----------- -------- -------- ------ ------\n";
}

inline std::string format_row(const Row& r) {
    auto mb = [](std::size_t b) { return b / (1024.0 * 1024.0); };
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "%6d   %7.3f ms %7.3f ms %5.2fx  %7.1f MB %7.1f MB %5.2fx  %7.2f  %.3f  %.3f\n",
                  r.seq_len,
                  r.baseline_attn_ms,
                  r.tq_attn_ms,
                  r.attn_speedup,
                  mb(r.baseline_mem_bytes),
                  mb(r.tq_mem_bytes),
                  r.compression_ratio,
                  r.encode_ms,
                  r.attn_score_cosine_sim,
                  r.attn_output_rel_l2);
    return std::string(buf);
}

}  // namespace turboquant_bench
