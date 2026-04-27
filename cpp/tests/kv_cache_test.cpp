// Buffer-portion parity test for TurboQuantKVCache::attention_scores
// and ::attend. Locks in the numerical contract of the (currently
// scalar) inner loop before the NEON-vectorized version lands in F2.
//
// Approach: prefill with seq_len <= buffer_size so n_quant_ stays 0
// and the whole computation goes through the unquantized buffer path
// — exactly the code paths the NEON intrinsics will replace.

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"
#include "tq_test.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace turboquant;

namespace {

void fill_uniform(std::vector<float>& v, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (auto& x : v) x = u(rng);
}

void reference_scores(
    const float* q, const float* k,
    int BH, int n_q, int n_buf, int D,
    float scale, std::vector<float>& out) {
    out.assign(static_cast<size_t>(BH) * n_q * n_buf, 0.f);
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            const float* qv = q + ((static_cast<size_t>(b) * n_q + t) * D);
            for (int j = 0; j < n_buf; ++j) {
                const float* kv = k + ((static_cast<size_t>(b) * n_buf + j) * D);
                double s = 0.0;
                for (int d = 0; d < D; ++d) s += static_cast<double>(qv[d]) * kv[d];
                out[(static_cast<size_t>(b) * n_q + t) * n_buf + j] =
                    static_cast<float>(s) * scale;
            }
        }
    }
}

void reference_attend(
    const float* w, const float* v,
    int BH, int n_q, int n_buf, int D,
    std::vector<float>& out) {
    out.assign(static_cast<size_t>(BH) * n_q * D, 0.f);
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            const float* wv = w + ((static_cast<size_t>(b) * n_q + t) * n_buf);
            float* o = out.data() + ((static_cast<size_t>(b) * n_q + t) * D);
            for (int j = 0; j < n_buf; ++j) {
                const float* vv = v + ((static_cast<size_t>(b) * n_buf + j) * D);
                float wj = wv[j];
                for (int d = 0; d < D; ++d) o[d] += wj * vv[d];
            }
        }
    }
}

bool run_buffer_parity(int D, int BH, int n_buf, int n_q) {
    std::printf("[kv_cache] buffer-only D=%d BH=%d n_buf=%d n_q=%d\n",
                D, BH, n_buf, n_q);

    TurboQuantKVCache::Config cfg;
    cfg.head_dim         = D;
    cfg.key_bits         = 4;
    cfg.value_bits       = 2;
    cfg.value_group_size = 32;
    // buffer_size >= n_buf so prefill keeps everything in the buffer (n_quant_ = 0).
    cfg.buffer_size      = std::max(64, n_buf);

    auto backend = create_backend(BackendKind::CpuScalar);
    TQ_CHECK(backend != nullptr);
    if (!backend) return false;

    auto Pi = generate_pi_qr(D, 42);
    auto S  = generate_qjl_S(D, 1042);

    TurboQuantKVCache kv(cfg, backend.get(), std::move(Pi), std::move(S));

    std::vector<float> keys  (static_cast<size_t>(BH) * n_buf * D);
    std::vector<float> values(static_cast<size_t>(BH) * n_buf * D);
    fill_uniform(keys,   0xa17e7170ULL);
    fill_uniform(values, 0xb1ce5e7dULL);
    kv.prefill(keys.data(), values.data(), BH, n_buf);
    TQ_CHECK_EQ(kv.seq_len(), n_buf);

    std::vector<float> q(static_cast<size_t>(BH) * n_q * D);
    fill_uniform(q, 0xc09d7011ULL);

    const float scale = 1.f / std::sqrt(static_cast<float>(D));

    // ----- attention_scores -----
    std::vector<float> scores(static_cast<size_t>(BH) * n_q * n_buf);
    kv.attention_scores(q.data(), BH, n_q, scores.data(), scale);

    std::vector<float> ref_scores;
    reference_scores(q.data(), keys.data(), BH, n_q, n_buf, D, scale, ref_scores);

    double max_abs = 0.0;
    for (size_t i = 0; i < scores.size(); ++i) {
        double diff = std::fabs(double(scores[i]) - double(ref_scores[i]));
        if (diff > max_abs) max_abs = diff;
    }
    std::fprintf(stderr, "  scores max-abs-err = %.3e\n", max_abs);
    for (size_t i = 0; i < scores.size(); ++i) {
        TQ_CHECK_NEAR(scores[i], ref_scores[i], 1e-4);
    }

    // ----- attend -----
    // Synthesize softmax-like weights (per row >= 0, sums to 1).
    std::vector<float> w(static_cast<size_t>(BH) * n_q * n_buf);
    fill_uniform(w, 0xd0aded71ULL);
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            float* row = w.data() + ((static_cast<size_t>(b) * n_q + t) * n_buf);
            float sum = 0.f;
            for (int j = 0; j < n_buf; ++j) { row[j] = std::fabs(row[j]); sum += row[j]; }
            float inv = 1.f / std::max(sum, 1e-30f);
            for (int j = 0; j < n_buf; ++j) row[j] *= inv;
        }
    }

    std::vector<float> out(static_cast<size_t>(BH) * n_q * D);
    kv.attend(w.data(), BH, n_q, out.data());

    std::vector<float> ref_out;
    reference_attend(w.data(), values.data(), BH, n_q, n_buf, D, ref_out);

    double max_abs_out = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        double diff = std::fabs(double(out[i]) - double(ref_out[i]));
        if (diff > max_abs_out) max_abs_out = diff;
    }
    std::fprintf(stderr, "  attend max-abs-err = %.3e\n", max_abs_out);
    for (size_t i = 0; i < out.size(); ++i) {
        TQ_CHECK_NEAR(out[i], ref_out[i], 1e-4);
    }

    return true;
}

}  // namespace

double cosine_similarity_f32(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Stress append() past buffer_size and check that:
//   (1) memory_bytes() stays bounded (proves flush is moving tokens out
//       of the unquantized buffer into the compressed portion);
//   (2) attention_scores still tracks an FP32 reference well enough
//       (cosine >= 0.85 against full-precision Q@K^T over all tokens).
//
// With the legacy grow-only append() this test fails on (1): memory grows
// linearly with `total_tokens`, blowing past half of the unquantized size.
bool run_flush_overflow(int D, int BH, int buffer_size, int total_tokens, int n_q) {
    std::printf("[kv_cache] flush-overflow D=%d BH=%d buf=%d total=%d n_q=%d\n",
                D, BH, buffer_size, total_tokens, n_q);

    TurboQuantKVCache::Config cfg;
    cfg.head_dim         = D;
    cfg.key_bits         = 4;
    cfg.value_bits       = 2;
    cfg.value_group_size = 32;
    cfg.buffer_size      = buffer_size;

    auto backend = create_backend(BackendKind::CpuScalar);
    TQ_CHECK(backend != nullptr);
    if (!backend) return false;

    auto Pi = generate_pi_qr(D, 42);
    auto S  = generate_qjl_S(D, 1042);

    TurboQuantKVCache kv(cfg, backend.get(), std::move(Pi), std::move(S));

    // Ground-truth full sequence per BH: [BH, total_tokens, D].
    std::vector<float> all_k(static_cast<size_t>(BH) * total_tokens * D);
    std::vector<float> all_v(static_cast<size_t>(BH) * total_tokens * D);
    fill_uniform(all_k, 0xa5a5a5a5ULL);
    fill_uniform(all_v, 0x5a5a5a5aULL);

    auto extract_step = [&](int t, std::vector<float>& kt, std::vector<float>& vt) {
        kt.resize(static_cast<size_t>(BH) * D);
        vt.resize(static_cast<size_t>(BH) * D);
        for (int b = 0; b < BH; ++b) {
            std::memcpy(kt.data() + static_cast<size_t>(b) * D,
                        all_k.data() + (static_cast<size_t>(b) * total_tokens + t) * D,
                        static_cast<size_t>(D) * sizeof(float));
            std::memcpy(vt.data() + static_cast<size_t>(b) * D,
                        all_v.data() + (static_cast<size_t>(b) * total_tokens + t) * D,
                        static_cast<size_t>(D) * sizeof(float));
        }
    };

    std::vector<float> kt, vt;
    extract_step(0, kt, vt);
    kv.prefill(kt.data(), vt.data(), BH, 1);
    TQ_CHECK_EQ(kv.seq_len(), 1);

    for (int t = 1; t < total_tokens; ++t) {
        extract_step(t, kt, vt);
        kv.append(kt.data(), vt.data(), BH);
    }
    TQ_CHECK_EQ(kv.seq_len(), total_tokens);

    // Memory bound: must be strictly less than HALF of the unquantized
    // (key + value) FP32 footprint at this seq_len. The legacy grow-only
    // append() hits ~the full unquantized size so this comfortably bites;
    // the post-flush footprint is buffer_bytes + compressed_quantized,
    // which scales with token count by ~36 bytes/vector (D=64, 4-bit MSE
    // + 1-bit QJL + 2-bit value), well under the ~512 bytes/vector that
    // FP32 K+V costs.
    const size_t unq_bytes = 2 * static_cast<size_t>(BH) * total_tokens * D * sizeof(float);
    const size_t bound     = unq_bytes / 2;
    std::printf("  memory_bytes()=%zu vs bound=%zu (unq=%zu)\n",
                kv.memory_bytes(), bound, unq_bytes);
    TQ_CHECK(kv.memory_bytes() < bound);

    // Attention score quality vs FP32 reference computed over the full
    // ground-truth sequence (no quantization). Cosine threshold matches
    // the bench's 0.85 acceptance band.
    std::vector<float> q(static_cast<size_t>(BH) * n_q * D);
    fill_uniform(q, 0xbeefcafeULL);
    const float scale = 1.f / std::sqrt(static_cast<float>(D));

    std::vector<float> scores(static_cast<size_t>(BH) * n_q * total_tokens);
    kv.attention_scores(q.data(), BH, n_q, scores.data(), scale);

    std::vector<float> ref;
    reference_scores(q.data(), all_k.data(), BH, n_q, total_tokens, D, scale, ref);

    double min_cos = 1.0;
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            const float* a = scores.data() + (static_cast<size_t>(b) * n_q + t) * total_tokens;
            const float* c = ref   .data() + (static_cast<size_t>(b) * n_q + t) * total_tokens;
            double cos = cosine_similarity_f32(a, c, static_cast<size_t>(total_tokens));
            if (cos < min_cos) min_cos = cos;
        }
    }
    std::printf("  min cosine across (BH,n_q) rows = %.4f\n", min_cos);
    TQ_CHECK(min_cos >= 0.85);
    return true;
}

int main() {
    run_buffer_parity(/*D=*/64,  /*BH=*/1, /*n_buf=*/16, /*n_q=*/1);
    run_buffer_parity(/*D=*/128, /*BH=*/8, /*n_buf=*/64, /*n_q=*/4);

    // Flush-on-overflow: small buffer (8), large total (128) so n_quant_
    // grows to 120 — well past anything the legacy grow-strategy could
    // have handled inside the memory bound.
    run_flush_overflow(/*D=*/64, /*BH=*/4, /*buffer_size=*/8,
                       /*total_tokens=*/128, /*n_q=*/4);
    return tq_test::report_and_exit();
}
