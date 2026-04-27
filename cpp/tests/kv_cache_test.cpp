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

int main() {
    run_buffer_parity(/*D=*/64,  /*BH=*/1, /*n_buf=*/16, /*n_q=*/1);
    run_buffer_parity(/*D=*/128, /*BH=*/8, /*n_buf=*/64, /*n_q=*/4);
    return tq_test::report_and_exit();
}
