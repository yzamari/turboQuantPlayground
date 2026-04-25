#include "baseline_kv_cache.hpp"

#include <cstring>

namespace turboquant_bench {

void BaselineKVCache::attention_scores(const float* query, int BH, int n_q,
                                       float* out_scores, float scale) const {
    const int D = D_;
    const int S = seq_len_;
    for (int b = 0; b < BH; ++b) {
        const float* k_bh = keys_.data() + static_cast<size_t>(b) * S * D;
        for (int t = 0; t < n_q; ++t) {
            const float* qv = query + (static_cast<size_t>(b) * n_q + t) * D;
            float* dst = out_scores + (static_cast<size_t>(b) * n_q + t) * S;
            for (int j = 0; j < S; ++j) {
                const float* kv = k_bh + static_cast<size_t>(j) * D;
                double s = 0.0;
                for (int k = 0; k < D; ++k) {
                    s += static_cast<double>(qv[k]) * kv[k];
                }
                dst[j] = static_cast<float>(s) * scale;
            }
        }
    }
}

void BaselineKVCache::attend(const float* weights, int BH, int n_q,
                             float* out) const {
    const int D = D_;
    const int S = seq_len_;
    std::memset(out, 0, static_cast<size_t>(BH) * n_q * D * sizeof(float));
    for (int b = 0; b < BH; ++b) {
        const float* v_bh = values_.data() + static_cast<size_t>(b) * S * D;
        for (int t = 0; t < n_q; ++t) {
            const float* w = weights + (static_cast<size_t>(b) * n_q + t) * S;
            float* o       = out     + (static_cast<size_t>(b) * n_q + t) * D;
            for (int j = 0; j < S; ++j) {
                const float* vj = v_bh + static_cast<size_t>(j) * D;
                float wj = w[j];
                for (int k = 0; k < D; ++k) o[k] += wj * vj[k];
            }
        }
    }
}

}  // namespace turboquant_bench
