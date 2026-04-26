// attention_turboquant — stateless single-call attention bridge (Path 2.1c
// Step 1). Composes the existing TurboQuantKVCache + softmax to produce a
// drop-in replacement for the standard Q@K.T → soft_max_ext → attn@V triple.
//
// Why a fresh throwaway cache per call: our llama.cpp fork's substituted
// attention op runs at graph-execution time with K/V tensors that already
// hold the full attention window (post-RoPE, post-cache-write). The cache
// class is the cleanest way to reuse all the algorithm code (PolarQuant
// rotate + groupwise quantize, MSE + 1-bit QJL scoring, weighted V dequant)
// without re-implementing those primitives here.

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace turboquant {

void attention_turboquant(
    const float* q, const float* k, const float* v,
    int BH, int n_q, int n_kv, int D,
    float scale, const float* mask, float* out,
    int key_bits, int value_bits, uint64_t seed) {

    if (BH <= 0 || n_q <= 0 || n_kv <= 0 || D <= 0) {
        throw std::invalid_argument("attention_turboquant: non-positive dim");
    }
    if (!q || !k || !v || !out) {
        throw std::invalid_argument("attention_turboquant: null pointer");
    }

    if (scale <= 0.f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    // Pi (rotation) + S (QJL sketch). Same seeds the Python reference uses,
    // offset by +1000 between Pi and S so they stay independent.
    auto Pi = generate_pi_qr (D, seed);
    auto S  = generate_qjl_S (D, seed + 1000);

    TurboQuantKVCache::Config cfg;
    cfg.head_dim          = D;
    cfg.key_bits          = key_bits;
    cfg.value_bits        = value_bits;
    cfg.value_group_size  = 32;
    // Force everything into the quantized branch — no recent-buffer
    // unquantized fast path. The whole point is to measure / use the
    // compressed representation end-to-end.
    cfg.buffer_size       = 0;
    cfg.layer_idx         = 0;

    // Backend choice: best available at runtime. Falls back to cpu_scalar.
    auto backend = create_best_backend();
    if (!backend) {
        throw std::runtime_error("attention_turboquant: no backend available");
    }

    TurboQuantKVCache cache(cfg, backend.get(), std::move(Pi), std::move(S));
    cache.prefill(k, v, BH, n_kv);

    // Scores: [BH, n_q, n_kv].
    std::vector<float> scores(static_cast<size_t>(BH) * n_q * n_kv);
    cache.attention_scores(q, BH, n_q, scores.data(), scale);

    // Mask + softmax in place, row-by-row over the n_kv axis.
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int b = 0; b < BH; ++b) {
        for (int t = 0; t < n_q; ++t) {
            float* row = &scores[(static_cast<size_t>(b) * n_q + t) * n_kv];

            if (mask) {
                const float* m = &mask[static_cast<size_t>(t) * n_kv];
                for (int j = 0; j < n_kv; ++j) row[j] += m[j];
            }

            // Numerically stable softmax: subtract max, exp, normalize.
            float row_max = neg_inf;
            for (int j = 0; j < n_kv; ++j) {
                if (row[j] > row_max) row_max = row[j];
            }
            if (row_max == neg_inf) {
                // All masked → uniform-zero output for this row.
                for (int j = 0; j < n_kv; ++j) row[j] = 0.f;
                continue;
            }

            float sum = 0.f;
            for (int j = 0; j < n_kv; ++j) {
                row[j] = std::exp(row[j] - row_max);
                sum   += row[j];
            }
            const float inv = 1.f / std::max(sum, 1e-30f);
            for (int j = 0; j < n_kv; ++j) row[j] *= inv;
        }
    }

    // attend produces [BH * n_q * D] from weights [BH * n_q * n_kv].
    cache.attend(scores.data(), BH, n_q, out);
}

}  // namespace turboquant
