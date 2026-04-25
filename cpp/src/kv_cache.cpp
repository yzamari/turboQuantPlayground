#include "turboquant/api.hpp"

#include "packing.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace turboquant {

// -----------------------------------------------------------------------------
// Per-group asymmetric value quantization (matches Python kv_cache.quantize_values).
// -----------------------------------------------------------------------------
ValueQuantized quantize_values(const float* v, int n_vec, int d,
                               int bits, int group_size) {
    if (d % group_size != 0) {
        throw std::invalid_argument("quantize_values: d must be divisible by group_size");
    }
    if (bits != 2 && bits != 4 && bits != 8) {
        throw std::invalid_argument("quantize_values: bits must be 2, 4, or 8");
    }

    ValueQuantized q;
    q.n_vec       = n_vec;
    q.d           = d;
    q.bits        = bits;
    q.group_size  = group_size;
    q.n_groups    = d / group_size;
    q.packed_d    = (bits == 2) ? d / 4 : (bits == 4 ? d / 2 : d);
    q.data .assign(static_cast<size_t>(n_vec) * q.packed_d, 0);
    q.scales.assign(static_cast<size_t>(n_vec) * q.n_groups, 0.f);
    q.zeros .assign(static_cast<size_t>(n_vec) * q.n_groups, 0.f);

    const int n_levels = (1 << bits) - 1;

    for (int vec = 0; vec < n_vec; ++vec) {
        const float* in = v + static_cast<size_t>(vec) * d;
        uint8_t*     out = q.data.data() + static_cast<size_t>(vec) * q.packed_d;
        float*       sc  = q.scales.data() + static_cast<size_t>(vec) * q.n_groups;
        float*       ze  = q.zeros .data() + static_cast<size_t>(vec) * q.n_groups;

        // Encode each coord into a temporary uint8 buffer, then bit-pack.
        std::vector<uint8_t> tmp(d);
        for (int g = 0; g < q.n_groups; ++g) {
            const float* gv = in + g * group_size;
            float vmin = gv[0], vmax = gv[0];
            for (int j = 1; j < group_size; ++j) {
                vmin = std::min(vmin, gv[j]);
                vmax = std::max(vmax, gv[j]);
            }
            float scale = (vmax - vmin) / static_cast<float>(n_levels);
            if (scale < 1e-10f) scale = 1e-10f;
            float zero  = vmin;
            sc[g] = scale;
            ze[g] = zero;

            float inv = 1.0f / scale;
            for (int j = 0; j < group_size; ++j) {
                float qv = std::round((gv[j] - zero) * inv);
                if (qv < 0.f)                              qv = 0.f;
                if (qv > static_cast<float>(n_levels))     qv = static_cast<float>(n_levels);
                tmp[g * group_size + j] = static_cast<uint8_t>(qv);
            }
        }

        // Bit-pack
        if (bits == 8) {
            std::memcpy(out, tmp.data(), d);
        } else if (bits == 4) {
            for (int j = 0; j < d; j += 2) {
                out[j >> 1] = static_cast<uint8_t>(tmp[j] | (tmp[j + 1] << 4));
            }
        } else /* bits == 2 */ {
            for (int j = 0; j < d; j += 4) {
                out[j >> 2] = static_cast<uint8_t>(
                    tmp[j] | (tmp[j + 1] << 2) | (tmp[j + 2] << 4) | (tmp[j + 3] << 6));
            }
        }
    }
    return q;
}

void dequantize_values(const ValueQuantized& vq, float* out) {
    const int d           = vq.d;
    const int group_size  = vq.group_size;
    const int n_groups    = vq.n_groups;
    const int packed_d    = vq.packed_d;
    const int bits        = vq.bits;

    for (int v = 0; v < vq.n_vec; ++v) {
        const uint8_t* in = vq.data.data()   + static_cast<size_t>(v) * packed_d;
        const float*   sc = vq.scales.data() + static_cast<size_t>(v) * n_groups;
        const float*   ze = vq.zeros.data()  + static_cast<size_t>(v) * n_groups;
        float*         o  = out              + static_cast<size_t>(v) * d;

        for (int j = 0; j < d; ++j) {
            int qv;
            if (bits == 8) {
                qv = in[j];
            } else if (bits == 4) {
                qv = (in[j >> 1] >> ((j & 1) * 4)) & 0xF;
            } else /* bits == 2 */ {
                qv = (in[j >> 2] >> ((j & 3) * 2)) & 0x3;
            }
            int g = j / group_size;
            o[j] = static_cast<float>(qv) * sc[g] + ze[g];
        }
    }
}

// -----------------------------------------------------------------------------
// TurboQuantKVCache
// -----------------------------------------------------------------------------
TurboQuantKVCache::TurboQuantKVCache(const Config& cfg, IBackend* backend,
                                     std::vector<float> Pi, std::vector<float> S)
    : cfg_(cfg), backend_(backend),
      key_quantizer_(cfg.head_dim, cfg.key_bits, backend, std::move(Pi), std::move(S)) {}

void TurboQuantKVCache::prefill(const float* keys, const float* values,
                                int BH, int seq_len) {
    BH_      = BH;
    seq_len_ = seq_len;

    if (seq_len <= cfg_.buffer_size) {
        size_t n = static_cast<size_t>(BH) * seq_len * cfg_.head_dim;
        key_buf_  .assign(keys,   keys   + n);
        value_buf_.assign(values, values + n);
        n_buf_   = seq_len;
        n_quant_ = 0;
        return;
    }

    n_quant_ = seq_len - cfg_.buffer_size;
    n_buf_   = cfg_.buffer_size;

    // Quantize the oldest n_quant tokens (per BH).
    std::vector<float> keys_to_q  (static_cast<size_t>(BH) * n_quant_ * cfg_.head_dim);
    std::vector<float> values_to_q(static_cast<size_t>(BH) * n_quant_ * cfg_.head_dim);
    for (int b = 0; b < BH; ++b) {
        std::memcpy(keys_to_q.data()   + static_cast<size_t>(b) * n_quant_ * cfg_.head_dim,
                    keys   + static_cast<size_t>(b) * seq_len * cfg_.head_dim,
                    static_cast<size_t>(n_quant_) * cfg_.head_dim * sizeof(float));
        std::memcpy(values_to_q.data() + static_cast<size_t>(b) * n_quant_ * cfg_.head_dim,
                    values + static_cast<size_t>(b) * seq_len * cfg_.head_dim,
                    static_cast<size_t>(n_quant_) * cfg_.head_dim * sizeof(float));
    }
    key_q_   = key_quantizer_.quantize(keys_to_q.data(), BH * n_quant_);
    value_q_ = quantize_values(values_to_q.data(), BH * n_quant_,
                               cfg_.head_dim, cfg_.value_bits, cfg_.value_group_size);

    // Buffer holds the most recent buffer_size tokens.
    key_buf_  .assign(static_cast<size_t>(BH) * n_buf_ * cfg_.head_dim, 0.f);
    value_buf_.assign(static_cast<size_t>(BH) * n_buf_ * cfg_.head_dim, 0.f);
    for (int b = 0; b < BH; ++b) {
        std::memcpy(key_buf_.data()   + static_cast<size_t>(b) * n_buf_ * cfg_.head_dim,
                    keys   + (static_cast<size_t>(b) * seq_len + n_quant_) * cfg_.head_dim,
                    static_cast<size_t>(n_buf_) * cfg_.head_dim * sizeof(float));
        std::memcpy(value_buf_.data() + static_cast<size_t>(b) * n_buf_ * cfg_.head_dim,
                    values + (static_cast<size_t>(b) * seq_len + n_quant_) * cfg_.head_dim,
                    static_cast<size_t>(n_buf_) * cfg_.head_dim * sizeof(float));
    }
}

void TurboQuantKVCache::append(const float* /*key*/, const float* /*value*/, int /*BH*/) {
    // Decode-loop append + flush is implemented in P0.4 follow-up; the bench
    // exercises prefill-only timing for now.
    throw std::logic_error("TurboQuantKVCache::append not yet implemented (post-P0)");
}

void TurboQuantKVCache::flush_buffer_() {
    throw std::logic_error("TurboQuantKVCache::flush_buffer_ not yet implemented (post-P0)");
}

void TurboQuantKVCache::attention_scores(const float* query, int BH, int n_q,
                                         float* out_scores, float scale) const {
    if (BH != BH_) throw std::invalid_argument("attention_scores: BH mismatch");
    if (scale == 0.f) scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    const int S = seq_len_;

    // Quantized portion: out[bh, t, j] for j in [0, n_quant)
    if (n_quant_ > 0) {
        std::vector<float> tmp(static_cast<size_t>(BH) * n_q * n_quant_);
        key_quantizer_.attention_score(query, BH, n_q, key_q_, n_quant_, tmp.data());
        for (int b = 0; b < BH; ++b) {
            for (int t = 0; t < n_q; ++t) {
                float* dst = out_scores + ((static_cast<size_t>(b) * n_q + t) * S);
                const float* src = tmp.data() + ((static_cast<size_t>(b) * n_q + t) * n_quant_);
                for (int j = 0; j < n_quant_; ++j) dst[j] = src[j] * scale;
            }
        }
    }

    // Buffer portion: out[bh, t, n_quant + j] = q[bh,t] . key_buf[bh, j]
    if (n_buf_ > 0) {
        for (int b = 0; b < BH; ++b) {
            const float* q_bh = query + static_cast<size_t>(b) * n_q * cfg_.head_dim;
            const float* k_bh = key_buf_.data() + static_cast<size_t>(b) * n_buf_ * cfg_.head_dim;
            for (int t = 0; t < n_q; ++t) {
                const float* qv = q_bh + static_cast<size_t>(t) * cfg_.head_dim;
                float* dst = out_scores + ((static_cast<size_t>(b) * n_q + t) * S) + n_quant_;
                for (int j = 0; j < n_buf_; ++j) {
                    const float* kv = k_bh + static_cast<size_t>(j) * cfg_.head_dim;
                    double s = 0.0;
                    for (int k = 0; k < cfg_.head_dim; ++k) {
                        s += static_cast<double>(qv[k]) * kv[k];
                    }
                    dst[j] = static_cast<float>(s) * scale;
                }
            }
        }
    }
}

void TurboQuantKVCache::attend(const float* weights, int BH, int n_q,
                               float* out) const {
    if (BH != BH_) throw std::invalid_argument("attend: BH mismatch");
    const int D = cfg_.head_dim;
    const int S = seq_len_;
    std::memset(out, 0, static_cast<size_t>(BH) * n_q * D * sizeof(float));

    // Quantized values: dequantize per-BH (n_quant tokens), then per (t)
    // accumulate weights * dequantized.
    if (n_quant_ > 0) {
        std::vector<float> v_dq(static_cast<size_t>(BH) * n_quant_ * D);
        dequantize_values(value_q_, v_dq.data());
        for (int b = 0; b < BH; ++b) {
            for (int t = 0; t < n_q; ++t) {
                const float* w = weights + ((static_cast<size_t>(b) * n_q + t) * S);
                float*       o = out     + ((static_cast<size_t>(b) * n_q + t) * D);
                for (int j = 0; j < n_quant_; ++j) {
                    const float* vj = v_dq.data() + ((static_cast<size_t>(b) * n_quant_ + j) * D);
                    float wj = w[j];
                    for (int k = 0; k < D; ++k) o[k] += wj * vj[k];
                }
            }
        }
    }

    // Buffer values
    if (n_buf_ > 0) {
        for (int b = 0; b < BH; ++b) {
            for (int t = 0; t < n_q; ++t) {
                const float* w = weights + ((static_cast<size_t>(b) * n_q + t) * S) + n_quant_;
                float*       o = out     + ((static_cast<size_t>(b) * n_q + t) * D);
                for (int j = 0; j < n_buf_; ++j) {
                    const float* vj = value_buf_.data()
                                    + ((static_cast<size_t>(b) * n_buf_ + j) * D);
                    float wj = w[j];
                    for (int k = 0; k < D; ++k) o[k] += wj * vj[k];
                }
            }
        }
    }
}

size_t TurboQuantKVCache::memory_bytes() const {
    size_t total = 0;
    total += key_q_.mse_indices   .size();
    total += key_q_.qjl_signs     .size();
    total += key_q_.residual_norms.size() * sizeof(float);
    total += key_q_.norms         .size() * sizeof(float);
    total += value_q_.data        .size();
    total += value_q_.scales      .size() * sizeof(float);
    total += value_q_.zeros       .size() * sizeof(float);
    total += key_buf_             .size() * sizeof(float);
    total += value_buf_           .size() * sizeof(float);
    return total;
}

const char* version_string() { return "turboquant 0.1.0"; }

}  // namespace turboquant
