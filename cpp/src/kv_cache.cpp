#include "turboquant/api.hpp"

#include "packing.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define TQ_HAVE_NEON 1
#else
#  define TQ_HAVE_NEON 0
#endif

#if defined(__ANDROID__)
#  include <android/log.h>
#  define TQ_KV_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "turboquant_kv", __VA_ARGS__)
#else
#  define TQ_KV_LOGI(...) ((void)0)
#endif

namespace turboquant {

namespace {

// Horizontal dot product: sum_k a[k] * b[k]. NEON path uses two parallel
// f32x4 accumulators (better ILP on Cortex-A7xx / Apple Firestorm) and
// folds the tail scalar. Scalar fallback keeps the original double-precision
// accumulator to preserve the host x86 ctest baseline.
inline float dot_f32(const float* a, const float* b, int n) {
#if TQ_HAVE_NEON
    float32x4_t acc0 = vdupq_n_f32(0.f);
    float32x4_t acc1 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + k),     vld1q_f32(b + k));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
    }
    float32x4_t acc = vaddq_f32(acc0, acc1);
    if (k + 4 <= n) {
        acc = vfmaq_f32(acc, vld1q_f32(a + k), vld1q_f32(b + k));
        k += 4;
    }
#  if defined(__aarch64__)
    float s = vaddvq_f32(acc);
#  else
    float32x2_t s2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    s2 = vpadd_f32(s2, s2);
    float s = vget_lane_f32(s2, 0);
#  endif
    for (; k < n; ++k) s += a[k] * b[k];
    return s;
#else
    double s = 0.0;
    for (int k = 0; k < n; ++k) s += static_cast<double>(a[k]) * b[k];
    return static_cast<float>(s);
#endif
}

// In-place SAXPY: y[k] += a * x[k] for k in [0, n).
inline void saxpy_inplace_f32(float* y, float a, const float* x, int n) {
#if TQ_HAVE_NEON
    const float32x4_t a_v = vdupq_n_f32(a);
    int k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t y_v = vld1q_f32(y + k);
        float32x4_t x_v = vld1q_f32(x + k);
        y_v = vfmaq_f32(y_v, x_v, a_v);
        vst1q_f32(y + k, y_v);
    }
    for (; k < n; ++k) y[k] += a * x[k];
#else
    for (int k = 0; k < n; ++k) y[k] += a * x[k];
#endif
}

// Append `new_q` (n_vec=BH, one row per BH) into `old_q` (n_vec=BH*old_per_bh,
// per-BH rows contiguous) so the result holds (old_per_bh + 1) rows per BH
// in the same BH-major layout TurboQuantKVCache uses for key_q_/value_q_.
ProdQuantized merge_prod_append_per_bh(const ProdQuantized& old_q,
                                       const ProdQuantized& new_q,
                                       int BH, int old_per_bh) {
    const int new_per_bh   = old_per_bh + 1;
    const size_t mse_row   = static_cast<size_t>(old_q.packed_len_mse);
    const size_t sig_row   = static_cast<size_t>(old_q.packed_len_signs);

    ProdQuantized r;
    r.n_vec            = BH * new_per_bh;
    r.d                = old_q.d;
    r.mse_bits         = old_q.mse_bits;
    r.packed_len_mse   = old_q.packed_len_mse;
    r.packed_len_signs = old_q.packed_len_signs;
    r.mse_indices   .assign(static_cast<size_t>(r.n_vec) * mse_row, 0);
    r.qjl_signs     .assign(static_cast<size_t>(r.n_vec) * sig_row, 0);
    r.norms         .assign(r.n_vec, 0.f);
    r.residual_norms.assign(r.n_vec, 0.f);

    for (int b = 0; b < BH; ++b) {
        const size_t old_off_mse = static_cast<size_t>(b) * old_per_bh * mse_row;
        const size_t new_off_mse = static_cast<size_t>(b) * new_per_bh * mse_row;
        std::memcpy(r.mse_indices.data() + new_off_mse,
                    old_q.mse_indices.data() + old_off_mse,
                    static_cast<size_t>(old_per_bh) * mse_row);
        std::memcpy(r.mse_indices.data() + new_off_mse + static_cast<size_t>(old_per_bh) * mse_row,
                    new_q.mse_indices.data() + static_cast<size_t>(b) * mse_row,
                    mse_row);

        const size_t old_off_sig = static_cast<size_t>(b) * old_per_bh * sig_row;
        const size_t new_off_sig = static_cast<size_t>(b) * new_per_bh * sig_row;
        std::memcpy(r.qjl_signs.data() + new_off_sig,
                    old_q.qjl_signs.data() + old_off_sig,
                    static_cast<size_t>(old_per_bh) * sig_row);
        std::memcpy(r.qjl_signs.data() + new_off_sig + static_cast<size_t>(old_per_bh) * sig_row,
                    new_q.qjl_signs.data() + static_cast<size_t>(b) * sig_row,
                    sig_row);

        const size_t old_norms_off = static_cast<size_t>(b) * old_per_bh;
        const size_t new_norms_off = static_cast<size_t>(b) * new_per_bh;
        std::memcpy(r.norms.data() + new_norms_off,
                    old_q.norms.data() + old_norms_off,
                    static_cast<size_t>(old_per_bh) * sizeof(float));
        r.norms[new_norms_off + old_per_bh] = new_q.norms[b];
        std::memcpy(r.residual_norms.data() + new_norms_off,
                    old_q.residual_norms.data() + old_norms_off,
                    static_cast<size_t>(old_per_bh) * sizeof(float));
        r.residual_norms[new_norms_off + old_per_bh] = new_q.residual_norms[b];
    }
    return r;
}

ValueQuantized merge_value_append_per_bh(const ValueQuantized& old_v,
                                         const ValueQuantized& new_v,
                                         int BH, int old_per_bh) {
    const int new_per_bh = old_per_bh + 1;
    const size_t data_row = static_cast<size_t>(old_v.packed_d);
    const size_t grp_row  = static_cast<size_t>(old_v.n_groups);

    ValueQuantized r;
    r.n_vec      = BH * new_per_bh;
    r.d          = old_v.d;
    r.bits       = old_v.bits;
    r.group_size = old_v.group_size;
    r.n_groups   = old_v.n_groups;
    r.packed_d   = old_v.packed_d;
    r.data  .assign(static_cast<size_t>(r.n_vec) * data_row, 0);
    r.scales.assign(static_cast<size_t>(r.n_vec) * grp_row,  0.f);
    r.zeros .assign(static_cast<size_t>(r.n_vec) * grp_row,  0.f);

    for (int b = 0; b < BH; ++b) {
        const size_t old_off_d = static_cast<size_t>(b) * old_per_bh * data_row;
        const size_t new_off_d = static_cast<size_t>(b) * new_per_bh * data_row;
        std::memcpy(r.data.data() + new_off_d,
                    old_v.data.data() + old_off_d,
                    static_cast<size_t>(old_per_bh) * data_row);
        std::memcpy(r.data.data() + new_off_d + static_cast<size_t>(old_per_bh) * data_row,
                    new_v.data.data() + static_cast<size_t>(b) * data_row,
                    data_row);

        const size_t old_off_g = static_cast<size_t>(b) * old_per_bh * grp_row;
        const size_t new_off_g = static_cast<size_t>(b) * new_per_bh * grp_row;
        std::memcpy(r.scales.data() + new_off_g,
                    old_v.scales.data() + old_off_g,
                    static_cast<size_t>(old_per_bh) * grp_row * sizeof(float));
        std::memcpy(r.scales.data() + new_off_g + static_cast<size_t>(old_per_bh) * grp_row,
                    new_v.scales.data() + static_cast<size_t>(b) * grp_row,
                    grp_row * sizeof(float));
        std::memcpy(r.zeros.data() + new_off_g,
                    old_v.zeros.data() + old_off_g,
                    static_cast<size_t>(old_per_bh) * grp_row * sizeof(float));
        std::memcpy(r.zeros.data() + new_off_g + static_cast<size_t>(old_per_bh) * grp_row,
                    new_v.zeros.data() + static_cast<size_t>(b) * grp_row,
                    grp_row * sizeof(float));
    }
    return r;
}

}  // namespace

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

// Append one decode-step (key, value) of shape [BH, D] to the cache.
//
// Strategy: fixed-buffer + flush-on-overflow. When the new slot would push
// n_buf_ above cfg_.buffer_size, the oldest slot is quantized into
// key_q_/value_q_ first (see flush_buffer_) and then the new slot is
// grown onto the buffer. This bounds memory at
// O(BH * buffer_size * D + BH * (seq_len - buffer_size) * <compressed>)
// regardless of how long the decode runs.
//
// The `while` loop covers two cases: (a) steady state — at most one flush
// per append; (b) legacy-state recovery — a session that ran under the
// previous grow-only append has n_buf_ >> buffer_size on entry and bleeds
// down one slot per call until the invariant is restored.
void TurboQuantKVCache::append(const float* key, const float* value, int BH) {
    if (BH != BH_) {
        throw std::invalid_argument("append: BH mismatch");
    }
    while (n_buf_ + 1 > cfg_.buffer_size) {
        flush_buffer_();
    }

    const int D = cfg_.head_dim;
    const int new_n_buf = n_buf_ + 1;

    // Grow buffers from [BH, n_buf, D] to [BH, n_buf+1, D]. Rebuild
    // because the bh-major layout puts each BH's tokens contiguously,
    // so inserting one slot per BH means rewriting every BH segment.
    // O(BH * n_buf * D) per call — amortized OK for chat-length decodes.
    std::vector<float> new_key_buf  (static_cast<size_t>(BH) * new_n_buf * D);
    std::vector<float> new_value_buf(static_cast<size_t>(BH) * new_n_buf * D);

    for (int b = 0; b < BH; ++b) {
        const size_t old_off = static_cast<size_t>(b) * n_buf_   * D;
        const size_t new_off = static_cast<size_t>(b) * new_n_buf * D;

        if (n_buf_ > 0) {
            std::memcpy(new_key_buf  .data() + new_off,
                        key_buf_     .data() + old_off,
                        static_cast<size_t>(n_buf_) * D * sizeof(float));
            std::memcpy(new_value_buf.data() + new_off,
                        value_buf_   .data() + old_off,
                        static_cast<size_t>(n_buf_) * D * sizeof(float));
        }

        const size_t tail_off = new_off + static_cast<size_t>(n_buf_) * D;
        std::memcpy(new_key_buf  .data() + tail_off,
                    key   + static_cast<size_t>(b) * D,
                    static_cast<size_t>(D) * sizeof(float));
        std::memcpy(new_value_buf.data() + tail_off,
                    value + static_cast<size_t>(b) * D,
                    static_cast<size_t>(D) * sizeof(float));
    }

    key_buf_   = std::move(new_key_buf);
    value_buf_ = std::move(new_value_buf);
    n_buf_     = new_n_buf;
    ++seq_len_;
}

// Move the oldest buffer slot (per BH) into the compressed quantized
// portion. Called by append() when the buffer would otherwise overflow
// cfg_.buffer_size.
//
// Steps:
//  1. Gather the oldest slot from each BH into a contiguous [BH, D] dense
//     array (one for keys, one for values).
//  2. Run the existing TurboQuantProd / quantize_values pipelines on those
//     BH rows.
//  3. Append the new BH rows into key_q_ / value_q_ at position n_quant_
//     for each BH, preserving the BH-major row layout the rest of the
//     class relies on.
//  4. Shift each BH's buffer left by 1 (drop the slot we just flushed).
//  5. Update counters: ++n_quant_, --n_buf_. seq_len_ is unchanged —
//     flush moves data within the cache, it doesn't add or drop tokens.
void TurboQuantKVCache::flush_buffer_() {
    if (n_buf_ <= 0) {
        throw std::logic_error("flush_buffer_: nothing to flush (n_buf_ == 0)");
    }
    const int BH = BH_;
    const int D  = cfg_.head_dim;

    std::vector<float> oldest_k(static_cast<size_t>(BH) * D);
    std::vector<float> oldest_v(static_cast<size_t>(BH) * D);
    for (int b = 0; b < BH; ++b) {
        std::memcpy(oldest_k.data() + static_cast<size_t>(b) * D,
                    key_buf_  .data() + static_cast<size_t>(b) * n_buf_ * D,
                    static_cast<size_t>(D) * sizeof(float));
        std::memcpy(oldest_v.data() + static_cast<size_t>(b) * D,
                    value_buf_.data() + static_cast<size_t>(b) * n_buf_ * D,
                    static_cast<size_t>(D) * sizeof(float));
    }

    auto new_kq = key_quantizer_.quantize(oldest_k.data(), BH);
    auto new_vq = quantize_values(oldest_v.data(), BH, D,
                                  cfg_.value_bits, cfg_.value_group_size);

    if (n_quant_ == 0) {
        // First flush — nothing to merge with; the new BH-row arrays already
        // sit in BH-major layout (one row per BH) which is exactly what the
        // rest of the class expects for n_quant_ == 1.
        key_q_   = std::move(new_kq);
        value_q_ = std::move(new_vq);
    } else {
        key_q_   = merge_prod_append_per_bh (key_q_,   new_kq, BH, n_quant_);
        value_q_ = merge_value_append_per_bh(value_q_, new_vq, BH, n_quant_);
    }

    const int new_n_buf = n_buf_ - 1;
    std::vector<float> new_kbuf(static_cast<size_t>(BH) * new_n_buf * D);
    std::vector<float> new_vbuf(static_cast<size_t>(BH) * new_n_buf * D);
    for (int b = 0; b < BH; ++b) {
        std::memcpy(new_kbuf.data() + static_cast<size_t>(b) * new_n_buf * D,
                    key_buf_.data() + static_cast<size_t>(b) * n_buf_ * D + D,
                    static_cast<size_t>(new_n_buf) * D * sizeof(float));
        std::memcpy(new_vbuf.data() + static_cast<size_t>(b) * new_n_buf * D,
                    value_buf_.data() + static_cast<size_t>(b) * n_buf_ * D + D,
                    static_cast<size_t>(new_n_buf) * D * sizeof(float));
    }
    key_buf_   = std::move(new_kbuf);
    value_buf_ = std::move(new_vbuf);

    ++n_quant_;
    --n_buf_;

    TQ_KV_LOGI("flush: seq_len=%d BH=%d D=%d n_buf=%d n_quant=%d",
               seq_len_, BH, D, n_buf_, n_quant_);
}

void TurboQuantKVCache::attention_scores(const float* query, int BH, int n_q,
                                         float* out_scores, float scale) const {
    if (BH != BH_) throw std::invalid_argument("attention_scores: BH mismatch");
    if (scale == 0.f) scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

    const int S = seq_len_;

    // Quantized portion: out[bh, t, j] for j in [0, n_quant)
    if (n_quant_ > 0) {
        std::vector<float> tmp(static_cast<size_t>(BH) * n_q * n_quant_);
        // P2 Stage B: forward the cached opaque GPU-key handle to the
        // quantizer so OpenCL can use the cl_mem pool. nullptr selects
        // the unpooled fallback (default backend impl).
        key_quantizer_.attention_score(query, BH, n_q, key_q_, n_quant_,
                                       tmp.data(), gpu_key_handle_);
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
                    dst[j] = dot_f32(qv, kv, cfg_.head_dim) * scale;
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
                    saxpy_inplace_f32(o, w[j], vj, D);
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
