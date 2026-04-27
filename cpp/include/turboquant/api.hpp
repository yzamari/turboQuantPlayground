// Public algorithm classes — the C++ analogue of the Python TurboQuantMSE,
// TurboQuantProd, and TurboQuantKVCache. Backend-agnostic; takes an IBackend
// at construction time.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "turboquant/backend.hpp"
#include "turboquant/codebook.hpp"
#include "turboquant/types.hpp"

namespace turboquant {

// Generate a random orthogonal rotation matrix Pi via QR. Returns row-major
// [d, d] floats. Uses C++ std::mt19937 + Gram–Schmidt — reproducible across
// platforms but does NOT match numpy.linalg.qr output. For exact parity with
// the Python reference (e.g. golden tests), load Pi from the golden corpus
// and pass it to TurboQuantMSE/Prod constructors directly.
std::vector<float> generate_pi_qr(int d, uint64_t seed);

// Generate the QJL Gaussian projection matrix S, row-major [d, d].
// Same caveat: reproducible C++ but not numpy-matching.
std::vector<float> generate_qjl_S(int d, uint64_t seed);

// TurboQuantMSE — Algorithm 1 (MSE-optimal scalar quantization).
//
// Per the Python reference this supports 'qr' rotation only in the v1 C++ port
// (WHT can be added later). Codebooks must be embedded at build time for the
// requested (d, bits) pair. Pi is provided by the caller — generated via
// generate_pi_qr() for runtime use, or loaded from a golden corpus for tests.
class TurboQuantMSE {
public:
    TurboQuantMSE(int dim, int bits, IBackend* backend, std::vector<float> Pi);

    int dim()  const { return dim_;  }
    int bits() const { return bits_; }

    // Quantize x of shape (n_vec, d). The returned MSEQuantized owns its
    // packed bytes and norms.
    MSEQuantized quantize(const float* x, int n_vec) const;

    // Reconstruct float vectors from quantized form. Output is
    // n_vec * d floats, written into `out` (caller-provided).
    void dequantize(const MSEQuantized& q, float* out) const;

    // Accessors used by TurboQuantProd / kernels.
    const std::vector<float>& Pi()                  const { return Pi_; }            // [D, D] row-major
    const std::vector<float>& centroids()           const { return cb_->centroids; }
    const std::vector<float>& decision_boundaries() const { return cb_->decision_boundaries; }
    const Codebook&           codebook()            const { return *cb_; }

private:
    int             dim_;
    int             bits_;
    IBackend*       backend_;
    const Codebook* cb_;
    std::vector<float> Pi_;       // [D, D] row-major
};

// TurboQuantProd — Algorithm 2 (unbiased inner-product quantization).
//
// Pi (rotation) and S (QJL projection) are caller-provided, same as
// TurboQuantMSE.
class TurboQuantProd {
public:
    TurboQuantProd(int dim, int bits, IBackend* backend,
                   std::vector<float> Pi, std::vector<float> S);

    int   dim()        const { return dim_;        }
    int   bits()       const { return bits_;       }
    float qjl_scale()  const { return qjl_scale_;  }

    ProdQuantized quantize(const float* x, int n_vec) const;
    void          dequantize(const ProdQuantized& q, float* out) const;

    // Compute attention scores between query[BH, n_q, D] and quantized keys
    // (already shaped to BH groups of N keys each).
    //
    // out shape: [BH, n_q, N], row-major.
    void attention_score(const float* query, int BH, int n_q,
                         const ProdQuantized& key,  int N,
                         float* out) const;

    const TurboQuantMSE& mse() const { return mse_; }
    const std::vector<float>& S() const { return S_; }   // [D, D] row-major

private:
    int               dim_;
    int               bits_;
    [[maybe_unused]]
    IBackend*         backend_;  // used by attention_score (out-of-line def)
    TurboQuantMSE     mse_;
    std::vector<float> S_;        // QJL projection [D, D]
    float             qjl_scale_;
};

// TurboQuantKVCache — drop-in KV cache. Stores keys via TurboQuantProd and
// values via per-group asymmetric min-max quantization (matching Python
// kv_cache.quantize_values).
//
// Shapes throughout: keys/values arrive as [B, H, S, D]; the cache flattens
// to BH = B*H internally. The cache holds one (key_quantized, value_quantized,
// recent-buffer) triple.
class TurboQuantKVCache {
public:
    struct Config {
        int  head_dim         = 128;
        int  key_bits         = 3;
        int  value_bits       = 2;
        int  value_group_size = 32;
        int  buffer_size      = 128;
        int  layer_idx        = 0;
    };

    // Pi/S: rotation + QJL matrices for the key quantizer at this layer.
    // Caller is responsible for using the correct seed for layer_idx.
    TurboQuantKVCache(const Config& cfg, IBackend* backend,
                      std::vector<float> Pi, std::vector<float> S);

    // Replace the current state with a prefill batch.
    // keys, values: [BH, S, D] row-major.
    void prefill(const float* keys, const float* values, int BH, int seq_len);

    // Append a single decode token: [BH, 1, D].
    void append(const float* key, const float* value, int BH);

    // Compute attention logits for query[BH, n_q, D]; output [BH, n_q, S]
    // where S is the current cache length. The optional `scale` defaults to
    // 1/sqrt(head_dim).
    void attention_scores(const float* query, int BH, int n_q,
                          float* out_scores, float scale = 0.f) const;

    // Compute the attention output: weights[BH, n_q, S] @ V -> [BH, n_q, D].
    // The caller is responsible for softmax.
    void attend(const float* weights, int BH, int n_q,
                float* out) const;

    int  seq_len()      const { return seq_len_; }
    int  batch_heads()  const { return BH_;       }

    // Memory footprint in bytes — useful for the bench's compression-ratio
    // calculation.
    size_t memory_bytes() const;

private:
    void flush_buffer_();

    Config            cfg_;
    IBackend*         backend_;
    TurboQuantProd    key_quantizer_;
    int               BH_      = 0;
    int               seq_len_ = 0;

    // Quantized portion (oldest tokens).
    ProdQuantized     key_q_;
    ValueQuantized    value_q_;
    int               n_quant_ = 0;

    // Unquantized buffer (most recent tokens).
    std::vector<float> key_buf_;     // [BH, n_buf, D]
    std::vector<float> value_buf_;   // [BH, n_buf, D]
    int                n_buf_ = 0;
};

// ---- Helper: per-group asymmetric value quantization (matches Python
//      kv_cache.quantize_values). Exposed so the bench can call it directly. ----
ValueQuantized quantize_values(const float* v, int n_vec, int d,
                               int bits, int group_size);

void dequantize_values(const ValueQuantized& vq, float* out);

// ---- Single-call attention via TurboQuant (Path 2.1c Step 1, A2) ----
//
// One-shot attention: feed Q, K, V, get masked + softmaxed attention output
// back. Internally builds a TurboQuantKVCache, fills K/V (compressed),
// computes attention_scores against Q, applies the optional additive mask,
// softmaxes, and finally calls attend().
//
// Two modes:
//
//   Stateless (default):  session_id == 0 OR layer_il < 0
//     Constructs a throwaway TurboQuantKVCache per call. Used by the host
//     bench / parity tests where each call is independent.
//
//   Session-cached (Phase A2):  session_id != 0 AND layer_il >= 0
//     Looks up an existing TurboQuantKVCache by (session_id, layer_il)
//     and re-uses it. On first call: prefill with all n_kv tokens. On
//     later calls with a larger n_kv: append only the *new* tokens
//     (positions [old_n_kv, n_kv)). Saves the rotate + quantize cost
//     for the K/V slice that's already in the cache, and on the OpenCL
//     backend keeps the K-side cl_mem allocations alive across calls.
//     session_id is intended to be the kv_cache_turboquant instance
//     pointer (cast to uintptr_t) — stable across decode steps and
//     unique per (model, kv-cache).
//
// This is the bridge function our llama.cpp fork calls from the substituted
// attention path (`ggml_custom_4d` op forward in the fork's
// llama-graph.cpp) so the on-device chat decoder can route through TurboQuant
// without taking a libturboquant link dependency on llama.cpp itself.
//
// Layouts (all row-major):
//   q     : [BH * n_q  * D]
//   k     : [BH * n_kv * D]   — full K cache for the current attention call
//   v     : [BH * n_kv * D]   — full V cache for the current attention call
//   mask  : [n_q * n_kv] or NULL — additive (-INF for masked positions),
//           consumed pre-softmax exactly like ggml_soft_max_ext
//   out   : [BH * n_q  * D]   — attention output
//   BH    : batch * n_head_kv (caller folds the GQA ratio if needed)
//   scale : 0.f → defaults to 1/sqrt(D)
//   seed  : per-layer RNG seed for the rotation Pi + QJL S matrices.
//           Convention used elsewhere in the project is 42 + 7*layer_idx.
void attention_turboquant(
    const float* q, const float* k, const float* v,
    int BH, int n_q, int n_kv, int D,
    float scale, const float* mask, float* out,
    int key_bits   = 3,
    int value_bits = 2,
    uint64_t seed  = 42,
    uint64_t session_id = 0,
    int      layer_il   = -1);

// Drop the cached TurboQuantKVCache for one (session_id, layer_il) pair.
// Called when a model unloads. layer_il < 0 means "drop all layers for
// this session". If neither matches anything, this is a no-op.
void attention_turboquant_invalidate(uint64_t session_id, int layer_il = -1);

// ---- Build info ----
const char* version_string();

}  // namespace turboquant
