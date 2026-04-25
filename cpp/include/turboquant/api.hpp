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

// ---- Build info ----
const char* version_string();

}  // namespace turboquant
