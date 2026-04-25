// Plain FP32 KV cache reference — the "no TurboQuant" arm of the A/B
// benchmark. Stores keys and values as raw float32, computes attention scores
// via straight Q @ K^T and attention output via weights @ V.
//
// Lives in bench/ (not in the core library) so the production library stays
// lean.

#pragma once

#include <cstddef>
#include <vector>

namespace turboquant_bench {

class BaselineKVCache {
public:
    BaselineKVCache(int head_dim) : D_(head_dim) {}

    void prefill(const float* keys, const float* values, int BH, int seq_len) {
        BH_      = BH;
        seq_len_ = seq_len;
        size_t n = static_cast<size_t>(BH) * seq_len * D_;
        keys_  .assign(keys,   keys   + n);
        values_.assign(values, values + n);
    }

    // out_scores[BH, n_q, S] = (query @ keys^T) * scale
    void attention_scores(const float* query, int BH, int n_q,
                          float* out_scores, float scale) const;

    // out[BH, n_q, D] = weights @ values
    void attend(const float* weights, int BH, int n_q, float* out) const;

    int    seq_len()      const { return seq_len_; }
    int    head_dim()     const { return D_; }
    size_t memory_bytes() const { return (keys_.size() + values_.size()) * sizeof(float); }

    // FP16 footprint (what production would actually allocate). FP32 storage
    // here is for arithmetic fidelity in the comparison.
    size_t memory_bytes_fp16() const { return (keys_.size() + values_.size()) * 2; }

private:
    int D_;
    int BH_      = 0;
    int seq_len_ = 0;
    std::vector<float> keys_;
    std::vector<float> values_;
};

}  // namespace turboquant_bench
