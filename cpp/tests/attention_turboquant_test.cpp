// Parity test for the new stateless `attention_turboquant()` entry point
// (Path 2.1c Step 1). Compares output against a hand-rolled FP32 reference
// at BH=8, D=128, n_q=8, n_kv=512, key_bits=3.
//
// Acceptance: cosine similarity ≥ 0.85 on the attention output against the
// FP32 baseline. The 0.92 cosine the README quotes is on attention SCORES;
// the per-output-row cosine after V-dequantize is naturally lower because
// V is groupwise-quantized too. 0.85 is the bench's `--check` threshold.

#include "turboquant/api.hpp"
#include "tq_test.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// xorshift-style FP32 random generator on a fixed seed. Matches no
// particular Python RNG — these inputs are local to this test.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0xdeadbeefULL) {}
    float next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Symmetric in [-1, 1].
        return static_cast<float>(static_cast<int64_t>(s) % 2'000'001) * 1e-6f - 1.f;
    }
};

void fill_random(std::vector<float>& v, uint64_t seed) {
    Rng r(seed);
    for (auto& x : v) x = r.next();
}

// FP32 reference: scores = Q@K^T * scale; mask + softmax; out = scores @ V.
// Shapes are flat row-major:
//   q   : [BH * n_q  * D]
//   k   : [BH * n_kv * D]
//   v   : [BH * n_kv * D]
//   out : [BH * n_q  * D]
void reference_attention(
    const float* q, const float* k, const float* v,
    int BH, int n_q, int n_kv, int D,
    float scale, std::vector<float>& out) {

    out.assign(static_cast<size_t>(BH) * n_q * D, 0.f);
    std::vector<float> row(n_kv);

    for (int b = 0; b < BH; ++b) {
        const float* qb = q + static_cast<size_t>(b) * n_q  * D;
        const float* kb = k + static_cast<size_t>(b) * n_kv * D;
        const float* vb = v + static_cast<size_t>(b) * n_kv * D;
        float*       ob = out.data() + static_cast<size_t>(b) * n_q * D;

        for (int t = 0; t < n_q; ++t) {
            const float* qv = qb + static_cast<size_t>(t) * D;

            // Scores row.
            for (int j = 0; j < n_kv; ++j) {
                const float* kv = kb + static_cast<size_t>(j) * D;
                float dot = 0.f;
                for (int d = 0; d < D; ++d) dot += qv[d] * kv[d];
                row[j] = dot * scale;
            }

            // Softmax in-place.
            float m = row[0];
            for (int j = 1; j < n_kv; ++j) if (row[j] > m) m = row[j];
            float sum = 0.f;
            for (int j = 0; j < n_kv; ++j) { row[j] = std::exp(row[j] - m); sum += row[j]; }
            float inv = 1.f / std::max(sum, 1e-30f);
            for (int j = 0; j < n_kv; ++j) row[j] *= inv;

            // Output row = row @ V.
            float* ov = ob + static_cast<size_t>(t) * D;
            for (int j = 0; j < n_kv; ++j) {
                const float w = row[j];
                const float* vv = vb + static_cast<size_t>(j) * D;
                for (int d = 0; d < D; ++d) ov[d] += w * vv[d];
            }
        }
    }
}

double cosine_similarity(const float* a, const float* b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main() {
    using namespace turboquant;

    constexpr int BH   = 8;
    constexpr int D    = 128;
    constexpr int n_q  = 8;
    constexpr int n_kv = 512;
    const float scale  = 1.f / std::sqrt(static_cast<float>(D));

    std::vector<float> q (static_cast<size_t>(BH) * n_q  * D);
    std::vector<float> k (static_cast<size_t>(BH) * n_kv * D);
    std::vector<float> v (static_cast<size_t>(BH) * n_kv * D);
    fill_random(q, 0xa17e7170u);
    fill_random(k, 0xb1ce5e7du);
    fill_random(v, 0xc09d7011u);

    // FP32 baseline.
    std::vector<float> ref;
    reference_attention(q.data(), k.data(), v.data(),
                        BH, n_q, n_kv, D, scale, ref);

    // TurboQuant single-call.
    std::vector<float> tq(ref.size(), 0.f);
    attention_turboquant(
        q.data(), k.data(), v.data(),
        BH, n_q, n_kv, D,
        scale, /*mask=*/nullptr, tq.data(),
        /*key_bits=*/3, /*value_bits=*/2, /*seed=*/42);

    // Cosine across the entire output blob; should be ≥ 0.85 with bits=3
    // (matches the bench's `--check` threshold on quantize+dequantize).
    const double cos_full = cosine_similarity(ref.data(), tq.data(), ref.size());
    std::fprintf(stderr,
        "attention_turboquant_test: BH=%d D=%d n_q=%d n_kv=%d "
        "key_bits=3 value_bits=2 → cosine=%.4f\n",
        BH, D, n_q, n_kv, cos_full);
    TQ_CHECK(cos_full >= 0.85);

    // Sanity: non-NaN output.
    bool all_finite = true;
    for (float x : tq) if (!std::isfinite(x)) { all_finite = false; break; }
    TQ_CHECK(all_finite);

    // Mask sanity: with all positions masked off, output must be all zero.
    std::vector<float> all_masked(static_cast<size_t>(n_q) * n_kv,
                                  -std::numeric_limits<float>::infinity());
    std::vector<float> tq_masked(ref.size(), 1.f);  // pre-fill with garbage
    attention_turboquant(
        q.data(), k.data(), v.data(),
        BH, n_q, n_kv, D,
        scale, all_masked.data(), tq_masked.data(),
        3, 2, 42);
    bool mask_zero = true;
    for (float x : tq_masked) if (std::fabs(x) > 1e-6f) { mask_zero = false; break; }
    TQ_CHECK(mask_zero);

    return tq_test::report_and_exit();
}
