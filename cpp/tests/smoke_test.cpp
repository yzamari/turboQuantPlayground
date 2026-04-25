// End-to-end smoke test: build a TurboQuantProd, run quantize -> dequantize,
// verify the dequantized vectors are close to the originals (loose tolerance —
// 3-bit quantization is lossy by design). Also runs attention_score and
// confirms the produced scores are in the same ballpark as the brute-force
// query.key inner products.
//
// This is NOT a parity-vs-Python golden test (that's parity_test.cpp once
// gen_golden.py runs). This is just "the C++ pipeline runs end-to-end without
// crashing".

#include "tq_test.hpp"

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace turboquant;

int main() {
    auto backend = create_backend(BackendKind::CpuScalar);
    TQ_CHECK(backend != nullptr);
    if (!backend) return tq_test::report_and_exit();

    constexpr int D = 128;
    constexpr int N = 64;
    constexpr int BH = 1;
    constexpr int n_q = 1;

    auto Pi = generate_pi_qr(D, /*seed=*/42);
    auto S  = generate_qjl_S(D, /*seed=*/1042);

    TurboQuantProd prod(D, /*bits=*/3, backend.get(), Pi, S);

    // Generate N random keys (unit-ish norm) and a query.
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> keys (static_cast<size_t>(N) * D);
    std::vector<float> query(static_cast<size_t>(n_q) * D);
    for (auto& v : keys ) v = nd(rng);
    for (auto& v : query) v = nd(rng);

    auto q = prod.quantize(keys.data(), N);
    TQ_CHECK_EQ(q.n_vec, N);
    TQ_CHECK_EQ(q.d, D);

    // Round-trip check: cosine similarity between original and dequantized.
    std::vector<float> reco(static_cast<size_t>(N) * D);
    prod.dequantize(q, reco.data());
    double cos_avg = 0.0;
    for (int v = 0; v < N; ++v) {
        double dot = 0, na = 0, nb = 0;
        for (int j = 0; j < D; ++j) {
            dot += static_cast<double>(keys[v * D + j]) * reco[v * D + j];
            na  += static_cast<double>(keys[v * D + j]) * keys[v * D + j];
            nb  += static_cast<double>(reco[v * D + j]) * reco[v * D + j];
        }
        cos_avg += dot / (std::sqrt(na * nb) + 1e-12);
    }
    cos_avg /= N;
    TQ_CHECK(cos_avg > 0.9);  // 3-bit quantization should retain ≥0.9 cosine on Gaussian data

    // Attention score sanity: produced scores should correlate well with the
    // exact query.key inner products.
    std::vector<float> scores_q (static_cast<size_t>(BH) * n_q * N);
    std::vector<float> scores_gt(static_cast<size_t>(BH) * n_q * N);
    prod.attention_score(query.data(), BH, n_q, q, N, scores_q.data());
    for (int n = 0; n < N; ++n) {
        double s = 0.0;
        for (int j = 0; j < D; ++j) s += static_cast<double>(query[j]) * keys[n * D + j];
        scores_gt[n] = static_cast<float>(s);
    }
    // Pearson correlation
    double m_q = 0, m_g = 0;
    for (int n = 0; n < N; ++n) { m_q += scores_q[n]; m_g += scores_gt[n]; }
    m_q /= N; m_g /= N;
    double cov = 0, vq = 0, vg = 0;
    for (int n = 0; n < N; ++n) {
        double dq = scores_q [n] - m_q;
        double dg = scores_gt[n] - m_g;
        cov += dq * dg; vq += dq * dq; vg += dg * dg;
    }
    double corr = cov / (std::sqrt(vq * vg) + 1e-12);
    std::printf("[smoke] cosine(roundtrip) = %.4f, corr(scores) = %.4f\n",
                cos_avg, corr);
    // Threshold is loose because:
    //   1) we use a C++-side Gram-Schmidt Pi that isn't numpy.linalg.qr exact
    //      (orthogonal but different from Python's golden Pi);
    //   2) the 1-bit QJL sketch adds variance to individual scores.
    // The strict golden parity test (parity_test.cpp, post-gen_golden.py)
    // tightens this to 1e-4.
    TQ_CHECK(corr > 0.85);

    return tq_test::report_and_exit();
}
