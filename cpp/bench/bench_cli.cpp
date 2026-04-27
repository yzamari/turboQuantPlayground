// turboquant_bench — adb-pushable CLI for parity checks and A/B benchmarks.
//
// Modes:
//   --check                    parity check (smoke level; full golden parity
//                               lands when gen_golden.py is wired up)
//   --bench                    A/B sweep: baseline (no TurboQuant) vs
//                               TurboQuant on the chosen backend
//   --check-cross              cross-backend equivalence (P1+; only one
//                               backend in P0)
//   --check-stateless          parity sweep for the stateless
//                               attention_turboquant() entry-point across a
//                               grid of {BH, D, n_q, n_kv, key_bits,
//                               value_bits} (Path 2.1c Step 6).
//
// Backend selection:  --backend cpu_scalar | cpu_neon | qnn_htp | opencl | vulkan
// Sweep:              --seq-lens 128,256,512,1024,2048,4096
//                     --bits 3
//                     --bh 8  --d 128
//                     --baseline-dtype fp16|fp32
// Output:             --csv <path>   (also prints a human-readable table)

#include "bench_runner.hpp"

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace turboquant;
using turboquant_bench::Row;
using turboquant_bench::RunConfig;

namespace {

struct Args : public RunConfig {
    std::string mode      = "bench";    // bench | check | check-cross | check-stateless
    std::string csv_path;
    std::vector<int> seq_lens = {128, 256, 512, 1024, 2048};
    float threshold = 0.85f;            // min cosine for --check-stateless
};

bool parse_int_list(const std::string& s, std::vector<int>* out) {
    out->clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out->push_back(std::atoi(tok.c_str()));
    }
    return !out->empty();
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) return "";
            return argv[++i];
        };
        if      (k == "--check")          a.mode = "check";
        else if (k == "--bench")          a.mode = "bench";
        else if (k == "--check-cross")    a.mode = "check-cross";
        else if (k == "--check-stateless") a.mode = "check-stateless";
        else if (k == "--backend")        a.backend = next();
        else if (k == "--csv")            a.csv_path = next();
        else if (k == "--seq-lens")       parse_int_list(next(), &a.seq_lens);
        else if (k == "--bits")           a.bits = std::atoi(next().c_str());
        else if (k == "--bh")             a.bh   = std::atoi(next().c_str());
        else if (k == "--d")              a.d    = std::atoi(next().c_str());
        else if (k == "--n-q")            a.n_q  = std::atoi(next().c_str());
        else if (k == "--warmup")         a.warmup = std::atoi(next().c_str());
        else if (k == "--iters")          a.iters  = std::atoi(next().c_str());
        else if (k == "--baseline-dtype") a.baseline_dtype = next();
        else if (k == "--threshold")      a.threshold = std::atof(next().c_str());
        else {
            std::fprintf(stderr, "Unknown argument: %s\n", k.c_str());
        }
    }
    return a;
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "Could not open %s for writing\n", path.c_str());
        return;
    }
    out << "device,backend,seq_len,bh,d,bits,"
           "baseline_attn_ms,tq_attn_ms,attn_speedup,"
           "baseline_mem_bytes,tq_mem_bytes,compression_ratio,"
           "encode_ms,attn_score_cosine_sim,attn_output_rel_l2\n";
    for (const auto& r : rows) {
        out << r.device << ',' << r.backend << ',' << r.seq_len << ','
            << r.bh << ',' << r.d << ',' << r.bits << ','
            << r.baseline_attn_ms << ',' << r.tq_attn_ms << ',' << r.attn_speedup << ','
            << r.baseline_mem_bytes << ',' << r.tq_mem_bytes << ','
            << r.compression_ratio << ',' << r.encode_ms << ','
            << r.attn_score_cosine_sim << ',' << r.attn_output_rel_l2 << '\n';
    }
}

// ---- --check-stateless helpers ----------------------------------------------
// Mirrors the FP32 reference + xorshift RNG in
// cpp/tests/attention_turboquant_test.cpp so the bench can sweep parity
// without pulling the test TU in.

struct StatelessRng {
    uint64_t s;
    explicit StatelessRng(uint64_t seed) : s(seed ? seed : 0xdeadbeefULL) {}
    float next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return static_cast<float>(static_cast<int64_t>(s) % 2'000'001) * 1e-6f - 1.f;
    }
};

void fill_random_stateless(std::vector<float>& v, uint64_t seed) {
    StatelessRng r(seed);
    for (auto& x : v) x = r.next();
}

// scores = Q@K^T * scale; softmax along n_kv; out = scores @ V.
void reference_attention_stateless(
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
            for (int j = 0; j < n_kv; ++j) {
                const float* kv = kb + static_cast<size_t>(j) * D;
                float dot = 0.f;
                for (int d = 0; d < D; ++d) dot += qv[d] * kv[d];
                row[j] = dot * scale;
            }
            float m = row[0];
            for (int j = 1; j < n_kv; ++j) if (row[j] > m) m = row[j];
            float sum = 0.f;
            for (int j = 0; j < n_kv; ++j) { row[j] = std::exp(row[j] - m); sum += row[j]; }
            float inv = 1.f / std::max(sum, 1e-30f);
            for (int j = 0; j < n_kv; ++j) row[j] *= inv;

            float* ov = ob + static_cast<size_t>(t) * D;
            for (int j = 0; j < n_kv; ++j) {
                const float w = row[j];
                const float* vv = vb + static_cast<size_t>(j) * D;
                for (int d = 0; d < D; ++d) ov[d] += w * vv[d];
            }
        }
    }
}

// Per-row cosine across [n_rows × D]; returns (min, mean) ignoring zero-norm
// rows (which can't happen with our random inputs but guard anyway).
struct CosStats {
    double cos_min;
    double cos_mean;
};

CosStats per_row_cosine(const float* a, const float* b, int n_rows, int D) {
    double cos_min = 1.0;
    double cos_sum = 0.0;
    int    counted = 0;
    for (int r = 0; r < n_rows; ++r) {
        const float* ar = a + static_cast<size_t>(r) * D;
        const float* br = b + static_cast<size_t>(r) * D;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int d = 0; d < D; ++d) {
            dot += static_cast<double>(ar[d]) * br[d];
            na  += static_cast<double>(ar[d]) * ar[d];
            nb  += static_cast<double>(br[d]) * br[d];
        }
        if (na <= 0.0 || nb <= 0.0) continue;
        double c = dot / (std::sqrt(na) * std::sqrt(nb));
        if (c < cos_min) cos_min = c;
        cos_sum += c;
        ++counted;
    }
    CosStats s;
    s.cos_min  = (counted > 0) ? cos_min : 0.0;
    s.cos_mean = (counted > 0) ? (cos_sum / counted) : 0.0;
    return s;
}

struct StatelessRow {
    int BH, D, n_q, n_kv, key_bits, value_bits;
    double cos_min;
    double cos_mean;
    double ms_per_call;
};

int run_check_stateless(const Args& a) {
    const std::vector<int> bh_grid       = {4, 8, 16};
    const std::vector<int> d_grid        = {64, 128};
    const std::vector<int> nq_grid       = {1, 8};
    const std::vector<int> nkv_grid      = {128, 512, 2048};
    const std::vector<int> kbits_grid    = {2, 3};
    const std::vector<int> vbits_grid    = {2};

    std::printf("--check-stateless: %s\n", version_string());
    std::printf("Sweeping BH x D x n_q x n_kv x key_bits x value_bits = "
                "%zu x %zu x %zu x %zu x %zu x %zu = %zu configs (threshold=%.3f)\n",
                bh_grid.size(), d_grid.size(), nq_grid.size(),
                nkv_grid.size(), kbits_grid.size(), vbits_grid.size(),
                bh_grid.size() * d_grid.size() * nq_grid.size() *
                    nkv_grid.size() * kbits_grid.size() * vbits_grid.size(),
                static_cast<double>(a.threshold));
    // Threshold gates on per-row cos_mean — that's the metric that mirrors
    // the existing single-config test's whole-blob 0.85 (per-row min is
    // strictly tighter and is reported as a diagnostic).
    std::printf("%4s %5s %5s %6s %4s %4s   %9s %9s %10s\n",
                "BH", "D", "n_q", "n_kv", "kb", "vb",
                "cos_min", "cos_mean", "ms/call");

    std::vector<StatelessRow> rows;
    rows.reserve(72);
    double grid_min  = 1.0;
    double grid_sum  = 0.0;
    int    grid_n    = 0;
    int    n_failed  = 0;

    for (int BH : bh_grid)
    for (int D  : d_grid)
    for (int nq : nq_grid)
    for (int nkv: nkv_grid)
    for (int kb : kbits_grid)
    for (int vb : vbits_grid) {
        const float scale = 1.f / std::sqrt(static_cast<float>(D));

        std::vector<float> q (static_cast<size_t>(BH) * nq  * D);
        std::vector<float> k (static_cast<size_t>(BH) * nkv * D);
        std::vector<float> v (static_cast<size_t>(BH) * nkv * D);
        // Vary seeds with config so we don't get an accidentally-easy grid.
        const uint64_t seed_q = 0xa17e7170u + 7u * (BH + 13 * D + 31 * nkv + nq);
        const uint64_t seed_k = 0xb1ce5e7du + 11u * (BH + 13 * D + 31 * nkv + nq);
        const uint64_t seed_v = 0xc09d7011u + 17u * (BH + 13 * D + 31 * nkv + nq);
        fill_random_stateless(q, seed_q);
        fill_random_stateless(k, seed_k);
        fill_random_stateless(v, seed_v);

        std::vector<float> ref;
        reference_attention_stateless(q.data(), k.data(), v.data(),
                                      BH, nq, nkv, D, scale, ref);

        std::vector<float> tq(ref.size(), 0.f);
        const auto t0 = std::chrono::steady_clock::now();
        attention_turboquant(
            q.data(), k.data(), v.data(),
            BH, nq, nkv, D,
            scale, /*mask=*/nullptr, tq.data(),
            kb, vb, /*seed=*/42);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        const int n_rows = BH * nq;
        const CosStats cs = per_row_cosine(ref.data(), tq.data(), n_rows, D);

        StatelessRow row;
        row.BH = BH; row.D = D; row.n_q = nq; row.n_kv = nkv;
        row.key_bits = kb; row.value_bits = vb;
        row.cos_min = cs.cos_min; row.cos_mean = cs.cos_mean;
        row.ms_per_call = ms;
        rows.push_back(row);

        if (cs.cos_min < grid_min) grid_min = cs.cos_min;
        grid_sum += cs.cos_mean;
        ++grid_n;
        if (cs.cos_mean < a.threshold) ++n_failed;

        std::printf("%4d %5d %5d %6d %4d %4d   %9.4f %9.4f %10.3f%s\n",
                    BH, D, nq, nkv, kb, vb,
                    cs.cos_min, cs.cos_mean, ms,
                    (cs.cos_mean < a.threshold) ? "  FAIL" : "");
    }

    const double grid_mean = (grid_n > 0) ? (grid_sum / grid_n) : 0.0;
    std::printf("\nSummary: %d configs, grid cos_min=%.4f grid cos_mean=%.4f, "
                "%d below threshold %.3f\n",
                grid_n, grid_min, grid_mean, n_failed,
                static_cast<double>(a.threshold));

    if (!a.csv_path.empty()) {
        std::ofstream out(a.csv_path);
        if (!out) {
            std::fprintf(stderr, "Could not open %s for writing\n",
                         a.csv_path.c_str());
        } else {
            out << "BH,D,n_q,n_kv,key_bits,value_bits,cos_min,cos_mean,ms_per_call\n";
            for (const auto& r : rows) {
                out << r.BH << ',' << r.D << ',' << r.n_q << ',' << r.n_kv
                    << ',' << r.key_bits << ',' << r.value_bits << ','
                    << r.cos_min << ',' << r.cos_mean << ',' << r.ms_per_call
                    << '\n';
            }
        }
    }

    return (n_failed == 0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    // --check-stateless exercises the host stateless attention_turboquant()
    // entry-point directly; it doesn't go through the backend interface and
    // shouldn't fail when only one backend is compiled in.
    if (a.mode == "check-stateless") {
        return run_check_stateless(a);
    }

    auto kind    = backend_kind_from_name(a.backend.c_str());
    auto backend = create_backend(kind);
    if (!backend) {
        std::fprintf(stderr, "Backend '%s' not available in this build.\n",
                     a.backend.c_str());
        return 2;
    }
    std::printf("Backend: %s   %s\n", backend->name(), version_string());

    if (a.mode == "check") {
        auto Pi = generate_pi_qr(a.d, 42);
        auto S  = generate_qjl_S(a.d, 1042);
        TurboQuantProd prod(a.d, a.bits, backend.get(), Pi, S);
        std::vector<float> x(static_cast<size_t>(8) * a.d, 0.f);
        std::mt19937 rng(0); std::normal_distribution<float> nd(0,1);
        for (auto& v : x) v = nd(rng);
        auto q = prod.quantize(x.data(), 8);
        std::vector<float> y(8 * a.d);
        prod.dequantize(q, y.data());
        double cos = turboquant_bench::cosine_sim(x.data(), y.data(), 8 * a.d);
        std::printf("smoke quantize/dequantize cosine = %.4f (expect > 0.85 for bits=%d)\n",
                    cos, a.bits);
        return (cos > 0.85) ? 0 : 1;
    }

    if (a.mode == "bench") {
        std::vector<Row> rows;
        std::printf("%s", turboquant_bench::format_header().c_str());
        for (int s : a.seq_lens) {
            Row r = turboquant_bench::run_one(a, backend.get(), s);
            std::printf("%s", turboquant_bench::format_row(r).c_str());
            rows.push_back(r);
        }
        if (!a.csv_path.empty()) write_csv(a.csv_path, rows);
        return 0;
    }

    if (a.mode == "check-cross") {
        std::fprintf(stderr, "--check-cross requires multiple backends; only one in P0.\n");
        return 0;
    }
    return 0;
}
