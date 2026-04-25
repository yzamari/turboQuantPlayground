// turboquant_bench — adb-pushable CLI for parity checks and A/B benchmarks.
//
// Modes:
//   --check                    parity check (smoke level; full golden parity
//                               lands when gen_golden.py is wired up)
//   --bench                    A/B sweep: baseline (no TurboQuant) vs
//                               TurboQuant on the chosen backend
//   --check-cross              cross-backend equivalence (P1+; only one
//                               backend in P0)
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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace turboquant;
using turboquant_bench::Row;
using turboquant_bench::RunConfig;

namespace {

struct Args : public RunConfig {
    std::string mode      = "bench";    // bench | check | check-cross
    std::string csv_path;
    std::vector<int> seq_lens = {128, 256, 512, 1024, 2048};
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

}  // namespace

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

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
