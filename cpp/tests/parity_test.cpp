// Byte-exact parity test against the Python reference (turboquant_mac).
//
// The golden corpus is produced by `cpp/tools/gen_golden.py` (run with the
// repo's venv on the pytorch backend, which is deterministic on CPU).
//
// For each (d, bits) config we:
//   1) load Pi and S from disk (so we don't depend on numpy.linalg.qr parity)
//   2) construct TurboQuantProd with the loaded matrices
//   3) load the golden inputs and call quantize()
//   4) byte-compare mse_indices and qjl_signs against the golden files
//   5) compare norms / residual_norms within 1e-5
//   6) load the golden query, call attention_score, and compare scores within 1e-4
//
// The configs and filenames are hard-coded to match gen_golden.py's CONFIGS
// list — keeping the test free of a JSON dependency. The golden directory
// can be overridden with argv[1]; otherwise we use TQ_GOLDEN_DIR (set by
// CMake) so `ctest` works without arguments.

#include "tq_test.hpp"

#include "turboquant/api.hpp"
#include "turboquant/backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TQ_GOLDEN_DIR
#define TQ_GOLDEN_DIR "tests/golden"
#endif

using namespace turboquant;

namespace {

struct Config {
    int d;
    int bits;
    int N;
    int n_q;
};

constexpr Config kConfigs[] = {
    {64,  2, 32, 1},
    {64,  3, 32, 1},
    {64,  4, 32, 1},
    {128, 2, 32, 1},
    {128, 3, 32, 1},
    {128, 4, 32, 1},
};

constexpr int kRotSeed = 42;
constexpr int kQjlSeed = 1042;

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

bool read_file(const std::string& path, std::vector<uint8_t>& bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "[parity] cannot open %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(sz));
    if (sz > 0 && !f.read(reinterpret_cast<char*>(bytes.data()), sz)) {
        std::fprintf(stderr, "[parity] read failed %s\n", path.c_str());
        return false;
    }
    return true;
}

bool read_floats(const std::string& path, std::vector<float>& out) {
    std::vector<uint8_t> bytes;
    if (!read_file(path, bytes)) return false;
    if (bytes.size() % sizeof(float) != 0) {
        std::fprintf(stderr, "[parity] %s: not a multiple of float32\n", path.c_str());
        return false;
    }
    out.resize(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}

bool run_one(const Config& cfg, const std::string& dir, IBackend* backend) {
    std::printf("[parity] d=%d bits=%d N=%d n_q=%d\n",
                cfg.d, cfg.bits, cfg.N, cfg.n_q);

    char namebuf[256];

    // Pi
    std::snprintf(namebuf, sizeof(namebuf), "pi_d%d_seed%d.bin", cfg.d, kRotSeed);
    std::vector<float> Pi;
    TQ_CHECK(read_floats(join_path(dir, namebuf), Pi));
    TQ_CHECK_EQ(static_cast<int>(Pi.size()), cfg.d * cfg.d);

    // S
    std::snprintf(namebuf, sizeof(namebuf), "s_d%d_seed%d.bin", cfg.d, kQjlSeed);
    std::vector<float> S;
    TQ_CHECK(read_floats(join_path(dir, namebuf), S));
    TQ_CHECK_EQ(static_cast<int>(S.size()), cfg.d * cfg.d);

    // inputs
    std::snprintf(namebuf, sizeof(namebuf), "inputs_d%d_n%d.bin", cfg.d, cfg.N);
    std::vector<float> inputs;
    TQ_CHECK(read_floats(join_path(dir, namebuf), inputs));
    TQ_CHECK_EQ(static_cast<int>(inputs.size()), cfg.N * cfg.d);

    // query
    std::snprintf(namebuf, sizeof(namebuf), "query_d%d_n_q%d.bin", cfg.d, cfg.n_q);
    std::vector<float> query;
    TQ_CHECK(read_floats(join_path(dir, namebuf), query));
    TQ_CHECK_EQ(static_cast<int>(query.size()), cfg.n_q * cfg.d);

    // golden mse_indices / qjl_signs / norms / residual_norms / scores
    std::snprintf(namebuf, sizeof(namebuf),
                  "prod_b%d_d%d.mse_indices.bin", cfg.bits, cfg.d);
    std::vector<uint8_t> golden_mse;
    TQ_CHECK(read_file(join_path(dir, namebuf), golden_mse));

    std::snprintf(namebuf, sizeof(namebuf),
                  "prod_b%d_d%d.qjl_signs.bin", cfg.bits, cfg.d);
    std::vector<uint8_t> golden_signs;
    TQ_CHECK(read_file(join_path(dir, namebuf), golden_signs));

    std::snprintf(namebuf, sizeof(namebuf),
                  "prod_b%d_d%d.norms.bin", cfg.bits, cfg.d);
    std::vector<float> golden_norms;
    TQ_CHECK(read_floats(join_path(dir, namebuf), golden_norms));
    TQ_CHECK_EQ(static_cast<int>(golden_norms.size()), cfg.N);

    std::snprintf(namebuf, sizeof(namebuf),
                  "prod_b%d_d%d.residual_norms.bin", cfg.bits, cfg.d);
    std::vector<float> golden_res;
    TQ_CHECK(read_floats(join_path(dir, namebuf), golden_res));
    TQ_CHECK_EQ(static_cast<int>(golden_res.size()), cfg.N);

    std::snprintf(namebuf, sizeof(namebuf),
                  "prod_b%d_d%d.scores.bin", cfg.bits, cfg.d);
    std::vector<float> golden_scores;
    TQ_CHECK(read_floats(join_path(dir, namebuf), golden_scores));
    TQ_CHECK_EQ(static_cast<int>(golden_scores.size()), cfg.n_q * cfg.N);

    // ----- Run C++ pipeline with golden Pi / S -----
    TurboQuantProd prod(cfg.d, cfg.bits, backend, Pi, S);

    auto q = prod.quantize(inputs.data(), cfg.N);
    TQ_CHECK_EQ(q.n_vec, cfg.N);
    TQ_CHECK_EQ(q.d, cfg.d);
    TQ_CHECK_EQ(q.mse_bits, cfg.bits - 1);

    // Bit-exact byte compare for packed payloads.
    TQ_CHECK_EQ(q.mse_indices.size(), golden_mse.size());
    if (q.mse_indices.size() == golden_mse.size()) {
        int diffs = 0;
        for (size_t i = 0; i < golden_mse.size(); ++i) {
            if (q.mse_indices[i] != golden_mse[i]) {
                if (diffs < 4) {
                    std::fprintf(stderr,
                        "[parity] mse_indices mismatch at byte %zu: cpp=0x%02x golden=0x%02x\n",
                        i, q.mse_indices[i], golden_mse[i]);
                }
                ++diffs;
            }
        }
        TQ_CHECK_EQ(diffs, 0);
    }

    TQ_CHECK_EQ(q.qjl_signs.size(), golden_signs.size());
    if (q.qjl_signs.size() == golden_signs.size()) {
        int diffs = 0;
        for (size_t i = 0; i < golden_signs.size(); ++i) {
            if (q.qjl_signs[i] != golden_signs[i]) {
                if (diffs < 4) {
                    std::fprintf(stderr,
                        "[parity] qjl_signs mismatch at byte %zu: cpp=0x%02x golden=0x%02x\n",
                        i, q.qjl_signs[i], golden_signs[i]);
                }
                ++diffs;
            }
        }
        TQ_CHECK_EQ(diffs, 0);
    }

    // Norms — float compare with 1e-5 tolerance.
    TQ_CHECK_EQ(static_cast<int>(q.norms.size()), cfg.N);
    TQ_CHECK_EQ(static_cast<int>(q.residual_norms.size()), cfg.N);
    for (int i = 0; i < cfg.N; ++i) {
        TQ_CHECK_NEAR(q.norms[i],          golden_norms[i], 1e-5);
        TQ_CHECK_NEAR(q.residual_norms[i], golden_res[i],   1e-5);
    }

    // Attention scores — flat layout [BH=1, n_q, N].
    std::vector<float> scores(static_cast<size_t>(cfg.n_q) * cfg.N);
    prod.attention_score(query.data(), /*BH=*/1, cfg.n_q, q, cfg.N, scores.data());
    for (int i = 0; i < cfg.n_q * cfg.N; ++i) {
        TQ_CHECK_NEAR(scores[i], golden_scores[i], 1e-4);
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? std::string(argv[1]) : std::string(TQ_GOLDEN_DIR);
    std::printf("[parity] golden dir = %s\n", dir.c_str());

    auto backend = create_backend(BackendKind::CpuScalar);
    TQ_CHECK(backend != nullptr);
    if (!backend) return tq_test::report_and_exit();

    for (const auto& cfg : kConfigs) {
        run_one(cfg, dir, backend.get());
    }

    return tq_test::report_and_exit();
}
