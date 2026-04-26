// QnnSigLipBlock parity test (Path C, Phase 1).
//
// Runs one transformer block of SigLIP-base on the QNN HTP backend and
// compares the output against the FP32 reference produced by
// `cpp/tools/siglip_block_ref.py`.
//
// Acceptance:
//   * cosine similarity >= 0.99
//   * absolute error    < 1e-3 (FP16 path) / < 1e-4 (FP32 path)
//
// Phase 1 status: this test is **expected to fail** at runtime — the
// QnnSigLipBlock::forward() method is currently a std::logic_error stub.
// We intentionally do *not* register the test with ctest yet (see
// cpp/tests/CMakeLists.txt) so CI doesn't go red. The binary lives in
// the build tree and can be invoked manually:
//
//     ./cpp/build-host/cpp/tests/tq_qnn_siglip_block_test \
//         cpp/tests/golden/siglip_block_d768_h12.bin
//
// Once the QNN runtime methods land op-by-op the expected outcome flips
// to "passes," at which point this test joins the ctest suite.

#include "tq_test.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(TQ_WITH_QNN)
#  include "../backends/qnn_htp/qnn_siglip.hpp"
#  include "../backends/qnn_htp/qnn_loader.hpp"
#endif

#ifndef TQ_GOLDEN_DIR
#define TQ_GOLDEN_DIR "tests/golden"
#endif

namespace {

// SigLIP-base block 0 — must match cpp/tools/siglip_block_ref.py.
constexpr int kSeqLen     = 196;
constexpr int kHiddenDim  = 768;
#if defined(TQ_WITH_QNN)
constexpr int kNumHeads   = 12;
constexpr int kMlpDim     = 3072;
constexpr float kLayerNormEps = 1e-6f;
#endif

// Layout of the .bin file produced by the Python ref. Each field is a
// contiguous run of float32. Shapes match SigLIPBlockConfig.
struct GoldenLayout {
    std::vector<float> input;          // [seq, h]
    std::vector<float> ln1_out;        // [seq, h]
    std::vector<float> attn_out;       // [seq, h]
    std::vector<float> attn_residual;  // [seq, h]
    std::vector<float> ln2_out;        // [seq, h]
    std::vector<float> mlp_out;        // [seq, h]
    std::vector<float> output;         // [seq, h]
};

bool read_floats(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "[siglip-test] cannot open %s\n", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0 || sz % static_cast<std::streamsize>(sizeof(float)) != 0) {
        std::fprintf(stderr, "[siglip-test] %s: bad size %lld\n",
                     path.c_str(), static_cast<long long>(sz));
        return false;
    }
    out.resize(static_cast<size_t>(sz) / sizeof(float));
    if (!f.read(reinterpret_cast<char*>(out.data()), sz)) {
        std::fprintf(stderr, "[siglip-test] read failed %s\n", path.c_str());
        return false;
    }
    return true;
}

// Slice the flat float32 buffer into the 7 named tensors. The Python
// generator concatenates them in this exact order (see the `arrays` dict
// in siglip_block_ref.py::main).
bool unpack_golden(const std::vector<float>& flat, GoldenLayout& g) {
    const size_t per_tensor = static_cast<size_t>(kSeqLen) * kHiddenDim;
    const size_t expected   = 7 * per_tensor;
    if (flat.size() != expected) {
        std::fprintf(stderr, "[siglip-test] flat size mismatch: got %zu, want %zu\n",
                     flat.size(), expected);
        return false;
    }
    auto slice = [&](size_t idx) {
        return std::vector<float>(flat.begin() + idx * per_tensor,
                                  flat.begin() + (idx + 1) * per_tensor);
    };
    g.input         = slice(0);
    g.ln1_out       = slice(1);
    g.attn_out      = slice(2);
    g.attn_residual = slice(3);
    g.ln2_out       = slice(4);
    g.mlp_out       = slice(5);
    g.output        = slice(6);
    return true;
}

#if defined(TQ_WITH_QNN)
double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0., na = 0., nb = 0.;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0. || nb == 0.) return 0.;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.;
    const size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return m;
}
#endif  // TQ_WITH_QNN

}  // namespace

int main(int argc, char** argv) {
    std::string golden_dir = (argc > 1) ? std::string(argv[1])
                                        : std::string(TQ_GOLDEN_DIR);
    std::string golden_path = golden_dir + "/siglip_block_d768_h12.bin";
    std::printf("[siglip-test] golden = %s\n", golden_path.c_str());

    std::vector<float> flat;
    if (!read_floats(golden_path, flat)) {
        std::fprintf(stderr,
            "[siglip-test] golden .bin not found — run "
            "`python cpp/tools/siglip_block_ref.py` on a Linux box first.\n");
        return tq_test::report_and_exit();
    }

    GoldenLayout g;
    TQ_CHECK(unpack_golden(flat, g));
    if (g.input.empty()) return tq_test::report_and_exit();

#if defined(TQ_WITH_QNN)
    using namespace turboquant::qnn;

    QnnApi api{};
    if (!load_qnn(&api)) {
        std::fprintf(stderr, "[siglip-test] QNN runtime unavailable on this host — "
                             "skipping (Phase 1 expected: macOS host or no SDK).\n");
        // Not a hard failure on Phase 1 hosts. Re-enable as a hard CHECK
        // once we have CI on a Linux box with QAIRT installed.
        return tq_test::report_and_exit();
    }

    SigLipBlockConfig cfg{};
    cfg.seq_len       = kSeqLen;
    cfg.hidden_dim    = kHiddenDim;
    cfg.num_heads     = kNumHeads;
    cfg.mlp_dim       = kMlpDim;
    cfg.layernorm_eps = kLayerNormEps;

    // Phase 1 scaffold: we don't have weight buffers wired through yet
    // (weights live in siglip_block_d768_h12.weights.bin and need a
    // loader pass; that lands together with build_attention_()).
    // Pass null pointers — the constructor only stores them today.
    SigLipBlockWeights weights{};

    GraphOptions opts{};

    QnnSigLipBlock block(api, cfg, weights, opts);

    std::vector<float> got(static_cast<size_t>(kSeqLen) * kHiddenDim, 0.f);

    bool threw = false;
    try {
        block.forward(g.input.data(), got.data());
    } catch (const std::logic_error& e) {
        threw = true;
        std::printf("[siglip-test] forward() threw (expected at Phase 1): %s\n",
                    e.what());
    }
    // Phase 1: forward() *must* throw. Once steps 1-7 are filled in, flip
    // this assertion to !threw and let the cosine/abs checks below do the
    // real work.
    TQ_CHECK(threw);

    if (!threw) {
        const double cs = cosine(got, g.output);
        const double ma = max_abs_diff(got, g.output);
        std::printf("[siglip-test] cosine = %.6f, max_abs = %.6g\n", cs, ma);
        TQ_CHECK(cs >= 0.99);
        TQ_CHECK(ma < (opts.use_fp32_fallback ? 1e-4 : 1e-3));
    }
#else
    std::printf("[siglip-test] built without TQ_WITH_QNN — nothing to compare.\n");
#endif

    return tq_test::report_and_exit();
}
