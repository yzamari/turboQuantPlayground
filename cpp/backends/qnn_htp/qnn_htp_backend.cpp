// QnnHtpBackend — IBackend implementation that runs rotate + value_dequant
// on the Hexagon NPU and forwards everything else to a composed CPU backend.
//
// Composition over inheritance: we *own* an internal IBackend instance (NEON
// if compiled in, scalar otherwise) and forward by pointer. This is the
// pattern recommended by the Qualcomm sample apps for "hybrid" deployments.
//
// Numerical tolerance vs the scalar reference:
//   * FP16 path (default):           absolute deviation < 1e-3
//   * FP32 fallback (ASIL builds):   absolute deviation < 1e-4

#include "turboquant/backend.hpp"

#include "qnn_graph.hpp"
#include "qnn_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace turboquant {

// Forward declarations of the CPU-fallback factories. We pick one at compile
// time via the macro set in CMakeLists.txt.
#if defined(TQ_QNN_FALLBACK_NEON)
std::unique_ptr<IBackend> create_cpu_neon_backend();
static std::unique_ptr<IBackend> make_qnn_fallback() { return create_cpu_neon_backend(); }
#elif defined(TQ_QNN_FALLBACK_SCALAR)
std::unique_ptr<IBackend> create_cpu_scalar_backend();
static std::unique_ptr<IBackend> make_qnn_fallback() { return create_cpu_scalar_backend(); }
#else
#  error "qnn_htp_backend.cpp built without a CPU fallback macro; check CMakeLists.txt"
#endif

namespace {

// Cache key for shape-specialized graphs. We re-use a graph for every call
// with the same dims so that graphFinalize() is one-time-only.
struct RotateKey { int n, D; };
struct DequantKey { int N, D, bits, group_size; };

inline bool operator==(const RotateKey& a, const RotateKey& b) {
    return a.n == b.n && a.D == b.D;
}
inline bool operator==(const DequantKey& a, const DequantKey& b) {
    return a.N == b.N && a.D == b.D && a.bits == b.bits && a.group_size == b.group_size;
}

struct RotateKeyHash {
    std::size_t operator()(const RotateKey& k) const noexcept {
        return (static_cast<std::size_t>(k.n) << 16) ^ static_cast<std::size_t>(k.D);
    }
};
struct DequantKeyHash {
    std::size_t operator()(const DequantKey& k) const noexcept {
        std::size_t h = static_cast<std::size_t>(k.N);
        h = h * 1315423911u ^ static_cast<std::size_t>(k.D);
        h = h * 1315423911u ^ static_cast<std::size_t>(k.bits);
        h = h * 1315423911u ^ static_cast<std::size_t>(k.group_size);
        return h;
    }
};

class QnnHtpBackend final : public IBackend {
public:
    const char* name() const override { return "qnn_htp"; }

    bool init() override {
        if (!qnn::load_qnn(&api_)) {
            // load_qnn already wrote a stderr line.
            return false;
        }

        // Compose the CPU fallback. If even *that* fails we cannot recover —
        // refuse init and let the factory pick the next backend.
        cpu_fallback_ = make_qnn_fallback();
        if (!cpu_fallback_ || !cpu_fallback_->init()) {
            std::fprintf(stderr, "[turboquant.qnn] CPU fallback init failed; "
                                 "qnn_htp backend disabled\n");
            return false;
        }

        // Optional: opt into FP32 for safety-critical builds via env var. The
        // bench CLI / app can also flip this directly through a future setter.
        const char* fp32 = std::getenv("TQ_QNN_FP32");
        opts_.use_fp32_fallback = (fp32 && fp32[0] == '1');

        return true;
    }

    // ----- HTP-accelerated paths -----

    void rotate(const float* in, const float* Pi,
                int n, int D, float* out) override {
        auto* g = get_or_build_rotate(n, D, Pi);
        if (!g || !g->valid()) {
            // Graph build failed at runtime; transparently fall back so the
            // application keeps running with degraded perf.
            cpu_fallback_->rotate(in, Pi, n, D, out);
            return;
        }
        g->execute(in, out);
    }

    void value_dequant(const uint8_t* packed,
                       const float* scales,
                       const float* zeros,
                       int N, int D, int bits, int group_size,
                       float* out) override {
        // Sub-byte bit widths aren't expressible with the QNN core op set yet
        // (see TODO in qnn_graph.hpp). Stay on CPU for those.
        if (bits != 8) {
            cpu_fallback_->value_dequant(packed, scales, zeros, N, D, bits, group_size, out);
            return;
        }
        auto* g = get_or_build_dequant(N, D, bits, group_size);
        if (!g || !g->valid()) {
            cpu_fallback_->value_dequant(packed, scales, zeros, N, D, bits, group_size, out);
            return;
        }
        g->execute(packed, scales, zeros, out);
    }

    // ----- CPU-only paths (forwarded) -----

    void mse_encode(const float* rotated,
                    const float* boundaries,
                    int N, int D, int bits,
                    void* packed_out) override {
        cpu_fallback_->mse_encode(rotated, boundaries, N, D, bits, packed_out);
    }

    void mse_score(const float* q_rot,
                   const void* mse_packed,
                   const float* norms,
                   const float* centroids,
                   int BH, int N, int D, int bits,
                   float* out) override {
        cpu_fallback_->mse_score(q_rot, mse_packed, norms, centroids, BH, N, D, bits, out);
    }

    void qjl_score(const float* q_sketch,
                   const uint8_t* signs,
                   const float* res_norms,
                   const float* mse_in,
                   int BH, int N, int D,
                   float qjl_scale,
                   float* out) override {
        cpu_fallback_->qjl_score(q_sketch, signs, res_norms, mse_in, BH, N, D, qjl_scale, out);
    }

private:
    qnn::RotateGraph* get_or_build_rotate(int n, int D, const float* Pi) {
        std::lock_guard<std::mutex> lk(mu_);
        RotateKey key{n, D};
        auto it = rotate_cache_.find(key);
        if (it != rotate_cache_.end()) return it->second.get();
        auto g = std::make_unique<qnn::RotateGraph>(api_, n, D, Pi, opts_);
        auto* raw = g.get();
        rotate_cache_.emplace(key, std::move(g));
        return raw;
    }

    qnn::ValueDequantGraph* get_or_build_dequant(int N, int D, int bits, int group_size) {
        std::lock_guard<std::mutex> lk(mu_);
        DequantKey key{N, D, bits, group_size};
        auto it = dequant_cache_.find(key);
        if (it != dequant_cache_.end()) return it->second.get();
        auto g = std::make_unique<qnn::ValueDequantGraph>(api_, N, D, bits, group_size, opts_);
        auto* raw = g.get();
        dequant_cache_.emplace(key, std::move(g));
        return raw;
    }

    qnn::QnnApi               api_{};
    qnn::GraphOptions         opts_{};
    std::unique_ptr<IBackend> cpu_fallback_;

    std::mutex mu_;
    std::unordered_map<RotateKey,  std::unique_ptr<qnn::RotateGraph>,        RotateKeyHash>  rotate_cache_;
    std::unordered_map<DequantKey, std::unique_ptr<qnn::ValueDequantGraph>,  DequantKeyHash> dequant_cache_;
};

}  // namespace

std::unique_ptr<IBackend> create_qnn_htp_backend() {
    return std::make_unique<QnnHtpBackend>();
}

}  // namespace turboquant
