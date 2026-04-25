// Scalar (portable C++17) reference backend.
//
// Translates each Metal kernel from src/turboquant_mac/backends/metal/*.py
// into straight C++. No SIMD intrinsics, no platform-specific code — this is
// the byte-exact numerical reference that all other backends must match.

#include "turboquant/backend.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

namespace turboquant {
namespace {

class CpuScalarBackend final : public IBackend {
public:
    const char* name() const override { return "cpu_scalar"; }
    bool init() override { return true; }

    void rotate(const float* in, const float* Pi,
                int n, int D, float* out) override {
        // out[i, j] = sum_k in[i, k] * Pi[j, k]    (out = in @ Pi^T)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < D; ++j) {
                double s = 0.0;
                for (int k = 0; k < D; ++k) {
                    s += static_cast<double>(in[i * D + k]) * Pi[j * D + k];
                }
                out[i * D + j] = static_cast<float>(s);
            }
        }
    }

    // -------------------------------------------------------------------------
    // mse_encode (mirrors metal/mse_encode.py)
    // -------------------------------------------------------------------------
    void mse_encode(const float* rotated, const float* boundaries,
                    int N, int D, int bits, void* packed_out) override {
        const int n_boundaries = (1 << bits) - 1;

        if (bits == 3) {
            // 10 vals per uint32, 30 of 32 bits used.
            const int n_words = (D + 9) / 10;
            uint32_t* out = static_cast<uint32_t*>(packed_out);
            for (int v = 0; v < N; ++v) {
                uint32_t* o = out + static_cast<size_t>(v) * n_words;
                for (int w = 0; w < n_words; ++w) o[w] = 0;
                for (int j = 0; j < D; ++j) {
                    float val = rotated[v * D + j];
                    int idx = 0;
                    for (int b = 0; b < n_boundaries; ++b) {
                        if (val >= boundaries[b]) ++idx;
                    }
                    int word = j / 10;
                    int sub  = j % 10;
                    o[word] |= (static_cast<uint32_t>(idx) & 0x7u) << (sub * 3);
                }
            }
            return;
        }

        // 1/2/4/8-bit cases — uint8 output.
        int vals_per_byte;
        int eff_bits;
        switch (bits) {
            case 1: vals_per_byte = 8; eff_bits = 1; break;
            case 2: vals_per_byte = 4; eff_bits = 2; break;
            case 4: vals_per_byte = 2; eff_bits = 4; break;
            case 8: default:
                    vals_per_byte = 1; eff_bits = 8; break;
        }
        const int packed_d = (D + vals_per_byte - 1) / vals_per_byte;
        uint8_t* out = static_cast<uint8_t*>(packed_out);
        for (int v = 0; v < N; ++v) {
            uint8_t* o = out + static_cast<size_t>(v) * packed_d;
            std::memset(o, 0, packed_d);
            for (int j = 0; j < D; ++j) {
                float val = rotated[v * D + j];
                int idx = 0;
                for (int b = 0; b < n_boundaries; ++b) {
                    if (val >= boundaries[b]) ++idx;
                }
                int byte = j / vals_per_byte;
                int sub  = j % vals_per_byte;
                o[byte] |= static_cast<uint8_t>((idx & ((1 << eff_bits) - 1)) << (sub * eff_bits));
            }
        }
    }

    // -------------------------------------------------------------------------
    // mse_score (mirrors metal/mse_score.py)
    // -------------------------------------------------------------------------
    void mse_score(const float* q_rot, const void* mse_packed,
                   const float* norms, const float* centroids,
                   int BH, int N, int D, int bits,
                   float* out) override {
        if (bits == 3) {
            const int n_words = (D + 9) / 10;
            const uint32_t* mse = static_cast<const uint32_t*>(mse_packed);
            for (int bh = 0; bh < BH; ++bh) {
                for (int n = 0; n < N; ++n) {
                    const uint32_t* row = mse + (static_cast<size_t>(bh) * N + n) * n_words;
                    double score = 0.0;
                    for (int j = 0; j < D; ++j) {
                        int word = j / 10;
                        int sub  = j % 10;
                        int idx  = static_cast<int>((row[word] >> (sub * 3)) & 0x7u);
                        score += static_cast<double>(q_rot[bh * D + j]) * centroids[idx];
                    }
                    out[bh * N + n] = static_cast<float>(score) * norms[bh * N + n];
                }
            }
            return;
        }

        int vals_per_byte;
        int eff_bits;
        int mask;
        switch (bits) {
            case 1: vals_per_byte = 8; eff_bits = 1; mask = 0x1; break;
            case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3; break;
            case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF; break;
            case 8: default:
                    vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
        }
        const int packed_d = (D + vals_per_byte - 1) / vals_per_byte;
        const uint8_t* mse = static_cast<const uint8_t*>(mse_packed);

        for (int bh = 0; bh < BH; ++bh) {
            for (int n = 0; n < N; ++n) {
                const uint8_t* row = mse + (static_cast<size_t>(bh) * N + n) * packed_d;
                double score = 0.0;
                for (int byte = 0; byte < packed_d; ++byte) {
                    uint8_t packed = row[byte];
                    for (int sub = 0; sub < vals_per_byte; ++sub) {
                        int coord = byte * vals_per_byte + sub;
                        if (coord >= D) break;
                        int idx = (packed >> (sub * eff_bits)) & mask;
                        score += static_cast<double>(q_rot[bh * D + coord]) * centroids[idx];
                    }
                }
                out[bh * N + n] = static_cast<float>(score) * norms[bh * N + n];
            }
        }
    }

    // -------------------------------------------------------------------------
    // qjl_score (mirrors metal/qjl_score.py)
    // -------------------------------------------------------------------------
    void qjl_score(const float* q_sketch, const uint8_t* signs,
                   const float* res_norms, const float* mse_in,
                   int BH, int N, int D,
                   float qjl_scale,
                   float* out) override {
        const int packed_d_signs = (D + 7) / 8;
        for (int bh = 0; bh < BH; ++bh) {
            for (int n = 0; n < N; ++n) {
                const uint8_t* row = signs + (static_cast<size_t>(bh) * N + n) * packed_d_signs;
                double dot = 0.0;
                for (int byte = 0; byte < packed_d_signs; ++byte) {
                    uint8_t packed = row[byte];
                    for (int bit = 0; bit < 8; ++bit) {
                        int coord = byte * 8 + bit;
                        if (coord >= D) break;
                        float sign_val = ((packed >> bit) & 1) ? 1.0f : -1.0f;
                        dot += static_cast<double>(q_sketch[bh * D + coord]) * sign_val;
                    }
                }
                float qjl_term = static_cast<float>(dot) * res_norms[bh * N + n] * qjl_scale;
                out[bh * N + n] = mse_in[bh * N + n] + qjl_term;
            }
        }
    }

    // -------------------------------------------------------------------------
    // value_dequant (mirrors metal/value_dequant.py)
    // -------------------------------------------------------------------------
    void value_dequant(const uint8_t* packed,
                       const float* scales, const float* zeros,
                       int N, int D, int bits, int group_size,
                       float* out) override {
        int vals_per_byte;
        int mask;
        int eff_bits;
        switch (bits) {
            case 2: vals_per_byte = 4; eff_bits = 2; mask = 0x3; break;
            case 4: vals_per_byte = 2; eff_bits = 4; mask = 0xF; break;
            case 8: default:
                    vals_per_byte = 1; eff_bits = 8; mask = 0xFF; break;
        }
        const int n_groups = D / group_size;
        const int packed_d = D / vals_per_byte;

        for (int v = 0; v < N; ++v) {
            const uint8_t* in = packed + static_cast<size_t>(v) * packed_d;
            const float*   sc = scales + static_cast<size_t>(v) * n_groups;
            const float*   ze = zeros  + static_cast<size_t>(v) * n_groups;
            float*         o  = out    + static_cast<size_t>(v) * D;
            for (int coord = 0; coord < D; ++coord) {
                int byte = coord / vals_per_byte;
                int sub  = coord % vals_per_byte;
                int qval = (in[byte] >> (sub * eff_bits)) & mask;
                int g    = coord / group_size;
                o[coord] = static_cast<float>(qval) * sc[g] + ze[g];
            }
        }
    }
};

}  // namespace

std::unique_ptr<IBackend> create_cpu_scalar_backend() {
    return std::make_unique<CpuScalarBackend>();
}

}  // namespace turboquant
