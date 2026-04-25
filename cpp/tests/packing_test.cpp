// Bit-packing roundtrip tests. Verifies that every supported bit width packs
// and unpacks back to the original integer indices, and that QJL signs
// roundtrip through pack/unpack to ±1.0 floats.

#include "tq_test.hpp"

#include "packing.hpp"

#include <random>
#include <vector>

using namespace turboquant;

static void test_indices_roundtrip(int d, int bits) {
    std::mt19937 rng(0xBEEF + d * 100 + bits);
    const int n_levels = (1 << bits);
    std::uniform_int_distribution<int> dist(0, n_levels - 1);

    const int n_vec = 17;
    std::vector<int32_t> in(static_cast<size_t>(n_vec) * d);
    for (auto& x : in) x = dist(rng);

    std::vector<uint8_t> packed(static_cast<size_t>(n_vec) * packed_len_for(d, bits));
    pack_indices(in.data(), n_vec, d, bits, packed.data());

    std::vector<int32_t> out(static_cast<size_t>(n_vec) * d);
    unpack_indices(packed.data(), n_vec, d, bits, out.data());

    for (size_t i = 0; i < in.size(); ++i) {
        TQ_CHECK_EQ(in[i], out[i]);
    }
}

static void test_qjl_signs_roundtrip(int d) {
    std::mt19937 rng(0xC0FFEE + d);
    const int n_vec = 11;
    std::vector<float> in(static_cast<size_t>(n_vec) * d);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& x : in) x = dist(rng);

    std::vector<uint8_t> packed(static_cast<size_t>(n_vec) * packed_signs_len(d));
    pack_qjl_signs(in.data(), n_vec, d, packed.data());

    std::vector<float> out(static_cast<size_t>(n_vec) * d);
    unpack_qjl_signs_to_float(packed.data(), n_vec, d, out.data());

    for (int v = 0; v < n_vec; ++v) {
        for (int j = 0; j < d; ++j) {
            float expected = (in[v * d + j] > 0.f) ? 1.0f : -1.0f;
            TQ_CHECK_EQ(static_cast<int>(out[v * d + j]), static_cast<int>(expected));
        }
    }
}

int main() {
    for (int d : {64, 128, 130 /* odd D, exercises padding */}) {
        for (int bits : {1, 2, 3, 4, 8}) {
            test_indices_roundtrip(d, bits);
        }
        test_qjl_signs_roundtrip(d);
    }
    return tq_test::report_and_exit();
}
