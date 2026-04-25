// Bit-packing reference implementation.
//
// Layout for MSE indices (bits parameter is the per-element bit width):
//   bits == 1  : 8 vals/byte, LSB = element 0          (packed_d = ceil(d/8))
//   bits == 2  : 4 vals/byte, LSB = element 0          (packed_d = ceil(d/4))
//   bits == 3  : 10 vals per uint32 little-endian word (packed_len_u32 =
//                ceil(d/10), byte buffer is 4× that)
//   bits == 4  : 2 vals/byte, LSB = element 0          (packed_d = ceil(d/2))
//   bits == 8  : 1 val /byte                            (packed_d = d)
//
// QJL signs: 1 bit per coord, 8 per byte LSB-first; bit=1 -> +1.0, bit=0 -> -1.0.

#include "packing.hpp"

#include <cassert>
#include <cstring>

namespace turboquant {

int packed_len_for(int d, int bits) {
    switch (bits) {
        case 1: return (d + 7) / 8;
        case 2: return (d + 3) / 4;
        case 3: {
            int n_words = (d + 9) / 10;
            return n_words * 4;  // bytes
        }
        case 4: return (d + 1) / 2;
        case 8: return d;
        default: return d;       // 8-bit fallback for unsupported widths
    }
}

int packed_signs_len(int d) {
    return (d + 7) / 8;
}

void pack_indices(const int32_t* idx, int n_vec, int d, int bits, uint8_t* out) {
    const int plen = packed_len_for(d, bits);
    for (int v = 0; v < n_vec; ++v) {
        const int32_t* in = idx + static_cast<size_t>(v) * d;
        uint8_t* o = out + static_cast<size_t>(v) * plen;
        std::memset(o, 0, plen);

        if (bits == 8) {
            for (int j = 0; j < d; ++j) o[j] = static_cast<uint8_t>(in[j] & 0xFF);
        } else if (bits == 1) {
            for (int j = 0; j < d; ++j) {
                if (in[j] & 1) o[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
            }
        } else if (bits == 2) {
            for (int j = 0; j < d; ++j) {
                int byte = j >> 2;
                int sub  = j & 3;
                o[byte] |= static_cast<uint8_t>((in[j] & 0x3) << (sub * 2));
            }
        } else if (bits == 4) {
            for (int j = 0; j < d; ++j) {
                int byte = j >> 1;
                int sub  = j & 1;
                o[byte] |= static_cast<uint8_t>((in[j] & 0xF) << (sub * 4));
            }
        } else if (bits == 3) {
            // 10 vals per uint32, little-endian within the word.
            uint32_t* ow = reinterpret_cast<uint32_t*>(o);
            const int n_words = plen / 4;
            for (int w = 0; w < n_words; ++w) ow[w] = 0;
            for (int j = 0; j < d; ++j) {
                int word = j / 10;
                int sub  = j % 10;
                ow[word] |= (static_cast<uint32_t>(in[j]) & 0x7u) << (sub * 3);
            }
        }
    }
}

void unpack_indices(const uint8_t* packed, int n_vec, int d, int bits, int32_t* out) {
    const int plen = packed_len_for(d, bits);
    for (int v = 0; v < n_vec; ++v) {
        const uint8_t* in = packed + static_cast<size_t>(v) * plen;
        int32_t* o        = out + static_cast<size_t>(v) * d;

        if (bits == 8) {
            for (int j = 0; j < d; ++j) o[j] = in[j];
        } else if (bits == 1) {
            for (int j = 0; j < d; ++j) o[j] = (in[j >> 3] >> (j & 7)) & 0x1;
        } else if (bits == 2) {
            for (int j = 0; j < d; ++j) {
                int byte = j >> 2;
                int sub  = j & 3;
                o[j] = (in[byte] >> (sub * 2)) & 0x3;
            }
        } else if (bits == 4) {
            for (int j = 0; j < d; ++j) {
                int byte = j >> 1;
                int sub  = j & 1;
                o[j] = (in[byte] >> (sub * 4)) & 0xF;
            }
        } else if (bits == 3) {
            const uint32_t* iw = reinterpret_cast<const uint32_t*>(in);
            for (int j = 0; j < d; ++j) {
                int word = j / 10;
                int sub  = j % 10;
                o[j] = static_cast<int32_t>((iw[word] >> (sub * 3)) & 0x7u);
            }
        }
    }
}

void pack_qjl_signs(const float* projected, int n_vec, int d, uint8_t* out) {
    const int plen = packed_signs_len(d);
    for (int v = 0; v < n_vec; ++v) {
        const float* in = projected + static_cast<size_t>(v) * d;
        uint8_t* o      = out + static_cast<size_t>(v) * plen;
        for (int b = 0; b < plen; ++b) o[b] = 0;
        for (int j = 0; j < d; ++j) {
            if (in[j] > 0.f) {
                o[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
            }
        }
    }
}

void unpack_qjl_signs_to_float(const uint8_t* packed, int n_vec, int d, float* out) {
    const int plen = packed_signs_len(d);
    for (int v = 0; v < n_vec; ++v) {
        const uint8_t* in = packed + static_cast<size_t>(v) * plen;
        float* o          = out + static_cast<size_t>(v) * d;
        for (int j = 0; j < d; ++j) {
            int bit = (in[j >> 3] >> (j & 7)) & 0x1;
            o[j] = bit ? 1.0f : -1.0f;
        }
    }
}

}  // namespace turboquant
