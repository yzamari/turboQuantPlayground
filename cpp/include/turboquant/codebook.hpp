// Lloyd-Max codebook lookup. Codebooks ship as embedded byte arrays compiled
// in by CMake from src/turboquant_mac/codebooks/*.json. No filesystem access
// at runtime — required for automotive deployment.

#pragma once

#include <vector>

namespace turboquant {

struct Codebook {
    int                d = 0;
    int                bits = 0;
    std::vector<float> centroids;          // size = 2^bits
    std::vector<float> boundaries;         // size = 2^bits + 1, includes ±1.0
    std::vector<float> decision_boundaries;// size = 2^bits - 1, interior only
    float              mse_per_coord = 0.f;
    float              mse_total     = 0.f;
};

// Returns nullptr if (d, bits) was not embedded at build time.
const Codebook* get_codebook(int d, int bits);

// For diagnostics — list of (d, bits) pairs available.
struct CodebookKey { int d; int bits; };
std::vector<CodebookKey> available_codebooks();

}  // namespace turboquant
