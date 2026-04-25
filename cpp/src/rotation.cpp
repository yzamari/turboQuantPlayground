#include "turboquant/api.hpp"

#include <cmath>
#include <random>

namespace turboquant {

namespace {

// Box–Muller normal sample using a std::mt19937_64 PRNG. We do NOT use
// std::normal_distribution because its implementation is unspecified across
// libc++ vs libstdc++; this gives us byte-exact reproducibility across hosts.
struct Mt19937Normal {
    std::mt19937_64 rng;
    bool   has_spare = false;
    double spare     = 0.0;

    explicit Mt19937Normal(uint64_t seed) : rng(seed) {}

    double next() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        // Two uniform doubles in (0, 1).
        constexpr double kInv = 1.0 / 18446744073709551616.0;  // 2^-64
        double u1, u2;
        do {
            u1 = (static_cast<double>(rng()) + 1.0) * kInv;
        } while (u1 <= 0.0);
        u2 = static_cast<double>(rng()) * kInv;

        double r     = std::sqrt(-2.0 * std::log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        double s     = std::sin(theta);
        double c     = std::cos(theta);
        spare        = r * s;
        has_spare    = true;
        return r * c;
    }
};

// In-place QR via modified Gram–Schmidt. Input A is row-major [m, n] (m == n
// for our use); output overwrites A so its rows form an orthonormal basis,
// and writes diag of R into r_diag (length n). Returns Q (= modified A) by
// reference. This is sufficient for generating an orthogonal Pi; we don't
// need the full R.
void mgs_qr(float* A, int n, float* r_diag) {
    // Treat rows as the vectors; the resulting matrix is orthogonal both ways
    // when iterated through completion (since transpose of orthogonal is
    // orthogonal).
    for (int k = 0; k < n; ++k) {
        // Compute the norm of row k.
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += static_cast<double>(A[k * n + j]) * A[k * n + j];
        }
        double norm = std::sqrt(sum);
        if (norm < 1e-30) norm = 1e-30;
        r_diag[k] = static_cast<float>(norm);

        float inv = static_cast<float>(1.0 / norm);
        for (int j = 0; j < n; ++j) A[k * n + j] *= inv;

        // Subtract projections from later rows.
        for (int i = k + 1; i < n; ++i) {
            double dot = 0.0;
            for (int j = 0; j < n; ++j) {
                dot += static_cast<double>(A[k * n + j]) * A[i * n + j];
            }
            float dotf = static_cast<float>(dot);
            for (int j = 0; j < n; ++j) {
                A[i * n + j] -= dotf * A[k * n + j];
            }
        }
    }
}

}  // namespace

std::vector<float> generate_pi_qr(int d, uint64_t seed) {
    std::vector<float> A(static_cast<size_t>(d) * d);
    Mt19937Normal rng(seed);
    for (auto& v : A) v = static_cast<float>(rng.next());

    std::vector<float> r_diag(d);
    mgs_qr(A.data(), d, r_diag.data());

    // Apply diag(sign(r_diag)) to columns to mirror the numpy reference's
    // convention (Q = Q * sign(diag(R))). After mgs_qr the rows of A form Q.
    // To apply on the column side of Q we'd flip signs of A's columns; we
    // instead flip signs of A's rows (equivalent to choosing a different
    // chirality — Pi is still orthogonal). Doesn't change correctness.
    for (int k = 0; k < d; ++k) {
        if (r_diag[k] < 0.f) {
            for (int j = 0; j < d; ++j) A[k * d + j] = -A[k * d + j];
        }
    }
    return A;
}

std::vector<float> generate_qjl_S(int d, uint64_t seed) {
    std::vector<float> S(static_cast<size_t>(d) * d);
    Mt19937Normal rng(seed);
    for (auto& v : S) v = static_cast<float>(rng.next());
    return S;
}

}  // namespace turboquant
