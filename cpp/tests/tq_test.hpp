// Tiny test helper — no external test framework. Keeps the build
// dependency-free for automotive toolchains.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace tq_test {

inline int& failures() { static int n = 0; return n; }
inline int& checks()   { static int n = 0; return n; }

#define TQ_CHECK(expr)                                                          \
    do {                                                                         \
        ::tq_test::checks()++;                                                   \
        if (!(expr)) {                                                           \
            ::tq_test::failures()++;                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        }                                                                        \
    } while (0)

#define TQ_CHECK_EQ(a, b)                                                       \
    do {                                                                         \
        ::tq_test::checks()++;                                                   \
        auto va_ = (a); auto vb_ = (b);                                          \
        if (!(va_ == vb_)) {                                                     \
            ::tq_test::failures()++;                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s  (got %lld vs %lld)\n",   \
                         __FILE__, __LINE__, #a, #b,                             \
                         static_cast<long long>(va_),                            \
                         static_cast<long long>(vb_));                           \
        }                                                                        \
    } while (0)

#define TQ_CHECK_NEAR(a, b, eps)                                                \
    do {                                                                         \
        ::tq_test::checks()++;                                                   \
        double va_ = (a); double vb_ = (b); double e_ = (eps);                   \
        if (std::fabs(va_ - vb_) > e_) {                                         \
            ::tq_test::failures()++;                                             \
            std::fprintf(stderr,                                                 \
                "FAIL %s:%d: %s ~ %s  (|%g - %g| = %g > %g)\n",                  \
                __FILE__, __LINE__, #a, #b, va_, vb_, std::fabs(va_-vb_), e_);   \
        }                                                                        \
    } while (0)

inline int report_and_exit() {
    std::fprintf(stderr, "%d / %d checks passed (%d failures)\n",
                 checks() - failures(), checks(), failures());
    return failures() == 0 ? 0 : 1;
}

}  // namespace tq_test
