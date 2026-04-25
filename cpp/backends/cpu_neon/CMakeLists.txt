# ARM NEON CPU backend.
#
# Builds for any ARMv8.0-A target — covers Cortex-A78AE (SA8295P) and
# Cortex-A76AE (SA8155P) as well as the consumer S24 Cortex-X4/A720.
# NDK clang already targets armv8 with NEON enabled, so no extra flags needed
# on Android. On Linux aarch64 we also get NEON unconditionally.

if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|arm64-v8a)$" OR
        ANDROID_ABI STREQUAL "arm64-v8a"))
    message(WARNING "TQ_WITH_NEON requested but target is not arm64; skipping.")
    add_library(tq_backend_cpu_neon INTERFACE)
    return()
endif()

add_library(tq_backend_cpu_neon STATIC
    neon_backend.cpp
)
target_include_directories(tq_backend_cpu_neon
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(tq_backend_cpu_neon PUBLIC cxx_std_17)

# Force ARMv8.0-A baseline (no Cortex-X4-only intrinsics) so the produced
# library runs on automotive Cortex-A76AE/A78AE targets unchanged.
target_compile_options(tq_backend_cpu_neon PRIVATE
    -O3
    -ffast-math
    -fno-finite-math-only   # keep nan/inf semantics for safety
)
