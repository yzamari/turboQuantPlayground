# Pure C++17 scalar reference backend.
# Builds on every platform (host CI, Android, Linux, QNX) — used as the
# numerical reference and as a fallback when no accelerator is available.
add_library(tq_backend_cpu_scalar STATIC
    scalar_backend.cpp
)
target_include_directories(tq_backend_cpu_scalar
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(tq_backend_cpu_scalar PUBLIC cxx_std_17)
