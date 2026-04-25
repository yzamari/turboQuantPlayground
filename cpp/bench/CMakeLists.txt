add_executable(turboquant_bench
    bench_cli.cpp
    baseline_kv_cache.cpp
)
target_link_libraries(turboquant_bench PRIVATE turboquant)
target_compile_features(turboquant_bench PRIVATE cxx_std_17)
