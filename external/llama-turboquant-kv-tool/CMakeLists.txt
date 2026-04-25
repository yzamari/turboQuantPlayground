# llama-turboquant-kv — proof-of-concept demo binary that integrates the
# TurboQuant KV-cache compression library with llama.cpp.
#
# Builds when TURBOQUANT_KV_DIR points at the turboQuantPlayground/cpp
# source tree. We intentionally use add_subdirectory so we always build
# turboquant from source against the same toolchain (especially important
# for the Android NDK cross-build).
#
# Usage from the parent CMake configure step:
#   -DLLAMA_TOOL_TURBOQUANT_KV=ON
#   -DTURBOQUANT_KV_DIR=/Users/yahavzamari/Projects/GitHub/turboQuantPlayground/cpp

if(NOT DEFINED TURBOQUANT_KV_DIR)
    set(TURBOQUANT_KV_DIR
        "/Users/yahavzamari/Projects/GitHub/turboQuantPlayground/cpp"
        CACHE PATH "Path to turboQuantPlayground/cpp")
endif()

if(NOT EXISTS "${TURBOQUANT_KV_DIR}/CMakeLists.txt")
    message(WARNING
        "TURBOQUANT_KV_DIR='${TURBOQUANT_KV_DIR}' does not point at a valid "
        "turboQuantPlayground/cpp checkout — skipping llama-turboquant-kv.")
    return()
endif()

# Build the turboquant static library inline. We turn off its tests/bench so
# we don't drag in extra targets into the llama.cpp build tree.
set(TQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TQ_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(TQ_WITH_CPU_SCALAR ON CACHE BOOL "" FORCE)

add_subdirectory(${TURBOQUANT_KV_DIR}
                 ${CMAKE_BINARY_DIR}/turboquant_external)

# The turboquant cpu_scalar backend hardcodes its include path as
# ${CMAKE_SOURCE_DIR}/include, which inside this nested build resolves to
# /tmp/llama.cpp/include — wrong. Patch it back to the real turboquant
# headers without modifying the read-only cpp/ tree.
if(TARGET tq_backend_cpu_scalar)
    target_include_directories(tq_backend_cpu_scalar
        PUBLIC "${TURBOQUANT_KV_DIR}/include")
endif()

set(TARGET llama-turboquant-kv)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama turboquant
                                        ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
