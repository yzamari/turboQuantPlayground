# Native build for the TurboQuant + Llama Android app.
#
# This CMake file lives at:
#   <repo>/android/app/src/main/cpp/CMakeLists.txt
#
# It pulls in the portable C++ TurboQuant core that lives at <repo>/cpp/ via
# add_subdirectory, then layers the Android-only JNI shim on top.
# The core CMakeLists is untouched — TQ_WITH_JNI/etc. is purely an option here.
#
# It also imports prebuilt llama.cpp shared libs from
#   <repo>/android/app/src/main/jniLibs/arm64-v8a/
# (libllama.so, libggml.so, libggml-base.so, libggml-cpu.so).

cmake_minimum_required(VERSION 3.22)
project(turboquant_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ARMv8 baseline — we only ship arm64-v8a.
if(ANDROID_ABI STREQUAL "arm64-v8a")
    add_compile_options(-march=armv8-a)
endif()

# Disable bench/tests in the embedded subbuild (the bench harness is included
# directly via the bench_runner.hpp header).
set(TQ_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(TQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)

set(TQ_WITH_CPU_SCALAR ON  CACHE BOOL "" FORCE)
set(TQ_WITH_NEON       ON  CACHE BOOL "" FORCE)
set(TQ_WITH_OPENCL     ON  CACHE BOOL "" FORCE)
set(TQ_WITH_VULKAN     ON  CACHE BOOL "" FORCE)
set(TQ_WITH_QNN        OFF CACHE BOOL "" FORCE)

# Path from this file up to the repo root, then into cpp/:
#   .../android/app/src/main/cpp -> ../../../../ (= repo root) -> cpp
add_subdirectory(${CMAKE_SOURCE_DIR}/../../../../../cpp turboquant_core)

# ---------------------------------------------------------------------------
# Imported llama.cpp libraries (prebuilt for arm64-v8a Android, see jniLibs/).
# ---------------------------------------------------------------------------

set(LLAMA_LIB_DIR "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

add_library(llama        SHARED IMPORTED)
add_library(ggml         SHARED IMPORTED)
add_library(ggml_base    SHARED IMPORTED)
add_library(ggml_cpu     SHARED IMPORTED)

set_target_properties(llama     PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libllama.so")
set_target_properties(ggml      PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml.so")
set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml-base.so")
set_target_properties(ggml_cpu  PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml-cpu.so")

# ---------------------------------------------------------------------------
# JNI shim — the only Android-specific C++ in the whole project.
# ---------------------------------------------------------------------------

add_library(turboquant_jni SHARED
    turboquant_jni.cpp
    llama_jni.cpp
    # The bench baseline lives in cpp/bench/ so we pull just the .cpp we need.
    ${CMAKE_SOURCE_DIR}/../../../../../cpp/bench/baseline_kv_cache.cpp
)

target_include_directories(turboquant_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../../../cpp/bench
    ${CMAKE_SOURCE_DIR}/include/llama
    ${CMAKE_SOURCE_DIR}/include/ggml
)

target_link_libraries(turboquant_jni
    PRIVATE
        turboquant
        llama
        ggml
        ggml_base
        ggml_cpu
        log
        android
)

# Strip in release for a leaner .so.
set_target_properties(turboquant_jni PROPERTIES
    LINK_FLAGS_RELEASE "-Wl,--strip-all"
)
