# Native build for the TurboQuant + Llama Android app.
#
# Builds llama.cpp from the pinned submodule under external/llama.cpp/. The
# llama-mtmd-cli executable is post-built into ../jniLibs/${ANDROID_ABI}/ as
# libllama-mtmd-cli.so so Gradle packs it alongside the other shared libs and
# the assistant's Runtime.exec() path keeps working.
#
# Layout reminder:
#   <repo>/android/app/src/main/cpp/CMakeLists.txt   <-- this file
#   <repo>/android/app/src/main/jniLibs/arm64-v8a/   <-- packaged libs
#   <repo>/cpp/                                      <-- libturboquant
#   <repo>/external/llama.cpp/                       <-- vendored fork

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

# ---------------------------------------------------------------------------
# libturboquant core.
# ---------------------------------------------------------------------------

set(TQ_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(TQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TQ_WITH_CPU_SCALAR ON  CACHE BOOL "" FORCE)
set(TQ_WITH_NEON       ON  CACHE BOOL "" FORCE)
# OpenCL/Vulkan/QNN backends in cpp/backends/ reference CMAKE_SOURCE_DIR
# (which points at this JNI tree under add_subdirectory) — disabled in the
# assistant build. The standalone bench keeps cpu_scalar + cpu_neon.
set(TQ_WITH_OPENCL     OFF CACHE BOOL "" FORCE)
set(TQ_WITH_VULKAN     OFF CACHE BOOL "" FORCE)
set(TQ_WITH_QNN        OFF CACHE BOOL "" FORCE)

# Path: this CMakeLists -> ../../../../../ (repo root) -> cpp
add_subdirectory(${CMAKE_SOURCE_DIR}/../../../../../cpp turboquant_core)

# ---------------------------------------------------------------------------
# llama.cpp — built from the pinned submodule.
# ---------------------------------------------------------------------------

# We only want the libraries we actually link against. llama.cpp gates
# subdirs on LLAMA_BUILD_COMMON / LLAMA_BUILD_TOOLS, so we have to enable
# those to get common+mtmd. The other targets (server, examples, tests,
# webui) stay off.
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_WEBUI    OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL  OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL  OFF CACHE BOOL "" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    ON  CACHE BOOL "" FORCE)

# Adreno OpenCL backend (the 5.6× prompt-eval win comes from here).
set(GGML_OPENCL                    ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)

# find_package(OpenCL) doesn't see the NDK's libOpenCL — point it at our
# vendored Khronos headers + the on-device ICD stub shipped under jniLibs.
set(OpenCL_INCLUDE_DIR
    "${CMAKE_SOURCE_DIR}/../../../../../external/OpenCL-Headers"
    CACHE PATH "" FORCE)
set(OpenCL_LIBRARY
    "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libOpenCL.so"
    CACHE FILEPATH "" FORCE)

add_subdirectory(
    ${CMAKE_SOURCE_DIR}/../../../../../external/llama.cpp
    llamacpp_build)

# Repackage llama-mtmd-cli (executable) as libllama-mtmd-cli.so for the
# APK. Gradle's android plugin packages everything in jniLibs/${ABI}/,
# and the existing assistant Runtime.exec()s this filename.
# Using a custom target rather than POST_BUILD because the mtmd-cli
# target is defined in another CMakeLists (cross-directory).
if(TARGET llama-mtmd-cli)
    add_custom_target(repackage_mtmd_cli ALL
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:llama-mtmd-cli>
            ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama-mtmd-cli.so
        DEPENDS llama-mtmd-cli
        COMMENT "[turboquant_jni] Repackaging llama-mtmd-cli -> libllama-mtmd-cli.so")
endif()

# ---------------------------------------------------------------------------
# JNI shim — the only Android-specific C++ in the whole project.
# ---------------------------------------------------------------------------

add_library(turboquant_jni SHARED
    turboquant_jni.cpp
    llama_jni.cpp
    ${CMAKE_SOURCE_DIR}/../../../../../cpp/bench/baseline_kv_cache.cpp
)

target_include_directories(turboquant_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/../../../../../cpp/bench
)

# llama / ggml targets export their public headers via INTERFACE.
target_link_libraries(turboquant_jni
    PRIVATE
        turboquant
        llama
        ggml
        log
        android
)

set_target_properties(turboquant_jni PROPERTIES
    LINK_FLAGS_RELEASE "-Wl,--strip-all"
)
