# Native build for the TurboQuant + Llama Android app.
#
# Two link paths controlled by BUILD_LLAMA_FROM_SOURCE:
#   OFF (default): import prebuilt llama+ggml shared libs from
#                  ../jniLibs/${ANDROID_ABI}/. Faster iteration; matches
#                  the historical workflow.
#   ON:            build llama.cpp from the pinned submodule under
#                  external/llama.cpp/. This is Path 2's foundation —
#                  required to patch the KV cache later.
#
# When ON, the llama-mtmd-cli executable produced by llama.cpp is copied
# into ../jniLibs/${ANDROID_ABI}/libllama-mtmd-cli.so so Gradle packs it
# into the APK alongside the other shared libs. (Existing chat path
# Runtime.exec()s this binary; preserved unchanged.)
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

option(BUILD_LLAMA_FROM_SOURCE
       "Build llama.cpp from external/llama.cpp submodule (Path 2 foundation)"
       OFF)

# ---------------------------------------------------------------------------
# libturboquant core (always built, unchanged from prebuilt path).
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
# llama.cpp / ggml libraries — from-source OR prebuilt.
# ---------------------------------------------------------------------------

if(BUILD_LLAMA_FROM_SOURCE)
    message(STATUS "[turboquant_jni] Building llama.cpp from source (Path 2)")

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
else()
    message(STATUS "[turboquant_jni] Using prebuilt llama+ggml from jniLibs/")
    set(LLAMA_LIB_DIR "${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}")

    add_library(llama        SHARED IMPORTED)
    add_library(ggml         SHARED IMPORTED)
    add_library(ggml_base    SHARED IMPORTED)
    add_library(ggml_cpu     SHARED IMPORTED)

    set_target_properties(llama     PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libllama.so")
    set_target_properties(ggml      PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml.so")
    set_target_properties(ggml_base PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml-base.so")
    set_target_properties(ggml_cpu  PROPERTIES IMPORTED_LOCATION "${LLAMA_LIB_DIR}/libggml-cpu.so")
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

if(BUILD_LLAMA_FROM_SOURCE)
    # llama / ggml targets export their public headers via INTERFACE.
    target_link_libraries(turboquant_jni
        PRIVATE
            turboquant
            llama
            ggml
            log
            android
    )
else()
    # Prebuilt libs ship without CMake-exported include paths; we vendor a
    # snapshot of the public headers under cpp/include/.
    target_include_directories(turboquant_jni PRIVATE
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
endif()

set_target_properties(turboquant_jni PROPERTIES
    LINK_FLAGS_RELEASE "-Wl,--strip-all"
)
