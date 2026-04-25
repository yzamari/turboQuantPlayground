# Qualcomm QNN / HTP NPU backend.
#
# This subproject is *gated* on the Qualcomm AI Engine Direct SDK ("QNN SDK"),
# which we cannot vendor: it requires a Qualcomm developer account and license
# acceptance. The expected layout is the standard `qairt/2.27.x` tree shipped
# by Qualcomm — i.e. the directory that *contains* `include/QNN/QnnInterface.h`,
# `lib/aarch64-android/libQnnHtp.so`, etc.
#
# Two build modes:
#
#   1. SDK NOT FOUND  -> graceful skip. We define an INTERFACE library with the
#      same target name so the rest of the tree links cleanly and CMake prints
#      a helpful message pointing at backends/qnn_htp/README.md.
#
#   2. SDK FOUND      -> build a STATIC library that dlopens libQnnHtp.so /
#      libQnnSystem.so at runtime and runs a hybrid pipeline (rotate +
#      value_dequant on the NPU; everything else delegated to the composed
#      CPU NEON / scalar backend).
#
# Activated via `-DTQ_WITH_QNN=ON -DQNN_SDK_ROOT=/path/to/qairt/2.27.x` or by
# exporting `QNN_SDK_ROOT` in the environment before invoking cmake.

# Resolve QNN_SDK_ROOT from cmake var, env var, or fail-soft.
if(NOT DEFINED QNN_SDK_ROOT AND DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT "$ENV{QNN_SDK_ROOT}" CACHE PATH "Qualcomm AI Engine Direct (QNN) SDK root")
endif()

if(NOT DEFINED QNN_SDK_ROOT OR QNN_SDK_ROOT STREQUAL "")
    message(STATUS "QNN_SDK_ROOT not set; skipping qnn_htp backend (instructions in cpp/backends/qnn_htp/README.md)")
    add_library(tq_backend_qnn_htp INTERFACE)
    return()
endif()

# Sanity-check the SDK layout. We don't hard-fail here; the user might have
# a slightly differently named SDK drop. We just warn and try to compile.
if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
    message(WARNING
        "QNN_SDK_ROOT='${QNN_SDK_ROOT}' does not contain include/QNN/QnnInterface.h. "
        "Continuing anyway, but compilation will likely fail. "
        "See cpp/backends/qnn_htp/README.md for the expected layout.")
endif()

message(STATUS "QNN HTP backend enabled: QNN_SDK_ROOT=${QNN_SDK_ROOT}")

add_library(tq_backend_qnn_htp STATIC
    qnn_loader.cpp
    qnn_graph.cpp
    qnn_htp_backend.cpp
)

target_include_directories(tq_backend_qnn_htp
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${QNN_SDK_ROOT}/include/QNN
            ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(tq_backend_qnn_htp PUBLIC cxx_std_17)

# QNN headers use designated initializers + variadic macros — same flags as
# Qualcomm's own SampleApp:
target_compile_definitions(tq_backend_qnn_htp PRIVATE
    QNN_API=
    TQ_QNN_SDK_ROOT="${QNN_SDK_ROOT}"
)

# dlopen lives in libdl on Linux/Android; bionic (Android) folds it into libc
# but linking against -ldl is still safe and explicit.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(tq_backend_qnn_htp PUBLIC dl)
endif()

# CPU fallback composition. The QNN backend forwards `mse_score`, `qjl_score`,
# and `mse_encode` to a composed CPU backend (NEON if available, else scalar).
# We export a compile def so the C++ source picks the right factory at build
# time without needing a runtime registry.
if(TQ_WITH_NEON AND TARGET tq_backend_cpu_neon)
    target_link_libraries(tq_backend_qnn_htp PRIVATE tq_backend_cpu_neon)
    target_compile_definitions(tq_backend_qnn_htp PRIVATE TQ_QNN_FALLBACK_NEON=1)
elseif(TQ_WITH_CPU_SCALAR AND TARGET tq_backend_cpu_scalar)
    target_link_libraries(tq_backend_qnn_htp PRIVATE tq_backend_cpu_scalar)
    target_compile_definitions(tq_backend_qnn_htp PRIVATE TQ_QNN_FALLBACK_SCALAR=1)
else()
    message(FATAL_ERROR
        "tq_backend_qnn_htp requires either TQ_WITH_NEON=ON (preferred) or "
        "TQ_WITH_CPU_SCALAR=ON to provide the CPU fallback for kernels not "
        "yet built as QNN graphs. Enable one and reconfigure.")
endif()
