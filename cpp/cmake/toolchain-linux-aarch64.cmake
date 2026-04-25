# Linux aarch64 toolchain — placeholder for Snapdragon automotive Linux
# (e.g. SA8775P running Linux). Activated when a cross-compiler is on PATH.
#
# Usage:
#   cmake -S cpp -B build-linux-arm64 \
#         -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-linux-aarch64.cmake \
#         -DTQ_WITH_NEON=ON -DTQ_WITH_OPENCL=ON

set(CMAKE_SYSTEM_NAME       Linux)
set(CMAKE_SYSTEM_PROCESSOR  aarch64)

if(NOT DEFINED CMAKE_C_COMPILER)
    find_program(_aarch64_gcc NAMES aarch64-linux-gnu-gcc)
    if(_aarch64_gcc)
        set(CMAKE_C_COMPILER   ${_aarch64_gcc})
        set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    else()
        message(FATAL_ERROR
            "aarch64-linux-gnu-gcc not found on PATH. Install the cross-compiler "
            "or set CMAKE_C_COMPILER / CMAKE_CXX_COMPILER explicitly.")
    endif()
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
