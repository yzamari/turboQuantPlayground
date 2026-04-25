# QNX aarch64 toolchain — stub for Snapdragon automotive QNX targets
# (SA8155P, SA8295P, etc. running QNX Neutrino).
#
# Activated when QNX_HOST and QNX_TARGET environment variables are set
# (standard QNX SDP layout). Currently a placeholder — fill in QCC paths once
# we have access to a QNX SDP install.

if(NOT DEFINED ENV{QNX_HOST} OR NOT DEFINED ENV{QNX_TARGET})
    message(FATAL_ERROR
        "QNX SDP not detected. Source qnxsdp-env.sh from your QNX SDP install "
        "to set QNX_HOST and QNX_TARGET, then re-run cmake.")
endif()

set(CMAKE_SYSTEM_NAME       QNX)
set(CMAKE_SYSTEM_PROCESSOR  aarch64)
set(CMAKE_SYSTEM_VERSION    7.1)

set(CMAKE_C_COMPILER        $ENV{QNX_HOST}/usr/bin/qcc)
set(CMAKE_CXX_COMPILER      $ENV{QNX_HOST}/usr/bin/q++)
set(CMAKE_C_COMPILER_TARGET   gcc_ntoaarch64le)
set(CMAKE_CXX_COMPILER_TARGET gcc_ntoaarch64le_cxx)

set(CMAKE_FIND_ROOT_PATH "$ENV{QNX_TARGET}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
