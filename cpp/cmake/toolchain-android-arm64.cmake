# Android arm64-v8a toolchain wrapper.
#
# Wraps the NDK's bundled android.toolchain.cmake so we can pin the API level
# and ABI in one place and keep the invocation short:
#
#   cmake -S cpp -B build-android \
#         -DCMAKE_TOOLCHAIN_FILE=cpp/cmake/toolchain-android-arm64.cmake \
#         -DTQ_WITH_NEON=ON
#
# Set ANDROID_NDK in your environment to point at the NDK root (or override
# the default below). NDK 26+ recommended.

if(NOT DEFINED ANDROID_NDK)
    if(DEFINED ENV{ANDROID_NDK_HOME})
        set(ANDROID_NDK "$ENV{ANDROID_NDK_HOME}")
    elseif(DEFINED ENV{ANDROID_NDK})
        set(ANDROID_NDK "$ENV{ANDROID_NDK}")
    elseif(EXISTS "$ENV{HOME}/Library/Android/sdk/ndk")
        # Pick the highest-versioned NDK in ~/Library/Android/sdk/ndk
        file(GLOB _ndks LIST_DIRECTORIES true "$ENV{HOME}/Library/Android/sdk/ndk/*")
        list(SORT _ndks)
        list(REVERSE _ndks)
        list(GET _ndks 0 ANDROID_NDK)
    endif()
endif()

if(NOT DEFINED ANDROID_NDK OR NOT EXISTS "${ANDROID_NDK}")
    message(FATAL_ERROR
        "Android NDK not found. Set ANDROID_NDK_HOME or ANDROID_NDK env var, "
        "or install the NDK into ~/Library/Android/sdk/ndk/.")
endif()

set(ANDROID_ABI         arm64-v8a CACHE STRING "Android ABI"        FORCE)
set(ANDROID_PLATFORM    android-29 CACHE STRING "Android API level" FORCE)
set(ANDROID_STL         c++_static CACHE STRING "C++ STL"           FORCE)

include("${ANDROID_NDK}/build/cmake/android.toolchain.cmake")
