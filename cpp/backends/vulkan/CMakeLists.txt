# Adreno Vulkan compute backend.
#
# Each .comp shader is compiled to SPIR-V at build time via glslc, then the
# resulting .spv is embedded into the static library as a const byte array
# using the shared embed_resource helper.
#
# Build is gated by -DTQ_WITH_VULKAN=ON in the parent CMakeLists.

find_package(Vulkan REQUIRED)
find_program(GLSLC NAMES glslc REQUIRED
    HINTS /opt/homebrew/bin /usr/local/bin
          $ENV{VULKAN_SDK}/bin
          ${ANDROID_NDK}/shader-tools/darwin-x86_64
          ${ANDROID_NDK}/shader-tools/linux-x86_64)

include(${CMAKE_SOURCE_DIR}/cmake/embed_resource.cmake)

set(_shaders_dir ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(_gen_dir     ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${_gen_dir})

set(_vulkan_shaders
    mse_score_uchar
    mse_score_u32_b3
    qjl_score
    mse_encode_uchar
    mse_encode_u32_b3
    value_dequant
    rotate
)

set(_vulkan_gen_srcs "")
foreach(s ${_vulkan_shaders})
    set(_in   ${_shaders_dir}/${s}.comp)
    set(_spv  ${_gen_dir}/${s}.spv)
    set(_out  ${_gen_dir}/${s}_spv.c)
    set(_sym  tq_vulkan_${s})

    # Step 1: glslc -> SPIR-V (re-runs only if .comp changes).
    add_custom_command(
        OUTPUT  ${_spv}
        COMMAND ${GLSLC} -O --target-env=vulkan1.1 -fshader-stage=compute
                ${_in} -o ${_spv}
        DEPENDS ${_in}
        COMMENT "Compiling Vulkan compute shader ${s}.comp -> ${s}.spv"
        VERBATIM
    )

    # Step 2: embed .spv as a const unsigned char[] symbol named tq_vulkan_<s>.
    # We can't use the embed_resource() function directly here because the
    # input file (.spv) only exists after the custom command above runs. Instead
    # generate the .c via a custom command that depends on the .spv and on the
    # embed_resource cmake script.
    add_custom_command(
        OUTPUT  ${_out}
        COMMAND ${CMAKE_COMMAND}
                -DEMBED_INPUT=${_spv}
                -DEMBED_OUTPUT=${_out}
                -DEMBED_SYMBOL=${_sym}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_runner.cmake
        DEPENDS ${_spv} ${CMAKE_CURRENT_SOURCE_DIR}/embed_runner.cmake
                ${CMAKE_SOURCE_DIR}/cmake/embed_resource.cmake
        COMMENT "Embedding ${s}.spv -> ${s}_spv.c (symbol ${_sym})"
        VERBATIM
    )

    list(APPEND _vulkan_gen_srcs ${_out})
endforeach()

add_library(tq_backend_vulkan STATIC
    vulkan_backend.cpp
    ${_vulkan_gen_srcs}
)

target_include_directories(tq_backend_vulkan
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${_gen_dir}
)
target_compile_features(tq_backend_vulkan PUBLIC cxx_std_17)

target_link_libraries(tq_backend_vulkan PRIVATE Vulkan::Vulkan)

if(ANDROID OR (UNIX AND NOT APPLE))
    target_link_libraries(tq_backend_vulkan PRIVATE dl)
endif()
