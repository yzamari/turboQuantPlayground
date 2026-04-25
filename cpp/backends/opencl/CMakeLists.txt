# Adreno OpenCL backend. We don't link against libOpenCL.so at build time;
# instead we dlopen it at runtime so the library still builds for hosts that
# have no OpenCL ICD.

# Embed each .cl source as a const byte array.
include(${CMAKE_SOURCE_DIR}/cmake/embed_resource.cmake)

set(_kernels_dir ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(_gen_dir     ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${_gen_dir})

set(_opencl_kernels mse_score qjl_score mse_encode value_dequant rotate)
set(_opencl_gen_srcs "")
foreach(k ${_opencl_kernels})
    set(_in  ${_kernels_dir}/${k}.cl)
    set(_out ${_gen_dir}/${k}_cl.c)
    embed_resource("${_in}" "${_out}" "tq_opencl_${k}")
    list(APPEND _opencl_gen_srcs ${_out})
endforeach()

add_library(tq_backend_opencl STATIC
    opencl_loader.cpp
    opencl_backend.cpp
    ${_opencl_gen_srcs}
)
target_include_directories(tq_backend_opencl
    PUBLIC  ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${_gen_dir}
            ${CMAKE_SOURCE_DIR}/src         # for packing.hpp if needed
)
target_compile_features(tq_backend_opencl PUBLIC cxx_std_17)

# We dlopen libOpenCL.so at runtime — only need libdl on Linux/Android.
if(ANDROID OR (UNIX AND NOT APPLE))
    target_link_libraries(tq_backend_opencl PRIVATE dl)
endif()
