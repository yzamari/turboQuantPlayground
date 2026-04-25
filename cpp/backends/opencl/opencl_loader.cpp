#include "opencl_loader.hpp"

#include <cstdio>

#if defined(__ANDROID__) || defined(__linux__)
#include <dlfcn.h>
#endif

namespace turboquant {

#if defined(__ANDROID__) || defined(__linux__)

namespace {
const char* kCandidates[] = {
    "libOpenCL.so",
    "/system/vendor/lib64/libOpenCL.so",
    "/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    "/system/vendor/lib/libOpenCL.so",
    "/vendor/lib/libOpenCL.so",
    "libOpenCL.so.1",
};
}  // namespace

bool load_opencl(OpenCLApi* api) {
    for (const char* path : kCandidates) {
        api->dl_handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
        if (api->dl_handle) break;
    }
    if (!api->dl_handle) {
        std::fprintf(stderr, "[opencl] could not dlopen libOpenCL.so: %s\n", dlerror());
        return false;
    }

#define _RESOLVE(field, sym)                                                   \
    api->field = reinterpret_cast<decltype(api->field)>(dlsym(api->dl_handle, sym)); \
    if (!api->field) {                                                          \
        std::fprintf(stderr, "[opencl] missing symbol: %s\n", sym);             \
        unload_opencl(api);                                                     \
        return false;                                                           \
    }
    _RESOLVE(GetPlatformIDs,           "clGetPlatformIDs");
    _RESOLVE(GetPlatformInfo,          "clGetPlatformInfo");
    _RESOLVE(GetDeviceIDs,             "clGetDeviceIDs");
    _RESOLVE(GetDeviceInfo,            "clGetDeviceInfo");
    _RESOLVE(CreateContext,            "clCreateContext");
    _RESOLVE(ReleaseContext,           "clReleaseContext");
    _RESOLVE(CreateCommandQueue,       "clCreateCommandQueue");
    _RESOLVE(ReleaseCommandQueue,      "clReleaseCommandQueue");
    _RESOLVE(CreateBuffer,             "clCreateBuffer");
    _RESOLVE(ReleaseMemObject,         "clReleaseMemObject");
    _RESOLVE(EnqueueWriteBuffer,       "clEnqueueWriteBuffer");
    _RESOLVE(EnqueueReadBuffer,        "clEnqueueReadBuffer");
    _RESOLVE(CreateProgramWithSource,  "clCreateProgramWithSource");
    _RESOLVE(BuildProgram,             "clBuildProgram");
    _RESOLVE(GetProgramBuildInfo,      "clGetProgramBuildInfo");
    _RESOLVE(ReleaseProgram,           "clReleaseProgram");
    _RESOLVE(CreateKernel,             "clCreateKernel");
    _RESOLVE(ReleaseKernel,            "clReleaseKernel");
    _RESOLVE(SetKernelArg,             "clSetKernelArg");
    _RESOLVE(EnqueueNDRangeKernel,     "clEnqueueNDRangeKernel");
    _RESOLVE(Finish,                   "clFinish");
    _RESOLVE(Flush,                    "clFlush");
#undef _RESOLVE
    return true;
}

void unload_opencl(OpenCLApi* api) {
    if (api->dl_handle) {
        dlclose(api->dl_handle);
        api->dl_handle = nullptr;
    }
    *api = OpenCLApi{};
}

#else  // host (macOS / Windows) — no OpenCL on the build host

bool load_opencl(OpenCLApi*) {
    std::fprintf(stderr, "[opencl] runtime loader not implemented on this host platform\n");
    return false;
}
void unload_opencl(OpenCLApi*) {}

#endif

}  // namespace turboquant
