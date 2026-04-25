// Runtime OpenCL function loader. The NDK doesn't ship libOpenCL.so, but
// almost every Snapdragon device has /vendor/lib64/libOpenCL.so. We dlopen
// it at backend init and resolve the function pointers we use.

#pragma once

#include "CL/cl.h"

#include <stdexcept>

namespace turboquant {

struct OpenCLApi {
    decltype(clGetPlatformIDs)*       GetPlatformIDs       = nullptr;
    decltype(clGetPlatformInfo)*      GetPlatformInfo      = nullptr;
    decltype(clGetDeviceIDs)*         GetDeviceIDs         = nullptr;
    decltype(clGetDeviceInfo)*        GetDeviceInfo        = nullptr;
    decltype(clCreateContext)*        CreateContext        = nullptr;
    decltype(clReleaseContext)*       ReleaseContext       = nullptr;
    decltype(clCreateCommandQueue)*   CreateCommandQueue   = nullptr;
    decltype(clReleaseCommandQueue)*  ReleaseCommandQueue  = nullptr;
    decltype(clCreateBuffer)*         CreateBuffer         = nullptr;
    decltype(clReleaseMemObject)*     ReleaseMemObject     = nullptr;
    decltype(clEnqueueWriteBuffer)*   EnqueueWriteBuffer   = nullptr;
    decltype(clEnqueueReadBuffer)*    EnqueueReadBuffer    = nullptr;
    decltype(clCreateProgramWithSource)* CreateProgramWithSource = nullptr;
    decltype(clBuildProgram)*         BuildProgram         = nullptr;
    decltype(clGetProgramBuildInfo)*  GetProgramBuildInfo  = nullptr;
    decltype(clReleaseProgram)*       ReleaseProgram       = nullptr;
    decltype(clCreateKernel)*         CreateKernel         = nullptr;
    decltype(clReleaseKernel)*        ReleaseKernel        = nullptr;
    decltype(clSetKernelArg)*         SetKernelArg         = nullptr;
    decltype(clEnqueueNDRangeKernel)* EnqueueNDRangeKernel = nullptr;
    decltype(clFinish)*               Finish               = nullptr;
    decltype(clFlush)*                Flush                = nullptr;

    void* dl_handle = nullptr;
};

// Tries common library paths; returns false (with details on stderr) if none
// of them resolve.
bool load_opencl(OpenCLApi* api);

void unload_opencl(OpenCLApi* api);

}  // namespace turboquant
