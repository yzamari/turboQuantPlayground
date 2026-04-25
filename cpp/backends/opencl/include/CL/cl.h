// Minimal Khronos OpenCL 1.2 C header for runtime dlopen on Android. We only
// declare the types and symbols we actually call — keeps the surface tiny and
// avoids pulling the full Khronos header. If you need more API surface, add
// the declaration here.
//
// References:
//   https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html

#ifndef _TQ_CL_H
#define _TQ_CL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_uint  cl_bool;
typedef cl_uint  cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_bitfield cl_mem_flags;
typedef cl_uint  cl_platform_info;
typedef cl_uint  cl_device_info;
typedef cl_uint  cl_program_info;
typedef cl_uint  cl_program_build_info;
typedef cl_uint  cl_kernel_info;
typedef cl_uint  cl_event_info;
typedef cl_uint  cl_command_type;
typedef cl_uint  cl_profiling_info;

typedef struct _cl_platform_id     *cl_platform_id;
typedef struct _cl_device_id       *cl_device_id;
typedef struct _cl_context         *cl_context;
typedef struct _cl_command_queue   *cl_command_queue;
typedef struct _cl_program         *cl_program;
typedef struct _cl_kernel          *cl_kernel;
typedef struct _cl_mem             *cl_mem;
typedef struct _cl_event           *cl_event;

typedef intptr_t cl_context_properties;

#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                       -1
#define CL_DEVICE_TYPE_GPU                  (1 << 2)
#define CL_DEVICE_TYPE_DEFAULT              (1 << 0)
#define CL_DEVICE_TYPE_ALL                  0xFFFFFFFF

#define CL_QUEUE_PROFILING_ENABLE           (1 << 1)

#define CL_MEM_READ_WRITE                   (1 << 0)
#define CL_MEM_WRITE_ONLY                   (1 << 1)
#define CL_MEM_READ_ONLY                    (1 << 2)
#define CL_MEM_USE_HOST_PTR                 (1 << 3)
#define CL_MEM_COPY_HOST_PTR                (1 << 5)

#define CL_TRUE                             1
#define CL_FALSE                            0

#define CL_PROGRAM_BUILD_LOG                0x1183

#define CL_DEVICE_NAME                      0x102B
#define CL_DEVICE_VENDOR                    0x102C
#define CL_DRIVER_VERSION                   0x102D
#define CL_PLATFORM_NAME                    0x0902
#define CL_PLATFORM_VENDOR                  0x0903

cl_int  clGetPlatformIDs(cl_uint, cl_platform_id*, cl_uint*);
cl_int  clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
cl_int  clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
cl_int  clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t*);
cl_context clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*,
                           void (*)(const char*, const void*, size_t, void*),
                           void*, cl_int*);
cl_int  clReleaseContext(cl_context);
cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
cl_int  clReleaseCommandQueue(cl_command_queue);
cl_mem  clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int*);
cl_int  clReleaseMemObject(cl_mem);
cl_int  clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                             const void*, cl_uint, const cl_event*, cl_event*);
cl_int  clEnqueueReadBuffer (cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                             void*, cl_uint, const cl_event*, cl_event*);
cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int*);
cl_int  clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*,
                       void (*)(cl_program, void*), void*);
cl_int  clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info,
                              size_t, void*, size_t*);
cl_int  clReleaseProgram(cl_program);
cl_kernel clCreateKernel(cl_program, const char*, cl_int*);
cl_int  clReleaseKernel(cl_kernel);
cl_int  clSetKernelArg(cl_kernel, cl_uint, size_t, const void*);
cl_int  clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                               const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
cl_int  clFinish(cl_command_queue);
cl_int  clFlush(cl_command_queue);

#ifdef __cplusplus
}
#endif

#endif  // _TQ_CL_H
