#include "turboquant/backend.hpp"

#include <cstring>
#include <memory>

namespace turboquant {

#if TQ_WITH_CPU_SCALAR
std::unique_ptr<IBackend> create_cpu_scalar_backend();
#endif
#if TQ_WITH_NEON
std::unique_ptr<IBackend> create_cpu_neon_backend();
#endif
#if TQ_WITH_QNN
std::unique_ptr<IBackend> create_qnn_htp_backend();
#endif
#if TQ_WITH_OPENCL
std::unique_ptr<IBackend> create_opencl_backend();
#endif
#if TQ_WITH_VULKAN
std::unique_ptr<IBackend> create_vulkan_backend();
#endif

const char* backend_kind_name(BackendKind k) {
    switch (k) {
        case BackendKind::CpuScalar: return "cpu_scalar";
        case BackendKind::CpuNeon:   return "cpu_neon";
        case BackendKind::QnnHtp:    return "qnn_htp";
        case BackendKind::OpenCL:    return "opencl";
        case BackendKind::Vulkan:    return "vulkan";
    }
    return "unknown";
}

BackendKind backend_kind_from_name(const char* name) {
    if (!name) return BackendKind::CpuScalar;
    if (std::strcmp(name, "cpu_scalar") == 0) return BackendKind::CpuScalar;
    if (std::strcmp(name, "cpu_neon")   == 0) return BackendKind::CpuNeon;
    if (std::strcmp(name, "qnn_htp")    == 0) return BackendKind::QnnHtp;
    if (std::strcmp(name, "opencl")     == 0) return BackendKind::OpenCL;
    if (std::strcmp(name, "vulkan")     == 0) return BackendKind::Vulkan;
    return BackendKind::CpuScalar;
}

std::unique_ptr<IBackend> create_backend(BackendKind kind) {
    std::unique_ptr<IBackend> b;
    switch (kind) {
        case BackendKind::CpuScalar:
#if TQ_WITH_CPU_SCALAR
            b = create_cpu_scalar_backend();
#endif
            break;
        case BackendKind::CpuNeon:
#if TQ_WITH_NEON
            b = create_cpu_neon_backend();
#endif
            break;
        case BackendKind::QnnHtp:
#if TQ_WITH_QNN
            b = create_qnn_htp_backend();
#endif
            break;
        case BackendKind::OpenCL:
#if TQ_WITH_OPENCL
            b = create_opencl_backend();
#endif
            break;
        case BackendKind::Vulkan:
#if TQ_WITH_VULKAN
            b = create_vulkan_backend();
#endif
            break;
    }
    if (b && !b->init()) b.reset();
    return b;
}

std::unique_ptr<IBackend> create_best_backend() {
    static const BackendKind kPriority[] = {
        BackendKind::QnnHtp,
        BackendKind::OpenCL,
        BackendKind::Vulkan,
        BackendKind::CpuNeon,
        BackendKind::CpuScalar,
    };
    for (BackendKind k : kPriority) {
        auto b = create_backend(k);
        if (b) return b;
    }
    return nullptr;
}

}  // namespace turboquant
