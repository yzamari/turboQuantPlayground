#include "turboquant/backend.hpp"

#include <cstring>
#include <memory>
#include <mutex>

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
    // Priority order: QnnHtp > CpuNeon > OpenCL > Vulkan > CpuScalar.
    //
    // CpuNeon is intentionally ahead of OpenCL/Vulkan for now. Both GPU
    // backends are upload-bound at ~17-23 ms per attention call (project
    // bench: cpp/bench/results/) because each call re-uploads Pi, the
    // quantized K-state, and centroids via CL_MEM_COPY_HOST_PTR. NEON is
    // ~2 ms per call. Until P2.1c Phase A2 ships persistent K-state on
    // GPU (IBackend::prepare_keys + cl_mem pool keyed by session_id),
    // NEON is the better default for end-to-end latency on Adreno
    // devices. QNN/HTP stays first because its rotate+value_dequant
    // graphs are already cached by shape via QNN's own graph cache —
    // it doesn't have the upload-per-call problem.
    //
    // Re-order this list once OpenCL ships prepare_keys().
    static const BackendKind kPriority[] = {
        BackendKind::QnnHtp,
        BackendKind::CpuNeon,
        BackendKind::OpenCL,
        BackendKind::Vulkan,
        BackendKind::CpuScalar,
    };
    for (BackendKind k : kPriority) {
        auto b = create_backend(k);
        if (b) return b;
    }
    return nullptr;
}

// ---- Singleton variant ----
//
// We keep the unique_ptr inside this TU rather than as a static-locals-in-
// function so we can guard initialization with a mutex (function-local
// statics handle the race themselves but we also want ensure_*() to act
// as an explicit "do init now" trigger).
namespace {
std::mutex                 g_singleton_mu;
std::unique_ptr<IBackend>  g_singleton_backend;  // nullptr until first init
}  // namespace

void ensure_backends_initialized() {
    std::lock_guard<std::mutex> lk(g_singleton_mu);
    if (g_singleton_backend) return;
    g_singleton_backend = create_best_backend();
}

IBackend * get_best_backend() {
    ensure_backends_initialized();
    std::lock_guard<std::mutex> lk(g_singleton_mu);
    return g_singleton_backend.get();
}

}  // namespace turboquant
