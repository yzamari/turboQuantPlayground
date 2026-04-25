// QNN dynamic loader.
//
// We never link directly against `libQnnHtp.so` because:
//   * its ABI is versioned per SDK and we want graceful fallback,
//   * on retail Android devices the vendor copy may live at a different path,
//   * on QNX the same dlopen pattern works against the QNX equivalent.
//
// `load_qnn()` resolves a small set of symbols and fills a `QnnApi` struct.
// On failure it logs to stderr and returns false; the backend then refuses to
// initialize and the factory falls back to NEON / scalar.

#pragma once

#include <cstdint>

// Forward-declared QNN types — we *do not* include the QNN headers here so
// that translation units that just want to call the backend through IBackend
// don't need QNN_SDK_ROOT on their include path.
extern "C" {
struct QnnInterface_t;
typedef struct QnnInterface_t QnnInterface_t;
}

namespace turboquant::qnn {

// Function pointer types for the handful of QNN entry points we actually use.
// Real signatures come from <QNN/QnnInterface.h>; we type-erase to void* here
// and reinterpret-cast at the call site inside qnn_graph.cpp where the QNN
// headers *are* included. This keeps qnn_loader.hpp SDK-header-free.
using QnnInterfaceGetProvidersFn =
    int (*)(const QnnInterface_t*** providers, std::uint32_t* num_providers);

struct QnnApi {
    void* htp_handle      = nullptr;   // dlopen handle for libQnnHtp.so
    void* system_handle   = nullptr;   // dlopen handle for libQnnSystem.so
    QnnInterfaceGetProvidersFn get_providers = nullptr;

    // Path actually loaded (for logs / debug). Owned by the loader.
    const char* htp_path    = nullptr;
    const char* system_path = nullptr;

    bool valid() const { return htp_handle != nullptr && get_providers != nullptr; }
};

// Try to dlopen the QNN HTP + System libraries from a prioritized list:
//   1. anything on $LD_LIBRARY_PATH (let dlopen do its thing with bare name)
//   2. /data/local/tmp/qnn_libs/   (developer push location)
//   3. /vendor/lib64/              (OEM-shipped runtimes, incl. libsnap_qnn.so)
//
// Returns true on success and fills *out. Returns false and emits a single
// stderr line on failure. Safe to call multiple times — second-call result is
// cached process-wide.
bool load_qnn(QnnApi* out);

// Release dlopen handles. Optional; the OS reclaims on exit. Mostly useful for
// test-suite teardown.
void unload_qnn(QnnApi* api);

}  // namespace turboquant::qnn
