// QNN dynamic loader — see qnn_loader.hpp for rationale.

#include "qnn_loader.hpp"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace turboquant::qnn {
namespace {

// Process-wide cache so multiple backend instances share one set of handles.
std::once_flag g_init_flag;
QnnApi         g_cached;
bool           g_cached_ok = false;

// Candidate library names. Order matters: first match wins.
struct Candidate {
    const char* htp;     // primary HTP runtime
    const char* system;  // graph-system / config support lib
    const char* note;    // human-readable identifier for the log line
};

const Candidate kCandidates[] = {
    // (1) Anything dlopen finds on the standard search path
    //     ($LD_LIBRARY_PATH, /system/lib64, /vendor/lib64, etc.).
    {"libQnnHtp.so",                  "libQnnSystem.so",                "default search path"},

    // (2) The conventional developer push location used by Qualcomm's own
    //     sample apps and by `adb push $QNN_SDK_ROOT/lib/aarch64-android/...`.
    {"/data/local/tmp/qnn_libs/libQnnHtp.so",
     "/data/local/tmp/qnn_libs/libQnnSystem.so",
     "/data/local/tmp/qnn_libs"},

    // (3) Vendor-shipped fallback. Samsung's "snap" wrapper exposes a
    //     QNN-compatible ABI so we can probe it as a last resort. libQnnSystem
    //     is rarely present here — we tolerate that and continue with system=null.
    {"/vendor/lib64/libsnap_qnn.so",  "/vendor/lib64/libQnnSystem.so",  "/vendor/lib64 (libsnap_qnn fallback)"},
};

// Try one candidate. On success populate *out (handle + symbol) and return true.
bool try_candidate(const Candidate& c, QnnApi* out) {
    void* htp = dlopen(c.htp, RTLD_NOW | RTLD_LOCAL);
    if (!htp) {
        // Not an error — most candidates will miss. Caller iterates.
        return false;
    }

    // libQnnSystem is *strongly recommended* but optional for the bare-bones
    // graph build path. If it's missing we soldier on with a null handle.
    void* sys = dlopen(c.system, RTLD_NOW | RTLD_LOCAL);

    // Resolve the single C entry point we need to enumerate providers; from
    // there everything else is reached through the returned QnnInterface_t.
    auto get_providers = reinterpret_cast<QnnInterfaceGetProvidersFn>(
        dlsym(htp, "QnnInterface_getProviders"));
    if (!get_providers) {
        std::fprintf(stderr,
                     "[turboquant.qnn] dlopen('%s') succeeded but "
                     "QnnInterface_getProviders missing: %s\n",
                     c.htp, dlerror());
        dlclose(htp);
        if (sys) dlclose(sys);
        return false;
    }

    out->htp_handle    = htp;
    out->system_handle = sys;
    out->get_providers = get_providers;
    out->htp_path      = c.htp;
    out->system_path   = c.system;
    return true;
}

void do_load_once() {
    QnnApi api;
    for (const auto& c : kCandidates) {
        if (try_candidate(c, &api)) {
            std::fprintf(stderr,
                         "[turboquant.qnn] loaded HTP runtime from %s (%s)\n",
                         api.htp_path, c.note);
            g_cached    = api;
            g_cached_ok = true;
            return;
        }
    }

    std::fprintf(stderr,
        "[turboquant.qnn] could not locate libQnnHtp.so. "
        "Tried: $LD_LIBRARY_PATH, /data/local/tmp/qnn_libs, /vendor/lib64. "
        "See cpp/backends/qnn_htp/README.md to install / push the runtime.\n");
    g_cached_ok = false;
}

}  // namespace

bool load_qnn(QnnApi* out) {
    std::call_once(g_init_flag, do_load_once);
    if (!g_cached_ok) return false;
    *out = g_cached;
    return true;
}

void unload_qnn(QnnApi* api) {
    if (!api) return;
    if (api->htp_handle)    dlclose(api->htp_handle);
    if (api->system_handle) dlclose(api->system_handle);
    api->htp_handle    = nullptr;
    api->system_handle = nullptr;
    api->get_providers = nullptr;
    // Note: process-wide cache is intentionally *not* cleared here so a
    // subsequent load_qnn() call still returns the same answer cheaply.
}

}  // namespace turboquant::qnn
