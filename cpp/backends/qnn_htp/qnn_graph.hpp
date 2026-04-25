// Pre-finalized QNN graphs used by QnnHtpBackend.
//
// Each graph is *shape-specialized*: we build one per (n, D) we actually run
// against. Building is one-time (a few ms on first call); execution is the
// hot path. We finalize at construction so the QNN runtime can do its full
// HTP partitioning + tile scheduling pass before the first inference.
//
// Three graphs:
//
//   1. RotateGraph        — single MatMul: out[n,D] = in[n,D] @ Pi^T
//   2. ValueDequantGraph  — Cast(uint8 -> float) -> Gather(scales/zeros) ->
//                           Mul -> Add. Group-wise dequant; bits in {2,4,8}.
//   3. MseScoreGraph      — STUB. Bit-unpack + gather + reduce is awkward in
//                           QNN core ops; planned as a custom HTP op later.
//                           See TODO inside qnn_graph.cpp.
//
// All graphs default to FP16 precision on HTP for throughput. Pass
// `use_fp32_fallback=true` to the constructor for ASIL-relevant deployments —
// keeps absolute deviation under 1e-4 vs the scalar reference (the FP16 path
// is bounded at <1e-3 instead).

#pragma once

#include "qnn_loader.hpp"

#include <cstdint>
#include <memory>

namespace turboquant::qnn {

// ----- common construction options ------------------------------------------------

struct GraphOptions {
    // FP16 (default) gives the best HTP throughput. FP32 is the compatibility
    // path for safety-critical deployments where bitwise-vs-CPU drift must be
    // bounded at 1e-4 — at the cost of ~2x latency on V73/V75 HTP.
    bool use_fp32_fallback = false;
};

// Common graph base: owns a Qnn graph handle + I/O tensor metadata.
// Implementation lives in qnn_graph.cpp where the QNN headers are pulled in.
class Graph {
public:
    virtual ~Graph();
    virtual const char* name() const = 0;
    bool valid() const { return graph_handle_ != nullptr; }

protected:
    explicit Graph(const QnnApi& api, GraphOptions opts);

    const QnnApi api_;
    GraphOptions opts_;
    void*        graph_handle_ = nullptr;   // Qnn_GraphHandle_t (opaque)
    void*        context_      = nullptr;   // Qnn_ContextHandle_t (opaque)
};

// ----- (1) rotate ----------------------------------------------------------------

class RotateGraph final : public Graph {
public:
    // Pi is row-major [D, D]. We build a graph whose constant weight tensor is
    // already Pi^T so the on-device op is a plain MatMul.
    RotateGraph(const QnnApi& api, int n, int D, const float* Pi, GraphOptions opts);
    const char* name() const override { return "rotate"; }

    // out[n, D] = in[n, D] @ Pi^T.
    // Shapes baked in at construction; reuse the same RotateGraph for every
    // inference of that shape.
    void execute(const float* in, float* out);

private:
    int n_;
    int D_;
};

// ----- (2) value dequant ---------------------------------------------------------

class ValueDequantGraph final : public Graph {
public:
    ValueDequantGraph(const QnnApi& api,
                      int N, int D, int bits, int group_size,
                      GraphOptions opts);
    const char* name() const override { return "value_dequant"; }

    // packed: [N, packed_d] uint8 (vals_per_byte = 8/bits, bits in {2,4,8})
    // scales/zeros: [N, n_groups]   where n_groups = D / group_size
    // out:    [N, D] float
    void execute(const std::uint8_t* packed,
                 const float* scales,
                 const float* zeros,
                 float* out);

private:
    int N_;
    int D_;
    int bits_;
    int group_size_;
};

// ----- (3) mse_score (STUB) ------------------------------------------------------
//
// TODO: build this once we have a custom HTP op for fused
//       bit-unpack + centroid gather + dot reduction. For now the backend
//       composes NEON for `mse_score` via QnnHtpBackend's CPU fallback.
//       This class is intentionally not provided.
//
// (We also leave qjl_score and mse_encode CPU-only for the same reason: their
// inner loop is dominated by bit-shuffles that QNN's core op set doesn't
// express efficiently.)

}  // namespace turboquant::qnn
