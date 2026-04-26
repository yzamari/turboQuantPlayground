// SigLIP single-block QNN graph — Phase 1 scaffold.
//
// All step methods are intentionally STUBS that throw std::logic_error. The
// goal of this commit is to land the framework so subsequent commits can
// fill in op-by-op (LayerNorm, Q/K/V Linear triple, attention, MLP, ...).
//
// Style template: qnn_graph.cpp. We pull the same QNN headers (with the
// __has_include guard so the file still parses on engineering hosts that
// don't have the SDK on the include path) and we use the same Graph base
// class for handle ownership.
//
// macOS host limitation: this file *will* compile on a macOS host whose
// QNN_SDK_ROOT exposes the headers (the SDK ships portable headers even
// though the converter binaries are Linux-only). Linking against
// libQnnHtp.so to actually run the graph is an on-device operation; that
// is irrelevant at Phase 1 since none of the methods here actually call
// the runtime.

#include "qnn_siglip.hpp"

#if __has_include(<QnnInterface.h>)
  #include <QnnInterface.h>
  #include <QnnGraph.h>
  #include <QnnTensor.h>
  #include <QnnOpDef.h>
  #include <QnnContext.h>
  #include <QnnTypes.h>
  #define TQ_QNN_HEADERS_AVAILABLE 1
#else
  #define TQ_QNN_HEADERS_AVAILABLE 0
#endif

#include <stdexcept>

namespace turboquant::qnn {

QnnSigLipBlock::QnnSigLipBlock(const QnnApi& api,
                               const SigLipBlockConfig& cfg,
                               const SigLipBlockWeights& weights,
                               GraphOptions opts)
    : Graph(api, opts), cfg_(cfg), weights_(weights) {
    // The constructor is intentionally a no-op for Phase 1. Once each
    // build_*_() helper is implemented, this body will:
    //   1. contextCreate
    //   2. graphCreate("tq_siglip_block_<dims>")
    //   3. build_ln1_()
    //   4. build_attention_()
    //   5. build_attention_residual_()
    //   6. build_ln2_()
    //   7. build_mlp_()
    //   8. build_block_residual_()
    //   9. graphFinalize
    //
    // Until then, valid() returns false (graph_handle_ stays nullptr) and
    // forward() will throw — which is what the parity test expects at this
    // phase.
    (void)api;
}

void QnnSigLipBlock::forward(const float* /*in*/, float* /*out*/) {
    // TODO Phase 1 step 7: bind input/output client buffers to the
    // pre-finalized graph and call api_.graphExecute(). Until the build_*_
    // helpers below are implemented, calling forward() is a programming
    // error — we throw so the parity test scaffold has something concrete
    // to fail against (logic_error, not silent zeros).
    throw std::logic_error(
        "QnnSigLipBlock::forward unimplemented — Phase 1 scaffold only. "
        "See cpp/backends/qnn_htp/qnn_siglip.hpp for the step plan.");
}

// ---------------------------------------------------------------------------
// Phase 1 step boundaries — every helper is a stub. Subsequent commits in
// this branch fill them in one at a time.
// ---------------------------------------------------------------------------

void QnnSigLipBlock::build_ln1_() {
    // TODO Phase 1 step 1: pre-attention LayerNorm.
    //
    // Inputs:
    //   - in       (APP_WRITE)  [seq, h] activation_dtype
    //   - ln1_gamma(STATIC)     [h]      activation_dtype  (from weights_)
    //   - ln1_beta (STATIC)     [h]      activation_dtype  (from weights_)
    // Output:
    //   - ln1_out  (NATIVE)     [seq, h] activation_dtype
    //
    // Use op QNN_OP_LAYER_NORM (axis=-1, epsilon=cfg_.layernorm_eps).
    throw std::logic_error("build_ln1_ unimplemented — Phase 1 step 1");
}

void QnnSigLipBlock::build_attention_() {
    // TODO Phase 1 step 2: Q/K/V projections + scaled-dot-product attention.
    //
    // Sub-steps (each a separate graphAddNode call):
    //   2a. q = ln1_out @ q_w^T + q_b   (MatMul + ElementWiseAdd, or fused
    //       FullyConnected if the runtime supports it)
    //   2b. k = ln1_out @ k_w^T + k_b
    //   2c. v = ln1_out @ v_w^T + v_b
    //   2d. reshape q,k,v to [seq, num_heads, head_dim] and transpose so
    //       the head axis is the leading one
    //   2e. scores = (q @ k^T) * (1/sqrt(head_dim))
    //   2f. softmax(scores)  (axis=-1)
    //   2g. attn = softmax_out @ v
    //   2h. attn_concat = transpose+reshape back to [seq, h]
    //   2i. attn_out = attn_concat @ o_w^T + o_b
    //
    // SigLIP vision has no causal mask — design doc §"Coverage gaps".
    throw std::logic_error("build_attention_ unimplemented — Phase 1 step 2");
}

void QnnSigLipBlock::build_attention_residual_() {
    // TODO Phase 1 step 3: residual = in + attn_out  (ElementWiseAdd).
    throw std::logic_error("build_attention_residual_ unimplemented — Phase 1 step 3");
}

void QnnSigLipBlock::build_ln2_() {
    // TODO Phase 1 step 4: LayerNorm on `residual` -> `ln2_out`.
    throw std::logic_error("build_ln2_ unimplemented — Phase 1 step 4");
}

void QnnSigLipBlock::build_mlp_() {
    // TODO Phase 1 step 5: MLP up + GELU + MLP down.
    //
    //   5a. mlp_up   = ln2_out @ mlp_up_w^T + mlp_up_b   (MatMul + Add)
    //   5b. mlp_gelu = GELU(mlp_up)
    //          v1: CPU fallback. The graph emits an APP_READ tensor for
    //          mlp_up, the host runs scalar/NEON GELU, and the result is
    //          fed back as an APP_WRITE tensor `mlp_gelu_in`. This split
    //          forces a graphExecute boundary — acceptable cost at Phase
    //          1, replaced by a custom HVX UDO in Phase 2.
    //   5c. mlp_dn   = mlp_gelu @ mlp_dn_w^T + mlp_dn_b
    throw std::logic_error("build_mlp_ unimplemented — Phase 1 step 5");
}

void QnnSigLipBlock::build_block_residual_() {
    // TODO Phase 1 step 6: out = residual + mlp_dn  (ElementWiseAdd).
    throw std::logic_error("build_block_residual_ unimplemented — Phase 1 step 6");
}

}  // namespace turboquant::qnn
