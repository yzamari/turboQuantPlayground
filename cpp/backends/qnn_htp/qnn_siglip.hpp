// SigLIP single-block QNN graph (Path C, Phase 1 scaffold).
//
// This header declares the *one transformer block* surface of the SigLIP
// vision encoder running on Hexagon HTP. Phase 1's deliverable is to land
// the framework — every method here is currently a stub that throws
// std::logic_error. Subsequent commits in this branch fill them in op-by-op:
//
//   Step 1: pre-attention LayerNorm
//   Step 2: Q/K/V Linear triple + scaled-dot-product attention
//   Step 3: residual add
//   Step 4: post-attention LayerNorm
//   Step 5: MLP up (Linear) -> GELU (CPU fallback v1) -> MLP down (Linear)
//   Step 6: final residual add
//
// Style template: this file mirrors qnn_graph.hpp — same Graph base class
// pattern, same shape-specialization-by-construction discipline, same
// FP16-default-with-FP32-fallback option struct.
//
// Numerical target (Phase 1 acceptance):
//   * cosine similarity vs FP32 reference >= 0.99
//   * absolute error < 1e-3 (FP16 path) / < 1e-4 (FP32 path)
//
// macOS host limitation: the actual weights have to come from the QAIRT
// converter on a Linux box. See `cpp/backends/qnn_htp/README.md` (SigLIP
// scaffold section) for the conversion pipeline.

#pragma once

#include "qnn_graph.hpp"
#include "qnn_loader.hpp"

#include <cstdint>
#include <memory>

namespace turboquant::qnn {

// Shape and config for one transformer block. SigLIP-base uses
// hidden_dim=768, num_heads=12, mlp_dim=3072, but the class is generic so
// we can unit-test smaller shapes too.
struct SigLipBlockConfig {
    int seq_len;     // e.g. 196 for 14x14 patches at 224x224
    int hidden_dim;  // e.g. 768
    int num_heads;   // e.g. 12 — head_dim = hidden_dim / num_heads
    int mlp_dim;     // e.g. 3072

    // Layer-norm epsilon. SigLIP-base uses 1e-6.
    float layernorm_eps = 1e-6f;
};

// Pointers to the eight float32 weight buffers a single block consumes.
// The caller owns the storage; QnnSigLipBlock copies what it needs into
// the graph's STATIC tensors at finalize time.
//
// Layouts (row-major):
//   ln1_gamma / ln1_beta : [hidden_dim]
//   q_w / k_w / v_w / o_w: [hidden_dim, hidden_dim]   (w in [out, in] order)
//   q_b / k_b / v_b / o_b: [hidden_dim]
//   ln2_gamma / ln2_beta : [hidden_dim]
//   mlp_up_w             : [mlp_dim, hidden_dim]
//   mlp_up_b             : [mlp_dim]
//   mlp_dn_w             : [hidden_dim, mlp_dim]
//   mlp_dn_b             : [hidden_dim]
struct SigLipBlockWeights {
    const float* ln1_gamma  = nullptr;
    const float* ln1_beta   = nullptr;

    const float* q_w        = nullptr;
    const float* q_b        = nullptr;
    const float* k_w        = nullptr;
    const float* k_b        = nullptr;
    const float* v_w        = nullptr;
    const float* v_b        = nullptr;
    const float* o_w        = nullptr;
    const float* o_b        = nullptr;

    const float* ln2_gamma  = nullptr;
    const float* ln2_beta   = nullptr;

    const float* mlp_up_w   = nullptr;
    const float* mlp_up_b   = nullptr;
    const float* mlp_dn_w   = nullptr;
    const float* mlp_dn_b   = nullptr;
};

// One pre-finalized HTP graph for a single SigLIP transformer block.
//
// Construction is *shape-specialized* by (seq_len, hidden_dim, num_heads,
// mlp_dim). For Phase 2 the surrounding QnnSigLipBackend will keep a cache
// keyed on these dims (analogous to QnnHtpBackend::rotate_cache_).
//
// Phase 1 surface: every step is a stub that throws std::logic_error on
// call. The constructor itself does NOT throw — we want the test harness
// to be able to *construct* the object so the parity test compiles and
// links cleanly even before any op is wired up.
class QnnSigLipBlock final : public Graph {
public:
    // Build the graph for the given config. `weights` is shallow-copied
    // (pointers only); the caller must keep the weight buffers alive until
    // the graph has been finalized (which today is a no-op stub).
    QnnSigLipBlock(const QnnApi& api,
                   const SigLipBlockConfig& cfg,
                   const SigLipBlockWeights& weights,
                   GraphOptions opts);

    const char* name() const override { return "siglip_block"; }

    // Forward pass for one block.
    //   in : [seq_len, hidden_dim] float32
    //   out: [seq_len, hidden_dim] float32
    //
    // Phase 1 status: throws std::logic_error("unimplemented"). The parity
    // test scaffold (cpp/tests/qnn_siglip_block_test.cpp) is explicitly
    // expected to fail at this point — the goal of this commit is to land
    // the harness, not the runtime.
    void forward(const float* in, float* out);

    // Accessors for the test harness.
    const SigLipBlockConfig& config() const { return cfg_; }

private:
    // ---- Phase 1 step boundaries (each method to be filled in by a
    //      follow-up commit; all currently stubs).

    // Step 1: pre-attention LayerNorm. Reads `in` [seq, h], writes a NATIVE
    // tensor `ln1_out` [seq, h].
    // TODO Phase 1 step 1
    void build_ln1_();

    // Step 2: Q/K/V projections + scaled-dot-product attention + output proj.
    // Inputs: ln1_out [seq, h]. Output: attn_out [seq, h].
    // TODO Phase 1 step 2
    void build_attention_();

    // Step 3: residual add (in + attn_out -> attn_residual [seq, h]).
    // TODO Phase 1 step 3
    void build_attention_residual_();

    // Step 4: post-attention LayerNorm.
    // TODO Phase 1 step 4
    void build_ln2_();

    // Step 5: MLP up -> GELU (CPU-fallback v1) -> MLP down.
    // GELU: see design doc §"Coverage gaps" — keep on CPU for v1, custom
    // HVX UDO is a v2 optimisation.
    // TODO Phase 1 step 5
    void build_mlp_();

    // Step 6: final residual add (attn_residual + mlp_out -> block out).
    // TODO Phase 1 step 6
    void build_block_residual_();

    SigLipBlockConfig  cfg_;
    SigLipBlockWeights weights_;
};

}  // namespace turboquant::qnn
