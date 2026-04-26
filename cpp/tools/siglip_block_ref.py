"""
SigLIP single-block FP32 reference + golden corpus generator (Phase 1).

Loads HuggingFace `openai/SigLIP-base-patch16-224` and runs *block 0* of
the vision tower on a fixed-seed input tensor on CPU FP32. Dumps the
inputs, all intermediates (post-LN1, attn output, post-residual, post-LN2,
post-MLP, final residual), and the block output to a single binary file
that the C++ parity test (`cpp/tests/qnn_siglip_block_test.cpp`) consumes.

Style follows `cpp/tools/gen_golden.py`: deterministic seeds, raw
little-endian float32 buffers concatenated in a known order, and a small
JSON manifest that documents shapes / offsets so the C++ side stays free
of a JSON dependency in the inner loop.

# When to run

This script is **Linux-only** in practice. macOS hosts can technically
execute it (PyTorch + transformers run fine on Darwin) but the heavy
~370 MB HuggingFace download and the QAIRT converter that turns the
weights into a QNN .bin both want a Linux box anyway. See
`.github/workflows/siglip-convert.yml` for the canonical CI invocation.

    python -m venv venv && . venv/bin/activate
    pip install torch transformers numpy
    python cpp/tools/siglip_block_ref.py            # writes golden/

# Output

    cpp/tests/golden/siglip_block_d768_h12.bin
    cpp/tests/golden/siglip_block_d768_h12.manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Match gen_golden.py: REPO_ROOT is two levels up from this file.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
GOLDEN_DIR = REPO_ROOT / "cpp" / "tests" / "golden"

# SigLIP-base-patch16-224 vision-tower constants. These match the public
# HuggingFace config; verify with `model.config.vision_config` before
# running.
MODEL_NAME = "google/siglip-base-patch16-224"
HIDDEN_DIM = 768
NUM_HEADS = 12
MLP_DIM = 3072
SEQ_LEN = 196          # 14 x 14 patches at 224 x 224 with patch=16
LAYERNORM_EPS = 1e-6   # SigLIP uses 1e-6, double-check from the model

# Fixed-seed deterministic input — same convention as gen_golden.py.
INPUT_SEED = 314159


def _to_numpy(t) -> np.ndarray:
    """torch / numpy / mlx tensor -> contiguous float32 numpy."""
    if isinstance(t, np.ndarray):
        return np.ascontiguousarray(t, dtype=np.float32)
    if hasattr(t, "detach"):  # torch.Tensor
        return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
    if hasattr(t, "numpy"):
        return np.ascontiguousarray(t.numpy(), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(t), dtype=np.float32)


def _write_concat(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    """Write all arrays back-to-back as raw float32 and return a manifest
    entry mapping name -> (offset_bytes, shape, dtype)."""
    manifest_entries = {}
    offset = 0
    with path.open("wb") as f:
        for name, arr in arrays.items():
            a = _to_numpy(arr)
            f.write(a.tobytes())
            manifest_entries[name] = {
                "offset": offset,
                "nbytes": int(a.nbytes),
                "shape": list(a.shape),
                "dtype": "float32",
            }
            offset += int(a.nbytes)
    return manifest_entries


def main() -> int:
    # Lazy imports so the file can be parsed (and lint-checked) on hosts
    # without torch / transformers installed.
    try:
        import torch
        import torch.nn.functional as F  # noqa: F401  (used implicitly)
        from transformers import AutoModel
    except ImportError as e:
        raise SystemExit(
            f"siglip_block_ref.py requires torch + transformers ({e}). "
            "Run on a Linux box with: pip install torch transformers"
        ) from e

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[siglip-ref] loading {MODEL_NAME} (CPU, FP32) ...")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
    vision = model.vision_model

    # SigLIP's HF vision tower uses a `SigLIPEncoderLayer` per block, with
    # attributes: layer_norm1, self_attn, layer_norm2, mlp.
    # We run only block 0 in isolation: feed it a fake [seq, hidden] tensor
    # bypassing the patch-embed entirely.
    block = vision.encoder.layers[0]

    # Validate model config matches our compile-time constants.
    cfg = model.config.vision_config
    assert cfg.hidden_size == HIDDEN_DIM, f"hidden mismatch: {cfg.hidden_size}"
    assert cfg.num_attention_heads == NUM_HEADS, f"heads mismatch: {cfg.num_attention_heads}"
    assert cfg.intermediate_size == MLP_DIM, f"mlp mismatch: {cfg.intermediate_size}"

    # Deterministic input tensor [batch=1, seq, hidden].
    rng = np.random.RandomState(INPUT_SEED)
    x_np = rng.randn(1, SEQ_LEN, HIDDEN_DIM).astype(np.float32)
    x_t = torch.from_numpy(x_np)

    # ----- Capture intermediates by running each sub-module manually so
    #       we don't have to monkey-patch the HF block.
    with torch.no_grad():
        ln1_out = block.layer_norm1(x_t)
        # SigLIPAttention returns (attn_out, attn_weights) — we want [0].
        attn_out = block.self_attn(ln1_out)[0]
        attn_residual = x_t + attn_out
        ln2_out = block.layer_norm2(attn_residual)
        mlp_out = block.mlp(ln2_out)
        block_out = attn_residual + mlp_out

    # ----- Dump everything ---------------------------------------------------
    arrays = {
        "input":           x_np[0],         # [seq, hidden]
        "ln1_out":         ln1_out[0],
        "attn_out":        attn_out[0],
        "attn_residual":   attn_residual[0],
        "ln2_out":         ln2_out[0],
        "mlp_out":         mlp_out[0],
        "output":          block_out[0],
    }
    out_bin = GOLDEN_DIR / f"siglip_block_d{HIDDEN_DIM}_h{NUM_HEADS}.bin"
    manifest_entries = _write_concat(out_bin, arrays)

    # Also dump the eight weight buffers a single block consumes — the C++
    # graph builder needs these to populate STATIC tensors during graphAdd.
    sa = block.self_attn
    weights = {
        "ln1_gamma":  block.layer_norm1.weight,
        "ln1_beta":   block.layer_norm1.bias,
        "q_w":        sa.q_proj.weight,
        "q_b":        sa.q_proj.bias,
        "k_w":        sa.k_proj.weight,
        "k_b":        sa.k_proj.bias,
        "v_w":        sa.v_proj.weight,
        "v_b":        sa.v_proj.bias,
        "o_w":        sa.out_proj.weight,
        "o_b":        sa.out_proj.bias,
        "ln2_gamma":  block.layer_norm2.weight,
        "ln2_beta":   block.layer_norm2.bias,
        "mlp_up_w":   block.mlp.fc1.weight,
        "mlp_up_b":   block.mlp.fc1.bias,
        "mlp_dn_w":   block.mlp.fc2.weight,
        "mlp_dn_b":   block.mlp.fc2.bias,
    }
    weight_bin = GOLDEN_DIR / f"siglip_block_d{HIDDEN_DIM}_h{NUM_HEADS}.weights.bin"
    weight_entries = _write_concat(weight_bin, weights)

    manifest = {
        "format_version": 1,
        "model": MODEL_NAME,
        "block_index": 0,
        "seq_len": SEQ_LEN,
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "mlp_dim": MLP_DIM,
        "layernorm_eps": LAYERNORM_EPS,
        "input_seed": INPUT_SEED,
        "tensors_file": out_bin.name,
        "tensors": manifest_entries,
        "weights_file": weight_bin.name,
        "weights": weight_entries,
        # Phase 1 acceptance criteria — duplicated here so the C++ test can
        # echo the targets in its log line.
        "acceptance": {
            "cosine_min": 0.99,
            "abs_error_max_fp16": 1e-3,
            "abs_error_max_fp32": 1e-4,
        },
    }
    manifest_path = GOLDEN_DIR / f"siglip_block_d{HIDDEN_DIM}_h{NUM_HEADS}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[siglip-ref] wrote {out_bin}")
    print(f"[siglip-ref] wrote {weight_bin}")
    print(f"[siglip-ref] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
