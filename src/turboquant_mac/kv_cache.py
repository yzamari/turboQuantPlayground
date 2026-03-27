"""
TurboQuant KV Cache — drop-in for transformer attention layers.

Handles:
  - Keys: TurboQuant_prod quantization (unbiased inner product estimation)
  - Values: Group quantization (asymmetric, per-group min-max)
  - Buffer: recent tokens kept unquantized for quality

Backend-agnostic: uses the backend abstraction layer.
"""

import math
from typing import NamedTuple, Optional

from turboquant_mac.quantizer import TurboQuantProd, ProdQuantized
from turboquant_mac.backends import get_backend


class ValueQuantized(NamedTuple):
    """Quantized value cache (bit-packed)."""
    data: object       # (..., n_tokens, packed_d) bit-packed quantized values
    scales: object     # (..., n_tokens, n_groups) scale per group
    zeros: object      # (..., n_tokens, n_groups) zero point per group
    bits: int = 2


def quantize_values(v, bits: int = 2, group_size: int = 32, backend: str = None) -> ValueQuantized:
    """
    Asymmetric group quantization for value vectors.

    Args:
        v: (..., seq_len, d) value vectors
        bits: quantization bits (2 or 4)
        group_size: number of elements per quantization group
    """
    B = get_backend(backend)
    orig_shape = v.shape
    d = orig_shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    # Reshape to groups
    v_grouped = B.reshape(v, (*orig_shape[:-1], n_groups, group_size))

    # Compute scale and zero per group (asymmetric)
    v_min = B.min_(v_grouped, dim=-1)
    v_max = B.max_(v_grouped, dim=-1)

    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    scale = B.clamp(scale, min_val=1e-10)
    zero = v_min

    # Quantize
    v_q = B.to_uint8(B.clamp(B.round_((v_grouped - zero) / scale), min_val=0, max_val=n_levels))
    v_q_flat = B.reshape(v_q, (*orig_shape[:-1], d))

    # Bit-pack
    if bits == 2:
        assert d % 4 == 0
        v_4 = B.reshape(v_q_flat, (*orig_shape[:-1], d // 4, 4))
        packed = B.bitwise_or(
            B.bitwise_or(v_4[..., 0], B.left_shift(v_4[..., 1], 2)),
            B.bitwise_or(B.left_shift(v_4[..., 2], 4), B.left_shift(v_4[..., 3], 6))
        )
        v_q_flat = packed
    elif bits == 4:
        assert d % 2 == 0
        v_2 = B.reshape(v_q_flat, (*orig_shape[:-1], d // 2, 2))
        packed = B.bitwise_or(v_2[..., 0], B.left_shift(v_2[..., 1], 4))
        v_q_flat = packed

    # Squeeze the keepdim from min/max
    scales_out = B.reshape(scale, (*orig_shape[:-1], n_groups))
    zeros_out = B.reshape(zero, (*orig_shape[:-1], n_groups))

    return ValueQuantized(data=v_q_flat, scales=scales_out, zeros=zeros_out, bits=bits)


def _unpack_values(vq: ValueQuantized, backend: str = None):
    """Unpack bit-packed value data to per-element."""
    B = get_backend(backend)
    bits = vq.bits
    packed = vq.data

    if bits == 2:
        v0 = B.bitwise_and(packed, 0x03)
        v1 = B.bitwise_and(B.right_shift(packed, 2), 0x03)
        v2 = B.bitwise_and(B.right_shift(packed, 4), 0x03)
        v3 = B.bitwise_and(B.right_shift(packed, 6), 0x03)
        return B.reshape(
            B.stack([v0, v1, v2, v3], dim=-1),
            (*packed.shape[:-1], packed.shape[-1] * 4)
        )
    elif bits == 4:
        v0 = B.bitwise_and(packed, 0x0F)
        v1 = B.bitwise_and(B.right_shift(packed, 4), 0x0F)
        return B.reshape(
            B.stack([v0, v1], dim=-1),
            (*packed.shape[:-1], packed.shape[-1] * 2)
        )
    return packed


def dequantize_values(vq: ValueQuantized, group_size: int = 32, backend: str = None):
    """Dequantize value vectors from bit-packed format."""
    B = get_backend(backend)
    data = B.to_float(_unpack_values(vq, backend))
    d = data.shape[-1]
    batch_shape = data.shape[:-1]

    n_groups = d // group_size
    data = B.reshape(data, (*batch_shape, n_groups, group_size))
    scales = B.unsqueeze(vq.scales, -1)
    zeros = B.unsqueeze(vq.zeros, -1)

    v = data * scales + zeros
    return B.reshape(v, (*batch_shape, d))


class TurboQuantKVCache:
    """
    KV cache using TurboQuant for keys and group quantization for values.

    Usage:
        cache = TurboQuantKVCache(head_dim=128, key_bits=3, value_bits=2)
        cache.prefill(key_states, value_states)
        cache.append(new_key, new_value)
        scores = cache.attention_scores(query_states)
        output = cache.attend(softmax(scores))
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 3,
        value_bits: int = 2,
        value_group_size: int = 32,
        buffer_size: int = 128,
        layer_idx: int = 0,
        backend: str = None,
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.value_group_size = value_group_size
        self.buffer_size = buffer_size
        self._backend_name = backend

        self.key_quantizer = TurboQuantProd(
            dim=head_dim, bits=key_bits, seed=42 + layer_idx * 7, backend=backend,
        )

        # State
        self.seq_len: int = 0
        self.key_quantized: Optional[ProdQuantized] = None
        self.value_quantized: Optional[ValueQuantized] = None
        self.key_buffer = None
        self.value_buffer = None

    @property
    def B(self):
        return get_backend(self._backend_name)

    def prefill(self, keys, values):
        """
        Process prefill tokens.

        Args:
            keys: (batch, n_heads, seq_len, head_dim)
            values: (batch, n_heads, seq_len, head_dim)
        """
        B = self.B
        seq_len = keys.shape[-2]
        self.seq_len = seq_len

        if seq_len <= self.buffer_size:
            self.key_buffer = keys
            self.value_buffer = values
            return

        n_quant = seq_len - self.buffer_size
        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]

        self.key_buffer = keys[..., n_quant:, :]
        self.value_buffer = values[..., n_quant:, :]

        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)
        self.value_quantized = quantize_values(
            values_to_quant, bits=self.value_bits,
            group_size=self.value_group_size, backend=self._backend_name,
        )

    def append(self, key, value):
        """Append a single decode token: (batch, n_heads, 1, head_dim)."""
        B = self.B
        self.seq_len += 1

        if self.key_buffer is not None:
            self.key_buffer = B.cat([self.key_buffer, key], dim=-2)
            self.value_buffer = B.cat([self.value_buffer, value], dim=-2)
        else:
            self.key_buffer = key
            self.value_buffer = value

        if self.key_buffer.shape[-2] > self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Move oldest tokens from buffer to quantized storage."""
        B = self.B
        n_flush = self.key_buffer.shape[-2] - self.buffer_size

        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]
        self.key_buffer = self.key_buffer[..., n_flush:, :]
        self.value_buffer = self.value_buffer[..., n_flush:, :]

        new_key_q = self.key_quantizer.quantize(keys_flush)
        new_val_q = quantize_values(
            values_flush, bits=self.value_bits,
            group_size=self.value_group_size, backend=self._backend_name,
        )

        if self.key_quantized is None:
            self.key_quantized = new_key_q
            self.value_quantized = new_val_q
        else:
            self.key_quantized = ProdQuantized(
                mse_indices=B.cat([self.key_quantized.mse_indices, new_key_q.mse_indices], dim=-2),
                qjl_signs=B.cat([self.key_quantized.qjl_signs, new_key_q.qjl_signs], dim=-2),
                residual_norms=B.cat([self.key_quantized.residual_norms, new_key_q.residual_norms], dim=-1),
                norms=B.cat([self.key_quantized.norms, new_key_q.norms], dim=-1),
                mse_bits=new_key_q.mse_bits,
            )
            self.value_quantized = ValueQuantized(
                data=B.cat([self.value_quantized.data, new_val_q.data], dim=-2),
                scales=B.cat([self.value_quantized.scales, new_val_q.scales], dim=-2),
                zeros=B.cat([self.value_quantized.zeros, new_val_q.zeros], dim=-2),
                bits=self.value_bits,
            )

    def attention_scores(self, query, scale: float = None):
        """
        Compute attention logits: score[i,j] = <query_i, key_j> / sqrt(d).

        Args:
            query: (batch, n_heads, n_q, head_dim)
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            scores: (batch, n_heads, n_q, seq_len)
        """
        B = self.B
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        scores_parts = []

        if self.key_quantized is not None:
            scores_quant = self.key_quantizer.attention_score(query, self.key_quantized)
            scores_parts.append(scores_quant * scale)

        if self.key_buffer is not None:
            scores_buf = B.matmul(query, B.transpose(self.key_buffer, -2, -1))
            scores_parts.append(scores_buf * scale)

        return B.cat(scores_parts, dim=-1)

    def attend(self, attn_weights):
        """
        Compute attention output: out = softmax(scores) @ values.

        Args:
            attn_weights: (batch, n_heads, n_q, seq_len) — already softmaxed

        Returns:
            output: (batch, n_heads, n_q, head_dim)
        """
        B = self.B
        output_parts = []
        col_offset = 0

        if self.value_quantized is not None:
            n_quant = self.value_quantized.data.shape[-2]
            w_quant = attn_weights[..., col_offset:col_offset + n_quant]
            v_dequant = dequantize_values(
                self.value_quantized, self.value_group_size, self._backend_name
            )
            output_parts.append(B.matmul(w_quant, v_dequant))
            col_offset += n_quant

        if self.value_buffer is not None:
            n_buf = self.value_buffer.shape[-2]
            w_buf = attn_weights[..., col_offset:col_offset + n_buf]
            output_parts.append(B.matmul(w_buf, self.value_buffer))

        return sum(output_parts)

    def memory_bytes(self) -> dict:
        """Estimate memory usage of the cache."""
        B = self.B
        info = {"quantized_keys": 0, "quantized_values": 0, "buffer": 0, "total": 0}

        if self.key_quantized is not None:
            info["quantized_keys"] += B.nelement(self.key_quantized.mse_indices)
            info["quantized_keys"] += B.nelement(self.key_quantized.qjl_signs)
            info["quantized_keys"] += B.nelement(self.key_quantized.residual_norms) * 2
            info["quantized_keys"] += B.nelement(self.key_quantized.norms) * 2

        if self.value_quantized is not None:
            info["quantized_values"] += B.nelement(self.value_quantized.data)
            info["quantized_values"] += B.nelement(self.value_quantized.scales) * 2
            info["quantized_values"] += B.nelement(self.value_quantized.zeros) * 2

        if self.key_buffer is not None:
            info["buffer"] += B.nelement(self.key_buffer) * 2
        if self.value_buffer is not None:
            info["buffer"] += B.nelement(self.value_buffer) * 2

        info["total"] = info["quantized_keys"] + info["quantized_values"] + info["buffer"]
        return info

    def get_seq_length(self) -> int:
        return self.seq_len
