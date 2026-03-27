"""
TurboQuant for Apple Silicon.

Port of 0xSero/turboquant from NVIDIA Triton to MLX Metal + PyTorch CPU.
Implements the TurboQuant algorithm (Zandieh et al., ICLR 2026) for
KV cache compression with near-optimal distortion rates.
"""

__version__ = "0.1.0"

from turboquant_mac.codebook import get_codebook, compute_lloyd_max_codebook
from turboquant_mac.quantizer import TurboQuantMSE, TurboQuantProd, MSEQuantized, ProdQuantized
from turboquant_mac.kv_cache import TurboQuantKVCache, ValueQuantized

__all__ = [
    "get_codebook",
    "compute_lloyd_max_codebook",
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantKVCache",
    "MSEQuantized",
    "ProdQuantized",
    "ValueQuantized",
]
