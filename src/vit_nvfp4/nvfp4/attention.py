"""Low-precision (FP8 / NVFP4) scaled-dot-product attention — emulated.

Per the precision policy, attention (QKᵀ / softmax / AV) tries NVFP4, falling
back to FP8 (SageAttention-style). Following SP1, this is the *measure-first*
layer: each low-precision matmul is emulated as quantize→dequantize→matmul to
expose the accuracy floor a fused FP8/FP4 tensor-core kernel would hit. softmax
always runs in fp32. A custom fused (online-softmax) CUDA kernel is a later step.

SOTA levers implemented (SageAttention2/3):
  - microscaling NVFP4 (E2M1 + per-16 E4M3 scale) on QKᵀ and PV (SageAttention3),
  - per-row FP8 E4M3 fallback,
  - smooth-K / smooth-V: subtract the per-channel mean of K (resp. V) over the
    token axis. Both are *exact* under a normalized softmax (the K-mean adds a
    per-query constant across keys → softmax-invariant; the V-mean is added back
    analytically since attention rows sum to 1), yet they strip the channel-wise
    outliers that wreck low-bit quantization.
"""
import math

import torch

from .quant import quantize_to_nvfp4, dequantize_nvfp4
from .fp8 import fp8_e4m3_quant_dequant

_MODES = ("bf16", "fp8", "nvfp4")


def _nvfp4_qdq(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """Round-trip ``x`` through NVFP4 along the last dim (zero-pad to ``block``)."""
    k = x.shape[-1]
    pad = (-k) % block
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    codes, bs, gs = quantize_to_nvfp4(x, block)
    deq = dequantize_nvfp4(codes, bs, gs)
    return deq[..., :k] if pad else deq


def _qdq(x: torch.Tensor, mode: str, axis: int = -1) -> torch.Tensor:
    """Emulate quantizing ``x`` to ``mode`` with the contraction along ``axis``."""
    if mode == "bf16":
        return x.to(torch.bfloat16).to(torch.float32)
    if axis not in (-1, x.ndim - 1):
        y = _qdq(x.transpose(axis, -1).contiguous(), mode, -1)
        return y.transpose(axis, -1)
    if mode == "fp8":
        return fp8_e4m3_quant_dequant(x)
    if mode == "nvfp4":
        return _nvfp4_qdq(x)
    raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


def quant_sdpa(query, key, value, attn_mask=None, is_causal=False, scale=None,
               qk="fp8", pv="fp8", smooth_k=True, smooth_v=True):
    """Drop-in low-precision ``F.scaled_dot_product_attention`` (emulated).

    ``query/key/value``: ``(..., S, D)`` (e.g. ``(B, H, S, D)``). ``qk`` selects
    the QKᵀ operand precision, ``pv`` the P·V operand precision; each is one of
    ``'bf16' | 'fp8' | 'nvfp4'``. Returns ``(..., S, D)`` in ``query.dtype``.
    """
    assert qk in _MODES and pv in _MODES
    D = query.shape[-1]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    q, k, v = query.float(), key.float(), value.float()

    if smooth_k:
        k = k - k.mean(dim=-2, keepdim=True)            # exact under softmax
    scores = torch.matmul(_qdq(q, qk), _qdq(k, qk).transpose(-1, -2)) * scale

    if is_causal:
        S_q, S_k = scores.shape[-2], scores.shape[-1]
        causal = torch.ones(S_q, S_k, dtype=torch.bool, device=scores.device).tril(S_k - S_q)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool \
            else scores + attn_mask
    p = torch.softmax(scores, dim=-1)

    mu_v = None
    if smooth_v:
        mu_v = v.mean(dim=-2, keepdim=True)             # added back after PV (rows sum to 1)
        v = v - mu_v
    out = torch.matmul(_qdq(p, pv, axis=-1), _qdq(v, pv, axis=-2))
    if mu_v is not None:
        out = out + mu_v
    return out.to(query.dtype)
