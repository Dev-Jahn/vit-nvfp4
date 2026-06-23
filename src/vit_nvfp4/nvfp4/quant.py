import torch

from . import format as fmt


def _encode_block(xb, amax_b, divisor, s_enc, global_scale):
    """Quantize per-16 blocks scaling each block max to FP4 ``divisor`` (6 or 4).

    Returns ``(codes, block_scale_e4m3, dequant)`` for the candidate, all
    block-shaped ``(..., K/blk, blk)`` except the E4M3 scale ``(..., K/blk, 1)``.
    """
    block_scale = ((amax_b / divisor) * s_enc).clamp(max=fmt.E4M3_MAX)
    block_scale_e4m3 = block_scale.to(torch.float8_e4m3fn)
    decoded = block_scale_e4m3.to(torch.float32) * global_scale             # effective per-block scale
    codes = fmt.round_to_e2m1_code(xb / decoded)
    dequant = fmt.code_to_value(codes) * decoded
    return codes, block_scale_e4m3, dequant


def quantize_to_nvfp4(x: torch.Tensor, block: int = 16, global_scale: torch.Tensor | None = None,
                      block_select: str = "six"):
    """Quantize ``x`` to NVFP4 with two-level scaling along the last dim.

    Returns ``(codes, block_scale_e4m3, global_scale)`` where:
      - ``codes``: uint8 E2M1 codes, same shape as ``x`` (last dim = K).
      - ``block_scale_e4m3``: float8_e4m3fn per-16 block scales, shape ``(..., K // block)``.
      - ``global_scale``: fp32 per-tensor scalar.
    Decode: ``x ≈ code_value * block_scale_e4m3 * global_scale``.

    ``block_select``:
      - ``'six'`` (default): always scale each block max to FP4 6.0 (the dead zone
        between 66.6%-100% of block max is left poorly represented).
      - ``'mse'`` (Four Over Six, arXiv:2512.02010): quantize each block twice —
        scaling max to 6.0 and to 4.0 — dequantize both, and keep the lower
        per-block-MSE variant. Scaling to 4.0 makes the FP4 grid denser near the
        block max (3 lands at 75%). On-wire format is unchanged: still standard
        E2M1 codes + per-16 E4M3 block scale (=amax_b/4 or amax_b/6, times 1/global)
        + FP32 global=amax/(6*448), so the backend GEMM decode is untouched.
    """
    assert x.shape[-1] % block == 0, "K must be a multiple of block; pad first"
    assert block_select in ("six", "mse")
    xf = x.to(torch.float32)
    if global_scale is None:
        amax = xf.abs().amax().clamp(min=1e-12)
        global_scale = (amax / (fmt.E2M1_MAX * fmt.E4M3_MAX)).to(torch.float32)
    s_enc = 1.0 / global_scale
    xb = xf.reshape(*xf.shape[:-1], xf.shape[-1] // block, block)
    amax_b = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)            # (..., K/blk, 1)

    codes6, bse6, deq6 = _encode_block(xb, amax_b, fmt.E2M1_MAX, s_enc, global_scale)
    if block_select == "six":
        codes, block_scale_e4m3 = codes6, bse6
    else:
        codes4, bse4, deq4 = _encode_block(xb, amax_b, 4.0, s_enc, global_scale)
        pick4 = (((deq4 - xb) ** 2).mean(dim=-1, keepdim=True)
                 < ((deq6 - xb) ** 2).mean(dim=-1, keepdim=True))           # (..., K/blk, 1)
        codes = torch.where(pick4, codes4, codes6)
        block_scale_e4m3 = torch.where(pick4.squeeze(-1), bse4.squeeze(-1), bse6.squeeze(-1)).unsqueeze(-1)

    codes = codes.reshape(x.shape)                                          # uint8 (..., K)
    return codes, block_scale_e4m3.squeeze(-1), global_scale


def dequantize_nvfp4(codes: torch.Tensor, block_scale_e4m3: torch.Tensor, global_scale: torch.Tensor):
    block = codes.shape[-1] // block_scale_e4m3.shape[-1]
    vals = fmt.code_to_value(codes).reshape(*codes.shape[:-1], block_scale_e4m3.shape[-1], block)
    scale = (block_scale_e4m3.to(torch.float32) * global_scale).unsqueeze(-1)
    return (vals * scale).reshape(codes.shape)
