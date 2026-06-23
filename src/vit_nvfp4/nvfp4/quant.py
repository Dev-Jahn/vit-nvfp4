import torch

from . import format as fmt


def quantize_to_nvfp4(x: torch.Tensor, block: int = 16, global_scale: torch.Tensor | None = None):
    """Quantize ``x`` to NVFP4 with two-level scaling along the last dim.

    Returns ``(codes, block_scale_e4m3, global_scale)`` where:
      - ``codes``: uint8 E2M1 codes, same shape as ``x`` (last dim = K).
      - ``block_scale_e4m3``: float8_e4m3fn per-16 block scales, shape ``(..., K // block)``.
      - ``global_scale``: fp32 per-tensor scalar.
    Decode: ``x ≈ code_value * block_scale_e4m3 * global_scale``.
    """
    assert x.shape[-1] % block == 0, "K must be a multiple of block; pad first"
    xf = x.to(torch.float32)
    if global_scale is None:
        amax = xf.abs().amax().clamp(min=1e-12)
        global_scale = (amax / (fmt.E2M1_MAX * fmt.E4M3_MAX)).to(torch.float32)
    s_enc = 1.0 / global_scale
    xb = xf.reshape(*xf.shape[:-1], xf.shape[-1] // block, block)
    amax_b = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)            # (..., K/blk, 1)
    block_scale = ((amax_b / fmt.E2M1_MAX) * s_enc).clamp(max=fmt.E4M3_MAX)
    block_scale_e4m3 = block_scale.to(torch.float8_e4m3fn)
    decoded = block_scale_e4m3.to(torch.float32) * global_scale             # ≈ amax_b / 6
    codes = fmt.round_to_e2m1_code(xb / decoded).reshape(x.shape)           # uint8 (..., K)
    return codes, block_scale_e4m3.squeeze(-1), global_scale


def dequantize_nvfp4(codes: torch.Tensor, block_scale_e4m3: torch.Tensor, global_scale: torch.Tensor):
    block = codes.shape[-1] // block_scale_e4m3.shape[-1]
    vals = fmt.code_to_value(codes).reshape(*codes.shape[:-1], block_scale_e4m3.shape[-1], block)
    scale = (block_scale_e4m3.to(torch.float32) * global_scale).unsqueeze(-1)
    return (vals * scale).reshape(codes.shape)
