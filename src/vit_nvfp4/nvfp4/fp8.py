import torch

from . import format as fmt


def fp8_e4m3_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Round-trip ``x`` through FP8 E4M3 with a per-row (last-dim) scale.

    scale = amax_row / 448; q = round_e4m3(x/scale); returns ``q * scale`` in
    fp32. This is the emulation of an FP8 operand (the accuracy floor a real
    ``torch._scaled_mm`` FP8 matmul would see), per the measure-first approach.
    """
    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / fmt.E4M3_MAX
    q = (xf / scale).clamp(-fmt.E4M3_MAX, fmt.E4M3_MAX).to(torch.float8_e4m3fn)
    return q.to(torch.float32) * scale
