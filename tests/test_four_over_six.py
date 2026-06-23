"""Four Over Six (per-block 6-vs-4 MSE scale selection) — arXiv:2512.02010.

Verifies the technique preserves the NVFP4 on-wire format invariants and never
increases per-block MSE versus the default scale-to-6 policy.
"""
import torch

from vit_nvfp4.nvfp4 import quantize_to_nvfp4, dequantize_nvfp4
from vit_nvfp4.nvfp4 import format as fmt


def _blocks(x, blk=16):
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // blk, blk)


def test_invariants_and_mse_monotone():
    torch.manual_seed(0)
    x = torch.randn(128, 768) * 0.1
    x[0, :16] += torch.randn(16) * 2.0  # block with a wide spread -> exercises 4-vs-6 dead zone

    c6, b6, g6 = quantize_to_nvfp4(x, 16, block_select="six")
    cm, bm, gm = quantize_to_nvfp4(x, 16, block_select="mse")

    # Format invariant: FP32 global = amax/(6*448), identical for both policies.
    assert torch.equal(g6, gm)
    assert torch.allclose(g6, x.abs().amax() / (fmt.E2M1_MAX * fmt.E4M3_MAX))
    # Codes stay standard E2M1 (0..15); block scale stays E4M3.
    assert cm.dtype == torch.uint8 and int(cm.max()) <= 15
    assert bm.dtype == torch.float8_e4m3fn and bm.shape == b6.shape

    # Per-block MSE is never worse with the MSE pick.
    xb = _blocks(x)
    mse6 = ((dequantize_nvfp4(c6, b6, g6).reshape(xb.shape) - xb) ** 2).mean(-1)
    msem = ((dequantize_nvfp4(cm, bm, gm).reshape(xb.shape) - xb) ** 2).mean(-1)
    assert (msem <= mse6 + 1e-9).all()
    # And strictly better in aggregate on this tensor.
    assert msem.mean() < mse6.mean()


def test_each_block_scale_is_six_or_four_variant():
    torch.manual_seed(1)
    x = torch.randn(64, 256)
    _, bm, gm = quantize_to_nvfp4(x, 16, block_select="mse")
    _, b6, _ = quantize_to_nvfp4(x, 16, block_select="six")
    _, b4, _ = quantize_to_nvfp4(x, 16, block_select="mse")  # sanity dtype

    # Every chosen block scale must equal either the scale-to-6 or scale-to-4 E4M3 value.
    s_enc = 1.0 / gm
    xb = _blocks(x)
    amax_b = xb.abs().amax(-1).clamp(min=1e-12)
    cand6 = ((amax_b / 6.0) * s_enc).clamp(max=fmt.E4M3_MAX).to(torch.float8_e4m3fn).float()
    cand4 = ((amax_b / 4.0) * s_enc).clamp(max=fmt.E4M3_MAX).to(torch.float8_e4m3fn).float()
    chosen = bm.float()
    assert (torch.isclose(chosen, cand6) | torch.isclose(chosen, cand4)).all()


if __name__ == "__main__":
    test_invariants_and_mse_monotone()
    test_each_block_scale_is_six_or_four_variant()
    print("OK")
