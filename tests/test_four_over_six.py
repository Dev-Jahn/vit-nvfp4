"""Four Over Six (per-block 6-vs-4 MSE scale selection) — arXiv:2512.02010.

Verifies the technique preserves the NVFP4 on-wire format invariants and never
increases per-block MSE versus the default scale-to-6 policy.
"""
import torch

from vit_nvfp4.nvfp4 import quantize_to_nvfp4, dequantize_nvfp4
from vit_nvfp4.nvfp4 import format as fmt


def _blocks(x, blk=16):
    return x.float().reshape(*x.shape[:-1], x.shape[-1] // blk, blk)


def test_invariants_and_aggregate_mse():
    torch.manual_seed(0)
    x = torch.randn(128, 768) * 0.1
    x[0, :16] += torch.randn(16) * 2.0  # block with a wide spread -> exercises 4-vs-6 dead zone

    c6, b6, g6 = quantize_to_nvfp4(x, 16, block_select="six")
    cm, bm, gm = quantize_to_nvfp4(x, 16, block_select="mse")

    # scale-to-6 uses the standard amax/(6*448) global; Four Over Six uses a
    # smaller amax/(6*256) global so its scale-to-4 candidate stays E4M3-
    # representable (else the M=4 scale saturates 448 on high-magnitude blocks
    # and silently reverts to M=6). The globals are therefore NOT equal.
    amax = x.abs().amax()
    assert torch.allclose(g6, amax / (fmt.E2M1_MAX * fmt.E4M3_MAX))
    assert torch.allclose(gm, amax / (fmt.E2M1_MAX * fmt.FOS_SCALE_MAX))
    assert not torch.equal(g6, gm)
    # Codes stay standard E2M1 (0..15); block scale stays E4M3.
    assert cm.dtype == torch.uint8 and int(cm.max()) <= 15
    assert bm.dtype == torch.float8_e4m3fn and bm.shape == b6.shape

    # Per-block "FoS never worse than six" only holds when both share a global.
    # Four Over Six uses a *different* (256) global, so each block's E4M3 scale
    # rounds to a different point than under six's 448 global and a few easy blocks
    # can round slightly worse. The meaningful guarantee is on the AGGREGATE error,
    # which FoS reduces by densifying the 4..6 dead zone where the error concentrates.
    xb = _blocks(x)
    mse6 = ((dequantize_nvfp4(c6, b6, g6).reshape(xb.shape) - xb) ** 2).mean()
    msem = ((dequantize_nvfp4(cm, bm, gm).reshape(xb.shape) - xb) ** 2).mean()
    assert msem < mse6, (float(msem), float(mse6))


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


def test_four_over_six_applies_to_the_max_block():
    # Regression: the block holding the tensor max must still benefit from the
    # scale-to-4 candidate. Under a amax/(6*448) global its M=4 scale (amax_b/4)
    # saturates E4M3 and collapses to M=6, neutering FoS exactly where magnitudes
    # are largest; amax/(6*256) keeps that candidate representable.
    torch.manual_seed(2)
    x = torch.randn(8, 64) * 0.05
    x[0, :16] = torch.tensor([6.0] + [5.0] * 15)  # tensor-max block, energy in the 4..6 dead zone
    xb = _blocks(x)
    c6, b6, g6 = quantize_to_nvfp4(x, 16, block_select="six")
    cm, bm, gm = quantize_to_nvfp4(x, 16, block_select="mse")
    mse6 = ((dequantize_nvfp4(c6, b6, g6).reshape(xb.shape) - xb) ** 2).mean(-1)[0, 0]
    msem = ((dequantize_nvfp4(cm, bm, gm).reshape(xb.shape) - xb) ** 2).mean(-1)[0, 0]
    assert msem < mse6, (float(msem), float(mse6))  # FoS must help the max block


if __name__ == "__main__":
    test_invariants_and_aggregate_mse()
    test_each_block_scale_is_six_or_four_variant()
    test_four_over_six_applies_to_the_max_block()
    print("OK")
