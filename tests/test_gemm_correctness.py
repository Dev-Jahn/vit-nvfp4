import torch
import pytest
from vit_nvfp4.nvfp4.quant import quantize_to_nvfp4, dequantize_nvfp4
from vit_nvfp4.nvfp4 import gemm as G


def _make_operands(M, K, N, dist, dtype=torch.float32, dev="cpu"):
    g = torch.Generator(device=dev).manual_seed(1234)
    a = torch.randn(M, K, generator=g, device=dev, dtype=dtype)
    b = torch.randn(N, K, generator=g, device=dev, dtype=dtype)
    if dist == "outlier":
        a[:2] *= 30.0
    a_codes, a_bs, a_gs = quantize_to_nvfp4(a, 16)
    b_codes, b_bs, b_gs = quantize_to_nvfp4(b, 16)
    return (a_codes, a_bs, a_gs), (b_codes, b_bs, b_gs)


def _fp4_emulated_ref(aq, bq):
    a = dequantize_nvfp4(*aq)
    b = dequantize_nvfp4(*bq)
    return a @ b.T


@pytest.mark.parametrize("M,K,N", [(64, 256, 64), (128, 1024, 256)])
def test_reference_matches_fp4_emulation(M, K, N):
    aq, bq = _make_operands(M, K, N, "normal")
    out = G.nvfp4_gemm(*aq, *bq, out_dtype=torch.float32, backend="reference")
    ref = _fp4_emulated_ref(aq, bq)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos > 0.9999, cos


def test_nvfp4_linear_shapes():
    x = torch.randn(8, 64)
    w = torch.randn(32, 64)
    w_codes, w_bs, w_gs = quantize_to_nvfp4(w, 16)
    y = G.nvfp4_linear(x, w_codes, w_bs, w_gs, bias=torch.zeros(32), backend="reference")
    assert y.shape == (8, 32)
