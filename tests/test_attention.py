import math

import torch
import torch.nn.functional as F

from conftest import requires_sm120
from vit_nvfp4.nvfp4 import quant_sdpa, quantize_to_nvfp4, nvfp4_gemm, dequantize_nvfp4


def _qkv(B=2, H=4, S=64, D=64, seed=0, outlier=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g, device="cuda")
    k = torch.randn(B, H, S, D, generator=g, device="cuda")
    v = torch.randn(B, H, S, D, generator=g, device="cuda")
    if outlier:  # ViT-like: a couple of high-norm tokens + a heavy channel
        k[..., :2, :] *= 12.0
        k[..., 7] *= 8.0
        v[..., :2, :] *= 10.0
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def _cos(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


@requires_sm120
def test_quant_sdpa_shape_and_dtype():
    q, k, v = _qkv()
    out = quant_sdpa(q, k, v, qk="fp8", pv="fp8")
    ref = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == ref.shape == (2, 4, 64, 64)
    assert out.dtype == q.dtype


@requires_sm120
def test_quant_sdpa_bf16_matches_reference():
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = quant_sdpa(q, k, v, qk="bf16", pv="bf16", smooth_k=False, smooth_v=False)
    assert _cos(out, ref) > 0.999, _cos(out, ref)


@requires_sm120
def test_quant_sdpa_causal_matches_reference():
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    out = quant_sdpa(q, k, v, is_causal=True, qk="bf16", pv="bf16", smooth_k=False, smooth_v=False)
    assert _cos(out, ref) > 0.999, _cos(out, ref)


@requires_sm120
def test_quant_sdpa_fp8_high_fidelity():
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = quant_sdpa(q, k, v, qk="fp8", pv="fp8")
    assert _cos(out, ref) > 0.99, _cos(out, ref)


@requires_sm120
def test_quant_sdpa_nvfp4_reasonable():
    q, k, v = _qkv()
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    out = quant_sdpa(q, k, v, qk="nvfp4", pv="nvfp4")
    assert _cos(out, ref) > 0.95, _cos(out, ref)


@requires_sm120
def test_smooth_k_is_softmax_exact():
    # Subtracting the per-channel mean of K over tokens leaves softmax(QKᵀ) unchanged.
    q, k, _ = _qkv()
    q, k = q.float(), k.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    p0 = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
    ks = k - k.mean(dim=-2, keepdim=True)
    p1 = torch.softmax(torch.matmul(q, ks.transpose(-1, -2)) * scale, dim=-1)
    assert torch.allclose(p0, p1, atol=1e-5), (p0 - p1).abs().max()


@requires_sm120
def test_smooth_v_is_exact():
    # P@(V-μ) + μ == P@V for any row-stochastic P (attention rows sum to 1).
    _, _, v = _qkv()
    v = v.float()
    B, H, S, _ = v.shape
    p = torch.softmax(torch.randn(B, H, S, S, device="cuda"), dim=-1)
    mu = v.mean(dim=-2, keepdim=True)
    out = torch.matmul(p, v - mu) + mu
    assert torch.allclose(out, torch.matmul(p, v), atol=1e-4), (out - torch.matmul(p, v)).abs().max()


@requires_sm120
def test_nvfp4_emulation_matches_kernel():
    # The emulated NVFP4 QKᵀ (qdq→matmul) tracks the real nvfp4_gemm tensor-core
    # path — i.e. the measured accuracy floor is representative of the kernel.
    g = torch.Generator(device="cuda").manual_seed(3)
    qf = torch.randn(64, 64, generator=g, device="cuda")
    kf = torch.randn(64, 64, generator=g, device="cuda")
    qc, qbs, qgs = quantize_to_nvfp4(qf, 16)
    kc, kbs, kgs = quantize_to_nvfp4(kf, 16)
    kernel = nvfp4_gemm(qc, qbs, qgs, kc, kbs, kgs, out_dtype=torch.float32)  # Q @ Kᵀ
    emul = dequantize_nvfp4(qc, qbs, qgs) @ dequantize_nvfp4(kc, kbs, kgs).t()
    assert _cos(kernel, emul) > 0.999, _cos(kernel, emul)
