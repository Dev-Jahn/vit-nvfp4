import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import requires_sm120
from vit_nvfp4.nvfp4 import format as fmt
from vit_nvfp4.nvfp4.triton_cast import cast_nvfp4
from vit_nvfp4.nvfp4.backends.torch_scaled_mm import gemm_packed
from vit_nvfp4.ptq.fused_mlp import FusedNVFP4MLP, fuse_mlps


def _g(t):
    return (t.abs().amax().clamp(min=1e-12) / (fmt.E2M1_MAX * fmt.E4M3_MAX)).to(torch.float32)


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


# (M, K, N) — K, N multiples of 16; M % 128 == 0 for the Triton fast path.
SHAPES = [(8192, 1024, 4096), (8192, 768, 3072), (4096, 1152, 4608)]


@requires_sm120
@pytest.mark.parametrize("M,K,N", SHAPES)
def test_cast_nvfp4_gemm_packed_matches_bf16(rand_tensor, M, K, N):
    x = rand_tensor(M, K, dtype=torch.bfloat16)
    w = rand_tensor(N, K, dtype=torch.bfloat16, seed=1)
    aq, asf = cast_nvfp4(x, _g(x))
    wq, wsf = cast_nvfp4(w, _g(w))
    out = gemm_packed(aq, asf, _g(x), wq, wsf, _g(w))
    assert aq.dtype == torch.float4_e2m1fn_x2 and asf.dtype == torch.float8_e4m3fn
    assert _cos(F.linear(x, w), out) >= 0.99


@requires_sm120
@pytest.mark.parametrize("M,K,N", [(17, 512, 768), (100, 1024, 4096), (10550, 1024, 1024)])
def test_cast_nvfp4_non_128_aligned_M(rand_tensor, M, K, N):
    """Real per-frame ViT M is rarely %128 (e.g. 10550); the masked tail kernel path must
    stay correct (just slower), not silently corrupt."""
    x = rand_tensor(M, K, dtype=torch.bfloat16)
    w = rand_tensor(N, K, dtype=torch.bfloat16, seed=1)
    aq, asf = cast_nvfp4(x, _g(x))
    wq, wsf = cast_nvfp4(w, _g(w))
    out = gemm_packed(aq, asf, _g(x), wq, wsf, _g(w))
    assert _cos(F.linear(x, w), out) >= 0.99


@requires_sm120
@pytest.mark.parametrize("M,K,N", SHAPES)
def test_cast_nvfp4_gelu_fused(rand_tensor, M, K, N):
    """cast_nvfp4(x, apply_gelu=True) must equal a separate GELU then cast (the whole point)."""
    x = rand_tensor(M, K, dtype=torch.bfloat16)
    w = rand_tensor(N, K, dtype=torch.bfloat16, seed=1)
    gh = _g(F.gelu(x))
    hq, hsf = cast_nvfp4(x, gh, apply_gelu=True)
    wq, wsf = cast_nvfp4(w, _g(w))
    fused = gemm_packed(hq, hsf, gh, wq, wsf, _g(w))
    assert _cos(F.linear(F.gelu(x), w), fused) >= 0.99


@requires_sm120
def test_fused_nvfp4_mlp_matches_bf16(rand_tensor):
    M, E = 8192, 1024
    x = rand_tensor(M, E, dtype=torch.bfloat16)
    fc1 = nn.Linear(E, 4 * E).cuda().bfloat16()
    fc2 = nn.Linear(4 * E, E).cuda().bfloat16()
    ref = fc2(F.gelu(fc1(x)))
    mlp = FusedNVFP4MLP(fc1, fc2)
    out = mlp(x)
    assert mlp.gx is not None and mlp.gh is not None          # calibrated on first forward
    assert _cos(ref, out) >= 0.97
    # custom-op wrapped forward must survive torch.compile (the small-model unlock)
    assert _cos(ref, torch.compile(mlp)(x)) >= 0.97


@requires_sm120
def test_fuse_mlps_skips_ends():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 1024)
            self.act = nn.GELU()                          # exact GELU -> passes require_gelu
            self.fc2 = nn.Linear(1024, 256)

    class Net(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.blocks = nn.ModuleList(Block() for _ in range(n))

    model = Net(6).cuda().bfloat16()
    n = fuse_mlps(model, skip_ends=2)
    assert n == 2                                              # 6 blocks - 2 ends each side
    fused = [isinstance(b, FusedNVFP4MLP) for b in model.blocks]
    assert fused == [False, False, True, True, False, False]


@requires_sm120
def test_fuse_mlps_skips_gated_and_non_gelu():
    class Gated(nn.Module):                                    # SwiGLU — must NOT be fused (wrong math)
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(256, 1024)
            self.up_proj = nn.Linear(256, 1024)
            self.down_proj = nn.Linear(1024, 256)

    class TanhMLP(nn.Module):                                  # tanh-GELU — must NOT be fused (kernel is erf-GELU)
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(256, 1024)
            self.act = nn.GELU(approximate="tanh")
            self.fc2 = nn.Linear(1024, 256)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Gated(), TanhMLP(), Gated(), TanhMLP()])

    model = Net().cuda().bfloat16()
    with pytest.warns(UserWarning, match="not confirmed exact GELU"):
        assert fuse_mlps(model, skip_ends=0) == 0             # gated + tanh both skipped
    assert not any(isinstance(b, FusedNVFP4MLP) for b in model.blocks)
