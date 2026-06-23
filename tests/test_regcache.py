"""Unit tests for RegCache hooks (shape preservation, deletion, range narrowing).

Uses a tiny stub model that mimics the HF DINOv2 container surface RegCache
touches (``model.encoder.layer`` of blocks taking/returning (B,T,H)) so the tests
run on CPU without downloading weights.
"""
import torch
import torch.nn as nn

from vit_nvfp4.ptq.regcache import curate_register, RegCache


class _Block(nn.Module):
    """Minimal block: identity residual + an 'attention' submodule (B,T,H)->(B,T,H)."""
    def __init__(self, h):
        super().__init__()
        self.attention = nn.Identity()

    def forward(self, x):
        return self.attention(x)


class _Encoder(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.layer = nn.ModuleList(_Block(h) for _ in range(n))

    def forward(self, x):
        for blk in self.layer:
            x = blk(x)
        return x


class _Model(nn.Module):
    def __init__(self, n=12, h=32):
        super().__init__()
        self.encoder = _Encoder(n, h)

    def forward(self, pixel_values):
        return self.encoder(pixel_values)


def _make_input(B=2, T=20, H=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, H, generator=g)
    # plant a few huge-norm "sink" tokens (never CLS=pos 0)
    x[:, 5] *= 50.0
    x[:, 11] *= 80.0
    x[:, 17] *= 30.0
    return x


def test_curate_register_picks_high_norm():
    m = _Model()
    x = _make_input()
    reg = curate_register(m, [{"pixel_values": x}], sensitive_at=3, top_k=4)
    assert reg.shape == (32,)
    # register should resemble the planted high-norm tokens, i.e. be large-norm
    assert reg.norm() > 5.0


def test_deletion_preserves_shape_and_removes_sinks():
    m = _Model().eval()
    x = _make_input()
    with torch.no_grad():
        base = m(pixel_values=x)
    reg = torch.zeros(32)
    k = 3
    rc = RegCache(m, reg, sensitive_at=4, k_tilde=k).install()
    # capture what block 4 actually receives (post-deletion)
    seen = {}
    h = m.encoder.layer[4].register_forward_pre_hook(lambda _m, a: seen.__setitem__("x", a[0]))
    with torch.no_grad():
        out = m(pixel_values=x)
    h.remove()
    rc.remove()
    # output length shrinks by exactly k (permanent deletion)
    assert out.shape == (base.shape[0], base.shape[1] - k, base.shape[2])
    # the deleted block-4 input no longer contains the largest-norm tokens
    in_linf = seen["x"].abs().amax(dim=-1)              # (B, T-k)
    full_linf = x.abs().amax(dim=-1)                    # (B, T)
    assert in_linf.max() < full_linf.max()              # biggest sink removed
    # CLS (pos 0) is never deleted -> survives unchanged into block 4
    assert torch.allclose(seen["x"][:, 0], x[:, 0])


def test_kv_prefix_disabled_by_default():
    m = _Model().eval()
    x = _make_input()
    reg = torch.randn(32)
    # tau=0 default -> only deletion hooks, attention untouched
    rc = RegCache(m, reg, sensitive_at=4, k_tilde=2)
    rc.install()
    # exactly one hook (the deletion pre-hook) installed
    assert len(rc._handles) == 1
    rc.remove()
    assert len(rc._handles) == 0
