import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import requires_sm120
from vit_nvfp4.ptq.qattention import _quant_attention, vit_attn_policy


class FakeAttention(nn.Module):  # *Attention class name + a `scaling` attr = interface-calling module
    def __init__(self):
        super().__init__()
        self.scaling = 0.125


def _qkv():
    q = torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.bfloat16)
    return q, torch.randn_like(q), torch.randn_like(q)


@requires_sm120
def test_quant_attention_fp8_matches_sdpa_and_returns_BSHD():
    q, k, v = _qkv()
    m = FakeAttention(); m._nvfp4_attn, m._nvfp4_qk, m._nvfp4_pv = True, "fp8", "fp8"
    out, w = _quant_attention(m, q, k, v, None, scaling=0.125)
    assert w is None and out.shape == (2, 16, 4, 64)  # (B, S, H, D) per the HF contract
    ref = F.scaled_dot_product_attention(q, k, v, scale=0.125).transpose(1, 2)
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0)
    assert cos > 0.99, cos


@requires_sm120
def test_quant_attention_unflagged_is_exact_sdpa():
    q, k, v = _qkv()
    out, _ = _quant_attention(FakeAttention(), q, k, v, None, scaling=0.125)  # no _nvfp4_attn flag
    ref = F.scaled_dot_product_attention(q, k, v, scale=0.125).transpose(1, 2).contiguous()
    assert torch.equal(out, ref)


def test_vit_attn_policy_selects_middle_attention_only():
    pol = vit_attn_policy(12, skip_first=2, skip_last=2, container="encoder.layer")
    a = FakeAttention()
    assert pol("encoder.layer.5.attention.attention", a)
    assert not pol("encoder.layer.1.attention.attention", a)        # skipped first
    assert not pol("encoder.layer.10.attention.attention", a)       # skipped last
    assert not pol("encoder.layer.5.mlp.fc1", nn.Linear(4, 4))      # not an attention module
    assert not pol("encoder.layer.5.attention.output", nn.Module()) # no scaling / wrong class
    assert not pol("predictor.layer.5.attention.attention", a)      # wrong container
