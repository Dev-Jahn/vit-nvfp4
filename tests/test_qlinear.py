import torch
import torch.nn as nn

from conftest import requires_sm120
from vit_nvfp4.ptq.qlinear import QuantLinear
from vit_nvfp4.nvfp4 import nvfp4_linear, quantize_to_nvfp4


@requires_sm120
def test_qlinear_matches_nvfp4_linear():
    lin = nn.Linear(256, 128).cuda().bfloat16()
    q = QuantLinear.from_linear(lin)
    x = torch.randn(4, 10, 256, device="cuda", dtype=torch.bfloat16)
    y = q(x)
    assert y.shape == (4, 10, 128)
    # from_linear now defaults to 'mse' (Four Over Six); reference must match that to test equivalence.
    wc, wb, wg = quantize_to_nvfp4(lin.weight.data.float(), 16, block_select="mse")
    ref = nvfp4_linear(x.reshape(-1, 256), wc, wb, wg, bias=lin.bias).reshape(4, 10, 128)
    assert torch.equal(y, ref)


@requires_sm120
def test_qlinear_close_to_bf16():
    lin = nn.Linear(512, 512).cuda().bfloat16()
    q = QuantLinear.from_linear(lin)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    y = q(x).float()
    ref = lin(x).float()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0)
    assert cos > 0.99, cos


@requires_sm120
def test_qlinear_static_global_scale_matches_dynamic_when_equal():
    # Setting the static activation scale to exactly the dynamic value reproduces dynamic output.
    lin = nn.Linear(256, 128).cuda().bfloat16()
    q = QuantLinear.from_linear(lin)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    y_dyn = q(x)
    _, _, gs = quantize_to_nvfp4(x.float(), 16)
    q.set_activation_scale(gs)
    y_static = q(x)
    assert torch.equal(y_dyn, y_static)
    assert q.x_global_scale is not None


@requires_sm120
def test_qlinear_no_bias():
    lin = nn.Linear(128, 64, bias=False).cuda().bfloat16()
    q = QuantLinear.from_linear(lin)
    assert q.bias is None
    y = q(torch.randn(8, 128, device="cuda", dtype=torch.bfloat16))
    assert y.shape == (8, 64)
