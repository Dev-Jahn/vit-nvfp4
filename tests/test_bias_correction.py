import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import requires_sm120
from vit_nvfp4.ptq.qlinear import QuantLinear
from vit_nvfp4.ptq.bias_correction import correct_bias


@requires_sm120
def test_correct_bias_zeros_mean_output_shift():
    """After correction, E_tokens[quant_out] matches E_tokens[ref_linear(x)]."""
    torch.manual_seed(0)
    lin = nn.Linear(64, 96).cuda().bfloat16()
    ref = nn.Sequential(lin).cuda().eval()
    quant = nn.Sequential(QuantLinear.from_linear(lin)).cuda().eval()

    batches = [torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

    # mean per-feature shift BEFORE correction
    with torch.no_grad():
        rs = torch.zeros(96, device="cuda")
        qs = torch.zeros(96, device="cuda")
        n = 0
        for b in batches:
            rs += ref(b).float().sum(0)
            qs += quant(b).float().sum(0)
            n += b.shape[0]
        shift_before = ((rs - qs) / n).abs().mean().item()

    n_corr = correct_bias(ref, quant, batches)
    assert n_corr == 1

    with torch.no_grad():
        rs = torch.zeros(96, device="cuda")
        qs = torch.zeros(96, device="cuda")
        n = 0
        for b in batches:
            rs += ref(b).float().sum(0)
            qs += quant(b).float().sum(0)
            n += b.shape[0]
        shift_after = ((rs - qs) / n).abs().mean().item()

    # correction must drive the mean shift toward zero (bf16-limited residual)
    assert shift_after < shift_before * 0.1, (shift_before, shift_after)


@requires_sm120
def test_correct_bias_creates_buffer_when_no_bias():
    torch.manual_seed(1)
    lin = nn.Linear(32, 32, bias=False).cuda().bfloat16()
    ref = nn.Sequential(lin).cuda().eval()
    quant = nn.Sequential(QuantLinear.from_linear(lin)).cuda().eval()
    assert quant[0].bias is None

    batches = [torch.randn(8, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    correct_bias(ref, quant, batches)

    assert quant[0].bias is not None
    assert "bias" in quant[0]._buffers
    assert quant[0].bias.shape == (32,)
