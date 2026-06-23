import torch

from conftest import requires_sm120
from vit_nvfp4.ptq.convert import quantize_model
from vit_nvfp4.ptq.policy import vit_block_policy
from vit_nvfp4.ptq.diagnostics import tensor_cosine

_NAME = "facebook/dinov2-base"


@requires_sm120
def test_dinov2_w4a4_swaps_and_tracks_bf16():
    from transformers import AutoModel

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16)

    ref = AutoModel.from_pretrained(_NAME, dtype=torch.bfloat16).cuda().eval()
    q = AutoModel.from_pretrained(_NAME, dtype=torch.bfloat16).cuda().eval()
    n = quantize_model(q, vit_block_policy(ref.config.num_hidden_layers, skip_first=2, skip_last=2))
    assert n == 48  # 8 middle blocks x 6 Linears

    with torch.no_grad():
        ref_out = ref(x).last_hidden_state
        q_out = q(x).last_hidden_state

    cos = tensor_cosine(ref_out, q_out)
    # PTQ-only baseline (dynamic activation scales, no calibration); observed ~0.95.
    # Guard is set above collapse and below observed to catch regressions, NOT at the
    # >=0.99 spec target — reaching that needs SP3 calibration + SP4/SP5 mitigation.
    assert cos >= 0.93, f"W4A4 DINOv2 last_hidden cosine regressed: {cos:.4f}"


@requires_sm120
def test_dinov2_w4a4_conservative_skip_is_higher_fidelity():
    from transformers import AutoModel

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    ref = AutoModel.from_pretrained(_NAME, dtype=torch.bfloat16).cuda().eval()
    q = AutoModel.from_pretrained(_NAME, dtype=torch.bfloat16).cuda().eval()
    quantize_model(q, vit_block_policy(ref.config.num_hidden_layers, skip_first=4, skip_last=4))
    with torch.no_grad():
        cos = tensor_cosine(ref(x).last_hidden_state, q(x).last_hidden_state)
    assert cos >= 0.97, f"skip(4,4) fidelity regressed: {cos:.4f}"
