import torch
import torch.nn as nn

from conftest import requires_sm120
from vit_nvfp4.ptq.qlinear import QuantLinear
from vit_nvfp4.ptq.calibrate import calibrate_activations, _ActObserver


def test_observer_percentile_clips_outlier():
    obs_max = _ActObserver("max")
    obs_pct = _ActObserver("percentile", 0.99)
    x = torch.randn(10000)
    x[0] = 1000.0  # extreme outlier
    obs_max.observe(x)
    obs_pct.observe(x)
    assert obs_max.amax() > 100        # raw max sees the outlier
    assert obs_pct.amax() < 10         # percentile clips it


@requires_sm120
def test_calibrate_sets_static_scales():
    lin = nn.Linear(64, 64).cuda().bfloat16()
    model = nn.Sequential(QuantLinear.from_linear(lin)).cuda()
    batches = [torch.randn(8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    n = calibrate_activations(model, batches, method="percentile")
    assert n == 1
    assert model[0].x_global_scale is not None
    assert model[0].x_global_scale.dtype == torch.float32


@requires_sm120
def test_dinov2_max_calibration_keeps_parity_with_dynamic():
    # 'max' calibration pins a static scale without losing fidelity vs dynamic (~0.95).
    from transformers import AutoModel
    from vit_nvfp4.ptq import quantize_model, vit_block_policy, tensor_cosine

    torch.manual_seed(0)
    calib = [torch.randn(4, 3, 224, 224, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    xeval = torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    ref = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.bfloat16).cuda().eval()
    q = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.bfloat16).cuda().eval()
    quantize_model(q, vit_block_policy(ref.config.num_hidden_layers, 2, 2))
    calibrate_activations(q, calib, method="max")
    with torch.no_grad():
        cos = tensor_cosine(ref(xeval).last_hidden_state, q(xeval).last_hidden_state)
    assert cos >= 0.93, f"max-calibrated static scale regressed vs dynamic: {cos:.4f}"
