import pytest
import torch

from conftest import requires_sm120
from vit_nvfp4.nvfp4 import gemm as G
from vit_nvfp4.nvfp4.check import assert_gemm_correct


# ViT-representative GEMM shapes (all K, N multiples of 16).
SHAPES = [
    (256, 512, 768),       # small
    (256, 1024, 1024),     # DINOv2-large hidden
    (1024, 1152, 4304),    # SigLIP2/Qwen3-VL SO400M MLP fc1
    (512, 1408, 6144),     # V-JEPA2 ViT-g MLP
]


@requires_sm120
def test_torch_backend_is_default():
    # auto-registered at import; must outrank reference in the ladder.
    assert "torch_scaled_mm_v2" in G.available_backends()
    assert G._default_backend() == "torch_scaled_mm_v2"


@requires_sm120
@pytest.mark.parametrize("M,K,N", SHAPES)
@pytest.mark.parametrize("dist", ["normal", "heavy", "outlier"])
def test_torch_nvfp4_matches_reference(M, K, N, dist):
    m = assert_gemm_correct("torch_scaled_mm_v2", M, K, N, dist=dist, device="cuda")
    assert m["cos_emul"] >= 0.999 and not m["all_zero"]
