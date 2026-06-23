import torch


def test_torch_sees_sm120():
    assert torch.cuda.is_available()
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) == (12, 0), f"expected sm_120, got {(major, minor)}"


def test_torch_cuda_is_13x():
    assert torch.version.cuda is not None
    assert torch.version.cuda.startswith("13"), torch.version.cuda


def test_bf16_matmul_runs():
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    assert c.shape == (64, 64) and torch.isfinite(c).all()
