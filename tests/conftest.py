import pytest
import torch


def _is_sm120():
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)


requires_sm120 = pytest.mark.skipif(not _is_sm120(), reason="requires sm_120 GPU")


@pytest.fixture
def rand_tensor():
    def _make(*shape, dist="normal", device="cuda", dtype=torch.float32, seed=0):
        g = torch.Generator(device=device).manual_seed(seed)
        if dist == "normal":
            return torch.randn(*shape, generator=g, device=device, dtype=dtype)
        if dist == "heavy":  # heavy-tailed
            return (torch.randn(*shape, generator=g, device=device, dtype=dtype)
                    * torch.empty(*shape, device=device, dtype=dtype).exponential_(0.5, generator=g))
        if dist == "outlier":  # inject a few large-magnitude rows (CLS/register-like)
            x = torch.randn(*shape, generator=g, device=device, dtype=dtype)
            x[..., :2, :] *= 30.0
            return x
        raise ValueError(dist)
    return _make
