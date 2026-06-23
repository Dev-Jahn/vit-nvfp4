import torch
import torch.nn.functional as F

from .quant import quantize_to_nvfp4, dequantize_nvfp4
from .gemm import nvfp4_gemm


def assert_gemm_correct(backend, M, K, N, *, dist="normal", device="cuda",
                        cos_emul=0.999, cos_bf16=0.99, seed=1234):
    """Validate a backend's NVFP4 GEMM against fp32-emulated and bf16 references.

    Returns a metrics dict. Raises AssertionError on all-zeros output (cf. flashinfer #2577)
    or on cosine-similarity below thresholds.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn(M, K, generator=g, device=device)
    b = torch.randn(N, K, generator=g, device=device)
    if dist == "outlier":
        a[:2] *= 30.0
    elif dist == "heavy":
        a = a * torch.empty(M, K, device=device).exponential_(0.5, generator=g)

    aq = quantize_to_nvfp4(a, 16)
    bq = quantize_to_nvfp4(b, 16)
    out = nvfp4_gemm(*aq, *bq, out_dtype=torch.float32, backend=backend)

    emul = dequantize_nvfp4(*aq) @ dequantize_nvfp4(*bq).T
    bf16 = (a.bfloat16() @ b.bfloat16().T).float()
    m = {
        "all_zero": bool((out == 0).all()),
        "cos_emul": float(F.cosine_similarity(out.flatten(), emul.flatten(), dim=0)),
        "cos_bf16": float(F.cosine_similarity(out.flatten(), bf16.flatten(), dim=0)),
        "rel_emul": float((out - emul).norm() / emul.norm().clamp(min=1e-9)),
    }
    assert not m["all_zero"], f"{backend} returned all zeros (cf. flashinfer #2577): {m}"
    assert m["cos_emul"] >= cos_emul, m
    assert m["cos_bf16"] >= cos_bf16, m
    return m
