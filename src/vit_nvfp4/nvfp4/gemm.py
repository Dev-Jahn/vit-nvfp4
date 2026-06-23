import torch

from .quant import quantize_to_nvfp4
from .backends import reference

_BACKENDS = {"reference": reference.gemm}

# ladder priority: real tensor-core kernels register ahead of reference once validated.
_PRIORITY = ("flashinfer_b12x", "cutlass79", "cublaslt", "reference")


def register_backend(name, fn):
    _BACKENDS[name] = fn


def available_backends():
    return list(_BACKENDS)


def _default_backend():
    for name in _PRIORITY:
        if name in _BACKENDS:
            return name
    return "reference"


def nvfp4_gemm(a_codes, a_bs, a_gs, b_codes, b_bs, b_gs, *, out_dtype=torch.bfloat16, backend=None):
    backend = backend or _default_backend()
    return _BACKENDS[backend](a_codes, a_bs, a_gs, b_codes, b_bs, b_gs, out_dtype=out_dtype)


def nvfp4_linear(x, w_codes, w_bs, w_gs, *, x_global_scale=None, bias=None, backend=None):
    """W4A4 linear: quantize activation online (dynamic block scale), call NVFP4 GEMM."""
    x_codes, x_bs, x_gs = quantize_to_nvfp4(x, 16, global_scale=x_global_scale)
    y = nvfp4_gemm(x_codes, x_bs, x_gs, w_codes, w_bs, w_gs, out_dtype=x.dtype, backend=backend)
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y
