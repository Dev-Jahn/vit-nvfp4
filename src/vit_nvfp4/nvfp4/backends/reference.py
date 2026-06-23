import torch

from ..quant import dequantize_nvfp4


def gemm(a_codes, a_bs, a_gs, b_codes, b_bs, b_gs, out_dtype=torch.bfloat16):
    """fp32 emulated NVFP4 GEMM oracle. A:(M,K), B:(N,K) row-major -> (M,N)."""
    a = dequantize_nvfp4(a_codes, a_bs, a_gs)   # (M, K) fp32
    b = dequantize_nvfp4(b_codes, b_bs, b_gs)   # (N, K) fp32
    return (a @ b.T).to(out_dtype)
