"""Verify GPTQ output-MSE <= RTN output-MSE on a tiny random linear."""
import torch

from vit_nvfp4.nvfp4 import quantize_to_nvfp4, dequantize_nvfp4
from vit_nvfp4.ptq.gptq import HessianObserver, gptq_quantize_weight


def _rtn_dequant(W):
    codes, bs, gs = quantize_to_nvfp4(W.float(), 16)
    return dequantize_nvfp4(codes, bs, gs)


def _gptq_dequant(W, H):
    codes, bs, gs = gptq_quantize_weight(W, H, block=16)
    return dequantize_nvfp4(codes, bs, gs)


def test_gptq_le_rtn():
    torch.manual_seed(0)
    N, K, M = 256, 256, 4096
    W = torch.randn(N, K)
    X = torch.randn(M, K)  # calibration inputs

    H = X.t() @ X

    W_rtn = _rtn_dequant(W)
    W_gptq = _gptq_dequant(W, H)

    # output MSE on the SAME calibration inputs
    Y = X @ W.t()
    mse_rtn = ((X @ W_rtn.t() - Y) ** 2).mean().item()
    mse_gptq = ((X @ W_gptq.t() - Y) ** 2).mean().item()
    print(f"output MSE: RTN={mse_rtn:.6f}  GPTQ={mse_gptq:.6f}  ratio={mse_gptq/mse_rtn:.4f}")
    assert mse_gptq <= mse_rtn, f"GPTQ {mse_gptq} > RTN {mse_rtn}"


if __name__ == "__main__":
    test_gptq_le_rtn()
    print("OK")
