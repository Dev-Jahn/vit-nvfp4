import torch
from vit_nvfp4.nvfp4.quant import quantize_to_nvfp4, dequantize_nvfp4


def test_roundtrip_error_bounded():
    x = torch.randn(4, 256, dtype=torch.float32)
    codes, bscale, gscale = quantize_to_nvfp4(x, block=16)
    xh = dequantize_nvfp4(codes, bscale, gscale)
    assert xh.shape == x.shape
    cos = torch.nn.functional.cosine_similarity(x.flatten(), xh.flatten(), dim=0)
    assert cos > 0.98, cos


def test_block_scale_dtype_and_shape():
    x = torch.randn(3, 64, dtype=torch.float32)
    codes, bscale, gscale = quantize_to_nvfp4(x, block=16)
    assert codes.dtype == torch.uint8 and codes.shape == (3, 64)
    assert bscale.dtype == torch.float8_e4m3fn and bscale.shape == (3, 4)
    assert gscale.dtype == torch.float32 and gscale.numel() == 1


def test_static_global_scale_path():
    x = torch.randn(2, 32, dtype=torch.float32)
    _, _, g = quantize_to_nvfp4(x, block=16)
    codes, bscale, g2 = quantize_to_nvfp4(x, block=16, global_scale=g)
    assert torch.equal(g, g2)
