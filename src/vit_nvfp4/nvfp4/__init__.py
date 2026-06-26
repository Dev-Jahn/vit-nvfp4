from .quant import quantize_to_nvfp4, dequantize_nvfp4
from .pack import pack_e2m1, unpack_e2m1, pad_to_block, as_float4_x2
from .gemm import nvfp4_gemm, nvfp4_linear, register_backend, available_backends
from .fp8 import fp8_e4m3_quant_dequant
from .attention import quant_sdpa

# NB: the fused Triton cast (``triton_cast.cast_nvfp4``) is intentionally NOT imported here
# so the eager quant/GEMM core stays importable without triton. Import it explicitly:
#   from vit_nvfp4.nvfp4.triton_cast import cast_nvfp4

__all__ = [
    "quantize_to_nvfp4", "dequantize_nvfp4",
    "pack_e2m1", "unpack_e2m1", "pad_to_block", "as_float4_x2",
    "nvfp4_gemm", "nvfp4_linear", "register_backend", "available_backends",
    "fp8_e4m3_quant_dequant", "quant_sdpa",
]
