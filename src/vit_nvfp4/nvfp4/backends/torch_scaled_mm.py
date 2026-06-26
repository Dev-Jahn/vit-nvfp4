"""W4A4 NVFP4 GEMM via torch native ``_scaled_mm_v2`` (BlockWise1x16 + per-tensor global).

Requires PyTorch >= 2.12 with cu13x on an sm_100+ (Blackwell) GPU.
"""
import torch
from torch.nn.functional import ScalingType, SwizzleType

from ..pack import pack_e2m1, as_float4_x2, to_blocked

_BLK = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
_SWZ = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]


def gemm(a_codes, a_bs, a_gs, b_codes, b_bs, b_gs, out_dtype=torch.bfloat16):
    """A:(M,K) codes, B:(N,K) codes (row-major weight) -> (M,N).

    Two-level NVFP4: per-16 E4M3 block scale (swizzled) + per-tensor FP32 global.
    """
    a_fp4 = as_float4_x2(pack_e2m1(a_codes.contiguous()))          # (M, K//2)
    b_fp4 = as_float4_x2(pack_e2m1(b_codes.contiguous())).t()      # (N,K//2) -> (K//2, N)
    a_sf = to_blocked(a_bs.contiguous())                           # (M, K//16) swizzled
    b_sf = to_blocked(b_bs.contiguous())                           # (N, K//16) swizzled (outer=N)
    a_g = a_gs.reshape(1).float()
    b_g = b_gs.reshape(1).float()
    return torch._scaled_mm_v2(
        a_fp4, b_fp4,
        [a_sf, a_g], _BLK, _SWZ,
        [b_sf, b_g], _BLK, _SWZ,
        None, out_dtype,
    )


def gemm_packed(a_fp4, a_sf, a_g, w_fp4, w_sf, w_g, out_dtype=torch.bfloat16):
    """W4A4 GEMM on operands already packed + swizzled by ``triton_cast.cast_nvfp4``.

    ``a_fp4``:(M, K//2) float4_e2m1fn_x2, ``a_sf``: swizzled E4M3 block scale, ``a_g``: fp32 scalar
    (activation). ``w_fp4``:(N, K//2) float4 (weight, row-major as cast), ``w_sf``: swizzled E4M3
    (N-outer), ``w_g``: fp32 scalar. Returns (M, N). Mirrors ``gemm`` but skips pack/swizzle since
    the fused cast already produced them — this is the fast inference path.
    """
    return torch._scaled_mm_v2(
        a_fp4, w_fp4.t(),                                   # (M,K//2) , (N,K//2)->(K//2,N)
        [a_sf, a_g.reshape(1).float()], _BLK, _SWZ,
        [w_sf, w_g.reshape(1).float()], _BLK, _SWZ,
        None, out_dtype,
    )
