"""Fused single-pass NVFP4 activation cast (Triton), optionally folding a producer
GELU into the cast so an MLP down-projection's wide-input quantization is free.

One @triton.jit kernel does: (optional GELU) -> per-16 block amax -> E4M3 block scale
-> E2M1 pack (2/byte) -> 128x4 scale swizzle -> store. Output (packed float4_e2m1fn_x2
qdata + swizzled E4M3 scale) feeds ``backends.torch_scaled_mm.gemm_packed`` directly.

The kernel and the ``nvfp4_scale_swizzle`` / ``convert_fp32_to_fp4_packed`` / ``_fp32_to_e8m0``
primitives are vendored from Meta's MSLK (meta-pytorch/MSLK,
``mslk/quantize/triton/legacy/{quantize,primitives}.py``), BSD-3-Clause, so the package has
no MSLK runtime dependency. Only local change: the ``APPLY_GELU`` constexpr branch.

K % 16 == 0 required. Triton fast path wants M % 128 == 0 and N % 64 == 0.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def nvfp4_scale_swizzle(offs_m):
    """
    Produces scale offsets swizzled according to the blackwell 128x4 scale layout.
    Each 128x4 layout can be viewed as 32 4x4 layouts, each of which we'll refer to below as a sub_layout.

    The returned offsets assume a 128x4 layout starting at 0. offs_m could be a subset of rows within a 128x4 layout.
    """
    # Offset of the 4x4 sub_layout within the 128x4 layout
    sub_layout_idx = offs_m % 32
    sub_layout_stride = 16
    sub_layout_off = sub_layout_idx * sub_layout_stride
    # Which row within the 4x4 sub_layout
    sub_layout_row = offs_m // 32
    # Offsets of the elements within 4x4 sub_layout
    elems = tl.arange(0, 4)[None, :]
    sub_layout_elem_offs = sub_layout_row * 4 + elems

    scale_offs = sub_layout_off + sub_layout_elem_offs

    return scale_offs


@triton.jit
def convert_fp32_to_fp4_packed(x_pairs):
    """Convert FP32 pairs to packed FP4 format.

    This function takes tensor where consecutive values along the last dimension
    are packed together into single bytes.

    Args:
        x_pairs: [Tensor, Tensor] both w/ shapes [..., 1] where zipped last dimension contains
                interleaved pairs of FP32 values to be packed together.

    Returns:
        Packed tensor with shape [...] (last dimension removed) where each
        element is an int8 containing 2 FP4 values:
        - First value of pair → low nibble (bits 0-3)
        - Second value of pair → high nibble (bits 4-7)

    Example:
        Input:  [128, 32, 2] containing FP32 pairs
        Output: [128, 32] containing packed FP4 bytes

    """

    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )

    return x_fp4x2


@triton.jit
def _fp32_to_e8m0(
    unscale,
    mbits: tl.constexpr,
    scale_round_mode: tl.constexpr,
):
    E8M0_EXPONENT_BIAS: tl.constexpr = 127  # type: ignore[Incompatible variable type]
    sign = tl.where(unscale < 0, -1.0, 1.0)
    abs_tensor = tl.abs(unscale)

    # MBITS_F32 = 23
    if scale_round_mode == "even":
        val_to_add = (1 << (23 - mbits - 1)) - 1
    elif scale_round_mode == "ceil":
        val_to_add = (1 << 23) - 1
    else:
        val_to_add = 0

    mask_exponent = ((1 << (8 + 1)) - 1) << 23
    mask_mantissa = (1 << 23) - 1

    fp32_bits = tl.extra.cuda.libdevice.float_as_int(abs_tensor)
    fp32_bits_exp = (fp32_bits + val_to_add) & mask_exponent
    exponent = (fp32_bits_exp >> 23) & 0xFF

    if scale_round_mode == "nv_round":
        mantissa = fp32_bits & mask_mantissa
        is_denormal = (exponent == 0) & (mantissa != 0)
        is_normal = ~is_denormal
        condition1 = is_normal & (exponent < 254) & (mantissa > 0)
        condition2 = is_denormal & (mantissa / (2**23) > 0.5)

        exponent = tl.where(condition1 | condition2, exponent + 1, exponent)

    exponent = exponent.to(tl.float32)
    e8m0_values = sign * tl.exp2(exponent - E8M0_EXPONENT_BIAS)

    unscale = e8m0_values
    # In case unscale=0 (scale will be inf), or unscale=inf or nan, we set the scale to 1.0
    unscale_invalid_mask = (
        (e8m0_values == 0)
        | (e8m0_values == float("inf"))
        | (e8m0_values == float("nan"))
    )
    unscale = tl.where(unscale_invalid_mask, 1.0, unscale)

    return unscale


def cast_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    apply_gelu: bool = False,
    use_e8m0_scale: bool = False,
    use_precise_math: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to NVFP4 in one fused Triton pass, optionally GELU-first.

    Args:
        x: Input tensor (last dim K must be a multiple of 16).
        per_tensor_scale: FP32 per-tensor global scale, same convention as
            ``quant.quantize_to_nvfp4`` (``amax / (6 * 448)``). The decode is
            ``x ≈ code_value * block_e4m3 * per_tensor_scale``. None -> no global (1.0).
        apply_gelu: If True, apply GELU to ``x`` inside the kernel before quantizing
            (folds an MLP down-projection's input cast into its producer GELU).
        use_e8m0_scale, use_precise_math: low-level kernel knobs (leave as default).

    Returns:
        ``(qdata, scale)`` — ``qdata`` packed E2M1 (``float4_e2m1fn_x2``, last dim K//2)
        and ``scale`` the per-16 E4M3 block scale already in 128x4 swizzled layout.
        Feed both straight to ``backends.torch_scaled_mm.gemm_packed``.
    """
    # The kernel multiplies each block scale by `global_scale`; for the two-level decode
    # above it must use 1 / per_tensor_scale (see torchao mslk wrapper convention).
    global_scale = per_tensor_scale.reciprocal() if per_tensor_scale is not None else None
    # reshape to 2d
    orig_leading_dims, orig_N = x.shape[:-2], x.shape[-1]
    x = x.reshape(-1, orig_N)

    M, N = x.shape
    assert N % 16 == 0, "N must be divisible by 16 for NVFP4 quantization"

    # Calculate blocks needed
    num_scales = N // 16
    n_row_blocks = triton.cdiv(M, 128)
    n_col_blocks = triton.cdiv(num_scales, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    xq = x.new_empty(M, N // 2, dtype=torch.uint8)
    scales = x.new_empty(padded_rows, padded_cols, dtype=torch.float8_e4m3fn)

    # For small M use lower M_PER_BLOCK to reduce wasted FP32 math
    M_PER_BLOCK = min(triton.next_power_of_2(M), 128)
    # We don't support multiple 128x4 layouts per block
    assert M_PER_BLOCK <= 128

    # If we are not aligned to M_PER_BLOCK * 64, use a mask
    USE_MASK = M % M_PER_BLOCK != 0 or N % 64 != 0

    grid = (triton.cdiv(N, 64), triton.cdiv(M, M_PER_BLOCK))
    # If M_PER_BLOCK is not 128 launch an extra set of blocks along M to handle zeroing scales.
    # This is needed as otherwise the kernel would not visit those scales, and the spec requires padded scales to be zero.
    if M_PER_BLOCK != 128:
        grid = (grid[0], grid[1] + 1)

    use_global_scale = global_scale is not None
    if not use_global_scale:
        # Pass a dummy pointer; the kernel won't load from it.
        global_scale = x.new_empty(())

    # Use int64 indexing when pointer offsets can exceed INT32_MAX
    use_int64_indexing = M * N > 2**31 - 1

    _cast_nvfp4_kernel[grid](
        x,
        global_scale,
        xq,
        scales,
        x.stride(0),
        x.stride(1),
        M,
        N,
        # pyre-ignore[6]
        M_PER_BLOCK=M_PER_BLOCK,
        # pyre-ignore[6]
        USE_MASK=USE_MASK,
        # pyre-ignore[6]
        USE_E8M0_SCALE=use_e8m0_scale,
        # pyre-ignore[6]
        USE_PRECISE_MATH=use_precise_math,
        # pyre-ignore[6]
        USE_GLOBAL_SCALE=use_global_scale,
        # pyre-ignore[6]
        USE_INT64_INDEXING=use_int64_indexing,
        # pyre-ignore[6]
        APPLY_GELU=apply_gelu,
    )

    # reshape back to original shape
    scales = scales.view(*orig_leading_dims, -1, padded_cols)
    xq = xq.view(*orig_leading_dims, -1, N // 2)

    return xq.view(torch.float4_e2m1fn_x2), scales


@triton.jit
def _cast_nvfp4_kernel(
    x_ptr,
    global_scale_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    USE_MASK: tl.constexpr,
    USE_E8M0_SCALE: tl.constexpr,
    USE_PRECISE_MATH: tl.constexpr,
    USE_GLOBAL_SCALE: tl.constexpr,
    USE_INT64_INDEXING: tl.constexpr,
    APPLY_GELU: tl.constexpr,
):
    E4M3_EPS = 1.5258789e-05
    FP8_E4M3_MAX = 448.0
    FP4_E2M1_MAX = 6.0
    INV_FP4_E2M1_MAX = 1.0 / 6.0

    NUM_ELEM_PER_LAYOUT = 128 * 4
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    # Special blocks that zeros out tail M scales if M_PER_BLOCK != 128
    # Technically this is a data race as we zero out scales another block has also zero'd out.
    # Since we write the same value, it shouldn't be an issue.
    if M_PER_BLOCK != 128 and pid_m * M_PER_BLOCK >= M:
        # This is only used (and supported) when M < 128.
        tl.device_assert(pid_m == 1, "pid_m != 1 when M_PER_BLOCK != 128")

        # Offset of the 128x4 layout
        layout_off = pid_n * NUM_ELEM_PER_LAYOUT
        offs_m = tl.arange(0, 128)[:, None]
        scale_offs = layout_off + nvfp4_scale_swizzle(offs_m)

        oob_mask = (offs_m >= M) & tl.full((4,), True, dtype=tl.int1)[None, :]
        zero_scales = tl.full([128, 4], 0, dtype=tl.float8e4nv)
        tl.store(s_ptr + scale_offs, zero_scales, mask=oob_mask)
        return

    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None

    if USE_GLOBAL_SCALE:
        global_scale = tl.load(global_scale_ptr)  # Scalar
    else:
        global_scale = 1.0

    load_offsets = offs_m * stride_xm + offs_n * stride_xn
    x = tl.load(x_ptr + load_offsets, mask=mask, other=other)  # [M_PER_BLOCK, 64]
    xf = x.to(tl.float32)
    if APPLY_GELU:  # fuse the producer GELU into the cast (one HBM pass)
        xf = 0.5 * xf * (1.0 + tl.erf(xf * 0.7071067811865476))
    x_blocks = xf.reshape(M_PER_BLOCK, 4, 16)  # [M_PER_BLOCK, 4, 16]

    # Block-wise max
    block_amax = tl.max(tl.abs(x_blocks), axis=2)  # [M_PER_BLOCK, 4]

    # To avoid expensive per-element tl.div_rn we can multiply by the reciprocal.
    # This could introduce ~1ULP differnce. However as the scales are casted
    # from FP32 to FP4 right after, for most FP32 values this is equivalent still.
    # We gate this to USE_PRECISE_MATH=False.
    if USE_E8M0_SCALE:
        if USE_PRECISE_MATH:
            scales = tl.div_rn(block_amax, 4.0) * global_scale
        else:
            scales = block_amax * 0.25 * global_scale
        scales = _fp32_to_e8m0(scales, mbits=1, scale_round_mode="even")
    else:
        if USE_PRECISE_MATH:
            scales = tl.div_rn(block_amax, FP4_E2M1_MAX) * global_scale
        else:
            scales = block_amax * INV_FP4_E2M1_MAX * global_scale
        scales = tl.clamp(scales, E4M3_EPS, FP8_E4M3_MAX)

    scales = scales.to(tl.float8e4nv)  # [M_PER_BLOCK, 4]

    # Apply combined scale to data
    total_scale = tl.div_rn(
        global_scale, scales.to(tl.float32)[:, :, None]
    )  # [M_PER_BLOCK, 4, 1]
    x_blocks = x_blocks * total_scale  # [M_PER_BLOCK, 4, 16]

    if USE_MASK:
        scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
        scale_mask = (offs_m < M) & (scale_offs_n < (N // 16))

        # Mask out scales to 0 if we are not aligned to M_PER_BLOCK x 64
        scales = tl.where(
            scale_mask,
            scales,
            0.0,
        )

    offs_m = (pid_m * M_PER_BLOCK % 128) + tl.arange(0, M_PER_BLOCK)[:, None]
    # Offset of the 128x4 layout
    layout_off = (
        (pid_m * M_PER_BLOCK) // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    scale_offs = layout_off + nvfp4_scale_swizzle(offs_m)
    tl.store(
        s_ptr + scale_offs,
        scales,
    )

    # Convert to FP4
    x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(M_PER_BLOCK, 32, 2).split())
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    offs_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N // 2)
    else:
        mask = None

    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    store_offsets = offs_m * (N // 2) + offs_n
    tl.store(q_ptr + store_offsets, x_fp4x2, mask=mask)
