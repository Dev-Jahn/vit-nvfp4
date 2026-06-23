import torch


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit E2M1 codes per byte. Low nibble = even index, high nibble = odd index."""
    assert codes.shape[-1] % 2 == 0 and codes.dtype == torch.uint8
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return ((hi << 4) | lo).contiguous()


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return out.to(torch.uint8)


def pad_to_block(x: torch.Tensor, block: int = 16, dim: int = -1):
    """Right-pad ``x`` with zeros along ``dim`` to a multiple of ``block``. Returns (padded, orig_len)."""
    orig = x.shape[dim]
    pad = (-orig) % block
    if pad:
        x = x.movedim(dim, -1)
        x = torch.nn.functional.pad(x, (0, pad))
        x = x.movedim(-1, dim)
    return x, orig


def as_float4_x2(packed: torch.Tensor) -> torch.Tensor:
    """Reinterpret a packed uint8 tensor as ``torch.float4_e2m1fn_x2`` (for kernels that require it)."""
    return packed.view(torch.float4_e2m1fn_x2)


def _ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Swizzle a (H, W) scale matrix into NVIDIA's 128x4 (internally 32x4x4) block-scale layout.

    Returns a flattened tensor as expected by block-scaled GEMM kernels (SWIZZLE_32_4_4).
    Vendored from the reference implementation (transformer_nuggets / torch test internals).
    See https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device,
                             dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()
