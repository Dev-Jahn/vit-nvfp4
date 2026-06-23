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
