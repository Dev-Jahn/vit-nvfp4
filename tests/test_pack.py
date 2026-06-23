import torch
from vit_nvfp4.nvfp4.pack import pack_e2m1, unpack_e2m1, pad_to_block


def test_pack_unpack_inverse():
    codes = torch.randint(0, 16, (3, 64), dtype=torch.uint8)
    packed = pack_e2m1(codes)
    assert packed.shape == (3, 32) and packed.dtype == torch.uint8
    assert torch.equal(unpack_e2m1(packed), codes)


def test_pack_nibble_order():
    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)  # lo=even idx, hi=odd idx
    packed = pack_e2m1(codes)
    assert packed.tolist() == [[(2 << 4) | 1, (4 << 4) | 3]]


def test_pad_to_block():
    x = torch.randn(2, 20)
    padded, orig = pad_to_block(x, block=16, dim=-1)
    assert padded.shape == (2, 32) and orig == 20
    assert torch.equal(padded[..., :20], x) and (padded[..., 20:] == 0).all()
