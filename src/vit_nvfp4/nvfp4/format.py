import torch

E2M1_MAX = 6.0
E4M3_MAX = 448.0

# magnitude per 3-bit index (E2M1 representable magnitudes)
_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
# midpoints between consecutive magnitudes (nearest-rounding boundaries)
_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def round_to_e2m1_code(scaled: torch.Tensor) -> torch.Tensor:
    """Round scaled values (expected |x| <~ 6) to the nearest E2M1 code (0..15).

    Bit layout: bit3 = sign, bits2..0 = magnitude index into _MAG.
    """
    sign = (scaled < 0).to(torch.uint8)
    mag = scaled.abs().clamp(max=E2M1_MAX)
    idx = torch.bucketize(mag, _MID.to(mag.device)).to(torch.uint8)  # 0..7
    return (sign << 3) | idx


def code_to_value(code: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 codes (0..15) back to float magnitudes with sign."""
    code = code.to(torch.long)
    sign = 1.0 - 2.0 * ((code >> 3) & 1).to(torch.float32)
    mag = _MAG.to(code.device)[code & 0x7]
    return sign * mag
