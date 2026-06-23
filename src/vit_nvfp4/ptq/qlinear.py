import torch
import torch.nn as nn

from ..nvfp4 import quantize_to_nvfp4, nvfp4_linear


class QuantLinear(nn.Module):
    """nn.Linear replacement with a statically NVFP4-quantized weight (W4A4).

    Weight is quantized once at construction; activations are quantized online
    inside ``nvfp4_linear`` per forward call. Bias stays in the activation dtype.
    """

    def __init__(self, w_codes, w_block_scale, w_global_scale, bias, in_features, out_features):
        super().__init__()
        assert in_features % 16 == 0, "NVFP4 requires in_features (K) divisible by 16"
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("w_codes", w_codes)
        self.register_buffer("w_block_scale", w_block_scale)
        self.register_buffer("w_global_scale", w_global_scale)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantLinear":
        w = linear.weight.data  # (out, in) = (N, K)
        codes, bscale, gscale = quantize_to_nvfp4(w.float(), 16)
        return cls(codes, bscale, gscale, linear.bias, linear.in_features, linear.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)
        x2d = x.reshape(-1, self.in_features)
        y = nvfp4_linear(x2d, self.w_codes, self.w_block_scale, self.w_global_scale, bias=self.bias)
        return y.reshape(out_shape)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4=W4A4"
