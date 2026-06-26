"""Fused NVFP4 W4A4 transformer MLP (``fc1 -> GELU -> fc2``) for fast inference.

The win over per-Linear ``QuantLinear`` is the **GELU→NVFP4 producer fusion**: fc2's
input is ``GELU(fc1_out)``, and a standalone NVFP4 cast of that wide (4·hidden) tensor
costs more than fc2's GEMM saves, so naive W4A4 *loses* on the down-projection. Folding
the GELU into the cast (``cast_nvfp4(..., apply_gelu=True)``) makes fc2's input quant free,
so the whole MLP runs ~1.3-1.5x vs BF16 (see ``FUSEDMLP_NVFP4_RESULT.md``).

The forward is wrapped in a ``torch.library.custom_op`` so ``torch.compile`` fuses the
*rest* of the model (LayerNorms, residuals, attention/RoPE prep) around it — without this
the eager overhead cancels the GEMM win on small models. Per-tensor activation global
scales are calibrated on the first forward and frozen (static).
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nvfp4 import format as fmt
from ..nvfp4.triton_cast import cast_nvfp4
from ..nvfp4.backends.torch_scaled_mm import gemm_packed


def _global_scale(t: torch.Tensor) -> torch.Tensor:
    """Per-tensor NVFP4 global scale (amax / (6 * 448)), matching ``quantize_to_nvfp4``."""
    if t.numel() == 0:                                          # empty (e.g. zero-length seq) -> no-op scale
        return torch.ones((), device=t.device, dtype=torch.float32)
    return (t.abs().amax().clamp(min=1e-12) / (fmt.E2M1_MAX * fmt.E4M3_MAX)).to(torch.float32)


def _quant_weight(linear: nn.Linear):
    g = _global_scale(linear.weight.data)
    q, sf = cast_nvfp4(linear.weight.data.to(torch.bfloat16), g)   # (N, K//2) float4 + swizzled E4M3
    return q, sf, g


@torch.library.custom_op("vit_nvfp4::fused_mlp", mutates_args=())
def _fused_mlp(
    x: torch.Tensor,
    w1q: torch.Tensor, w1sf: torch.Tensor, w1g: torch.Tensor, b1: torch.Tensor | None,
    w2q: torch.Tensor, w2sf: torch.Tensor, w2g: torch.Tensor, b2: torch.Tensor | None,
    gx: torch.Tensor, gh: torch.Tensor, k1: int, n2: int,
) -> torch.Tensor:
    M = x.numel() // k1
    x2 = x.reshape(M, k1).to(torch.bfloat16)
    aq, asf = cast_nvfp4(x2, gx)                       # up-projection input cast
    y1 = gemm_packed(aq, asf, gx, w1q, w1sf, w1g)
    if b1 is not None:
        y1 = y1 + b1
    hq, hsf = cast_nvfp4(y1, gh, apply_gelu=True)      # GELU fused into the down-proj input cast
    y2 = gemm_packed(hq, hsf, gh, w2q, w2sf, w2g)
    if b2 is not None:
        y2 = y2 + b2
    return y2.reshape(*x.shape[:-1], n2)


@_fused_mlp.register_fake
def _(x, w1q, w1sf, w1g, b1, w2q, w2sf, w2g, b2, gx, gh, k1, n2):
    # the real op always returns bf16 (gemm_packed out_dtype) regardless of x's dtype
    return x.new_empty(*x.shape[:-1], n2, dtype=torch.bfloat16)


class FusedNVFP4MLP(nn.Module):
    """Drop-in replacement for a ``fc1 -> GELU -> fc2`` MLP with NVFP4 W4A4 + GELU fusion.

    Assumes the block activation is (exact) GELU — true for DINOv2/3, V-JEPA2, etc.
    """

    def __init__(self, fc1: nn.Linear, fc2: nn.Linear):
        super().__init__()
        assert fc1.out_features == fc2.in_features, "not an fc1->fc2 MLP pair"
        assert fc1.in_features > 0 and fc1.in_features % 16 == 0 and fc2.in_features % 16 == 0, \
            "NVFP4 needs K > 0 and K % 16 == 0"
        assert fc1.weight.is_cuda, "FusedNVFP4MLP requires CUDA weights (sm_120 NVFP4 GEMM)"
        w1q, w1sf, w1g = _quant_weight(fc1)
        w2q, w2sf, w2g = _quant_weight(fc2)
        self.register_buffer("w1q", w1q); self.register_buffer("w1sf", w1sf); self.register_buffer("w1g", w1g)
        self.register_buffer("w2q", w2q); self.register_buffer("w2sf", w2sf); self.register_buffer("w2g", w2g)
        # bias in bf16 so ``y1 + b1`` stays bf16 (fp32 bias would silently up-cast the GEMM output)
        self.register_buffer("b1", None if fc1.bias is None else fc1.bias.detach().to(torch.bfloat16))
        self.register_buffer("b2", None if fc2.bias is None else fc2.bias.detach().to(torch.bfloat16))
        self.k1 = fc1.in_features
        self.n2 = fc2.out_features
        # static per-tensor activation globals, calibrated on first forward
        self.register_buffer("gx", None)
        self.register_buffer("gh", None)

    @torch.no_grad()
    def _calibrate(self, x: torch.Tensor) -> None:
        x2 = x.reshape(-1, self.k1).to(torch.bfloat16)
        gx = _global_scale(x2)
        aq, asf = cast_nvfp4(x2, gx)
        y1 = gemm_packed(aq, asf, gx, self.w1q, self.w1sf, self.w1g)
        if self.b1 is not None:
            y1 = y1 + self.b1
        self.gx = gx
        self.gh = _global_scale(F.gelu(y1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gx is None:
            self._calibrate(x)
        return _fused_mlp(x, self.w1q, self.w1sf, self.w1g, self.b1,
                          self.w2q, self.w2sf, self.w2g, self.b2,
                          self.gx, self.gh, self.k1, self.n2)

    def extra_repr(self) -> str:
        return f"in={self.k1}, hidden={self.w1q.shape[0]}, out={self.n2}, nvfp4=W4A4+gelu_fused"


# HF activation class names: exact erf-GELU (safe to fuse) vs approximations (must NOT).
_EXACT_GELU = {"GELUActivation"}
_APPROX_GELU = {"PytorchGELUTanh", "NewGELUActivation", "QuickGELUActivation",
                "FastGELUActivation", "GELUTanh", "ClippedGELUActivation"}


def _is_exact_gelu(module: nn.Module):
    """True if the MLP's activation child is exact erf-GELU, False if a known approximation,
    None if undetectable (no activation submodule found)."""
    for child in module.children():
        cn = type(child).__name__
        if cn in _EXACT_GELU:
            return True
        if cn in _APPROX_GELU:
            return False
        if isinstance(child, nn.GELU):
            return child.approximate == "none"
    return None


def _mlp_pair(module: nn.Module):
    """Return ``(fc1, fc2)`` if ``module`` is a plain (non-gated) fc1->GELU->fc2 MLP, else None."""
    if getattr(module, "gate_proj", None) is not None:         # SwiGLU/gated MLP — not fc1->GELU->fc2
        return None
    for up, down in (("fc1", "fc2"), ("up_proj", "down_proj")):
        f1, f2 = getattr(module, up, None), getattr(module, down, None)
        if (isinstance(f1, nn.Linear) and isinstance(f2, nn.Linear)
                and f1.out_features == f2.in_features and f1.in_features % 16 == 0):
            return f1, f2
    return None


def fuse_mlps(model: nn.Module, skip_ends: int = 2, require_gelu: bool = True) -> int:
    """In-place swap of each transformer block's GELU MLP with ``FusedNVFP4MLP``.

    ``skip_ends``: leave the first/last N blocks' MLPs in BF16 (the accuracy-sensitive
    ends — the standard PTQ rule; ``skip_ends=2`` lifts feature cos from ~0.87 to ~0.97).
    ``require_gelu`` (default True): only fuse MLPs whose activation is confirmed exact
    erf-GELU (the fused kernel applies erf-GELU); MLPs with a tanh/quick GELU variant or an
    undetectable activation are left in BF16 with a warning — set False to force-fuse.
    Gated/SwiGLU MLPs (``gate_proj``) are always skipped. Returns the number of MLPs fused.
    """
    cands = [(n, m, p) for n, m in model.named_modules() if (p := _mlp_pair(m)) is not None]
    if require_gelu:
        bad = [n for n, m, _ in cands if _is_exact_gelu(m) is not True]
        if bad:
            warnings.warn(f"fuse_mlps: leaving {len(bad)} MLP(s) in BF16 — activation not confirmed "
                          f"exact GELU (e.g. {bad[:3]}). Pass require_gelu=False to force.")
        cands = [(n, m, p) for n, m, p in cands if _is_exact_gelu(m) is True]
    if skip_ends and 2 * skip_ends >= len(cands):
        raise ValueError(f"skip_ends={skip_ends} too large for {len(cands)} fusible MLP block(s)")
    keep = cands[skip_ends:len(cands) - skip_ends] if skip_ends else cands
    for name, _module, pair in keep:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, FusedNVFP4MLP(*pair))
    return len(keep)
