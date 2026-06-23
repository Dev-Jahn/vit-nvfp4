"""GPTQ for NVFP4 (arXiv:2210.17323), ported minimally.

Replaces the inner RTN of ``quantize_to_nvfp4`` with column-wise (OBQ) quantization
plus inverse-Hessian error compensation onto the remaining columns. The on-wire format
is unchanged: E2M1 codes + per-16 E4M3 block scale + FP32 global = amax/(6*448).

The block scale of each 16-wide group (along the input/K dimension) and the per-tensor
global are computed once from the ORIGINAL weight and FROZEN, exactly as RTN does. GPTQ
only changes which E2M1 *code* each weight element gets (via error feedback), so the
decode path and ``torch._scaled_mm_v2`` backend see standard NVFP4 tensors.
"""
import torch

from ..nvfp4 import format as fmt


class HessianObserver:
    """Accumulates H = sum_x (x x^T) over calibration input vectors for one Linear."""

    def __init__(self, in_features: int):
        self.in_features = in_features
        self.H = torch.zeros(in_features, in_features, dtype=torch.float32)
        self.nsamples = 0

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        # x: (..., K) activation feeding the Linear. Flatten leading dims to rows.
        x = x.detach().reshape(-1, self.in_features).float()
        if x.shape[0] == 0:
            return
        if self.H.device != x.device:
            self.H = self.H.to(x.device)
        self.H += x.t() @ x
        self.nsamples += x.shape[0]


def _quantize_block_scales(w: torch.Tensor, block: int, global_scale: torch.Tensor):
    """Per-16-block E4M3 scales (frozen from original weight), matching quant.py.

    Returns ``dequant_scale`` of shape (N, K/block, 1): the float32 decode scale
    ``block_scale_e4m3 * global`` per block (== amax_b / 6 rounded through E4M3).
    """
    s_enc = 1.0 / global_scale
    wb = w.reshape(w.shape[0], w.shape[1] // block, block)
    amax_b = wb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)         # (N, K/blk, 1)
    block_scale = ((amax_b / fmt.E2M1_MAX) * s_enc).clamp(max=fmt.E4M3_MAX)
    block_scale_e4m3 = block_scale.to(torch.float8_e4m3fn)
    dequant_scale = block_scale_e4m3.to(torch.float32) * global_scale     # (N, K/blk, 1)
    return block_scale_e4m3, dequant_scale


@torch.no_grad()
def gptq_quantize_weight(weight: torch.Tensor, H: torch.Tensor,
                         block: int = 16, percdamp: float = 0.01, blocksize: int = 128):
    """GPTQ-quantize one Linear weight to NVFP4.

    Args:
        weight: (N, K) float weight (out, in).
        H:      (K, K) accumulated Hessian sum_x x x^T.
        block:  NVFP4 group size along K (16).
        percdamp: Hessian damping as a fraction of mean(diag).
        blocksize: lazy-batch column block size (GPTQ Algorithm 1).

    Returns ``(codes, block_scale_e4m3, global_scale)`` matching ``quantize_to_nvfp4``.
    """
    W = weight.float().clone()
    N, K = W.shape
    dev = W.device
    H = H.to(dev).clone()

    # Global = MAX over original weight (sole job: keep E4M3 block scales <= 448).
    amax = W.abs().amax().clamp(min=1e-12)
    global_scale = (amax / (fmt.E2M1_MAX * fmt.E4M3_MAX)).to(torch.float32)

    # Frozen per-block decode scale (block_scale_e4m3 * global), shape (N, K/block, 1).
    block_scale_e4m3, dequant_scale = _quantize_block_scales(W, block, global_scale)
    dq = dequant_scale.squeeze(-1)                                         # (N, K/block)

    def quant_col(col_vals: torch.Tensor, j: int) -> torch.Tensor:
        # Quantize one column j (vector over N rows) to its block's NVFP4 grid, return dequantized.
        b = j // block
        scale = dq[:, b]                                                   # (N,)
        codes = fmt.round_to_e2m1_code(col_vals / scale)
        return fmt.code_to_value(codes) * scale

    # --- Hessian preconditioning: damping + dead-column handling + Cholesky inverse ---
    diag = torch.arange(K, device=dev)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    H[diag, diag] += damp

    # Hinv = upper-Cholesky of H^{-1} (GPTQ trick: cholesky_inverse then cholesky upper).
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            j = i1 + i
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quant_col(w, j)
            Q1[:, i] = q
            err = (w - q) / d
            # propagate error to remaining columns within this block
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err

        Q[:, i1:i2] = Q1
        # propagate accumulated block error to all later columns (outside this block)
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    # --- Re-encode the compensated dequantized weights Q into standard NVFP4 codes ---
    # Q holds dequantized values that already lie on the per-block grid, so dividing by
    # the same frozen decode scale and rounding recovers the exact codes (idempotent).
    Qb = Q.reshape(N, K // block, block)
    codes = fmt.round_to_e2m1_code(Qb / dequant_scale).reshape(N, K)
    return codes, block_scale_e4m3.squeeze(-1), global_scale
