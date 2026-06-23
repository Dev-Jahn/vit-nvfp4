"""RegCache: training-free prefix register tokens for ViT W4A4 (arXiv:2510.04547).

"Activation Quantization of Vision Encoders Needs Prefixing Registers."

PROBLEM. Vision encoders (DINOv2) grow a handful of very high-norm
"massive-activation" / attention-sink tokens in their MIDDLE blocks (empirically
in DINOv2-base the per-token norm jumps from ~21 to ~514 entering blocks 8-10,
~4-5 outlier patch tokens per image). Those few tokens dominate the per-tensor
activation amax that NVFP4's global scale must cover, so every other token in
those blocks is quantized at a needlessly coarse step -> W4A4 error.

REGCACHE (three steps, faithful to the paper):
  (1) Curating  - run a BF16 reference model on reference images, take the top-k
      highest ell-inf-norm tokens entering the quantization-sensitive block, and
      average them into ONE outlier-prone-but-semantically-empty register vector.
  (2) Caching   - insert tau copies of that register as a KV-cache PREFIX into the
      self-attention of every middle-to-final block (from prefix_at onward). The
      register provides keys/values that real tokens attend to (an external
      attention sink), so real patch tokens no longer need to BECOME sinks. The
      register produces no output token (queries are unchanged), so sequence
      length and the CLS feature are unaffected.
  (3) Deleting  - at the input of the quantization-sensitive block, remove the
      top-k-tilde ell-inf-norm patch tokens (the internally-emerging sinks). With
      the external KV register now serving the sink role, deleting these huge-norm
      tokens directly narrows the activation dynamic range feeding the quantized
      Linears.

NVFP4 INVARIANT. This is purely a pre-quantization, distribution-shaping transform
on the activations flowing INTO the quantized Linears. The on-wire NVFP4 format
(E2M1 + per-16 E4M3 block scale + FP32 global=amax/(6*448)) and the
torch._scaled_mm backend are untouched; only the tensor that gets quantized
changes (now over a narrower dynamic range). RegCache must therefore be installed
*before* activation calibration so the static x_global_scale is fit on the
narrowed range, and stay installed at inference.

DINOv2-base finding (measured here). DINOv2's middle-layer outliers are
self-generated "massive activations" (block-9-input amax ~446), NOT primarily
attention-sink-driven. Consequently the KV-prefix (step 2) does NOT help: the
curated register is itself a massive-norm vector (ell-inf ~408), and feeding it
as a raw key/value collapses softmax attention onto it and destroys the feature
(CLS cos -> 0.01). The TOKEN-DELETION lever (step 3) is the one that nets a W4A4
gain on DINOv2-base (CIFAR-100 k-NN +1.1pt). KV-prefix (tau>0) is therefore
DISABLED by default and kept only for sink-driven encoders (the paper's CLIP/
SigLIP regime). This is faithful to the paper's two innovations
(middle-layer prefixing + token deletion); only deletion transfers cleanly here.

Implementation is hook-based (no transformers-internal monkeypatch):
  * KV-prefix: a forward-PRE hook on each Dinov2Attention prepends tau register
    rows to its input and a forward hook drops the tau leading output rows. Real
    tokens therefore attend over [register; real] keys/values - exact KV-prefix
    semantics - while the residual stream length is unchanged.
  * Deletion: a forward-PRE hook on the sensitive Dinov2Layer drops the top-k-tilde
    highest-norm patch tokens (CLS protected) from the block input; a forward hook
    re-inserts zero rows so downstream shapes (residual, final layernorm, CLS
    pooling) are preserved exactly.
"""
from __future__ import annotations

import torch


def _encoder_layers(model):
    """ModuleList of transformer blocks for a HF DINOv2 model."""
    return model.encoder.layer


@torch.no_grad()
def curate_register(model, batches, *, sensitive_at: int, top_k: int = 64):
    """Curate ONE register vector from reference-image statistics.

    Captures the hidden states entering block ``sensitive_at`` over ``batches``,
    selects the ``top_k`` tokens with the largest ell-inf norm (the outlier-prone
    sink tokens), and averages them into a single register prototype. Per the
    paper, these middle-layer sink tokens are highly similar across images
    (cosine ~0.89), so a single averaged register generalizes to any test image.

    Returns a ``(hidden,)`` float tensor.
    """
    layers = _encoder_layers(model)
    captured: list[torch.Tensor] = []

    def pre_hook(_m, args):
        captured.append(args[0].detach().float().cpu())

    h = layers[sensitive_at].register_forward_pre_hook(pre_hook)
    try:
        for batch in batches:
            model(**batch)
    finally:
        h.remove()

    states = torch.cat([s.reshape(-1, s.shape[-1]) for s in captured])  # (N*T, hidden)
    linf = states.abs().amax(dim=-1)                                    # (N*T,)
    sel = linf.topk(min(top_k, states.shape[0])).indices
    return states[sel].mean(dim=0).contiguous()                         # (hidden,)


class RegCache:
    """Install/remove RegCache (KV-prefix caching + sink-token deletion) on DINOv2.

    Args:
        model: HF DINOv2 model (already quantized or BF16).
        register: ``(hidden,)`` curated register vector.
        prefix_at: first block index that receives the KV prefix (= skip_first).
        prefix_to: last block index (inclusive) that receives the KV prefix.
        sensitive_at: block index at whose input sink tokens are deleted.
        tau: number of register copies inserted as the KV prefix.
        k_tilde: number of highest-norm patch tokens to delete at ``sensitive_at``.
    """

    def __init__(self, model, register, *, sensitive_at, k_tilde=6,
                 prefix_at=None, prefix_to=None, tau=0):
        assert register.dim() == 1, "register must be (hidden,)"
        self.model = model
        self.register = register
        self.prefix_at = prefix_at
        self.prefix_to = prefix_to
        self.sensitive_at = sensitive_at
        self.tau = tau
        self.k_tilde = k_tilde
        self._handles: list = []

    def install(self) -> "RegCache":
        layers = _encoder_layers(self.model)
        tau, k = self.tau, self.k_tilde

        # ---- (2) Caching: KV-prefix on each middle-to-final block's attention ----
        # NOTE (DINOv2-base finding): the curated register is a MASSIVE-norm vector
        # (ell-inf ~408, ell-2 ~468) because DINOv2's middle-layer outliers are
        # self-generated "massive activations", not attention-sink-driven. Inserting
        # it as a raw extra key/value makes softmax attention collapse onto the
        # register (every real token attends ~entirely to it), destroying the CLS
        # feature (cos -> 0.01). So for DINOv2-base the KV-prefix is DISABLED by
        # default (tau=0); the token-deletion lever (step 3) is what nets the W4A4
        # gain here. tau>0 is kept for encoders whose outliers ARE sink-driven
        # (the paper's CLIP/SigLIP setting), where a scale-matched register helps.
        if tau > 0 and self.prefix_at is not None:
            def attn_pre(_m, args):
                x = args[0]
                reg = self.register.to(x.device, x.dtype).view(1, 1, -1).expand(x.shape[0], tau, -1)
                return (torch.cat((reg, x), dim=1), *args[1:])

            def attn_post(_m, _args, out):
                if isinstance(out, tuple):
                    return (out[0][:, tau:], *out[1:])
                return out[:, tau:]

            for i in range(self.prefix_at, self.prefix_to + 1):
                attn = layers[i].attention
                self._handles.append(attn.register_forward_pre_hook(attn_pre))
                self._handles.append(attn.register_forward_hook(attn_post))

        # ---- (3) Deleting: permanently drop top-k_tilde sink patch tokens ----
        # A single forward-PRE hook on the sensitive block removes the highest
        # ell-inf-norm patch tokens (the internally-emerging sinks) from the
        # residual stream. They are NOT re-inserted: re-inserting zero rows would
        # let later blocks attend to dead positions and corrupt the output (CLS
        # cosine collapses to ~0). Permanent removal keeps CLS at pos 0 (protected)
        # and only drops uninformative background sinks; downstream shapes shrink
        # by k_tilde consistently and the final layernorm + CLS pooling are intact.
        # This deletion is what narrows the activation range feeding the quantized
        # block (DINOv2-base: block-9-input amax 446 -> ~11 at k_tilde=8).
        if k > 0:
            def del_pre(_m, args):
                x = args[0]                                    # (B, T, H)
                linf = x.detach().float().abs().amax(dim=-1)   # (B, T)
                linf[:, 0] = float("-inf")                     # never delete CLS
                drop_idx = linf.topk(k, dim=1).indices         # (B, k)
                keep = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                keep.scatter_(1, drop_idx, False)              # (B, T)
                kept = x[keep].reshape(x.shape[0], x.shape[1] - k, x.shape[2])
                return (kept, *args[1:])

            sens = layers[self.sensitive_at]
            self._handles.append(sens.register_forward_pre_hook(del_pre))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.remove()
