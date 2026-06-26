"""Route flagged ViT attention modules through SageAttention3 (FP4 Blackwell attention).

At 1080p the per-frame attention is a large fraction of the forward and is BF16; SA3
(``sageattn3``) runs it in FP4 for ~1.2x at batch 1 (it does NOT win at large batch,
where flash SDPA already saturates the GPU — so process frames one at a time).

SA3 is an **optional** dependency (``pip install 'vit-nvfp4[accel]'``). The kernel call is
wrapped in a ``torch.library.custom_op`` so ``torch.compile`` can compile the rest of the
model around it (the same reason ``FusedNVFP4MLP`` is wrapped). Constraint: head_dim ∈ {64, 128}.
"""
import torch
import torch.nn.functional as F

from .policy import block_index

_IMPL = "vit_nvfp4_sa3"
_SA3_FN = None


def _sa3_kernel(q, k, v, is_causal):
    global _SA3_FN
    if _SA3_FN is None:
        try:
            from sageattn3 import sageattn3_blackwell
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SA3 attention needs SageAttention3. Install the accel extra: "
                "pip install 'vit-nvfp4[accel]' (sm_120 build required)."
            ) from e
        _SA3_FN = sageattn3_blackwell
    return _SA3_FN(q, k, v, is_causal=is_causal)


@torch.library.custom_op("vit_nvfp4::sa3_attn", mutates_args=())
def _sa3_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    # q,k,v are (B, H, S, D); SA3 returns (B, H, S, D) -> transpose to (B, S, H, D) per the
    # transformers attention-interface return contract.
    out = _sa3_kernel(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), bool(is_causal))
    return out.transpose(1, 2).contiguous()


@_sa3_op.register_fake
def _(q, k, v, is_causal):
    b, h, s, d = q.shape
    return q.new_empty(b, s, h, d, dtype=torch.bfloat16)   # SA3 always returns bf16


def _sa3_attention(module, query, key, value, attention_mask=None, dropout=0.0,
                   scaling=None, is_causal=None, **kwargs):
    if getattr(module, "_sa3", False):
        return _sa3_op(query, key, value, bool(is_causal)), None
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                         dropout_p=dropout, scale=scaling, is_causal=bool(is_causal))
    return out.transpose(1, 2).contiguous(), None


def sa3_attn_policy(num_layers: int, skip_first: int = 0, skip_last: int = 0,
                    head_dims=(64, 128)):
    """``(name, module) -> bool`` selecting interface-calling attention modules whose
    head_dim is SA3-supported. Unlike the MLP, the attention ends are *not* especially
    sensitive, so skip_first/last default to 0."""
    lo, hi = skip_first, num_layers - skip_last

    def _head_dim(module):
        hd = getattr(module, "head_dim", None)
        if hd is None and getattr(module, "scaling", None):
            hd = round(1.0 / (module.scaling ** 2))   # scaling = 1/sqrt(head_dim)
        return hd

    def should(name, module) -> bool:
        if "Attention" not in type(module).__name__ or not hasattr(module, "scaling"):
            return False
        if _head_dim(module) not in head_dims:
            return False
        i = block_index(name)
        return i is not None and lo <= i < hi

    return should


def enable_sa3_attention(model, should_quantize_attn) -> int:
    """Route attention modules matched by ``should_quantize_attn`` through SA3, the rest
    through exact SDPA. Returns the number of attention modules switched to SA3."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if _IMPL not in ALL_ATTENTION_FUNCTIONS.valid_keys():
        ALL_ATTENTION_FUNCTIONS.register(_IMPL, _sa3_attention)
    n, cfgs = 0, set()
    for name, mod in model.named_modules():
        if "Attention" in type(mod).__name__ and getattr(mod, "config", None) is not None:
            if id(mod.config) not in cfgs:
                # don't clobber a pre-existing custom attention impl
                if getattr(mod.config, "_attn_implementation", None) in (None, "sdpa", "eager"):
                    mod.config._attn_implementation = _IMPL
                cfgs.add(id(mod.config))
        if should_quantize_attn(name, mod):
            mod._sa3 = True
            n += 1
    return n
