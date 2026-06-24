"""Wire low-precision NVFP4/FP8 attention (``quant_sdpa``) into HF ViT models.

Registers a custom attention impl in ``ALL_ATTENTION_FUNCTIONS`` that routes
*flagged* middle-block attention modules through ``quant_sdpa`` (FP8 by default,
per the precision policy: attention tries NVFP4 → falls back to FP8) and falls
back to plain SDPA everywhere else. Orthogonal to the ``QuantLinear`` weight quant
— enable either or both. HF looks up ``self.config._attn_implementation`` per
forward, so this can be switched on after the model is loaded.

The registered fn signature + (B,H,S,D) operands + ``out.transpose(1,2)`` return
contract follow transformers 5.x ``sdpa_attention_forward``.
"""
import torch.nn.functional as F

from .policy import block_index

_IMPL = "nvfp4_attn"


def _quant_attention(module, query, key, value, attention_mask=None, dropout=0.0,
                     scaling=None, is_causal=None, **kwargs):
    from ..nvfp4.attention import quant_sdpa
    if getattr(module, "_nvfp4_attn", False):
        out = quant_sdpa(query, key, value, attn_mask=attention_mask, is_causal=bool(is_causal),
                         scale=scaling, qk=getattr(module, "_nvfp4_qk", "fp8"),
                         pv=getattr(module, "_nvfp4_pv", "fp8"))
    else:  # not a quantized block — exact standard SDPA
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                             dropout_p=dropout, scale=scaling, is_causal=bool(is_causal))
    return out.transpose(1, 2).contiguous(), None


def vit_attn_policy(num_layers: int, skip_first: int = 2, skip_last: int = 2,
                    container: str | None = None):
    """``(name, module) -> bool`` selecting the interface-calling attention module
    of each *middle* block (mirrors ``vit_block_policy`` for Linears)."""
    lo, hi = skip_first, num_layers - skip_last

    def should(name, module) -> bool:
        # the module that calls the attention interface has a `scaling` attr and an *Attention class name
        if "Attention" not in type(module).__name__ or not hasattr(module, "scaling"):
            return False
        if container is not None:
            seg = container + "."
            if not (name.startswith(seg) or ("." + seg) in name):
                return False
        i = block_index(name)
        return i is not None and lo <= i < hi

    return should


def enable_quant_attention(model, should_quantize_attn, qk: str = "fp8", pv: str = "fp8") -> int:
    """Route the model's attention through ``quant_sdpa`` for the modules matched by
    ``should_quantize_attn`` (FP8 default). Returns the number of attention modules quantized.

    Registers the impl, flags the target modules, and points every attention module's
    config at the impl (so flagged ones quantize, the rest fall back to exact SDPA)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if _IMPL not in ALL_ATTENTION_FUNCTIONS.valid_keys():
        ALL_ATTENTION_FUNCTIONS.register(_IMPL, _quant_attention)
    n, cfgs = 0, set()
    for name, mod in model.named_modules():
        if "Attention" in type(mod).__name__ and getattr(mod, "config", None) is not None:
            if id(mod.config) not in cfgs:                 # one config may back many modules / a sub-config
                mod.config._attn_implementation = _IMPL
                cfgs.add(id(mod.config))
        if should_quantize_attn(name, mod):
            mod._nvfp4_attn, mod._nvfp4_qk, mod._nvfp4_pv = True, qk, pv
            n += 1
    return n
