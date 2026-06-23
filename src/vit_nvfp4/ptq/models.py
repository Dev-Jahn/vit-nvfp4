"""Model-aware config for the W4A4 PTQ frontend.

Derives the per-model knobs ``quantize_model`` / ``vit_block_policy`` need (which
ModuleList holds the transformer blocks, how many, how to pool a retrieval
feature) generically from the module tree, hardcoding only the few known
per-architecture branches. Validated on DINOv2, DINOv3, SigLIP, Qwen3-VL ViT,
V-JEPA2 (see SP2_EXPAND_RESULT.md)."""
from dataclasses import dataclass, field

import torch.nn as nn

from .policy import vit_block_policy
from .convert import quantize_model


@dataclass
class ModelSpec:
    arch: str
    block_container: str            # dotted path of the block stack (relative to the quantized module)
    num_layers: int
    feature: str                    # 'cls' | 'mean' — how to pool last_hidden_state for retrieval
    quirks: dict = field(default_factory=dict)


def _block_stacks(model: nn.Module):
    """(name, len) for every ModuleList whose blocks contain an nn.Linear."""
    return [(name, len(mod)) for name, mod in model.named_modules()
            if isinstance(mod, nn.ModuleList) and len(mod)
            and any(isinstance(s, nn.Linear) for s in mod[0].modules())]


def find_block_container(model: nn.Module) -> str:
    """Dotted path of the transformer block stack.

    The encoder stack is the longest Linear-bearing ModuleList; this beats
    auxiliary stacks (V-JEPA's ``predictor.layer``, Qwen3-VL's
    ``deepstack_merger_list``) by length."""
    stacks = _block_stacks(model)
    if not stacks:
        raise ValueError(f"no transformer block ModuleList found in {type(model).__name__}")
    return max(stacks, key=lambda x: x[1])[0]


def _detect_quirks(block: nn.Module) -> dict:
    lin = [n for n, s in block.named_modules() if isinstance(s, nn.Linear)]
    types = [type(c).__name__ for _, c in block.named_modules()]
    return {
        "fused_qkv": any("qkv" in n for n in lin),
        "swiglu": any("gate" in n for n in lin),
        "layerscale": any("LayerScale" in t for t in types),
        "rope": any(("Rope" in t or "RoPE" in t) for t in types),
        "out_proj": next((n.rsplit(".", 1)[-1] for n in lin
                          if n.rsplit(".", 1)[-1] in ("out_proj", "proj", "dense", "o_proj")), None),
    }


def _has_cls(arch: str) -> bool:
    # DINOv2/DINOv3 prepend a CLS token; SigLIP/V-JEPA/Qwen3-VL ViT are CLS-less.
    return any(k in arch.lower() for k in ("dinov2", "dinov3"))


def model_spec(model: nn.Module) -> ModelSpec:
    """Derive the PTQ config for ``model`` (the module you'll pass to ``quantize_model``)."""
    bc = find_block_container(model)
    blocks = model.get_submodule(bc)
    assert isinstance(blocks, nn.ModuleList)  # find_block_container only returns ModuleList paths
    arch = type(model).__name__
    return ModelSpec(arch=arch, block_container=bc, num_layers=len(blocks),
                     feature="cls" if _has_cls(arch) else "mean",
                     quirks=_detect_quirks(blocks[0]))


def quantize_vit(model: nn.Module, skip_first: int = 2, skip_last: int = 2,
                 w_block_select: str = "mse"):
    """Quantize the middle blocks of any supported ViT to W4A4. Returns (n_swapped, spec)."""
    spec = model_spec(model)
    policy = vit_block_policy(spec.num_layers, skip_first, skip_last, container=spec.block_container)
    n = quantize_model(model, policy, w_block_select=w_block_select)
    return n, spec
