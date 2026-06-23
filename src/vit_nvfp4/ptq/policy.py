import re

import torch.nn as nn

# Matches the per-block index in common ViT module paths:
# encoder.layer.7 / blocks.7 / layers.7 / h.7
_LAYER_RE = re.compile(r"(?:^|\.)(?:layer|layers|blocks|h)\.(\d+)(?:\.|$)")


def block_index(name: str):
    """Return the transformer block index encoded in a module path, or None."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def vit_block_policy(num_layers: int, skip_first: int = 2, skip_last: int = 2):
    """Quantize nn.Linear layers in the *middle* blocks only.

    Returns a ``(name, module) -> bool`` predicate. Linears in the first
    ``skip_first`` and last ``skip_last`` blocks, and any Linear without a
    block index (heads, embeddings), are kept in BF16.
    """
    lo, hi = skip_first, num_layers - skip_last

    def should_quantize(name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        i = block_index(name)
        return i is not None and lo <= i < hi

    return should_quantize
