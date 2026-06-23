import torch.nn as nn

from .qlinear import QuantLinear


def quantize_model(model: nn.Module, should_quantize, w_block_select: str = "mse") -> int:
    """In-place swap of selected nn.Linear modules with NVFP4 ``QuantLinear``.

    ``should_quantize`` is a ``(name, module) -> bool`` predicate (see policy.py).
    ``w_block_select`` ('six'|'mse') chooses the weight per-block 6-vs-4 scale
    policy (see ``quantize_to_nvfp4``). Returns the number of layers swapped.
    """
    targets = [(n, m) for n, m in model.named_modules() if should_quantize(n, m)]
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, QuantLinear.from_linear(module, w_block_select=w_block_select))
    return len(targets)
