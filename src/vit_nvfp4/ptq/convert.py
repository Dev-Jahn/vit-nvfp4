import torch.nn as nn

from .qlinear import QuantLinear


def quantize_model(model: nn.Module, should_quantize) -> int:
    """In-place swap of selected nn.Linear modules with NVFP4 ``QuantLinear``.

    ``should_quantize`` is a ``(name, module) -> bool`` predicate (see policy.py).
    Returns the number of layers swapped.
    """
    targets = [(n, m) for n, m in model.named_modules() if should_quantize(n, m)]
    for name, module in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, QuantLinear.from_linear(module))
    return len(targets)
