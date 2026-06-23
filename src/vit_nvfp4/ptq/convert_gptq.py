"""Convert a model to W4A4 NVFP4 using GPTQ weight quantization (arXiv:2210.17323).

Two passes: (1) attach Hessian observers to the target nn.Linear modules and run the
calibration batches to accumulate H = sum_x x x^T per layer; (2) GPTQ-quantize each
target weight and swap in a QuantLinear built from the resulting codes/scales.
"""
import torch
import torch.nn as nn

from .qlinear import QuantLinear
from .gptq import HessianObserver, gptq_quantize_weight


@torch.no_grad()
def quantize_model_gptq(model: nn.Module, should_quantize, batches,
                        percdamp: float = 0.01, blocksize: int = 128) -> int:
    """In-place swap of selected nn.Linear with GPTQ-quantized NVFP4 QuantLinear.

    ``should_quantize`` is a ``(name, module) -> bool`` predicate (see policy.py).
    ``batches`` are calibration inputs (dict of kwargs or positional tensor), run once
    to accumulate the per-layer Hessian. Returns the number of layers swapped.
    """
    targets = [(n, m) for n, m in model.named_modules() if should_quantize(n, m)]

    # Pass 1: accumulate Hessians via forward pre-hooks on the original Linears.
    observers = {n: HessianObserver(m.in_features) for n, m in targets}
    handles = []
    for name, module in targets:
        obs = observers[name]
        handles.append(module.register_forward_pre_hook(
            lambda m, args, o=obs: o.observe(args[0])))
    try:
        for batch in batches:
            if isinstance(batch, dict):
                model(**batch)
            else:
                model(batch)
    finally:
        for h in handles:
            h.remove()

    # Pass 2: GPTQ-quantize each target and swap.
    for name, module in targets:
        H = observers[name].H
        codes, bscale, gscale = gptq_quantize_weight(
            module.weight.data, H, block=16, percdamp=percdamp, blocksize=blocksize)
        ql = QuantLinear(codes.to(module.weight.device), bscale, gscale, module.bias,
                         module.in_features, module.out_features)
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, ql)

    return len(targets)
