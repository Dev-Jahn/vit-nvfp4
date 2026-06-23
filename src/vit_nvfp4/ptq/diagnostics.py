import torch
import torch.nn.functional as F


@torch.no_grad()
def tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Flattened cosine similarity between two tensors (computed in fp32)."""
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


@torch.no_grad()
def block_output_cosines(ref_model, quant_model, inputs, block_container=None):
    """Per-block output cosine between a reference and quantized model.

    Registers forward hooks on each child of ``ref_model.<block_container>`` and the
    corresponding quant block, runs both, and returns a list of per-block cosines.
    ``inputs`` is a dict of kwargs passed to both models. ``block_container`` is
    auto-detected per-model when None (e.g. DINOv3=``model.layer``, SigLIP=``encoder.layers``).
    """
    if block_container is None:
        from .models import find_block_container
        block_container = find_block_container(ref_model)
    ref_blocks = ref_model.get_submodule(block_container)
    q_blocks = quant_model.get_submodule(block_container)
    ref_out, q_out = {}, {}

    def _hook(store, i):
        def fn(_m, _inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    handles = []
    for i, (rb, qb) in enumerate(zip(ref_blocks, q_blocks)):
        handles.append(rb.register_forward_hook(_hook(ref_out, i)))
        handles.append(qb.register_forward_hook(_hook(q_out, i)))
    try:
        ref_model(**inputs)
        quant_model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return [tensor_cosine(ref_out[i], q_out[i]) for i in range(len(ref_blocks))]
