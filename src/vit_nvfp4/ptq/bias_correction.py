"""Bias correction (mean output-shift fix) for NVFP4 QuantLinear layers.

After quantization, each ``QuantLinear`` produces a systematic per-output-feature
mean shift relative to the unquantized layer. Bias correction measures that shift
on a calibration set and folds it into the layer bias, so the corrected layer's
mean output matches the float layer's mean output:

    correction = E_tokens[ ref_linear(x) - quant_linear(x) ]
    bias <- bias + correction

Both linears see the IDENTICAL input ``x`` (the input the quantized model produces
at that site, captured by a forward hook on the QuantLinear). The reference output
is recomputed from the original float weight/bias. Calibration-only, no training,
no Hessian, no format change. Ref: arXiv:2006.10518.
"""
import torch
import torch.nn.functional as F

from .qlinear import QuantLinear


@torch.no_grad()
def correct_bias(ref_model, quant_model, batches) -> int:
    """Fold the mean output shift of each QuantLinear into its bias.

    ``ref_model`` is the unquantized model; ``quant_model`` is the quantized model
    (same module names, ``nn.Linear`` swapped for ``QuantLinear``). ``batches`` is an
    iterable of model inputs (each a dict of kwargs or a positional tensor) used as
    calibration data. Compose AFTER weight quantization and activation calibration.

    Returns the number of QuantLinear layers corrected.
    """
    # Map each QuantLinear (by name) to the corresponding reference nn.Linear.
    ref_mods = dict(ref_model.named_modules())
    targets = {}  # name -> (qmod, ref_weight, ref_bias)
    for name, qmod in quant_model.named_modules():
        if isinstance(qmod, QuantLinear):
            ref_lin = ref_mods[name]
            targets[name] = (
                qmod,
                ref_lin.weight.detach(),
                None if ref_lin.bias is None else ref_lin.bias.detach(),
            )

    # Accumulate sum of (ref_out - quant_out) over all tokens, per output feature.
    diff_sum = {name: None for name in targets}
    tok_count = {name: 0 for name in targets}
    out_dtype = {name: None for name in targets}  # activation/compute dtype seen at this site

    def _make_hook(name):
        qmod, ref_w, ref_b = targets[name]

        def hook(_m, args, out):
            x = args[0]
            out_dtype[name] = out.dtype
            # ref output on the IDENTICAL input the quant layer saw, in fp32.
            ref_out = F.linear(x.float(), ref_w.float(),
                               None if ref_b is None else ref_b.float())
            d = (ref_out - out.float()).reshape(-1, qmod.out_features)
            s = d.sum(dim=0)
            diff_sum[name] = s if diff_sum[name] is None else diff_sum[name] + s
            tok_count[name] += d.shape[0]

        return hook

    handles = [targets[name][0].register_forward_hook(_make_hook(name)) for name in targets]
    try:
        for batch in batches:
            if isinstance(batch, dict):
                quant_model(**batch)
            else:
                quant_model(batch)
    finally:
        for h in handles:
            h.remove()

    for name, (qmod, _ref_w, _ref_b) in targets.items():
        correction = (diff_sum[name] / tok_count[name])  # (out_features,) fp32
        if qmod.bias is None:
            # create a bias buffer in the compute dtype (forward applies bias in y.dtype,
            # which is the activation dtype; weights/activations are bf16 here).
            # QuantLinear sets `self.bias = None` as a plain attr; free the name first.
            del qmod.bias
            qmod.register_buffer("bias", correction.to(out_dtype[name]))
        else:
            qmod.bias.add_(correction.to(qmod.bias.dtype))
    return len(targets)
