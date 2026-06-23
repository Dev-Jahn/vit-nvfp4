import torch

from .qlinear import QuantLinear

_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
_SUBSAMPLE = 1_000_000  # cap elements fed to torch.quantile


class _ActObserver:
    """Collects per-tensor activation magnitude statistics across calibration batches."""

    def __init__(self, method: str = "percentile", pct: float = 0.999):
        assert method in ("max", "percentile")
        self.method = method
        self.pct = pct
        self._stats = []

    def observe(self, x: torch.Tensor) -> None:
        a = x.detach().abs().float().reshape(-1)
        if self.method == "max":
            self._stats.append(a.amax())
        else:
            if a.numel() > _SUBSAMPLE:
                a = a[:: a.numel() // _SUBSAMPLE]
            self._stats.append(torch.quantile(a, self.pct))

    def amax(self) -> torch.Tensor:
        s = torch.stack(self._stats)
        return s.amax() if self.method == "max" else s.mean()


@torch.no_grad()
def calibrate_activations(model, batches, method: str = "max", pct: float = 0.999) -> int:
    """Calibrate a static per-tensor activation global scale for every QuantLinear.

    Runs ``model`` over ``batches`` (each a dict of kwargs or a positional tensor),
    collects activation stats via forward pre-hooks, and pins ``x_global_scale`` on
    each QuantLinear. Returns the number of layers calibrated.

    ``method='max'`` (default) is correct for NVFP4's two-level scheme: the global
    scale only needs to keep per-16 E4M3 block scales within range, so it must cover
    the true amax. Lowering it (``'percentile'``) saturates block scales at E4M3-max
    (448) for high-magnitude blocks and CLIPS them — empirically harmful on NVFP4
    (DINOv2: dynamic/max ~0.95 cosine vs percentile-0.999 ~0.66). ``'percentile'`` is
    kept for experiments with real per-token activation outliers, but is not the default.
    """
    observers = {}
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, QuantLinear):
            obs = _ActObserver(method, pct)
            observers[name] = (mod, obs)
            handles.append(mod.register_forward_pre_hook(lambda m, args, o=obs: o.observe(args[0])))
    try:
        for batch in batches:
            if isinstance(batch, dict):
                model(**batch)
            else:
                model(batch)
    finally:
        for h in handles:
            h.remove()

    for _name, (mod, obs) in observers.items():
        gs = (obs.amax() / (_E2M1_MAX * _E4M3_MAX)).float()
        mod.set_activation_scale(gs)
    return len(observers)
