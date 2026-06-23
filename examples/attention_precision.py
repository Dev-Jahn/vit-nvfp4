"""Per-stage low-precision attention accuracy on real DINOv2 Q/K/V.

Captures Q,K,V from a mid DINOv2 attention layer (real images if available, else
random pixel_values), then measures the output cosine of ``quant_sdpa`` vs the
fp32 reference for each (QKᵀ, P·V) precision combination and the smooth-K/V
ablations. This locates the per-stage accuracy floor (NVFP4 vs FP8).

Run: HF_HOME=/var/cache/huggingface TORCH_CUDA_ARCH_LIST=12.0a CUDA_VISIBLE_DEVICES=1 \
     uv run python examples/attention_precision.py
"""
import os

import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "/var/cache/huggingface")

from transformers import AutoModel, AutoImageProcessor  # noqa: E402

from vit_nvfp4.nvfp4 import quant_sdpa  # noqa: E402

NAME = "facebook/dinov2-base"
LAYER = 6  # mid block


def _pixel_values(proc, n=8):
    try:
        from datasets import load_dataset
        ds = load_dataset("dpdl-benchmark/oxford_flowers102", split="validation").shuffle(seed=0)
        imgs = [ds[i]["image"].convert("RGB") for i in range(n)]
        pv = proc(images=imgs, return_tensors="pt")["pixel_values"]
        print(f"using {n} real flowers images")
    except Exception as e:
        print(f"dataset unavailable ({type(e).__name__}); random pixel_values")
        pv = torch.randn(n, 3, 224, 224)
    return pv.to("cuda", torch.bfloat16)


@torch.no_grad()
def capture_qkv(model, pv):
    """Hook the q/k/v projections of one attention block; return (B,H,S,D) tensors."""
    cfg = model.config
    H, Dh = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads
    attn = model.encoder.layer[LAYER].attention.attention
    grab = {}

    def hook(name):
        def fn(_m, _i, out):
            B, S, _ = out.shape
            grab[name] = out.view(B, S, H, Dh).permute(0, 2, 1, 3).contiguous()
        return fn

    handles = [attn.query.register_forward_hook(hook("q")),
               attn.key.register_forward_hook(hook("k")),
               attn.value.register_forward_hook(hook("v"))]
    try:
        model(pixel_values=pv)
    finally:
        for h in handles:
            h.remove()
    return grab["q"], grab["k"], grab["v"]


def _cos(a, b):
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def main():
    torch.manual_seed(0)
    proc = AutoImageProcessor.from_pretrained(NAME)
    model = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    q, k, v = capture_qkv(model, _pixel_values(proc))
    print(f"{NAME} layer {LAYER}: Q/K/V {tuple(q.shape)} (B,H,S,D)")
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())

    def row(label, **kw):
        out = quant_sdpa(q, k, v, **kw)
        print(f"  {label:<34} cos={_cos(out, ref):.4f}")

    print("\n-- per-stage isolation (other stage = bf16) --")
    for m in ("fp8", "nvfp4"):
        row(f"QKᵀ {m:<6} (PV bf16)", qk=m, pv="bf16")
    for m in ("fp8", "nvfp4"):
        row(f"PV  {m:<6} (QKᵀ bf16)", qk="bf16", pv=m)

    print("\n-- combined schemes (smooth-K + smooth-V on) --")
    for qk in ("fp8", "nvfp4"):
        for pv in ("fp8", "nvfp4"):
            row(f"QKᵀ {qk} / PV {pv}", qk=qk, pv=pv)

    print("\n-- smoothing ablation (QKᵀ nvfp4 / PV nvfp4) --")
    row("no smoothing", qk="nvfp4", pv="nvfp4", smooth_k=False, smooth_v=False)
    row("smooth-K only", qk="nvfp4", pv="nvfp4", smooth_k=True, smooth_v=False)
    row("smooth-V only", qk="nvfp4", pv="nvfp4", smooth_k=False, smooth_v=True)
    row("smooth-K + smooth-V", qk="nvfp4", pv="nvfp4", smooth_k=True, smooth_v=True)
    print("\n-- recommended W4A4-attention (QKᵀ fp8 / PV fp8 + smoothing) --")
    row("QKᵀ fp8 / PV fp8 + smooth", qk="fp8", pv="fp8")


if __name__ == "__main__":
    main()
