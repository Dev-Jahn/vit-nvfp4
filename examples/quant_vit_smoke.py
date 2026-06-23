"""Structural + fidelity smoke test of W4A4 NVFP4 on the special-input ViTs
(Qwen3-VL fused-qkv vision encoder; V-JEPA2.1 RoPE video encoder + predictor).

These don't fit the image-only k-NN harness (patchified / video inputs), so we
validate: registry detection, swap count (fused qkv / predictor exclusion), and
BF16-vs-W4A4 output cosine on one real/realistic forward.

Run: HF_HOME=/var/cache/huggingface TORCH_CUDA_ARCH_LIST=12.0a CUDA_VISIBLE_DEVICES=0 \
       uv run python examples/quant_vit_smoke.py
"""
import os

import torch

os.environ.setdefault("HF_HOME", "/var/cache/huggingface")

from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

from vit_nvfp4.ptq import quantize_vit, calibrate_activations, tensor_cosine, model_spec  # noqa: E402


def real_image():
    ds = load_dataset("dpdl-benchmark/oxford_flowers102", split="validation").shuffle(seed=0)
    return ds[0]["image"].convert("RGB")


def report(tag, spec, n, c_ref, c_q):
    print(f"\n[{tag}] {spec.arch} container={spec.block_container} layers={spec.num_layers} "
          f"quirks={spec.quirks}")
    print(f"  W4A4 swaps={n} | output cosine (BF16 vs W4A4) = {tensor_cosine(c_ref, c_q):.4f}")


@torch.no_grad()
def qwen3vl():
    name = "Qwen/Qwen3-VL-2B-Instruct"
    proc = AutoProcessor.from_pretrained(name)
    px = proc.image_processor(images=[real_image()], return_tensors="pt")
    pv = px["pixel_values"].to("cuda", torch.bfloat16)
    grid = px["image_grid_thw"].to("cuda")

    ref = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval().visual
    print(f"qwen3vl spec: {model_spec(ref)}")
    out_ref = ref(pv, grid_thw=grid)[0].float().cpu()
    del ref; torch.cuda.empty_cache()

    quant = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval().visual
    n, spec = quantize_vit(quant)
    calibrate_activations(quant, [{"hidden_states": pv, "grid_thw": grid}], method="max")
    out_q = quant(pv, grid_thw=grid)[0].float().cpu()
    report("Qwen3-VL ViT", spec, n, out_ref, out_q)
    del quant; torch.cuda.empty_cache()


@torch.no_grad()
def vjepa():
    name = "Dev-Jahn/vjepa2.1-vitl-fpc64-384"
    torch.manual_seed(0)
    # fp32: this model's RoPE attention upcasts q,k to fp32 and SDPA rejects a bf16 v,
    # so the BF16 ref itself fails. QuantLinear follows the activation dtype (out=x.dtype).
    vid = torch.randn(1, 3, 16, 224, 224)  # (B,C,T,H,W); random clip for a relative fidelity check

    def feats(m):
        v = vid.to("cuda", torch.float32)
        o = m(pixel_values_videos=v, skip_predictor=True)
        h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        return h.float().cpu()

    ref = AutoModel.from_pretrained(name, dtype=torch.float32, trust_remote_code=True).cuda().eval()
    print(f"vjepa spec: {model_spec(ref)}")
    out_ref = feats(ref)
    del ref; torch.cuda.empty_cache()

    quant = AutoModel.from_pretrained(name, dtype=torch.float32, trust_remote_code=True).cuda().eval()
    n, spec = quantize_vit(quant)
    out_q = feats(quant)  # dynamic activation scale (skip_predictor avoids the mask path)
    report("V-JEPA2.1 (random clip)", spec, n, out_ref, out_q)
    del quant; torch.cuda.empty_cache()


if __name__ == "__main__":
    for fn in (qwen3vl, vjepa):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"\n[{fn.__name__}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
