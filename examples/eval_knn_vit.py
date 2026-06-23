"""Generic real-data k-NN accuracy of W4A4 NVFP4 vs BF16 for any supported image ViT.

The model registry (ptq/models.py) derives the block container / layer count /
feature pooling, so one harness covers DINOv2, DINOv3, SigLIP2, ...

Run: HF_HOME=/var/cache/huggingface TORCH_CUDA_ARCH_LIST=12.0a CUDA_VISIBLE_DEVICES=0 \
       uv run python examples/eval_knn_vit.py --model facebook/dinov3-vitb16-pretrain-lvd1689m
"""
import argparse
import os

import torch

os.environ.setdefault("HF_HOME", "/var/cache/huggingface")

from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoImageProcessor  # noqa: E402

from vit_nvfp4.ptq import quantize_vit, calibrate_activations, tensor_cosine, model_spec  # noqa: E402
from vit_nvfp4.eval.knn import knn_top1_accuracy  # noqa: E402

DATASET = "dpdl-benchmark/oxford_flowers102"
GALLERY_SPLIT, QUERY_SPLIT = "test", "validation"


def load_images(split, n, seed):
    ds = load_dataset(DATASET, split=split).shuffle(seed=seed).select(range(n))
    return [ex["image"].convert("RGB") for ex in ds], torch.tensor([ex["label"] for ex in ds])


def pool(out, feature):
    if feature == "cls":
        return out.last_hidden_state[:, 0]
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    return out.last_hidden_state.mean(1)


@torch.no_grad()
def extract(model, proc, images, feature, bs):
    feats = []
    for i in range(0, len(images), bs):
        pv = proc(images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        pv = pv.to("cuda", torch.bfloat16)
        feats.append(pool(model(pixel_values=pv), feature).float().cpu())
    return torch.cat(feats)


def get(name, submodule):
    m = AutoModel.from_pretrained(name, dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
    return getattr(m, submodule) if submodule else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--submodule", default="")  # e.g. "vision_model" for Siglip2Model
    ap.add_argument("--n-gallery", type=int, default=3060)
    ap.add_argument("--n-query", type=int, default=1020)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--skip", type=int, nargs=2, default=(2, 2))
    args = ap.parse_args()
    torch.manual_seed(0)

    proc = AutoImageProcessor.from_pretrained(args.model)
    g_imgs, g_lab = load_images(GALLERY_SPLIT, args.n_gallery, 0)
    q_imgs, q_lab = load_images(QUERY_SPLIT, args.n_query, 1)

    ref = get(args.model, args.submodule)
    spec = model_spec(ref)
    print(f"{args.model} | {spec.arch} | container={spec.block_container} layers={spec.num_layers} "
          f"feature={spec.feature} quirks={spec.quirks}")
    g_ref = extract(ref, proc, g_imgs, spec.feature, args.bs)
    q_ref = extract(ref, proc, q_imgs, spec.feature, args.bs)
    del ref; torch.cuda.empty_cache()

    quant = get(args.model, args.submodule)
    n, _ = quantize_vit(quant, skip_first=args.skip[0], skip_last=args.skip[1])
    calib = [{"pixel_values": proc(images=g_imgs[i:i + args.bs], return_tensors="pt")["pixel_values"]
              .to("cuda", torch.bfloat16)} for i in range(0, 8 * args.bs, args.bs)]
    calibrate_activations(quant, calib, method="max")
    g_q = extract(quant, proc, g_imgs, spec.feature, args.bs)
    q_q = extract(quant, proc, q_imgs, spec.feature, args.bs)

    print(f"W4A4 swaps: {n} | feature cosine (query): {tensor_cosine(q_ref, q_q):.4f}")
    for k in (10, 20):
        a_ref = knn_top1_accuracy(g_ref, g_lab, q_ref, q_lab, k=k)
        a_q = knn_top1_accuracy(g_q, g_lab, q_q, q_lab, k=k)
        print(f"k={k:>2}: BF16 top-1={a_ref:.4f} | W4A4 top-1={a_q:.4f} | drop={a_ref - a_q:+.4f}")


if __name__ == "__main__":
    main()
