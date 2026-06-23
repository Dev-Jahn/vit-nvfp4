"""Real-data accuracy of W4A4 NVFP4 DINOv2 vs BF16, via frozen-feature k-NN on Food-101.

Measures the metric that matters (downstream classification), not just feature cosine.
Run: HF_HOME=/var/cache/huggingface TORCH_CUDA_ARCH_LIST=12.0a uv run python examples/eval_knn_dinov2.py
"""
import os

import torch

os.environ.setdefault("HF_HOME", "/var/cache/huggingface")

from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoImageProcessor  # noqa: E402

from vit_nvfp4.ptq import quantize_model, vit_block_policy, calibrate_activations, tensor_cosine  # noqa: E402
from vit_nvfp4.eval.knn import knn_top1_accuracy  # noqa: E402

NAME = "facebook/dinov2-base"
# High-res standard DINOv2 transfer benchmark (~500px native, 102 classes); small (~330MB).
DATASET = "dpdl-benchmark/oxford_flowers102"
IMG_KEY, LBL_KEY = "image", "label"
GALLERY_SPLIT, QUERY_SPLIT = "test", "validation"  # test=6149 (gallery), validation=1020 (query)
N_GALLERY, N_QUERY, BS = 3060, 1020, 64


def load_images(split, n, seed):
    # full (non-streaming) load + random subset across all classes
    ds = load_dataset(DATASET, split=split).shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    imgs = [ex[IMG_KEY].convert("RGB") for ex in ds]
    labels = torch.tensor([ex[LBL_KEY] for ex in ds])
    return imgs, labels


@torch.no_grad()
def extract(model, processor, images, device="cuda", bs=BS):
    feats = []
    for i in range(0, len(images), bs):
        pv = processor(images=images[i:i + bs], return_tensors="pt")["pixel_values"]
        pv = pv.to(device=device, dtype=torch.bfloat16)
        feats.append(model(pixel_values=pv).last_hidden_state[:, 0].float().cpu())
    return torch.cat(feats)


def main():
    torch.manual_seed(0)
    proc = AutoImageProcessor.from_pretrained(NAME)
    g_imgs, g_lab = load_images(GALLERY_SPLIT, N_GALLERY, seed=0)
    q_imgs, q_lab = load_images(QUERY_SPLIT, N_QUERY, seed=1)
    print(f"{DATASET}: gallery={len(g_imgs)} query={len(q_imgs)}")

    ref = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    g_ref, q_ref = extract(ref, proc, g_imgs), extract(ref, proc, q_imgs)

    quant = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    n = quantize_model(quant, vit_block_policy(ref.config.num_hidden_layers, skip_first=2, skip_last=2))
    # calibrate static activation scale (max) on a few gallery batches
    calib = [{"pixel_values": proc(images=g_imgs[i:i + BS], return_tensors="pt")["pixel_values"]
              .to("cuda", torch.bfloat16)} for i in range(0, 8 * BS, BS)]
    calibrate_activations(quant, calib, method="max")
    g_q, q_q = extract(quant, proc, g_imgs), extract(quant, proc, q_imgs)

    print(f"W4A4 swaps: {n} | feature cosine (query): {tensor_cosine(q_ref, q_q):.4f}")
    for k in (10, 20):
        a_ref = knn_top1_accuracy(g_ref, g_lab, q_ref, q_lab, k=k)
        a_q = knn_top1_accuracy(g_q, g_lab, q_q, q_lab, k=k)
        print(f"k={k:>2}: BF16 top-1={a_ref:.4f} | W4A4 top-1={a_q:.4f} | drop={a_ref - a_q:+.4f}")


if __name__ == "__main__":
    main()
