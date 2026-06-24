"""Authoritative high-res comparison of calibration-only W4A4 error-reduction techniques.

DINOv2-base, skip(2,2), Food-101 (native high-res, 101 classes, real k-NN headroom).
Compares: BF16 / W4A4 baseline / +Four Over Six / +GPTQ / +FoS+bias-correction / +RegCache.
Run: TORCH_CUDA_ARCH_LIST=12.0a CUDA_VISIBLE_DEVICES=4 \
     uv run python examples/eval_techniques.py
"""
import torch

from datasets import load_dataset  # noqa: E402
from transformers import AutoModel, AutoImageProcessor  # noqa: E402

from vit_nvfp4.ptq import (  # noqa: E402
    quantize_model, quantize_model_gptq, vit_block_policy,
    calibrate_activations, correct_bias, curate_register, RegCache, tensor_cosine,
)
from vit_nvfp4.eval.knn import knn_top1_accuracy  # noqa: E402

NAME = "facebook/dinov2-base"
DATASET, IMG, LBL = "ethz/food101", "image", "label"
GAL_SPLIT, QRY_SPLIT = "train", "validation"
N_GAL, N_QRY, BS = 2020, 1010, 64
SKIP = (2, 2)
PROC = None
L = 12


def load_images(split, n, seed):
    ds = load_dataset(DATASET, split=split).shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [ex[IMG].convert("RGB") for ex in ds], torch.tensor([ex[LBL] for ex in ds])


@torch.no_grad()
def extract(model, images, bs=BS):
    feats = []
    for i in range(0, len(images), bs):
        pv = PROC(images=images[i:i + bs], return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)
        feats.append(model(pixel_values=pv).last_hidden_state[:, 0].float().cpu())
    return torch.cat(feats)


def fresh():
    return AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()


def main():
    global PROC, L
    torch.manual_seed(0)
    PROC = AutoImageProcessor.from_pretrained(NAME)
    g_imgs, g_lab = load_images(GAL_SPLIT, N_GAL, 0)
    q_imgs, q_lab = load_images(QRY_SPLIT, N_QRY, 1)
    print(f"{DATASET}: gallery={len(g_imgs)} query={len(q_imgs)}")

    ref = fresh()
    L = ref.config.num_hidden_layers
    policy = vit_block_policy(L, *SKIP)
    g_ref, q_ref = extract(ref, g_imgs), extract(ref, q_imgs)
    calib = [{"pixel_values": PROC(images=g_imgs[i:i + BS], return_tensors="pt")["pixel_values"].to("cuda", torch.bfloat16)}
             for i in range(0, 8 * BS, BS)]

    bf16_acc = knn_top1_accuracy(g_ref, g_lab, q_ref, q_lab, k=10)
    print(f"\n{'config':<24} {'k-NN top-1':>10} {'Δ vs bf16':>10} {'cosine':>8}")
    print(f"{'BF16 (ref)':<24} {bf16_acc:>10.4f} {0.0:>+10.4f} {1.0:>8.3f}")

    def evaluate(tag, build):
        try:
            m = build()
            gq, qq = extract(m, g_imgs), extract(m, q_imgs)
            acc = knn_top1_accuracy(gq, g_lab, qq, q_lab, k=10)
            print(f"{tag:<24} {acc:>10.4f} {acc - bf16_acc:>+10.4f} {tensor_cosine(q_ref, qq):>8.3f}")
        except Exception as e:
            print(f"{tag:<24} FAILED: {repr(e)[:80]}")

    def baseline():
        m = fresh(); quantize_model(m, policy); calibrate_activations(m, calib, "max"); return m

    def fos():
        m = fresh(); quantize_model(m, policy, w_block_select="mse"); calibrate_activations(m, calib, "max"); return m

    def gptq():
        m = fresh(); quantize_model_gptq(m, policy, calib); calibrate_activations(m, calib, "max"); return m

    def fos_bias():
        m = fresh(); quantize_model(m, policy, w_block_select="mse"); calibrate_activations(m, calib, "max")
        correct_bias(ref, m, calib); return m

    def regcache():
        m = fresh(); quantize_model(m, policy)
        reg = curate_register(m, calib, sensitive_at=8, top_k=64)
        RegCache(m, reg, sensitive_at=8, k_tilde=6, tau=0).install()
        calibrate_activations(m, calib, "max"); return m

    evaluate("W4A4 baseline", baseline)
    evaluate("W4A4 + FourOverSix", fos)
    evaluate("W4A4 + GPTQ", gptq)
    evaluate("W4A4 + FoS + bias", fos_bias)
    evaluate("W4A4 + RegCache(del)", regcache)


if __name__ == "__main__":
    main()
