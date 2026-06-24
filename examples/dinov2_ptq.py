"""Quantize DINOv2 to W4A4 NVFP4 and report fidelity vs the BF16 model.

Run: TORCH_CUDA_ARCH_LIST=12.0a uv run python examples/dinov2_ptq.py
"""
import torch

from transformers import AutoModel  # noqa: E402

from vit_nvfp4.ptq import quantize_model, vit_block_policy, tensor_cosine, block_output_cosines  # noqa: E402

NAME = "facebook/dinov2-base"


def main():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    ref = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        r = ref(x).last_hidden_state
    L = ref.config.num_hidden_layers

    print(f"{NAME} | {L} layers, hidden {ref.config.hidden_size}")
    print(f"{'skip':>8} {'swaps':>6} {'last_hidden cos':>16} {'CLS cos':>9}")
    for sf, sl in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]:
        q = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
        n = quantize_model(q, vit_block_policy(L, sf, sl))
        with torch.no_grad():
            qo = q(x).last_hidden_state
        print(f"({sf},{sl}){'':>3} {n:>6} {tensor_cosine(r, qo):>16.4f} {tensor_cosine(r[:, 0], qo[:, 0]):>9.4f}")

    q = AutoModel.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    quantize_model(q, vit_block_policy(L, 2, 2))
    cosns = block_output_cosines(ref, q, {"pixel_values": x})
    print("\nper-block output cos (skip 2,2):")
    print("  " + " ".join(f"{c:.3f}" for c in cosns))


if __name__ == "__main__":
    main()
