# Fused low-precision attention on sm_120 — custom kernel? SageAttention? (measured)

**Date:** 2026-06-24 · RTX PRO 6000 (sm_120) · CUDA 13.1 / torch 2.12.1+cu130. Question: should we write a custom fused FP4/FP8 attention CUDA kernel for the ViT towers, or use **SageAttention**?

## SageAttention3 builds cleanly on our stack
`sageattn3==1.0.0` (the FP4 Blackwell variant) built + installed in a **throwaway py3.12 venv** (never the project `.venv`) via `uv pip install --no-build-isolation .`, with `CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=12.0a` — **2m38s, zero patches** (contrast: WSL2/CUDA-13.0 reports needed ~6 patches). Apache-2.0; `install_requires` has **no torch pin**, so it does **not** downgrade torch (unlike the flashinfer incident). API: `sageattn3_blackwell(q, k, v, is_causal=False)`, `(B,H,S,D)` bf16/fp16, FP4 internally.

## But it does NOT help image ViT towers (measured vs torch BF16 SDPA, sm_120)
**Head-dim support** (B=1,H=8,S=1280): 64 ✓, 128 ✓, **96 ✗ CompilationError**, **192 ✗ CompilationError** (= Qwen3-VL vision head_dim), 256 → "Unsupported, falls back to SDPA". Accuracy where supported: cos ≈ 0.98 vs SDPA (FP4).

**Speed vs torch SDPA, D=128, B=1, H=8** (the speedup is `t_sdpa / t_sa3`):
| S | torch SDPA | SA3 FP4 | speedup |
|---|---|---|---|
| 1280 | 0.046 ms | 0.114 ms | **0.40×** (SA3 slower) |
| 4096 | 0.331 ms | 0.181 ms | 1.83× |
| 8192 | 0.972 ms | 0.503 ms | 1.93× |
| 16384 | 3.71 ms | 1.80 ms | 2.06× |
| 65536 | 52.5 ms | 22.7 ms | 2.31× |

The crossover is **S ≈ 2–4k**. Below it, SA3's runtime FP4 quantization overhead isn't amortized and torch's BF16 SDPA (cuDNN/flash) is faster.

Our ViT image encoders sit **below** the crossover: DINOv2 S=257, SigLIP, Qwen3-VL vision S≈1280 (and Qwen's head_dim 192 isn't even supported). Earlier profiling: attention is ~17% of (Linear+SDPA) in BF16, ~37% after W4A4 Linears — but a *slower* attention kernel makes that worse, not better.

## Verdict
1. **No custom attention kernel.** It would also lose to torch SDPA at ViT sequence lengths, and SA3 already covers the long-S regime — writing one is pure waste.
2. **Image ViT towers → keep torch BF16 SDPA.** It is already the fastest option at S ≤ 1280; SA3 is 2.5–4× slower there and lacks head_dim 192. Our emulated `quant_sdpa` (`ptq/qattention.py`) stays an **accuracy-floor measurement tool**, not a speed path (it adds quant overhead with no kernel benefit).
3. **Long-sequence regime only → SageAttention3.** At S ≥ ~4k it gives ~2× at cos 0.98. The one place this is relevant to us is **long-context video ViT (V-JEPA2, 64-frame 384px → S≈37k tokens, head_dim 64 — supported)**. If/when we target that, slot SA3 behind the existing `enable_quant_attention` hook (a `backend` switch); no custom kernel, no new integration surface.

**Bottom line:** the "fused attention CUDA kernel" roadmap item is **closed** — unnecessary for image ViTs (torch SDPA wins), and for video ViTs the answer is SA3 (buildable on sm_120, ~2× at long S), never a hand-written kernel.
