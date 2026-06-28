# Development log — what we tried, what worked, what didn't

`vit-nvfp4` is a general-purpose Vision Transformer **NVFP4 W4A4 PTQ + low-precision-attention**
toolkit for **sm_120** (RTX PRO 6000 Blackwell, CUDA 13.x, torch 2.12+cu130). This log is the
single place that records *every approach we tried and its verdict*, so future work doesn't
re-walk dead ends. Per-experiment detail lives under [`results/`](results/); external research
under [`research/`](research/); designs under [`design/`](design/).

Two distinct problems were solved, in order: **(1) accuracy** — does W4A4 NVFP4 preserve ViT
quality? (yes, with the right tricks); **(2) speed** — is it actually faster than BF16? (yes, but
only after killing the activation-cast overhead).

---

## TL;DR — what ships

| Layer | Shipping choice | Source |
|---|---|---|
| W4A4 GEMM | `torch._scaled_mm_v2` (BlockWise1x16 + TensorWise global, SWIZZLE_32_4_4) | [perf/PHASE0](results/perf/PHASE0_RESULT.md) |
| Weight quant | **Four Over Six** (per-block 6-vs-4 MSE select), global `amax/(6·256)` | [ptq/SP5](results/ptq/SP5_RESULT.md), [ptq/QWEN](results/ptq/QWEN_W4A4_VERDICT.md) |
| Activation global | **static "max"** calibration (parity with dynamic, deployment-safe) | [ptq/SP3](results/ptq/SP3_RESULT.md) |
| Block policy | skip first/last 2 blocks' MLPs (BF16); attention scores BF16 | [ptq/SP2](results/ptq/SP2_RESULT.md) |
| Attention (accuracy) | FP8 emulated `quant_sdpa` — lossless floor measurement | [attention/SP-A](results/attention/SPA_ATTENTION_RESULT.md) |
| Attention (speed, long-S) | **SageAttention3** FP4 (only S≳4k, head_dim 64/128, per-frame B=1) | [attention/SAGE](results/attention/SAGEATTENTION_SM120_EVAL.md) |
| Fast inference | **GELU→NVFP4 producer-fused MLP** + SA3, both `custom_op` + `torch.compile` | [perf/FUSEDMLP](results/perf/FUSEDMLP_NVFP4_RESULT.md) |

**Accuracy:** high-res k-NN ~lossless (Flowers-102 0.9931 = BF16; Food-101 −0.1 pt with FoS).
**Speed @1080p vs compiled BF16:** DINOv2-large 1.18–1.21×, DINOv2-base 1.15×, DINOv3-B 1.21×,
**V-JEPA2-L 1.40×** (S=130560) — feature cos 0.97–0.98.

---

## 1. W4A4 NVFP4 GEMM (the foundation) — [perf/PHASE0](results/perf/PHASE0_RESULT.md)

### ✅ Worked
- **`torch._scaled_mm_v2`** with `ScalingType.BlockWise1x16` + per-tensor `TensorWise` global +
  `SwizzleType.SWIZZLE_32_4_4` — numerically exact W4A4 on sm_120 (cos_emul = 1.0 vs fp32 of the
  same quantized operands), **no custom kernel build**. The pure FP4 GEMM is **1.3–3.1× faster
  than BF16** on large GEMMs (M·K·N big).
- Two-level scale that works: `global = amax/(6·448)` (FP32), per-16 `block = E4M3((amax_b/6)/global)`;
  decode `x ≈ e2m1 · block_e4m3 · global`. Scales **must** be 128×4 swizzled (`to_blocked`); K, N %16.

### ❌ Didn't work / dead ends
- **`torch._scaled_mm` (v1) NVFP4 path** — returns garbage (cos ≈ 0) on sm_120. Must use **v2**.
- **FlashInfer `b12x`** — `flashinfer-python` pins torch and **downgrades 2.12→2.10**, breaking the
  cu130 env (`undefined symbol: ncclCommResume`). Never install it.
- **CUTLASS-79 / cuBLASLt / hand-built kernel** — unnecessary; v2 passed the gate. (CUTLASS-79 also
  had a known misaligned-address bug, unmerged fix.)
- **Small/square GEMMs don't win** — FP4 only beats BF16 when the GEMM is large enough to be
  compute-bound (the recurring theme behind the whole speed story below).

---

## 2. PTQ accuracy — [ptq/](results/ptq/)

### ✅ Worked
- **Mixed precision by structure** (skip first/last 2 blocks' Linears → BF16; attention
  QK/softmax/PV → BF16 via SDPA) + NVFP4 per-16 block scaling → **no naive collapse** (the
  literature's "DINOv2-B 83→0" is an all-layer per-tensor artifact). DINOv2-base feature cos ~0.95
  at skip(2,2). [SP2](results/ptq/SP2_RESULT.md)
- **High-res downstream accuracy is ~lossless.** k-NN top-1: Flowers-102 0.9931 = BF16 (Δ0.0);
  Food-101 −1.4 pt baseline → **−0.1 pt with Four Over Six**. The earlier CIFAR −2.4 pt was a
  *low-resolution artifact* (32→224 upsampling), not a W4A4 cost. [SP4](results/ptq/SP4_EVAL_RESULT.md)
- **Four Over Six** (per-block, MSE-select scale-to-6 vs scale-to-4) — **the one robust accuracy
  lever**: free, weights-only, offline, format-preserving, never worse per-block. Recovers ~93% of
  the high-res gap. **Adopted as the default** (`w_block_select='mse'`). [SP5](results/ptq/SP5_RESULT.md)
  - **Required fix:** FoS must use global `amax/(6·256)`, not `448` — under 448 the scale-to-4
    candidate saturates E4M3 and silently collapses back to scale-to-6. The 448 bug made FoS
    *net-negative*. [QWEN verdict](results/ptq/QWEN_W4A4_VERDICT.md)
- **Static "max" activation calibration** — at parity with per-input dynamic (~0.95) but removes
  per-forward scale computation and is outlier-image robust. Deployment win. [SP3](results/ptq/SP3_RESULT.md)
- **One code path for all ViTs** — a small registry derives per-model knobs (block container,
  fused-qkv, RoPE, mean-pool) from the module tree; validated on DINOv2/v3, SigLIP2, Qwen3-VL,
  V-JEPA2. [SP2-expand](results/ptq/SP2_EXPAND_RESULT.md)

### ❌ Didn't work / dead ends
- **Percentile calibration of the global scale** — *catastrophic* on NVFP4 (cos 0.95 → 0.66 @99.9 →
  0.41 @99.0). Setting global below the true amax clips high-magnitude blocks (their E4M3 saturates
  at 448). The "use percentile for activations" rule is for *per-tensor* quant; it does **not**
  transfer to NVFP4's two-level scheme. Use **max**. [SP3](results/ptq/SP3_RESULT.md)
- **GPTQ** — lowers weight error (cosine ↑ everywhere) but the gain **doesn't transfer** to high-res
  k-NN (it lives along directions the L2-normalized cosine metric discards). Kept optional. [SP5](results/ptq/SP5_RESULT.md)
- **RegCache token-deletion** — task-dependent and can *hurt* (CIFAR +1.1 pt, Food-101 −0.5 pt). Off
  by default. [SP5](results/ptq/SP5_RESULT.md)
- **Bias correction** — neutral for cosine k-NN (mean shift annihilated by L2-norm). Near-free, kept
  as a cleanup for non-retrieval uses. [SP5](results/ptq/SP5_RESULT.md)
- **"Qwen3-VL needs mixed precision" — a metric artifact, reversed.** The 0.881 was `last_hidden`
  cosine, *upstream* of the merger's LayerNorm. At the real MLLM output W4A4 is **near-lossless**
  (response-position KL 0.024 nats, top-1 0.96, BF16-matching descriptions). Mixed precision
  **dropped** for Qwen. Activation-side FoS "improved" last_hidden (0.883→0.906) but **diverged the
  generated response** — a textbook case of optimizing the wrong metric. [QWEN](results/ptq/QWEN_W4A4_VERDICT.md)

---

## 3. Low-precision attention — [attention/](results/attention/)

### ✅ Worked
- **FP8 attention is effectively lossless** on ViT (QKᵀ 0.9998, P·V 0.9999, combined 0.9996). The
  recommended accuracy default. Emulated `quant_sdpa` + wired into HF via the attention-interface
  registry (per-module flag, exact-SDPA fallback elsewhere). [SP-A](results/attention/SPA_ATTENTION_RESULT.md),
  [SP-A wiring](results/attention/SPA_WIRING_RESULT.md)
- **SageAttention3** (FP4 flash, Blackwell) — builds clean on sm_120 (2m38s, no patches, no torch
  downgrade). ~2× over torch SDPA **at S ≳ 4k**, cos 0.98. The right tool for **long-sequence video
  ViT**. [SAGE eval](results/attention/SAGEATTENTION_SM120_EVAL.md)

### ❌ Didn't work / dead ends
- **A custom fused FP4/FP8 attention CUDA kernel** — rejected. It would lose to torch SDPA at ViT
  sequence lengths, and SA3 already covers long-S. Pure waste; roadmap item closed.
- **NVFP4 (vs FP8) attention** — viable but a real step down (combined 0.9951 vs 0.9996); QKᵀ logits
  feed an exponential so their FP4 error is amplified. If used, prefer **FP8 QKᵀ + NVFP4 P·V**.
- **SA3 below its crossover** — *slower* than BF16 SDPA at S ≲ 2–4k (FP4-quant overhead not
  amortized). Image ViTs at native res sit below it. Also **head_dim 96/192 unsupported** (= Qwen3-VL
  vision). So SA3 is **not** a blanket attention speedup — only long-S, head_dim 64/128.
- **smooth-K / smooth-V** — exact and near-free, but the benefit on ViT QKV is tiny (+0.0004): ViT
  outliers are per-token (already handled by per-token scaling), not the per-channel kind smoothing
  targets. Kept on as cheap insurance.

---

## 4. Fast inference — killing the activation-cast overhead — [perf/](results/perf/)

This is where "W4A4 is faster" was actually *earned*. The naive path is **slower** than BF16; the
win required a producer-fusion kernel + a compile-friendly wrapper.

### ❌ Didn't work / dead ends (the long road)
- **Naive per-Linear W4A4 in eager mode — 3.7× SLOWER.** The FP4 GEMM is fine; the killer is the
  **online activation cast** (amax → block-scale → round → FP4 pack → swizzle) done in eager
  PyTorch, costing **10–70× the GEMM** it feeds. [REAL_VIDEO_PERF](results/perf/REAL_VIDEO_PERF.md)
- **`torch.compile` alone** — fuses the *standalone* cast to ~BF16 parity (8.55→0.38 ms), but that
  **does not carry to the full model**; image ViT stayed ~1.0×.
- **torchao + MSLK fused cast + v2 GEMM** — *correct* (cos 0.99) and the fastest available cast, but
  full-model **capped ~1.07–1.10×**: `fc1` (wide output) wins 1.5×, but `fc2` (wide *input*) and
  square `qkv` **lose** — their cast cost (~15% of peak BW on sm_120) exceeds the GEMM benefit.
- **QuTLASS** (built from source for sm_120a) — same fc2/qkv loss pattern; its cast is also a
  separate HBM pass. No better than MSLK for the full model.
- **M-padding to %128 via `F.pad`** — the full-tensor copy costs more than the fast-path saves.
- **Inductor producer-fusion** — `torch.compile` does *not* fuse GELU+cast on its own; the
  `use_triton_kernel=False` plain-cast path didn't help either.
- **`torch._dynamo.disable` hack** — let the rest compile but regressed the large model. Reverted.

### ✅ Worked — the breakthrough
- **GELU→NVFP4 producer fusion.** `fc2`'s input is `GELU(fc1_out)`; fold the NVFP4 cast *into* the
  GELU (one Triton pass, `cast_nvfp4(apply_gelu=True)`) so fc2's wide-input cast is **free** → the
  whole MLP runs **1.3–1.5×** (cos 0.98).
- **Wrap the fused MLP *and* SA3 as `torch.library.custom_op`** (with `register_fake`) → `torch.compile`
  fuses the rest of the model (LayerNorms, RoPE, residuals). This removed the eager overhead that was
  sinking small models — **DINOv3 went 0.98× → 1.21×**. The single most important integration step.
- **Per-frame (B=1)** so SA3 helps the attention (at B=8 the batched SDPA is already saturated, SA3
  → 1.02×); **static** activation globals (calibrate once, freeze); **skip first/last 2** block MLPs
  for accuracy (feature cos 0.87→0.97 for ~0.03–0.05× of speed).
- **No MSLK/torchao runtime dep in the shipped code** — the Triton cast kernel and its swizzle/pack
  primitives are vendored (BSD, from Meta MSLK); the GEMM reuses our existing `_scaled_mm_v2` backend.

Final: all four model families meaningfully faster than compiled BF16 (1.15–1.40×), accuracy
preserved (cos ≥ 0.97). V-JEPA2 (largest M) wins most — the GELU-fused MLP scales with M.

---

## Cross-cutting lessons

1. **Measure the right metric.** `last_hidden` cosine lied about Qwen3-VL (merger LayerNorm absorbs
   magnitude distortion); the decisive metric was response-level KL/top-1. Feature cosine ≠ k-NN
   accuracy; low-res benchmarks (CIFAR) overstate the W4A4 cost vs native-res (Flowers/Food).
2. **The FP4 *GEMM* was never the bottleneck — the *activation cast* was.** Every speed dead-end
   above is a different way of paying that cast; the win was making it free (producer fusion).
3. **`torch.compile` is load-bearing**, but only once the opaque ops (FP4 GEMM, SA3) are wrapped as
   custom ops so it can fuse around them.
4. **sm_120 ≠ sm_100.** No tcgen05/TMEM; v1 `_scaled_mm` and several CUTLASS/FlashInfer paths are
   broken — always gate numerics on-device. (See [research/nvfp4-vit-findings](research/2026-06-23-nvfp4-vit-research-findings.md).)
5. **NVFP4's two-level scale breaks LLM intuitions** — percentile calibration and per-channel
   smoothing, both standard for per-tensor/INT quant, are useless-to-harmful here.
