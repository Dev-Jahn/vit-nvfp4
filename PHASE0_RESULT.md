# SP1 Phase-0 Result — W4A4 NVFP4 GEMM on sm_120

**Date:** 2026-06-23 · **GPU:** NVIDIA RTX PRO 6000 Blackwell (sm_120, CC 12.0) · **Stack:** torch 2.12.1+cu130, CUDA 13.0

## Verdict

**A Python-callable, numerically-correct W4A4 NVFP4 dense GEMM exists on this box — no custom kernel build required.**

Selected backend: **`torch._scaled_mm_v2`** (PyTorch native), `ScalingType.BlockWise1x16` + per-tensor `TensorWise` global, `SwizzleType.SWIZZLE_32_4_4` for block scales. Both operands FP4 (`float4_e2m1fn_x2`) → true W4A4.

## Backend ladder outcome

| Rung | Backend | Outcome |
|---|---|---|
| **1 (chosen)** | **torch `_scaled_mm_v2`** | ✅ Correct (cos_emul = 1.00000 vs fp32 reference) across all ViT shapes/dists. Zero extra deps. |
| — | FlashInfer b12x | ❌ Not used: `flashinfer-python` pins torch and **downgrades 2.12.1+cu130 → 2.10.0**, breaking the cu130 env (`undefined symbol: ncclCommResume`). Incompatible with the user-chosen cu130 stack. |
| — | CUTLASS-79 / cuBLASLt / hand-built | ⏭️ Unnecessary — rung 1 passed the correctness gate. |

**Plan adaptation:** the written plan had FlashInfer as rung 1; reality made torch-native the correct first rung. The plan's governing principle ("adopt the first backend that passes the on-device gate") is satisfied.

## Correctness (28 tests pass)

`assert_gemm_correct` compares the kernel against (a) an fp32 emulation of the *same* quantized operands and (b) a bf16 reference:
- **cos_emul = 1.00000** everywhere → packing, two-level scale math, and SWIZZLE_32_4_4 layout all match the kernel exactly (catches the flashinfer #2577 all-zeros failure mode — not present here).
- **cos_bf16 ≈ 0.991** → expected FP4 quantization error vs bf16.
- Distributions tested: normal, heavy-tailed, outlier-row (CLS/register-like).
- **M is unconstrained** (197, 577, 1370, even M=1 all cos_emul=1.0) → fits ViT variable sequence lengths. **K and N must be multiples of 16** (NVFP4 block).

## Throughput (pure kernel, operands pre-packed)

| M | K | N | FP4 TFLOP/s | BF16 | FP8 | FP4/BF16 | FP4/FP8 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 256 | 1024 | 1024 | 17.2 | 62.0 | 48.2 | 0.28× | 0.36× |
| 1024 | 1024 | 1024 | 72.5 | 173.3 | 196.3 | 0.42× | 0.37× |
| 256 | 1152 | 4304 | 81.9 | 175.0 | 224.4 | 0.47× | 0.37× |
| 1024 | 1152 | 4304 | 335.5 | 260.3 | 447.8 | 1.29× | 0.75× |
| 1024 | 1408 | 6144 | 563.6 | 222.3 | 477.6 | 2.54× | 1.18× |
| 4096 | 1408 | 6144 | 929.5 | 303.8 | 548.2 | 3.06× | 1.70× |
| 4096 | 4096 | 4096 | 1172.3 | 377.1 | 714.5 | 3.11× | 1.64× |

- **Compute-bound (large M·K·N): FP4 = 3.0–3.1× BF16, 1.6–1.7× FP8** — ≥2× target met. This is the prefill / large-batch / large-MLP regime.
- **Small shapes: FP4 slower than BF16** (overhead/launch-bound). Decode (M small) is memory-bound, where the win is weight-bandwidth (W4) not FP4 compute — consistent with the research's decode-vs-prefill split.

## Calling convention (the SSOT for SP2)

For A:(M,K) and weight B:(N,K) row-major → out (M,N):
```python
a_fp4 = as_float4_x2(pack_e2m1(a_codes))        # (M, K//2)  float4_e2m1fn_x2
b_fp4 = as_float4_x2(pack_e2m1(b_codes)).t()    # (N,K//2) -> (K//2, N)
a_sf  = to_blocked(a_bs)                         # (M, K//16) e4m3 -> swizzled flat
b_sf  = to_blocked(b_bs)                         # (N, K//16) e4m3 -> swizzled flat  (outer = N, NOT transposed)
torch._scaled_mm_v2(a_fp4, b_fp4,
    [a_sf, a_global.reshape(1)], [BlockWise1x16, TensorWise], [SWIZZLE_32_4_4, NO_SWIZZLE],
    [b_sf, b_global.reshape(1)], [BlockWise1x16, TensorWise], [SWIZZLE_32_4_4, NO_SWIZZLE],
    None, out_dtype)
```
- Two-level scale: `x ≈ e2m1_value * block_scale_e4m3 * global_fp32`; `global = amax/(6*448)`, `block = e4m3((amax_block/6)/global)`.
- Both A and B block scales are indexed `[outer_dim, K//16]` (A: outer=M, B: outer=N) and swizzled with `to_blocked`.
- Activation block scale is computed **on host** (per-16 amax) and pre-swizzled; the kernel does not derive it dynamically.

## Open questions (spec §11) — resolved

1. **Which backend is correct on this box?** → `torch._scaled_mm_v2` (BlockWise1x16). cos_emul = 1.0.
2. **CUTLASS-79 #2906 fix sufficient?** → N/A (not needed).
3. **FlashInfer b12x JIT on this box?** → N/A; flashinfer-python is incompatible with torch 2.12+cu130 (downgrades torch).
4. **Activation per-block scale: HW-dynamic or host pre-swizzle?** → **Host**: we quantize + `to_blocked`-swizzle the per-16 block scale and pass it; the kernel consumes it.
5. **cu130 torch wheel availability?** → ✅ `torch==2.12.1+cu130` stable wheel; sm_120 detected, native FP4 GEMM works.

## Implication for SP2+
- No custom CUDA/CUTLASS toolchain needed for dense W4A4 — build `QuantLinear` directly on `torch._scaled_mm_v2`.
- Pre-quantize/pre-swizzle weights once (static). Quantize activations online on host; consider fusing pack+swizzle to cut the small-shape overhead seen above.
- Attention (SP-A) still needs a separate path (FP8/FP4 attention); `_scaled_mm_v2` covers only the dense Linear GEMMs.
