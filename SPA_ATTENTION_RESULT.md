# SP-A Result — Low-precision (FP8 / NVFP4) attention (emulated, measured)

**Date:** 2026-06-23 · DINOv2-base layer 6, sm_120 · measure-first emulation (quantize→dequantize→matmul, softmax in fp32), format-preserving, no custom kernel yet.

Adds `quant_sdpa` — a drop-in `F.scaled_dot_product_attention` that runs QKᵀ and P·V in FP8 E4M3 or NVFP4 (E2M1 + per-16 E4M3 scale). Per SP1, this is the **accuracy-floor measurement layer**: each low-precision matmul is emulated so we know what a fused FP8/FP4 tensor-core kernel would hit before building one.

## SOTA grounding (web research)
- **SageAttention3** (arXiv:2505.11594, Blackwell): microscaling **NVFP4** (E2M1 + 1×16 E4M3 scale) on *both* QKᵀ and P·V; 5× over FlashAttention on RTX5090. The 1×16 group exactly matches our `quantize_to_nvfp4(block=16)`.
- **SageAttention2** (arXiv:2411.10958): Q,K→INT4 with outlier **smoothing**, P̃,V→**FP8** with two-level accumulation; uses 8-bit (INT8+FP8) for problematic layers, 4-bit otherwise.
- **torch-native feasibility on sm_120 today**: the GEMM primitives exist (`torch._scaled_mm` FP8 per-tensor, `torch._scaled_mm_v2` NVFP4 block-scaled — the SP1-validated path). What does **not** exist in torch is a *fused online-softmax flash kernel* in FP8/FP4 → that is the custom-CUDA follow-up (SageAttention3's kernel). Emulation is the correct measurement stand-in until then.

## Measured (real DINOv2 Q/K/V, B=8, H=12, S=257, D=64; cosine vs fp32 SDPA)

Per-stage isolation (the other stage kept bf16):
| stage | FP8 | NVFP4 |
|---|---|---|
| **QKᵀ** | **0.9998** | 0.9970 |
| **P·V** | **0.9999** | 0.9981 |

Combined (smooth-K + smooth-V on):
| QKᵀ \ P·V | FP8 | NVFP4 |
|---|---|---|
| **FP8** | **0.9996** | 0.9979 |
| **NVFP4** | 0.9969 | 0.9951 |

Smoothing ablation (QKᵀ nvfp4 / P·V nvfp4): no-smooth 0.9947 → +smooth-K 0.9949 → +smooth-V 0.9948 → both **0.9951**.

## Findings / recommendation
1. **FP8 attention is effectively lossless** on ViT (QKᵀ 0.9998, P·V 0.9999, combined **0.9996**). This is the **recommended default for attention** — it satisfies the precision policy's "try NVFP4 → fallback FP8" by landing squarely on the safe FP8 rung with no measurable cost.
2. **NVFP4 attention is viable but a real step down** (combined **0.9951**). P·V tolerates NVFP4 better (0.9981) than QKᵀ (0.9970): post-softmax P∈[0,1] is benign, while QKᵀ logits feed an exponential so their error is amplified. A sound **mixed** rung is **NVFP4 P·V + FP8 QKᵀ** (0.9979) — most of the FP4 memory/throughput win on the larger PV operands, FP8 safety on the error-sensitive logits. This mirrors SageAttention2 (4-bit QK + FP8 PV) adapted to NVFP4.
3. **smooth-K / smooth-V are exact** (proven + unit-tested: K-mean is softmax-invariant; V-mean adds back since attention rows sum to 1) and near-free, but their benefit on ViT QKV is small (+0.0004 here) because ViT attention outliers are *per-token* (CLS/register), already absorbed by per-token scaling — not the *per-channel* outliers smoothing targets (an LLM phenomenon). Kept **on by default** as cheap insurance for outlier-heavier models/layers.
4. **Emulation is faithful to the kernel**: emulated NVFP4 QKᵀ vs the real `nvfp4_gemm` tensor-core path → cosine > 0.999 (test), so these floors are representative (same conclusion as SP1's cos_emul≈1.0).

## API & integration (integration itself is a follow-up, not this task)
```python
quant_sdpa(query, key, value, attn_mask=None, is_causal=False, scale=None,
           qk="fp8", pv="fp8", smooth_k=True, smooth_v=True)  # (...,S,D) -> (...,S,D)
```
Drop-in for `F.scaled_dot_product_attention`. Later: monkeypatch a model's attention `F.scaled_dot_product_attention` call with `quant_sdpa`, or subclass the attention module — gated by the per-model policy (a follow-up owned by the model-expansion / mixed-precision work, NOT changed here). Default `qk="fp8", pv="fp8"`; drop PV (or both) to `"nvfp4"` for the aggressive rung.

## Blockers / notes
- No torch fused FP8/FP4 flash kernel on sm_120 → fused speedup needs custom CUDA (SageAttention3-style); this deliverable is the accuracy floor + drop-in API, not a perf kernel.
- `torch._scaled_mm` FP8 is 2-D / per-tensor-or-rowwise; batched attention (B·H, S, S) would loop or reshape per (B,H) when wiring the real kernel — fine for the emulation here.
- Measured on one mid layer (6) × 8 images; numbers are stable but a full per-layer sweep is a cheap future add.

## Deliverables
- `src/vit_nvfp4/nvfp4/fp8.py` (`fp8_e4m3_quant_dequant`), `src/vit_nvfp4/nvfp4/attention.py` (`quant_sdpa`); exported from `nvfp4/__init__.py`.
- Tests: `tests/test_attention.py` — 8 pass (shape, bf16/causal parity, FP8/NVFP4 fidelity, smooth-K/V exactness, emulation-vs-kernel).
- Harness: `examples/attention_precision.py` (real DINOv2 Q/K/V capture + per-stage table).
