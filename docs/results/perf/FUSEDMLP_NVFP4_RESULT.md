# Fused-MLP NVFP4 + SA3: meaningfully faster ViT inference at 1080p (sm_120)

**Goal:** make every kind of dense ViT (DINOv2-base/large, DINOv3, V-JEPA2) run *meaningfully*
faster than `torch.compile`'d BF16 at 1080p on RTX PRO 6000 (sm_120), with accuracy preserved.

**Result (vs compiled BF16, @1080p, per-frame):**

| model            | speedup       | feature cos | notes |
|------------------|---------------|-------------|-------|
| DINOv2-large     | **1.18–1.21×** | 0.974 | hidden 1024, 24 layers |
| DINOv2-base      | **1.13–1.17×** | 0.965 | hidden 768, 12 layers (smallest dense) |
| DINOv3-ViT-B/16  | **1.18–1.23×** | 0.984 | RoPE attention |
| V-JEPA2.1-L      | **1.40×**      | 0.984 | S=130560 (32 frames @1080p); biggest M → biggest MLP-fusion win |

All four beat compiled BF16 by ≥1.13× with feature cosine ≥0.96, all via the one
`fuse_mlps` + `enable_sa3_attention` path. V-JEPA2 (S=130560) is the largest win — the GELU-fused
MLP scales with M, so the heavy video model gains the most (eager 1.29×, compiled 1.40×).

## Why naive W4A4 did NOT win (the trap)

NVFP4 W4A4 only helps the GEMM (~1.5× compute). But each quantized Linear needs an **online
activation cast** (amax + per-16 block-scale + E2M1 pack + 128×4 swizzle). On sm_120 the available
cast kernels (torchao/MSLK, QuTLASS) run at ~15% of peak bandwidth, so the cast cost dominates for
any GEMM whose input is wide:

| GEMM (M=84480)         | fused cast + GEMM | verdict |
|------------------------|-------------------|---------|
| fc1 (K=hidden, N=4H)   | 1.45–1.60×        | win (cheap cast, big GEMM) |
| fc2 (K=4H, N=hidden)   | **0.69×**         | lose (wide-input cast dominates) |
| qkv (square)           | 0.73×             | lose |

So a drop-in `quantize_` of every Linear caps image ViT at **~1.07–1.10×** — not worth applying.
Neither torchao+MSLK nor QuTLASS fixes this; both cast as a separate HBM pass.

## What unlocked it

1. **GELU→NVFP4 producer fusion.** `fc2`'s input is `GELU(fc1_out)`. We fold the NVFP4 cast *into*
   the GELU (one Triton kernel: load fc1_out → GELU → block-amax → pack → swizzle → store), so fc2's
   wide-input cast becomes free and fc2 runs at its FP4-compute rate. The MLP (fc1 + GELU + fc2)
   then runs **1.31–1.51×** (cos 0.98). Kernel = MSLK's `triton_quantize_nvfp4_kernel` + one GELU
   line after the load; pass `per_tensor_scale.reciprocal()` as the kernel's global scale.

2. **`_scaled_mm_v2`, not `_scaled_mm`.** torch's v1 NVFP4 path returns garbage on sm_120 (cos≈0);
   `torch._scaled_mm_v2` (BlockWise1x16 + TensorWise global, SWIZZLE_32_4_4) is correct (cos 0.99).

3. **SA3 attention, per-frame (B=1).** At 1080p attention is ~35–55% of the forward. SageAttention3
   (FP4) gives ~1.23× at B=1 but only ~1.02× at B=8 (batched SDPA already saturates the GPU), so we
   process one frame at a time.

4. **Wrap both as `torch.library.custom_op` + `torch.compile`.** The fused MLP and SA3 are opaque to
   Dynamo and graph-break. Wrapping each as a custom op (with `register_fake`) lets `torch.compile`
   fuse the *rest* of the model (RoPE concat, LayerNorms, residuals, qkv projection). This is what
   removed the eager overhead that was sinking the small models — DINOv3 went **0.98× → 1.18–1.23×**.

5. **Static (calibrated) global scales** — the per-tensor amax is captured on the first forward and
   frozen, so steady-state forwards do no extra reduction pass.

## Accuracy

Quantizing every MLP gives cos 0.87–0.93. Skipping the **first/last 2 blocks' MLPs**
(`replace_mlp_raw(model, skip_ends=2)`, the standard "sensitive ends" PTQ rule) lifts cos to
**0.965–0.984** for ~0.03–0.05× of speed. Attention projections (qkv/out) stay BF16 — square GEMMs
don't benefit from FP4 anyway.

## Usage (integrated into `vit_nvfp4`)

The fused NVFP4 cast kernel is vendored (`nvfp4/triton_cast.py`, no MSLK runtime dep) and the
GEMM reuses the existing `_scaled_mm_v2` backend, so the fast path needs only **torch + triton**.
Only the FP4 *attention* needs SageAttention3 — the optional `accel` extra (`pip install
'vit-nvfp4[accel]'`; sm_120 source build, see notes below).

```python
from vit_nvfp4.ptq import fuse_mlps, enable_sa3_attention, sa3_attn_policy

model = AutoModel.from_pretrained(..., dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
fuse_mlps(model, skip_ends=2)                                   # GELU-fused NVFP4 W4A4 MLPs
enable_sa3_attention(model, sa3_attn_policy(model.config.num_hidden_layers))  # FP4 attention
model = torch.compile(model)                                   # fuses the rest (norms/RoPE/residuals)
# process frames one at a time (B=1) — SA3 only wins at batch 1
```

`fuse_mlps` assumes the block activation is exact GELU (true for DINOv2/3, V-JEPA2); it leaves the
first/last `skip_ends` blocks in BF16 (accuracy). `cast_nvfp4(x, per_tensor_scale, apply_gelu=)` and
`backends.torch_scaled_mm.gemm_packed` are the reusable primitives. Unit tests: `tests/test_fused_mlp.py`.
