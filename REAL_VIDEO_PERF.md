# Real-video use-case (repvis V-JEPA on test.mp4) — measured perf & accuracy

**Date:** 2026-06-24 · RTX PRO 6000 sm_120 · a real 32-frame clip of `repvis/samples/test.mp4` (1 h 1080p) through V-JEPA 2.1 ViT-L at the actual repvis spec (max_side 640, tubelet 2) → **attention sequence S = 14,080**.

This corrects earlier glib speed claims by measuring the real pipeline, per the spec we'd actually deploy (W4A4 NVFP4 Linears + low-precision attention).

## Accuracy — all configs preserve the repvis output (dense features → PCA-RGB video)
| config | feature cos | per-token mean | **PCA-RGB cos** |
|---|---|---|---|
| W4A4 linears | 0.976 | — | 0.9943 |
| SA3 FP4 attention | 0.9965 | 0.9962 | **0.9989** |
| W4A4 linears + SA3 attn | 0.970 | — | 0.9920 |

The PCA-RGB visualization (what repvis renders) is essentially unchanged (cos ≥ 0.992). **PTQ accuracy is sound on the real long-context video workload.**

## Speed — EAGER mode (torch.compile OFF)
| config | fwd ms | speedup |
|---|---|---|
| BF16 (ref) | 122 | 1.00× |
| SA3 FP4 attention | 111 | **1.10×** |
| W4A4 linears | 452 | **0.27× (3.7× SLOWER)** |
| W4A4 linears + SA3 | 440 | 0.28× |

W4A4 is a large net loss in eager mode. Root cause (microbench, V-JEPA Linear shapes, M=14080):
| shape | bf16 | nvfp4_linear (eager, online-quant+GEMM) | GEMM-only (pre-quantized) |
|---|---|---|---|
| K1024 N1024 | 0.105 ms | 1.578 ms (15×) | 0.117 ms (1.11×) |
| K1024 N4096 | 0.340 ms | 1.657 ms (4.9×) | 0.186 ms (**0.55× = 1.8× faster**) |
| K4096 N1024 | 0.327 ms | 8.524 ms (26×) | 0.255 ms (0.78×) |

The **FP4 GEMM itself is fine** (≈1.3–1.8× faster than bf16 on large GEMMs); the killer is the **online activation quantization done in eager PyTorch** (amax → per-block scale → round → FP4 pack → swizzle), which costs 10–70× the GEMM it feeds.

## The fix: torch.compile fuses the quant
| | eager | **torch.compile** | bf16 |
|---|---|---|---|
| nvfp4_linear (M14080 K4096 N1024) | 8.555 ms | **0.378 ms** | 0.329 ms |

`torch.compile` fuses the elementwise quant into ~bf16 parity (22× over eager) **for the STANDALONE function**. repvis exposes compile via `REPVIS_COMPILE=1`. **But this parity does NOT carry to the full model** — see the multi-model benchmark below.

## Full-model benchmark — all 4 repvis models, torch.compile ON, at repvis DEFAULT resolution
Real test.mp4 input (DINO: 8 frames, per-frame @max_side 1024; V-JEPA: 32-frame clip @640). W4A4 = NVFP4 Linears (middle blocks) + SA3 FP4 attention (all blocks). Steady-state fwd ms after compile warmup. **(At full resolution the picture flips — next section.)**

| model | S | BF16 (compiled) | W4A4+SA3 (compiled) | speedup | feat cos | PCA-RGB cos |
|---|---|---|---|---|---|---|
| dinov2-base | 2994 | 23.2 ms | 48.7 ms | **0.48×** | 0.948 | 0.992 |
| dinov2-large | 2994 | 69.8 ms | 126.2 ms | **0.55×** | 0.962 | 0.994 |
| dinov3-vitb16 | 2309 | 18.8 ms | 43.5 ms | **0.43×** | 0.805 | 0.912 |
| vjepa21-vitl | 14080 | 93.7 ms | 142.3 ms | **0.66×** | 0.970 | 0.992 |

**At this default resolution W4A4+SA3 is slower on every model (0.43–0.66×).** Cause (compile logs): `torch._scaled_mm_v2` and the SA3 op are **opaque to TorchDynamo (graph breaks)**, so the online activation quant doesn't fully fuse — at small S the residual overhead dominates. **But this flips with resolution.**

## Full resolution (max_side 1920, all 90,003 frames) — the speedup SCALES WITH S
Bigger inputs → bigger Linear GEMMs (FP4 wins) + longer attention (SA3 wins), and the fixed overhead amortizes. Compiled speedup (BF16 / W4A4+SA3), and full-video forward-only time over all 90,003 frames at source fps:

| model | S @1920 | speedup @default | speedup @1920 | feat cos | PCA-RGB | full-video fwd (BF16 → W4A4+SA3) |
|---|---|---|---|---|---|---|
| dinov2-base | 10.5k | 0.48× | 0.85× | 0.953 | 0.994 | 28.0 → 32.9 min |
| dinov2-large | 10.5k | 0.55× | 0.87× | 0.958 | 0.994 | 79.4 → 91.3 min |
| dinov3-vitb16 | 8.2k | 0.43× | 0.81× | 0.816 | 0.931 | 19.5 → 24.1 min |
| **vjepa21-vitl** | **130.6k** | 0.66× (S14k) | **1.26×** | 0.972 | 0.991 | **232.7 → 184.6 min (−48 min)** |

V-JEPA sequence-length sweep (compiled): **S=14k → 0.68×, S=37k → 0.98×, S=131k → 1.26×.** At full resolution the spatio-temporal V-JEPA workload (S≈131k, attends all tokens at once) is **1.26× faster** with W4A4+SA3 — saving ~48 min on the hour-long video (233→185 min forward-only), accuracy preserved (PCA 0.991). DINO stays per-frame (S ≤ 10.5k even at 1920), below the **~S≈37k crossover**, so it's still ~0.81–0.87×.

## Corrected conclusions
1. **"W4A4 = 3× faster" (PHASE0/SP1) was a GEMM-only microbench**; the real end-to-end speedup **SCALES WITH sequence length S** — bigger GEMMs + longer attention amortize the online-quant overhead and the `_scaled_mm_v2` graph-break.
2. **Crossover ≈ S 37k.** Below it (per-frame image ViT, low-res clips) W4A4+SA3 is *slower* (0.43–0.87×); above it it *wins* — **full-resolution V-JEPA (S≈131k) = 1.26× faster**, accuracy preserved.
3. **The toolkit delivers a real speedup on the heavy long-context case (full-res spatio-temporal video), not on short-sequence per-frame image ViT.** A fused quant+GEMM `torch.library` custom op (removing the dynamo graph-break) — or vLLM/TRT-LLM — would lower the crossover so image ViTs benefit too and widen the video win.
4. **SA3 FP4 attention** wins at long S (≈1.3× attention-only) and is part of the V-JEPA full-res win; it loses below ~S=4–7k (`SAGEATTENTION_SM120_EVAL.md`).
5. **Accuracy holds across all** (PCA-RGB ≥ 0.99 except DINOv3 0.93 / feat 0.82). The PTQ math is sound; speed is purely an S / kernel-fusion question.

## Reproduce
Scripts in the session scratchpad: `real_vjepa_sa3.py`, `real_vjepa_combined.py`, `nvfp4_lin_bench.py`, `compile_test.py` (run in a torch-2.12+cu130 venv with `sageattn3` built; `repvis/src` + `vit-nvfp4/src` on `sys.path`).
