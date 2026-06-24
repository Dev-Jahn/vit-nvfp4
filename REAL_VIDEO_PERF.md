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

`torch.compile` fuses the elementwise quant into ~bf16 parity (22× over eager). repvis already exposes this via `REPVIS_COMPILE=1`. Even compiled, the *net* W4A4 speedup is GEMM-size-dependent: large MLP up-projections win (~1.5–1.8×), small projections sit near parity.

## Corrected conclusions
1. **The "W4A4 NVFP4 = 3× faster" (PHASE0/SP1) was a GEMM-only microbench** (operands already quantized). End-to-end inference with the eager `nvfp4_linear` is 5–26× *slower* because the online activation quant is unfused.
2. **W4A4 inference requires `torch.compile`** to fuse the activation quant (or a fused quant+GEMM kernel / an optimized W4A4 runtime like vLLM/TRT-LLM). Our `nvfp4_linear` is **accuracy/measurement-grade, not speed-grade** as-is.
3. **The genuine "fused kernel" need is on the LINEAR activation-quant side, not attention** — and `torch.compile` largely covers it (no hand-written kernel required).
4. **SA3 FP4 attention** is the one drop-in fused win on this video workload: 1.10× end-to-end (1.30× attention-only; attention is ~45% of the V-JEPA forward), accuracy-safe (PCA 0.9989), no custom kernel. Worth it for long-context video; below ~S=4–7k it loses to torch SDPA (see `SAGEATTENTION_SM120_EVAL.md`).
5. **Accuracy of the PTQ holds** on the real workload — the toolkit correctly preserves the rendered output; the open work is making the W4A4 *inference path* fast (compile / fused quant), not accuracy.

## Reproduce
Scripts in the session scratchpad: `real_vjepa_sa3.py`, `real_vjepa_combined.py`, `nvfp4_lin_bench.py`, `compile_test.py` (run in a torch-2.12+cu130 venv with `sageattn3` built; `repvis/src` + `vit-nvfp4/src` on `sys.path`).
