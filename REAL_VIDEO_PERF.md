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

## Full-model benchmark — all 4 repvis models, torch.compile ON (the real deployment baseline)
Real test.mp4 input (DINO: 8 frames, per-frame @max_side 1024; V-JEPA: 32-frame clip @640). W4A4 = NVFP4 Linears (middle blocks) + SA3 FP4 attention (all blocks). Steady-state fwd ms after compile warmup.

| model | S | BF16 (compiled) | W4A4+SA3 (compiled) | speedup | feat cos | PCA-RGB cos |
|---|---|---|---|---|---|---|
| dinov2-base | 2994 | 23.2 ms | 48.7 ms | **0.48×** | 0.948 | 0.992 |
| dinov2-large | 2994 | 69.8 ms | 126.2 ms | **0.55×** | 0.962 | 0.994 |
| dinov3-vitb16 | 2309 | 18.8 ms | 43.5 ms | **0.43×** | 0.805 | 0.912 |
| vjepa21-vitl | 14080 | 93.7 ms | 142.3 ms | **0.66×** | 0.970 | 0.992 |

**W4A4+SA3 is SLOWER than compiled BF16 on every model (0.43–0.66×), even at V-JEPA's S=14080.** Accuracy holds (PCA-RGB ≥ 0.99 except DINOv3 0.912 / feat 0.805). The compile logs show the cause: `torch._scaled_mm_v2` and the SA3 custom op are **opaque to TorchDynamo (graph breaks)**, so the online activation quant does not fully fuse in-model — the standalone-function parity does not carry. And compiled BF16 is itself a strong baseline.

## Corrected conclusions
1. **"W4A4 NVFP4 = 3× faster" (PHASE0/SP1) was a GEMM-only microbench** (operands pre-quantized). It does not reflect inference: end-to-end our W4A4 path is *slower*.
2. **`torch.compile` does NOT rescue it in-model.** It fuses the *standalone* quant to bf16 parity, but `_scaled_mm_v2` graph-breaks dynamo inside the real model, so compiled W4A4+SA3 is **0.43–0.66× (1.5–2.3× slower)** across all four models.
3. **As built, this toolkit delivers PTQ accuracy, NOT inference speed.** Real speedup needs the quant+FP4-GEMM fused and registered as a `torch.library` custom op (so dynamo keeps it in-graph), or an external optimized W4A4 runtime (vLLM `nvfp4_scaled_mm_sm120`, TRT-LLM). `nvfp4_linear` (eager quant + `_scaled_mm_v2`) is a correctness/measurement reference.
4. **SA3 FP4 attention** speeds attention only (≈1.3× at long S; loses below ~S=4–7k), but attention is a minority of the forward, so it cannot offset the linear-quant overhead — net still slower here. (`SAGEATTENTION_SM120_EVAL.md`)
5. **Accuracy holds** (output preserved; DINOv3 weakest at feat 0.805 / PCA 0.912). The open problem is the inference *kernel*, not the PTQ math.

## Reproduce
Scripts in the session scratchpad: `real_vjepa_sa3.py`, `real_vjepa_combined.py`, `nvfp4_lin_bench.py`, `compile_test.py` (run in a torch-2.12+cu130 venv with `sageattn3` built; `repvis/src` + `vit-nvfp4/src` on `sys.path`).
