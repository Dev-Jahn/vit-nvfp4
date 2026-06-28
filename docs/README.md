# vit-nvfp4 docs

NVFP4 W4A4 PTQ + low-precision-attention toolkit for Vision Transformers on sm_120
(RTX PRO 6000 Blackwell, CUDA 13.x, torch 2.12+cu130).

**Start here:** [`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md) — every approach tried, what worked,
what didn't, and why. The one-page map of the whole project.

## Layout

```
docs/
  DEVELOPMENT_LOG.md   ← what worked / what didn't (read first)
  research/            ← external findings that informed the design
  design/              ← specs & plans (the intended design, pre-implementation)
  results/             ← measured outcomes of each work package
    ptq/               ·   quantization accuracy & methodology
    attention/         ·   low-precision (FP8/NVFP4/SA3) attention
    perf/              ·   GEMM kernel, real-video perf, fast-inference (FusedMLP)
```

## Index

### results/ptq — quantization accuracy
- [SP2_RESULT](results/ptq/SP2_RESULT.md) — NVFP4 PTQ frontend (DINOv2 vertical slice), block policy.
- [SP2_EXPAND_RESULT](results/ptq/SP2_EXPAND_RESULT.md) — model-aware PTQ across DINOv2/v3, SigLIP2, Qwen3-VL, V-JEPA2.
- [SP3_RESULT](results/ptq/SP3_RESULT.md) — activation calibration; **why "max", not percentile**.
- [SP4_EVAL_RESULT](results/ptq/SP4_EVAL_RESULT.md) — real-data k-NN accuracy (high-res ~lossless).
- [SP5_RESULT](results/ptq/SP5_RESULT.md) — 4 calibration error-reduction techniques; **Four Over Six adopted**.
- [QWEN_W4A4_VERDICT](results/ptq/QWEN_W4A4_VERDICT.md) — the "0.881 needs mixed precision" metric-artifact reversal.

### results/attention — low-precision attention
- [SPA_ATTENTION_RESULT](results/attention/SPA_ATTENTION_RESULT.md) — emulated FP8/NVFP4 `quant_sdpa`, accuracy floors.
- [SPA_WIRING_RESULT](results/attention/SPA_WIRING_RESULT.md) — wiring `quant_sdpa` into HF models.
- [SAGEATTENTION_SM120_EVAL](results/attention/SAGEATTENTION_SM120_EVAL.md) — SageAttention3 vs torch SDPA; custom-kernel decision.

### results/perf — kernels & speed
- [PHASE0_RESULT](results/perf/PHASE0_RESULT.md) — the W4A4 NVFP4 GEMM backend (`_scaled_mm_v2`).
- [REAL_VIDEO_PERF](results/perf/REAL_VIDEO_PERF.md) — repvis V-JEPA on test.mp4; why naive W4A4 is slower.
- [FUSEDMLP_NVFP4_RESULT](results/perf/FUSEDMLP_NVFP4_RESULT.md) — **GELU-fused MLP + SA3, the fast-inference win**.

### research / design
- [research/nvfp4-vit-research-findings](research/2026-06-23-nvfp4-vit-research-findings.md) — sm_120 NVFP4 facts, GEMM-path breakages.
- [research/calibration-error-reduction](research/2026-06-23-calibration-error-reduction.md) — survey behind SP5.
- [design/specs](design/specs/) · [design/plans](design/plans/) — pre-implementation designs.
