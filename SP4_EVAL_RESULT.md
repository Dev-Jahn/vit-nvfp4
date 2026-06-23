# SP4 (eval first) — Real-data accuracy of W4A4 NVFP4 DINOv2

**Date:** 2026-06-23 · DINOv2-base, sm_120 · metric: frozen-feature **k-NN top-1** (CLS embedding)

User directive: measure **real downstream accuracy** before building mixed-precision. Done.

## Result — CIFAR-100 (balanced, all 100 classes; gallery 2000 / query 1000)

| | k=10 | k=20 |
|---|---|---|
| BF16 k-NN top-1 | **0.822** | 0.812 |
| W4A4 k-NN top-1 (skip 2/2, max-calibrated) | **0.798** | 0.784 |
| **accuracy drop** | **−2.4 pt** | −2.8 pt |
| query feature cosine vs BF16 | 0.956 | |

## Reading
- **W4A4 PTQ costs ~2.5 percentage points of k-NN top-1 — a modest drop, not the collapse** the all-layer/naive-W4A4 literature reports for ViTs. This is what feature cosine 0.95 actually means downstream.
- The mixed-precision lever (SP4 proper) trades FP4 compute for fidelity: skip(4,4) gave feature cosine 0.98 (vs 0.95 at skip 2/2), so its accuracy drop should be smaller — to be quantified if the 2.5 pt gap matters for the target use case.
- Whether 2.5 pt is acceptable is a product call. If yes, W4A4 PTQ is already usable for DINOv2; if not, SP4 mixed-precision promotion and/or SP5 calibration-recovery (see the calibration-techniques research) close it.

## Methodology / caveats
- **k-NN, not linear-probe**: cosine k-NN (k=10/20) on CLS features, gallery=train subset, query=test subset. Standard DINOv2-style frozen-feature eval.
- **CIFAR-100 (32×32 upsampled to 224)**: balanced and non-degenerate (BF16 0.82 is healthy), but low native resolution. A high-res set (ImageNet/Food-101) would give higher absolute numbers; the **relative** W4A4-vs-BF16 drop is the measured quantity and is robust to this.
  - (An earlier Food-101 run was discarded: its class-ordered stream + small shuffle buffer sampled only ~2 classes → degenerate 11% task. CIFAR-100 full-load + random subset fixes class coverage.)
- Quant model: skip(2,2) → 48 W4A4 Linears, static activation scale calibrated with `method="max"` on 8 gallery batches (SP3 finding: max, not percentile).
- Harness: `src/vit_nvfp4/eval/knn.py` (hermetic unit-tested), runnable `examples/eval_knn_dinov2.py`.
