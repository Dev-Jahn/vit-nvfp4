# SP5 Result — Calibration-only W4A4 error-reduction (4 techniques, measured)

**Date:** 2026-06-23 · DINOv2-base, sm_120, skip(2,2) · all format-preserving (E2M1 + per-16 E4M3 + FP32 global=max, `torch._scaled_mm_v2` unchanged), composable, no scratch-QAT.

Implemented & integrated four calibration-only techniques (from the `docs/research/2026-06-23-calibration-error-reduction.md` survey), each measured on the **same k-NN harness**.

## Authoritative: Food-101 (native high-res, 101 classes, real headroom) — gallery 2020 / query 1010, k=10

| config | k-NN top-1 | Δ vs BF16 | cosine |
|---|---|---|---|
| BF16 (ref) | 0.7554 | — | 1.000 |
| W4A4 baseline (max-calib) | 0.7416 | **−1.39 pt** | 0.976 |
| **+ Four Over Six** | **0.7545** | **−0.10 pt** | 0.978 |
| + GPTQ | 0.7416 | −1.39 pt | 0.987 |
| + Four Over Six + bias correction | 0.7525 | −0.30 pt | 0.982 |
| + RegCache (token deletion) | 0.7366 | −1.88 pt | 0.972 |

## Cross-task view

| technique | CIFAR-100 (low-res, Δ vs baseline) | Food-101 (high-res, Δ vs baseline) | verdict |
|---|---|---|---|
| **Four Over Six** | **+1.20 pt** | **+1.29 pt** (−1.39→−0.10) | **robust winner — adopt** |
| GPTQ | +1.60 pt | +0.00 pt (cosine only) | task-dependent |
| RegCache (del) | +1.06 pt | −0.49 pt (hurts) | risky/task-dependent |
| bias correction | ~0 | ~0 | neutral for cosine-kNN |

(Flowers-102 high-res was saturated: W4A4 baseline already = BF16, no gap for any technique to close.)

## Findings
1. **Four Over Six is the technique to adopt.** It recovers ~93% of the high-res W4A4 gap and ~45% of the low-res gap, is **free** (weights-only, offline, no calibration data, ~2× one-shot quant compute), **format-preserving**, and **never worse per-block** (MSE-selected). It is the only lever that helped on BOTH tasks.
2. **GPTQ reduces weight error (cosine ↑ everywhere) but its k-NN benefit does not transfer to high-res** — like bias correction, the gain it produces is partly along directions the L2-normalized cosine k-NN metric discards. Keep as an optional lever (it may help linear-probe / logit tasks more than cosine retrieval).
3. **RegCache (token deletion) is task-dependent and can hurt** (helped CIFAR +1.1, hurt Food-101 −0.5): deleting high-norm tokens removes signal on some image distributions. Keep optional, off by default; its KV-prefix half is already disabled for DINOv2 (massive-activation, not sink-driven).
4. **bias correction is neutral for cosine k-NN** (mean shift annihilated by L2-norm); near-free, keep as a final cleanup for non-retrieval uses.

## Recommendation
- **Default weight quantization → Four Over Six** (`w_block_select='mse'` in `quantize_model` / `QuantLinear.from_linear`). It is currently opt-in (default `'six'` for backward-compat); flipping the PTQ-frontend default to `'mse'` is the suggested next change (strictly ≥ 'six' per-block, negligible one-shot cost).
- GPTQ / RegCache / bias remain available, composable options for tasks where they help.

## Deliverables
- `nvfp4/quant.py` (`block_select='mse'`), `ptq/gptq.py` + `ptq/convert_gptq.py`, `ptq/bias_correction.py`, `ptq/regcache.py`; threaded `w_block_select` through `qlinear`/`convert`.
- Tests: `test_four_over_six`, `test_gptq_mse`, `test_bias_correction`, `test_regcache` (50 total pass).
- Comparison harness: `examples/eval_techniques.py`.
