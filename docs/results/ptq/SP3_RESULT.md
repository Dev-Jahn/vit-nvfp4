# SP3 Result — Activation Calibration (static global scale)

**Date:** 2026-06-23 · DINOv2-base, sm_120, torch 2.12.1+cu130

## What shipped
- `QuantLinear.set_activation_scale()` + `x_global_scale` buffer → static (calibrated) or dynamic (per-input) activation global scale.
- `ptq/calibrate.py`: `calibrate_activations(model, batches, method)` — forward-pre-hook observers collect per-tensor activation stats, pin a static global scale on each QuantLinear. Methods: `max` (default), `percentile`.

## Key finding (non-obvious): for NVFP4, calibrate the global with **max**, not percentile

DINOv2-base (skip 2/2), last_hidden cosine vs BF16 (random eval input):

| activation global scale | cosine |
|---|---|
| dynamic (per-input amax) | 0.947 |
| static **max** (calibrated) | 0.945 |
| static percentile 99.9 | **0.660** |
| static percentile 99.0 | **0.408** |

**Percentile calibration of the global scale is harmful on NVFP4.** Mechanism: NVFP4's per-16 E4M3 block scales already localize range; the per-tensor global's only role is to keep those block scales within E4M3 (≤448). Setting global below the true amax (percentile) forces high-magnitude blocks' E4M3 scale to saturate at 448, which **clips** those blocks (representable max = percentile < block amax). The classic "use percentile for activations" advice applies to *per-tensor* quant where the scale directly scales elements — it does **not** transfer to NVFP4's two-level scheme.

→ Default method set to `max`. Percentile retained for future experiments with genuine per-token outliers on real images.

## Implication
- Calibration gives a **static** scale at **parity** with dynamic (~0.95) — a deployment win (no per-input scale computation, outlier-image robust) but **does not** move the 0.95→0.99 fidelity gap.
- The real fidelity lever is therefore **SP4 (mixed-precision promotion)** — promote the highest-error Linears (per `block_output_cosines`) back to BF16 to hit a target cosine at minimum FP4 budget — and/or per-block techniques (adaptive M=4/6 scaling, "Four Over Six").

## Caveat
Calibration here uses random images (no real per-token outlier structure). Whether percentile/per-token-aware calibration helps on real images (CLS/background high-norm tokens, per RegCache) needs the SP4 real-data eval harness; the machinery supports it.

## Tests
- `_ActObserver` percentile clips outliers (unit).
- `calibrate_activations` pins scales on all QuantLinears.
- DINOv2 max-calibration keeps cosine ≥ 0.93 (parity with dynamic).
- Full suite: 40 passed.
