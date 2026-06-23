# SP2 Result — NVFP4 PTQ Frontend (DINOv2 vertical slice)

**Date:** 2026-06-23 · **GPU:** RTX PRO 6000 (sm_120) · torch 2.12.1+cu130

## What shipped
- `QuantLinear` — static W4A4 NVFP4 weight, online activation quant via SP1's `nvfp4_linear`.
- `vit_block_policy` — quantizes nn.Linear in middle blocks only (BF16 for first/last N blocks, norms, heads, embeddings; attention QK·softmax·AV are SDPA → BF16 automatically).
- `quantize_model` — in-place `nn.Linear → QuantLinear` swap by predicate.
- `diagnostics` — `tensor_cosine`, `block_output_cosines`.
- 36 tests pass (28 SP1 + 8 SP2).

## DINOv2-base W4A4 PTQ-only fidelity (random input, **no calibration**, dynamic activation global scale)

| skip (first,last) | quantized Linears | last_hidden cosine | CLS cosine |
|---|---|---|---|
| (0,0) all 12 blocks | 72 | 0.904 | 0.873 |
| (1,1) | 60 | 0.929 | 0.913 |
| **(2,2)** (default) | **48** | **0.951** | 0.941 |
| (4,4) | 24 | 0.980 | 0.978 |

Per-block output cosine (skip 2,2): `1.00 1.00 0.99 0.98 0.96 0.95 0.94 0.93 0.99 0.99 0.98 0.95` — error accumulates monotonically through quantized middle blocks.

## Reading the result
- **Far from the naive-W4A4 collapse** the literature reports for ViTs (e.g. DINOv2-B 83→0 top-1 with all-layer per-tensor W4A4). Mixed precision (BF16 first/last blocks + BF16 attention scores) plus NVFP4's per-16 block scaling keeps last-hidden cosine at 0.90–0.98.
- **Below the spec's ≥0.99 target.** That target was an optimistic estimate; ~0.95 (skip 2,2) is the honest PTQ-only baseline. Closing the gap is the job of:
  - **SP3 (calibration):** static activation global scale via percentile/MSE (vs current per-input dynamic) to tame CLS/background-token outliers.
  - **SP4 (mixed-precision driver):** promote the most error-contributing Linears (per `block_output_cosines`) back to BF16 to hit a target cosine at minimum FP4 budget.
  - **SP5 (mitigation + light recovery):** Had16 rotation, RegCache-style prefix registers, and calibration-set block reconstruction.
- The test thresholds (skip 2,2 ≥ 0.93; skip 4,4 ≥ 0.97) are **regression guards set below observed**, not the end goal.

## Caveat
Cosine on features ≠ downstream accuracy. The decisive metric (ImageNet linear-probe / k-NN) is deferred to SP4's accuracy harness; this slice validates that the W4A4 PTQ pipeline runs end-to-end on a real model and produces a strong, non-collapsed signal.
