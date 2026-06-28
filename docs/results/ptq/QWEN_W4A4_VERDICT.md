# Qwen3-VL ViT W4A4 — verdict: the "0.881 = ceiling, needs mixed precision" was a metric artifact

**Date:** 2026-06-24 · Qwen/Qwen3-VL-2B-Instruct vision tower, sm_120, skip(2,2), W4A4 on vision **blocks** only (merger + LLM stay BF16).

SP2-expansion reported Qwen3-VL ViT W4A4 at **0.881 last_hidden-state flattened cosine** and called it "the W4A4-sensitive outlier, prime customer for a mixed-precision driver." A review challenged that on three grounds — all verified, and the conclusion reversed.

## Three issues found (all confirmed in code)
1. **Four Over Six global-scale saturation bug.** `quantize_to_nvfp4` used `global = amax/(6·448)` even for `block_select='mse'`. The scale-to-4 candidate (`amax_b/4` = 1.5× the scale-to-6 one) then saturates E4M3 at 448 on every block with `amax_b > ⅔·amax` and silently collapses back to scale-to-6 — neutering FoS on the high-magnitude blocks. **Fixed** (commit `792094e`): `'mse'` now uses `amax/(6·256)` so the max block's candidates (256, 384) stay E4M3-representable (arXiv:2512.02010).
2. **GPTQ was M6-only.** The GPTQ path froze block scale = `amax_b/6` (RTN) and global = `amax/(6·448)`, optimizing only E2M1 codes — so the earlier "GPTQ+bias → 0.921" was M6-GPTQ, not FoS-GPTQ.
3. **The metric was wrong.** `last_hidden_state` flattened cosine is upstream of a LayerNorm in the vision **merger**; comparing it to DINO/SigLIP's ~0.98 is apples-to-oranges.

## Error decomposition (single image; flattened cosine + per-token + norm error)
| config | flat_cos | tok_p1 | norm_err |
|---|---|---|---|
| W4A16 six | 0.9387 | 0.7828 | 0.473 |
| **W4A16 mse (256 fix)** | **0.9455** | 0.7949 | 0.438 |
| W4A16 mse (448 bug) | 0.9376 | 0.7723 | 0.442 |
| W16A4 | 0.9255 | 0.7142 | 0.568 |
| W4A4 mse (256 fix) | 0.8831 | 0.5898 | 0.807 |

- **Activation-dominated**, not weight: W16A4 (0.926) < W4A16 (0.946). Weight-side levers (GPTQ, weight reconstruction) have a low ceiling here.
- **The FoS fix works**: buggy `mse(448)` (0.9376) was *below* `six` (0.9387); fixed `mse(256)` (0.9455) is clearly best. The bug had made FoS net-negative on Qwen weights.
- **`norm_err` is huge on `last_hidden` (0.4–0.8)** while cosine looks moderate — cosine hides severe magnitude distortion. But the merger LayerNorm normalizes magnitude, so this may not propagate.

## The decision metric: response-level (the merger output + the LLM see the vision tower, not us)
Greedy-generate K=48 response tokens with BF16 (reference); per config report greedy token-match, exact-response count, and teacher-forced KL / top-1 at the response positions. **6 Oxford-Flowers images, "Describe this image in detail."**

| config | tok_match | exact_resp | resp_KL (nats) | resp_top1 |
|---|---|---|---|---|
| W4A4 six \| tensor | 0.497 | 1/6 | 0.0250 | 0.951 |
| **W4A4 mse \| tensor** | **0.785** | **2/6** | **0.0239** | **0.962** |

Single-image detail: W4A4 mse reproduced the BF16 response **token-for-token** (tok_match 1.000); BF16 and W4A4 both: *"This is a close-up photograph of several pink, lily-like flowers, likely a species of Lilium…"*.

## Verdict
- **Qwen3-VL W4A4 (vision blocks) is near-lossless at the MLLM output**: response-position KL ≈ **0.024 nats**, top-1 ≈ **0.96**, coherent BF16-matching descriptions. The **0.881 last_hidden cosine was a metric artifact** — the merger's LayerNorm + the LLM's robustness absorb the vision-tower magnitude distortion.
- **Mixed precision is NOT needed for Qwen3-VL.** Dropped. The corrected `W4A4 mse|tensor` (the default shipping path) is the simplest and best.
- **Corrected Four Over Six is the one real win** — it helps both weights (W4A16 0.9376→0.9455) and the response level (tok_match 0.497→0.785, KL 0.0250→0.0239, top-1 0.951→0.962) over `six`, with **no regression on the validated case** (DINOv2 Flowers-102 k-NN top-1 = 0.9931 = BF16, lossless, post-fix). Kept as the default.
- **Activation-side tweaks (per-row/tile global, activation FoS) are NOT adopted**: per-row global alone didn't help; activation FoS improved *last_hidden* cosine (0.883→0.906) but did **not** help — and in one case **diverged** the greedy response. A textbook case of optimizing the wrong metric actively hurting the real objective.

## Lesson
For VLM vision towers, evaluate PTQ at the **response / logit level** (teacher-forced KL + greedy token match on generated text), not raw hidden-state cosine. The hidden state passes through a normalization + a merger + a full LLM before anything observable; cosine/MSE on it can be off by a wide margin in both directions. tok_match (free-running greedy) is sensitive to early-token flips and overstates divergence; the robust signals are response-position **KL** and **top-1 agreement**.
