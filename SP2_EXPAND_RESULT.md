# SP2 Expansion Result — model-aware W4A4 PTQ beyond DINOv2

**Date:** 2026-06-23 · sm_120 · torch 2.12.1+cu130 / transformers 5.12.1 · all numbers W4A4 NVFP4 (weights Four Over Six `mse`, activations `six`, static max calibration).

Generalized the PTQ frontend so one code path quantizes any supported ViT. A small registry (`ptq/models.py`) derives the per-model knobs from the module tree; `vit_block_policy` gained a `container` scope; `block_output_cosines` auto-detects its block container. Validated on 5 architectures from the local HF cache.

## Per-model config (auto-derived) + validation

| model | arch | block_container | layers | block Linears | quirks detected | swaps (skip 2,2) |
|---|---|---|---|---|---|---|
| DINOv2-base | `Dinov2Model` | `encoder.layer` | 12 | q/k/v, output.dense, fc1/2 | layerscale, cls | 48 = 8×6 |
| DINOv3 ViT-B/16 | `DINOv3ViTModel` | **`model.layer`** | 12 | q/k/v/o_proj, up/down_proj | layerscale, cls, eps 1e-5 | 48 = 8×6 |
| SigLIP2-base/16 | `SiglipVisionModel` | `encoder.layers` | 12 | q/k/v/out_proj, fc1/2 | mean-pool (no cls) | 48 = 8×6 |
| Qwen3-VL-2B ViT | `Qwen3VLVisionModel` | `blocks` | 24 | **fused `attn.qkv`**, proj, linear_fc1/2 | fused_qkv, mean | 80 = 20×4 |
| V-JEPA2.1 ViT-L | `VJEPA21Model` | `encoder.layer` | 24 | q/k/v, proj, fc1/2 | rope, mean | 120 = 20×6 |

Swap counts confirm the hard cases: Qwen3-VL's **fused qkv** is swapped (4 Linears/block), and V-JEPA's **`predictor.layer` (12 blocks, 72 Linears) is excluded** by the `container="encoder.layer"` scope (it matches the `.layer.N` regex but is the wrong stack). SigLIP's SO400M-style non-4× MLP and DINOv3's `o_proj`/SwiGLU-naming are handled generically (any `nn.Linear` in a middle block is swapped regardless of name).

## Measured accuracy

**Image encoders — frozen-feature k-NN on Oxford Flowers-102** (gallery 3060 / query 1020, k-NN top-1), env `CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=12.0a uv run python examples/eval_knn_vit.py --model <name>`:

| model | feature cos | k=10 BF16→W4A4 | k=20 BF16→W4A4 | verdict |
|---|---|---|---|---|
| DINOv3 ViT-B/16 | 0.9839 | 0.9941 → 0.9941 (**+0.000**) | 0.9853 → 0.9843 (−0.001) | ~lossless |
| SigLIP2-base/16 (`--submodule vision_model`) | 0.9837 | 0.9608 → 0.9598 (−0.001) | 0.9343 → 0.9255 (−0.009) | ~lossless |

**Special-input encoders — BF16-vs-W4A4 output cosine** (one forward; `examples/quant_vit_smoke.py`):

| model | input | output cosine | note |
|---|---|---|---|
| Qwen3-VL-2B ViT | real flower image | **0.881** | sensitive — see below |
| V-JEPA2.1 ViT-L | random 16-frame clip, fp32 | 0.987 | RoPE path + predictor-exclusion OK |

### Qwen3-VL ViT is the W4A4-sensitive outlier

> **⚠️ CORRECTION (2026-06-24 — see `QWEN_W4A4_VERDICT.md`):** this subsection's premise is a **metric artifact**. The 0.881 is `last_hidden_state` flattened cosine, which sits *upstream* of the vision merger's LayerNorm. At the actual MLLM output — response-level teacher-forced **logit-KL ≈ 0.024 nats, top-1 ≈ 0.96**, and BF16-matching generated descriptions — Qwen3-VL **W4A4 is near-lossless and needs no mixed precision**. The cosine/skip-sweep numbers below stand as measured; their *interpretation* (sensitive outlier / prime mixed-precision target) is **superseded**. Also: the corrected Four Over Six (global 448→256, commit `792094e`) turns FoS from net-negative (0.9376, below `six`) to best (0.9455) on Qwen weights; the earlier "GPTQ+bias→0.921" used the buggy FoS and the wrong metric.

Sweeping skip depth (same image): skip(2,2)=0.881 · (4,4)=0.907 · (6,6)=0.915 · (8,8)=0.938. Monotonic with quantized-block count — distributed error accumulation, not a kernel bug (the GEMM is the SP1-validated path that yields ≥0.98 on every other model here). Unlike the SSL-pretrained encoders (DINOv2/v3, SigLIP), Qwen3-VL's vision tower does **not** reach ~0.98 even at skip(8,8). **This is the prime customer for the planned per-Linear mixed-precision driver** (promote high-error Linears to BF16 via `block_output_cosines`) or an FP8 vision-encoder fallback. A true downstream-VQA eval (not done here — too heavy) is needed to judge task impact.

**Remaining calibration-only levers measured (skip 2,2, 8-image calib, same eval image).** The 0.881 baseline applied only the *defaults* (Four Over Six weights + max activation calibration); the opt-in SP5 levers were not. Adding them:

| config | output cosine | Δ |
|---|---|---|
| Four Over Six + calib (baseline) | 0.890 | — |
| + bias correction | 0.894 | +0.004 |
| **GPTQ** + calib | 0.915 | +0.025 |
| **GPTQ + bias** + calib | **0.921** | **+0.031** |

GPTQ does the heavy lifting here — and this is exactly the SP5 nuance confirmed: GPTQ raises *raw weight/output cosine* but its gain didn't transfer to the L2-normalized k-NN top-1 of the SSL encoders; Qwen3-VL ViT is evaluated by raw cosine, so it *does* benefit. But **0.921 is the weights-only ceiling** — still short of the ~0.98 the SSL encoders hit, so the structural gap (24-layer accumulation, VLM-tower statistics, fused-qkv dynamic range) still needs the **mixed-precision driver**, not more calibration tricks.

## Blockers / notes
- **V-JEPA2.1 BF16 forward is broken upstream**, independent of quantization: its custom RoPE attention upcasts q,k to fp32 while v stays bf16, and `scaled_dot_product_attention` rejects the dtype mismatch. Workaround: run the model in **fp32** (QuantLinear follows `out_dtype=x.dtype`, so the W4A4 path runs in fp32 too). It is a 64-frame **video** model (`fpc64`); we used a 16-frame random clip for a relative fidelity check, so 0.987 is a sanity number, not a benchmark.
- **SigLIP2**: validated the real target `google/siglip2-base-patch16-224` (downloaded). Its `Siglip2Model.vision_model` resolves to a `SiglipVisionModel` whose attention-pool `pooler_output` is the retrieval feature.
- **Qwen3-VL** vision encoder takes patchified `(hidden_states, grid_thw)`, not `pixel_values`; driven via `proc.image_processor(...)`. The `deepstack_merger_list` ModuleList is correctly ignored (shorter than `blocks`, and its path doesn't match the block regex).

## Deliverables
- `src/vit_nvfp4/ptq/models.py` (new): `ModelSpec`, `find_block_container`, `model_spec`, `quantize_vit`.
- `src/vit_nvfp4/ptq/policy.py`: `vit_block_policy(..., container=None)` scope.
- `src/vit_nvfp4/ptq/diagnostics.py`: `block_output_cosines(block_container=None)` auto-detect.
- `src/vit_nvfp4/ptq/__init__.py`: export the registry symbols.
- `examples/eval_knn_vit.py` (new, generic image-ViT k-NN), `examples/quant_vit_smoke.py` (new, Qwen3-VL + V-JEPA structural/fidelity).
- `tests/test_models.py` (new): 6 synthetic (container detection, predictor/merger exclusion, fused-qkv swap, quirks) + 5 gated meta-device structural checks on the real cached models. **All 16 of {test_models, test_convert, test_dinov2_ptq} pass.**
