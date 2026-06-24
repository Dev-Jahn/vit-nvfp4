# SP-A wiring — FP8/NVFP4 attention integrated into HF ViT models

**Date:** 2026-06-24 · sm_120 · `quant_sdpa` (SP-A) wired into real HuggingFace models via the transformers attention-interface registry.

SP-A built and measured `quant_sdpa` (emulated FP8/NVFP4 attention) but left it unwired. This lands the integration: `ptq/qattention.py` routes selected (middle-block) attention modules through `quant_sdpa` while everything else falls back to exact SDPA.

## API
- `enable_quant_attention(model, should_quantize_attn, qk='fp8', pv='fp8') -> n` — registers a custom impl in `ALL_ATTENTION_FUNCTIONS`, flags the matched attention modules, and points each attention module's `config._attn_implementation` at it. Flagged modules run `quant_sdpa`; the rest run standard SDPA. Orthogonal to `QuantLinear` weight quant.
- `vit_attn_policy(num_layers, skip_first=2, skip_last=2, container=None)` — `(name, module) -> bool` selecting the interface-calling attention module (`*Attention` class + a `scaling` attr) of each middle block; mirrors `vit_block_policy`.

How it works: transformers 5.x attention modules look up `ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, ...)` **per forward**, so the impl can be switched after load. The registered fn matches the `(module, q, k, v, attention_mask, dropout, scaling, is_causal, **kwargs) -> (out.transpose(1,2), None)` contract (q/k/v are `(B,H,S,D)`). Per-module gating is via a flag the fn checks (transformers offers only global impl selection), so only middle blocks quantize. **For a VLM, call `enable_quant_attention` on the vision submodule** (e.g. `model.model.visual`) so the LLM's attention is untouched.

Default is **FP8** (`qk='fp8', pv='fp8'`) per the precision policy and SP-A (FP8 attention was effectively lossless; NVFP4 is a real step down). `qk`/`pv` can be set to `'nvfp4'` per stage.

## Validation (end-to-end, real models, the right metric per model)

**DINOv2-base — frozen-feature k-NN top-1 on Oxford Flowers-102** (BF16 ref = 0.9931):
| config | k-NN top-1 | drop | feat cos |
|---|---|---|---|
| W4A4 linears | 0.9931 | +0.000 | 0.9827 |
| **FP8 attention** | **0.9931** | **+0.000** | **0.9996** |
| W4A4 linears + FP8 attn | 0.9931 | +0.000 | 0.9825 |

**Qwen3-VL-2B — response-level** (vision-only quant, LLM bf16; 4 images, K=48, teacher-forced KL/top-1 on the BF16 greedy response):
| config | resp KL (nats) | resp top-1 |
|---|---|---|
| W4A4 linears | 0.0231 | 0.969 |
| W4A4 linears + FP8 attn | 0.0234 | 0.969 |

## Verdict
FP8 attention is **quality-free**: alone it's lossless (DINOv2 feat cos 0.9996, k-NN parity), and stacked on W4A4 Linears it leaves the response-level KL/top-1 unchanged (0.023 nats / 0.969). The wiring is clean (registry + per-module flag, exact-SDPA fallback elsewhere) and composes with `QuantLinear`. 73 tests pass. Note: this is an **emulated** FP8 path (quantize→dequantize→matmul) measuring the accuracy floor; a fused FP8/FP4 flash kernel for *speed* on sm_120 remains a custom-CUDA follow-up.

## Usage
```python
import torch
from transformers import AutoModel
from vit_nvfp4.ptq import (quantize_model, vit_block_policy, calibrate_activations,
                           enable_quant_attention, vit_attn_policy, model_spec)

m = AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.bfloat16).cuda().eval()
spec = model_spec(m)
quantize_model(m, vit_block_policy(spec.num_layers, 2, 2, spec.block_container))          # W4A4 Linears
enable_quant_attention(m, vit_attn_policy(spec.num_layers, 2, 2, spec.block_container))    # FP8 attention
calibrate_activations(m, calib, "max")   # for a VLM: enable_quant_attention(m.model.visual, ...)
```

## Deliverables
- `src/vit_nvfp4/ptq/qattention.py` (new): `enable_quant_attention`, `vit_attn_policy`, the registered adapter.
- `src/vit_nvfp4/ptq/__init__.py`: exports.
- `tests/test_qattention.py` (new): adapter shape/contract + FP8-vs-SDPA fidelity + exact-fallback + policy selection (73 total pass).
