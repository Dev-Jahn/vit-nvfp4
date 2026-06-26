from .qlinear import QuantLinear
from .policy import vit_block_policy, block_index
from .qattention import enable_quant_attention, vit_attn_policy
from .sa3_attention import enable_sa3_attention, sa3_attn_policy
try:
    from .fused_mlp import FusedNVFP4MLP, fuse_mlps   # fast path: needs triton (ships with cu130 torch)
    _HAS_FUSED_MLP = True
except ImportError:                                   # eager QuantLinear path stays usable without triton
    _HAS_FUSED_MLP = False
from .convert import quantize_model
from .convert_gptq import quantize_model_gptq
from .models import ModelSpec, model_spec, find_block_container, quantize_vit
from .diagnostics import tensor_cosine, block_output_cosines
from .calibrate import calibrate_activations
from .bias_correction import correct_bias
from .regcache import RegCache, curate_register

__all__ = [
    "QuantLinear",
    "vit_block_policy", "block_index",
    "enable_quant_attention", "vit_attn_policy",
    "enable_sa3_attention", "sa3_attn_policy",
    "quantize_model", "quantize_model_gptq",
    "ModelSpec", "model_spec", "find_block_container", "quantize_vit",
    "tensor_cosine", "block_output_cosines",
    "calibrate_activations",
    "correct_bias",
    "RegCache", "curate_register",
]

if _HAS_FUSED_MLP:
    __all__ += ["FusedNVFP4MLP", "fuse_mlps"]
