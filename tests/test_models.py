import os

import pytest
import torch
import torch.nn as nn

os.environ.setdefault("HF_HOME", "/var/cache/huggingface")

from vit_nvfp4.ptq.models import find_block_container, model_spec, quantize_vit
from vit_nvfp4.ptq.policy import vit_block_policy
from vit_nvfp4.ptq.qlinear import QuantLinear


# --- synthetic modules mirroring each architecture's container + naming -------

def _block(linears):
    b = nn.Module()
    for path, (i, o) in linears.items():
        parent, _, leaf = path.rpartition(".")
        p = b
        for seg in parent.split(".") if parent else []:
            if not hasattr(p, seg):
                setattr(p, seg, nn.Module())
            p = getattr(p, seg)
        setattr(p, leaf, nn.Linear(i, o))
    return b


_ATTN_MLP = {"attention.q_proj": (32, 32), "attention.k_proj": (32, 32),
             "attention.v_proj": (32, 32), "attention.o_proj": (32, 32),
             "mlp.fc1": (32, 64), "mlp.fc2": (64, 32)}


def _dinov3_like():  # block stack at model.layer
    m = nn.Module()
    m.model = nn.Module()
    m.model.layer = nn.ModuleList([_block(_ATTN_MLP) for _ in range(6)])
    return m


def _vjepa_like():  # encoder.layer (real) + predictor.layer (auxiliary, must be excluded)
    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layer = nn.ModuleList([_block({"mlp.fc1": (32, 64), "mlp.fc2": (64, 32)}) for _ in range(4)])
    m.predictor = nn.Module()
    m.predictor.layer = nn.ModuleList([_block({"mlp.fc1": (32, 64), "mlp.fc2": (64, 32)}) for _ in range(2)])
    return m


def _qwen_like():  # blocks (real, fused qkv) + deepstack_merger_list (auxiliary)
    m = nn.Module()
    m.blocks = nn.ModuleList([_block({"attn.qkv": (32, 96), "attn.proj": (32, 32),
                                      "mlp.linear_fc1": (32, 64), "mlp.linear_fc2": (64, 32)})
                              for _ in range(5)])
    m.deepstack_merger_list = nn.ModuleList([_block({"fc": (32, 32)}) for _ in range(3)])
    return m


def test_find_block_container_picks_longest_linear_stack():
    assert find_block_container(_dinov3_like()) == "model.layer"
    assert find_block_container(_vjepa_like()) == "encoder.layer"      # 4 > predictor's 2
    assert find_block_container(_qwen_like()) == "blocks"              # 5 > merger's 3


def test_container_scoped_policy_excludes_aux_stack():
    # V-JEPA predictor.layer.N matches the block regex but must NOT be quantized.
    pol = vit_block_policy(4, skip_first=1, skip_last=1, container="encoder.layer")
    lin = nn.Linear(32, 32)
    assert pol("encoder.layer.1.mlp.fc1", lin)
    assert pol("encoder.layer.2.mlp.fc1", lin)
    assert not pol("encoder.layer.0.mlp.fc1", lin)        # skipped first
    assert not pol("encoder.layer.3.mlp.fc1", lin)        # skipped last
    assert not pol("predictor.layer.1.mlp.fc1", lin)      # wrong stack


def test_container_scope_matches_with_parent_prefix():
    # When quantizing from a parent module the path is e.g. "visual.blocks.2.attn.qkv".
    pol = vit_block_policy(5, skip_first=1, skip_last=1, container="blocks")
    assert pol("visual.blocks.2.attn.qkv", nn.Linear(32, 96))
    assert pol("blocks.2.attn.qkv", nn.Linear(32, 96))


def test_quantize_vit_excludes_predictor_and_swaps_middle():
    m = _vjepa_like()
    n, spec = quantize_vit(m, skip_first=1, skip_last=1)
    assert spec.block_container == "encoder.layer" and spec.num_layers == 4
    assert n == 4  # 2 middle encoder blocks x 2 Linears
    assert isinstance(m.encoder.layer[1].mlp.fc1, QuantLinear)
    assert isinstance(m.encoder.layer[0].mlp.fc1, nn.Linear)   # skipped
    assert isinstance(m.predictor.layer[1].mlp.fc1, nn.Linear)  # never touched


def test_quantize_vit_swaps_fused_qkv():
    m = _qwen_like()
    n, spec = quantize_vit(m, skip_first=1, skip_last=1)
    assert n == 12  # 3 middle blocks x 4 Linears (fused qkv counts as one)
    assert spec.quirks["fused_qkv"] is True
    assert isinstance(m.blocks[2].attn.qkv, QuantLinear)        # fused qkv quantized
    assert isinstance(m.deepstack_merger_list[0].fc, nn.Linear)  # aux stack untouched


def test_model_spec_quirks_and_feature():
    s3 = model_spec(_dinov3_like())
    assert s3.quirks["out_proj"] == "o_proj"
    sq = model_spec(_qwen_like())
    assert sq.feature == "mean" and sq.quirks["fused_qkv"]


# --- gated structural checks on the real cached models (meta device, no GPU) --

def _meta_submodule(name, sub=None, trust=False):
    try:
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust)
        with torch.device("meta"):
            model = AutoModel.from_config(cfg, trust_remote_code=trust)
    except Exception as e:  # offline / arch unsupported in this transformers build
        pytest.skip(f"{name} unavailable: {type(e).__name__}")
    if sub:
        for p in sub.split("."):
            model = getattr(model, p)
    return model


@pytest.mark.parametrize("name,sub,trust,container,layers,fused", [
    ("facebook/dinov2-base", None, False, "encoder.layer", 12, False),
    ("facebook/dinov3-vitb16-pretrain-lvd1689m", None, False, "model.layer", 12, False),
    ("google/paligemma-3b-pt-224", "vision_tower", False, "encoder.layers", 27, False),
    ("Qwen/Qwen3-VL-2B-Instruct", "visual", False, "blocks", 24, True),
    ("Dev-Jahn/vjepa2.1-vitl-fpc64-384", None, True, "encoder.layer", 24, False),
])
def test_real_model_spec(name, sub, trust, container, layers, fused):
    model = _meta_submodule(name, sub, trust)
    spec = model_spec(model)
    assert spec.block_container == container, (name, spec.block_container)
    assert spec.num_layers == layers
    assert spec.quirks["fused_qkv"] == fused
