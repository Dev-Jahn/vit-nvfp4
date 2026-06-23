import torch.nn as nn

from vit_nvfp4.ptq.policy import vit_block_policy, block_index
from vit_nvfp4.ptq.convert import quantize_model
from vit_nvfp4.ptq.qlinear import QuantLinear


def _net(n_layers):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.fc1 = nn.Linear(32, 64)
            self.mlp.fc2 = nn.Linear(64, 32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList([Block() for _ in range(n_layers)])
            self.head = nn.Linear(32, 10)

    return Net()


def test_block_index_parsing():
    assert block_index("encoder.layer.7.mlp.fc1") == 7
    assert block_index("encoder.layer.0.attention.attention.query") == 0
    assert block_index("head") is None
    assert block_index("embeddings.patch_embeddings.projection") is None


def test_policy_skips_first_and_last_blocks():
    should = vit_block_policy(num_layers=12, skip_first=2, skip_last=2)
    lin = nn.Linear(8, 8)
    assert not should("encoder.layer.0.mlp.fc1", lin)   # first
    assert not should("encoder.layer.1.mlp.fc1", lin)
    assert should("encoder.layer.2.mlp.fc1", lin)        # middle
    assert should("encoder.layer.9.mlp.fc1", lin)
    assert not should("encoder.layer.10.mlp.fc1", lin)   # last 2
    assert not should("encoder.layer.11.mlp.fc1", lin)
    assert not should("head", lin)                        # no block index


def test_quantize_model_swaps_only_middle_blocks():
    net = _net(6)  # skip 2/2 -> quantize layers 2,3 (2 Linears each) = 4
    n = quantize_model(net, vit_block_policy(num_layers=6, skip_first=2, skip_last=2))
    assert n == 4
    assert isinstance(net.encoder.layer[2].mlp.fc1, QuantLinear)
    assert isinstance(net.encoder.layer[3].mlp.fc2, QuantLinear)
    assert isinstance(net.encoder.layer[0].mlp.fc1, nn.Linear)   # first block untouched
    assert isinstance(net.encoder.layer[5].mlp.fc2, nn.Linear)   # last block untouched
    assert isinstance(net.head, nn.Linear)
