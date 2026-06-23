import torch
from vit_nvfp4.nvfp4 import format as fmt


def test_grid_values_roundtrip():
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6],
                        dtype=torch.float32)
    code = fmt.round_to_e2m1_code(grid)
    val = fmt.code_to_value(code)
    assert torch.allclose(val, grid), (val, grid)


def test_nearest_rounding():
    x = torch.tensor([0.6, 0.8, 1.2, 5.5, 7.0, -0.6], dtype=torch.float32)
    val = fmt.code_to_value(fmt.round_to_e2m1_code(x))
    # 0.6->0.5, 0.8->1.0, 1.2->1.0(|1.0-1.2|<|1.5-1.2|), 5.5->6, 7->6(clamp), -0.6->-0.5
    assert torch.allclose(val, torch.tensor([0.5, 1.0, 1.0, 6.0, 6.0, -0.5]))
