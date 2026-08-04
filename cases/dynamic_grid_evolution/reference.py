"""Vectorized PyTorch reference for dynamic grid evolution."""

import torch

from cases.dynamic_grid_evolution import config


def reference_forward(inputs, params):
    """Apply the confirmed 3x3 zero-padded nonlinear grid update."""
    E = inputs["E"]
    k = params["k"]
    H, W = E.shape

    zero_row = E.new_zeros((1, W))
    padded_rows = torch.cat((zero_row, E, zero_row), dim=0)
    zero_col = E.new_zeros((H + 2, 1))
    padded = torch.cat((zero_col, padded_rows, zero_col), dim=1)

    neighbors = torch.stack(
        (
            padded[0:H, 0:W],
            padded[0:H, 1 : W + 1],
            padded[0:H, 2 : W + 2],
            padded[1 : H + 1, 0:W],
            padded[1 : H + 1, 1 : W + 1],
            padded[1 : H + 1, 2 : W + 2],
            padded[2 : H + 2, 0:W],
            padded[2 : H + 2, 1 : W + 1],
            padded[2 : H + 2, 2 : W + 2],
        ),
        dim=0,
    )

    weighted_sum = (torch.sigmoid(k * neighbors) * neighbors).sum(dim=0)
    return torch.sigmoid(weighted_sum)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    E = torch.randn(
        (config.H, config.W),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if requires_grad:
        E.requires_grad_(True)
    return {"E": E}
