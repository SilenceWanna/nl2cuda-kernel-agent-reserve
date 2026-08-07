"""Vectorized PyTorch gold standard for Warp-local sparse aggregation."""

import torch

from cases.warp_sparse_contrast import config


def reference_forward(inputs, params):
    t_values = inputs["T_values"]
    slot_to_nnz = inputs["slot_to_nnz"]
    weights = inputs["W"]
    edge_mask = inputs["edge_mask"]

    valid = slot_to_nnz >= 0
    zero_index = torch.full_like(slot_to_nnz, t_values.numel())
    gather_index = torch.where(valid, slot_to_nnz, zero_index)

    values_with_zero = torch.cat((t_values, t_values.new_zeros(1)))
    t_dense = values_with_zero.index_select(
        0, gather_index.reshape(-1)
    ).reshape(config.GROUPS, config.WARP_SIZE)

    effective_weights = weights * edge_mask.to(weights.dtype)
    summed = (effective_weights * t_dense.unsqueeze(1)).sum(dim=-1)
    r_dense = torch.tanh(summed)

    flat_valid = valid.reshape(-1)
    output_indices = slot_to_nnz.reshape(-1)[flat_valid]
    output_values = r_dense.reshape(-1)[flat_valid]
    return torch.zeros_like(t_values).scatter(
        0, output_indices, output_values
    )


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)

    valid = torch.rand(
        config.GROUPS,
        config.WARP_SIZE,
        device=device,
        generator=generator,
    ) < config.T_DENSITY
    nnz = int(valid.sum().item())

    slot_to_nnz = torch.full(
        (config.GROUPS, config.WARP_SIZE),
        -1,
        dtype=torch.int64,
        device=device,
    )
    slot_to_nnz[valid] = torch.arange(nnz, dtype=torch.int64, device=device)

    t_values = torch.randn(
        nnz, dtype=dtype, device=device, generator=generator
    )
    weights = torch.randn(
        config.GROUPS,
        config.WARP_SIZE,
        config.WARP_SIZE,
        dtype=dtype,
        device=device,
        generator=generator,
    ) / (config.WARP_SIZE ** 0.5)

    edge_mask = torch.rand(
        config.GROUPS,
        config.WARP_SIZE,
        config.WARP_SIZE,
        device=device,
        generator=generator,
    ) < config.EDGE_DENSITY
    edge_mask |= torch.eye(
        config.WARP_SIZE, dtype=torch.bool, device=device
    ).unsqueeze(0)

    t_values.requires_grad_(requires_grad)
    weights.requires_grad_(requires_grad)
    return {
        "T_values": t_values,
        "slot_to_nnz": slot_to_nnz,
        "W": weights,
        "edge_mask": edge_mask,
    }
