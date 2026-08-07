"""Autograd wrapper for the Warp-local sparse CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _module():
    return load_kernel(
        "warp_sparse_contrast_cuda",
        ["warp_sparse_contrast.cu"],
        base_dir=_KERNEL_DIR,
    )


class WarpSparseContrastFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t_values, slot_to_nnz, weights, edge_mask):
        r_values = _module().forward(
            t_values, slot_to_nnz, weights, edge_mask
        )
        ctx.save_for_backward(
            t_values, slot_to_nnz, weights, edge_mask, r_values
        )
        return r_values

    @staticmethod
    def backward(ctx, grad_r_values):
        t_values, slot_to_nnz, weights, edge_mask, r_values = ctx.saved_tensors
        grad_t_values, grad_weights = _module().backward(
            t_values,
            slot_to_nnz,
            weights,
            edge_mask,
            r_values,
            grad_r_values.contiguous(),
        )
        return grad_t_values, None, grad_weights, None


def candidate(inputs, params):
    return WarpSparseContrastFunction.apply(
        inputs["T_values"],
        inputs["slot_to_nnz"],
        inputs["W"],
        inputs["edge_mask"],
    )


def forward_only(inputs, params):
    return _module().forward(
        inputs["T_values"],
        inputs["slot_to_nnz"],
        inputs["W"],
        inputs["edge_mask"],
    )
