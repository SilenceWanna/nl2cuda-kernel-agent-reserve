"""Autograd wrapper for the custom CUDA dynamic grid evolution kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _forward_module():
    return load_kernel(
        "dynamic_grid_evolution_forward",
        ["dynamic_grid_evolution_forward.cu"],
        base_dir=_KERNEL_DIR,
    )


@functools.lru_cache(maxsize=1)
def _backward_module():
    return load_kernel(
        "dynamic_grid_evolution_backward",
        ["dynamic_grid_evolution_backward.cu"],
        base_dir=_KERNEL_DIR,
    )


class DynamicGridEvolutionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, k):
        output = _forward_module().dynamic_grid_evolution_forward(E, float(k))
        ctx.save_for_backward(E, output)
        ctx.k = float(k)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        E, output = ctx.saved_tensors
        grad_E = _backward_module().dynamic_grid_evolution_backward(
            E,
            output,
            grad_output.contiguous(),
            ctx.k,
        )
        return grad_E, None


def candidate(inputs, params):
    return DynamicGridEvolutionFunction.apply(inputs["E"], params["k"])


def forward_only(inputs, params):
    return _forward_module().dynamic_grid_evolution_forward(
        inputs["E"], float(params["k"])
    )
