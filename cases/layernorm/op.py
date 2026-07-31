"""Autograd wrapper for the custom LayerNorm CUDA kernels."""

import functools
import os

import torch

from framework.loader import load_kernel


_KERNEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _module():
    return load_kernel(
        "layernorm_cuda_ext",
        ["layernorm_forward.cu", "layernorm_backward.cu"],
        base_dir=_KERNEL_DIR,
        verbose=True,
    )


class _LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, eps):
        X_c = X.contiguous()
        gamma_c = gamma.contiguous()
        beta_c = beta.contiguous()
        Y, mean, inv_std = _module().layernorm_forward(
            X_c, gamma_c, beta_c, float(eps)
        )
        ctx.save_for_backward(X_c, gamma_c, mean, inv_std)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X, gamma, mean, inv_std = ctx.saved_tensors
        dX, dgamma, dbeta = _module().layernorm_backward(
            grad_Y.contiguous(), X, gamma, mean, inv_std
        )
        return dX, dgamma, dbeta, None


def candidate(inputs, params):
    return _LayerNormFunction.apply(
        inputs["X"], inputs["gamma"], inputs["beta"], float(params["eps"])
    )


def forward_only(inputs, params):
    return candidate(inputs, params)
