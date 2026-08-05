import functools
import os

import torch

from framework.loader import load_kernel


_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _module():
    return load_kernel("integral_image", ["integral_image.cu"], base_dir=_KDIR)


class IntegralImageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return _module().integral_image_forward(X)

    @staticmethod
    def backward(ctx, G):
        return _module().integral_image_backward(G.contiguous())


def candidate(inputs, params):
    return IntegralImageFunction.apply(inputs["X"])


def forward_only(inputs, params):
    return _module().integral_image_forward(inputs["X"])
