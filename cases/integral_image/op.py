"""integral image 候选实现封装（dict 接口，符合 framework 候选契约）。

candidate(inputs, params) -> S，对 inputs["X"] 提供梯度。
前向：2D 前缀和；反向：上游梯度 dS 的 2D 后缀和（数学上 ∂S/∂X 是下三角全 1 的累加）。
kernel 源在 cases/integral_image/kernels/，由 framework.loader 即时编译。float32-only。
"""

import os
import functools

import torch

from framework.loader import load_kernel

_KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")


@functools.lru_cache(maxsize=1)
def _fwd_module():
    return load_kernel("integral_image_forward", ["integral_image_forward.cu"], base_dir=_KDIR)


@functools.lru_cache(maxsize=1)
def _bwd_module():
    return load_kernel("integral_image_backward", ["integral_image_backward.cu"], base_dir=_KDIR)


class IntegralImageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        # 前向不依赖 X 的值做反向（2D 前缀和的雅可比是常量 0/1），无需 save X
        ctx.save_for_backward()
        return _fwd_module().integral_image_forward(X.contiguous())

    @staticmethod
    def backward(ctx, grad_S):
        # dX[p,q] = Σ_{i>=p, j>=q} dS[i,j]，即 dS 的 2D 后缀和
        dX = _bwd_module().integral_image_backward(grad_S.contiguous())
        return dX


def candidate(inputs, params):
    """framework 候选契约：inputs={"X"}, params={} -> S。"""
    return IntegralImageFunction.apply(inputs["X"])


def forward_only(inputs, params):
    """仅前向（no autograd），供前向单独对拍/调试。"""
    return _fwd_module().integral_image_forward(inputs["X"].contiguous())
