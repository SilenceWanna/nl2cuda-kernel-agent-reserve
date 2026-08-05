"""integral image（2D 前缀和）的 PyTorch 参考实现（正确性金标准，dict 接口）。

前向：S[i,j] = Σ_{p<=i, q<=j} X[p,q]，即沿两维依次做前缀和（含边界）。
只用基础张量算子（torch.cumsum，O(HW) 前缀原语），向量化、无 Python for；反向由 autograd 提供。
"""

import torch

from cases.integral_image import config


def reference_forward(inputs, params):
    """inputs={"X":[H,W]}, params={} -> S:[H,W]，含边界的 2D 前缀和。"""
    X = inputs["X"]
    # 沿行方向再沿列方向做前缀和（cumsum 是 O(N) 前缀原语，torch.compile 会用并行 scan）
    return torch.cumsum(torch.cumsum(X, dim=0), dim=1)


def make_inputs(seed, dtype, device, requires_grad=False):
    """按种子生成命名输入 {"X"}。用自然 randn 分布，不挑异常输入。"""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(config.H, config.W, dtype=dtype, device=device, generator=g)
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X}
