import torch

from cases.integral_image import config


def reference_forward(inputs, params):
    X = inputs["X"]
    return torch.cumsum(torch.cumsum(X, dim=1), dim=0)


def make_inputs(seed, dtype, device, requires_grad=False):
    generator = torch.Generator(device=device).manual_seed(seed)
    X = torch.randn(
        config.H,
        config.W,
        dtype=dtype,
        device=device,
        generator=generator,
    ).contiguous()
    if requires_grad:
        X.requires_grad_(True)
    return {"X": X}
