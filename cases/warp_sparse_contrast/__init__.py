"""Case registration for Warp-local sparse weighted contrast."""

import os

from framework.case import Case
from cases.warp_sparse_contrast import config
from cases.warp_sparse_contrast.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


CASE = Case(
    name="warp_sparse_contrast",
    description=_load_description(),
    params={},
    grad_inputs=["T_values", "W"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
