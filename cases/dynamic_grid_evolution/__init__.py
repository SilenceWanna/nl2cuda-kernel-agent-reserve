"""Dynamic grid evolution case definition."""

import os

from framework.case import Case
from cases.dynamic_grid_evolution import config
from cases.dynamic_grid_evolution.reference import make_inputs, reference_forward


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


CASE = Case(
    name="dynamic_grid_evolution",
    description=_load_description(),
    params={"k": config.K},
    grad_inputs=["E"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
