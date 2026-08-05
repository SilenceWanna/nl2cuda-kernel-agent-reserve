"""integral image case：暴露 CASE 实例供 framework 使用。"""

import os

from framework.case import Case
from cases.integral_image import config
from cases.integral_image.reference import reference_forward, make_inputs


def _load_description():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "description.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


CASE = Case(
    name="integral_image",
    description=_load_description(),
    params={},
    grad_inputs=["X"],
    dtype=config.DTYPE,
    make_inputs=make_inputs,
    reference_forward=reference_forward,
)
