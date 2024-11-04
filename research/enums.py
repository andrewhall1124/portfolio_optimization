from enum import Enum


class Optimizer(Enum):
    MVO = "mvo"
    CEIL = "ceil"
    QP = "qp"


class Rounding(Enum):
    CEIL = "ceil"
    FLOOR = "floor"
    MID = "mid"
