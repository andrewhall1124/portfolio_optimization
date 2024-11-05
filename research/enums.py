from enum import Enum

class Optimizer(Enum):
    MVO = "mvo"
    QP = "qp"
    TWO_STAGE = "two_stage"
    MIQP = "miqp"
    GA = "ga"

class Rounding(Enum):
    CEIL = "ceil"
    FLOOR = "floor"
    MID = "mid"
