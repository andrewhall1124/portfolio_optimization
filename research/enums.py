from enum import Enum

class Optimizer(Enum):
    MVO = "mvo"
    QP = "qp"
    MIQP = "miqp"
    GA = "ga"

class Rounding(Enum):
    CEIL = "ceil"
    FLOOR = "floor"
    MID = "mid"
