from enum import Enum


class Optimizer(Enum):
    SLSQP = "slsqp"
    QP = "qp"
    TWO_STAGE = "two_stage"
    MIQP = "miqp"
    GA = "ga"
    ITER_QP = "iter_qp"
    ITER_MIQP = "iter_miqp"


class Rounding(Enum):
    CEIL = "ceil"
    FLOOR = "floor"
    MID = "mid"


class ChartType(Enum):
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
