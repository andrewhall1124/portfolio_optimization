from enum import Enum


class Optimizer(Enum):
    SLSQP = "slsqp"
    QP = "qp"
    TWO_STAGE_SLSQP = "two_stage_slsqp"
    TWO_STAGE_QP = "two_stage_qp"
    MIQP = "miqp"
    GA = "ga"


class Rounding(Enum):
    CEIL = "ceil"
    FLOOR = "floor"
    MID = "mid"


class ChartType(Enum):
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
