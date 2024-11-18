from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.enums import Optimizer
from research.interfaces import AssetData

from .miqp import miqp
from .qp import qp
from .slsqp import slsqp
from .two_stage_qp import two_stage_qp
from .two_stage_slsqp import two_stage_slsqp


def optimize(
    optimizer: Optimizer, data: AssetData, weights: np.ndarray, **kwargs: Any
) -> NDArray[np.float64]:

    # Default kwargs
    budget = kwargs.get("budget", None)
    gamma = kwargs.get("gamma", 2)
    scale_weights = kwargs.get("scale_weights", True)

    match optimizer:

        case Optimizer.SLSQP:
            return slsqp(data, weights)

        case Optimizer.QP:
            return qp(data, gamma, scale_weights)

        case Optimizer.TWO_STAGE_SLSQP:
            return two_stage_slsqp(data, weights, budget)

        case Optimizer.TWO_STAGE_QP:
            return two_stage_qp(data, gamma, budget)

        case Optimizer.MIQP:
            return miqp(data, gamma, budget)

    return np.array([])
