from research.enums import Optimizer
from research.interfaces import AssetData
from .slsqp import slsqp
from .qp import qp
from .two_stage_slsqp import two_stage_slsqp
from .two_stage_qp import two_stage_qp
from .miqp import miqp

import numpy as np

def optimize(optimizer: Optimizer, data: AssetData, weights: np.ndarray, **kwargs):

    # Default kwargs
    budget = kwargs.get("budget", None)
    gamma = kwargs.get("gamma", 2)
    scale_weights = kwargs.get("scale_weights", True)

    match optimizer:

        case Optimizer.SLSQP:
            return slsqp(data, weights)
        
        case Optimizer.QP:
            return qp(data, weights, gamma, scale_weights)
        
        case Optimizer.TWO_STAGE_SLSQP:
            return two_stage_slsqp(data, weights, budget)
        
        case Optimizer.TWO_STAGE_QP:
            return two_stage_qp(data, weights, gamma, budget)

        case Optimizer.MIQP:
            return miqp(data, weights, gamma, budget)