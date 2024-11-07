from research.enums import Optimizer
from research.interfaces import AssetData
from .slsqp import slsqp
from .qp import qp

import numpy as np

def optimize(optimizer: Optimizer, data: AssetData, weights: np.ndarray, **args):

    match optimizer:

        case Optimizer.SLSQP:
            return slsqp(data, weights)
        
        case Optimizer.QP:
            return qp(data, weights)