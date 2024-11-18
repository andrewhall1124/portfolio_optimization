import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from research.interfaces import AssetData

from .qp import qp


def two_stage_qp(data: AssetData, gamma: float, budget: float) -> NDArray[np.float64]:
    optimal_weights = qp(data, gamma, scale_weights=True)
    optimal_values = optimal_weights * budget
    n_assets = len(data.names)

    shares = cp.Variable(n_assets, integer=True)
    values = cp.multiply(shares, data.prices)

    objective = cp.Minimize(cp.sum_squares(optimal_values - values))

    constraints = [
        cp.sum(cp.multiply(shares, data.prices)) <= budget,
    ]

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.GUROBI)

    approximate_weights = shares.value * data.prices / budget

    return approximate_weights
