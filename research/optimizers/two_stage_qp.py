from .qp import qp
import numpy as np
from research.interfaces import AssetData
import cvxpy as cp

def two_stage_qp(data: AssetData, weights:np.ndarray[float], gamma: float, budget: float):
    optimal_weights = qp(data, weights, gamma)
    n_assets = len(data.names)

    shares = cp.Variable(n_assets, integer=True)
    weights = cp.multiply(shares, data.prices / budget)

    objective = cp.Minimize(cp.sum_squares(optimal_weights - weights))

    constraints = [
        cp.sum(cp.multiply(shares, data.prices)) <= budget,
    ]

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.SCIP)

    weights = shares.value * data.prices / budget

    return weights