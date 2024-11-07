from .slsqp import slsqp
import numpy as np
from research.interfaces import AssetData
import cvxpy as cp

def two_stage_slsqp(data: AssetData, weights:np.ndarray[float], budget: float):
    optimal_weights = slsqp(data, weights)
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

    weights = shares.value * data.prices / budget
    
    return weights