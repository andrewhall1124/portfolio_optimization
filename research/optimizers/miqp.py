import cvxpy as cp
from research.interfaces import AssetData
import numpy as np

def miqp(data: AssetData, weights:np.ndarray[float], gamma, budget: float):
    n_assets = len(data.names)

    shares = cp.Variable(n_assets, integer=True)
    scale = data.prices / budget
    weights = cp.multiply(shares, scale)

    portfolio_return = cp.sum(cp.multiply(data.expected_returns, weights))
    portfolio_variance = cp.quad_form(weights, data.covariance_matrix)

    objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)

    constraints = [
        cp.sum(cp.multiply(shares, data.prices)) <= budget,
    ]

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.SCIP)

    weights = shares.value * data.prices / budget

    return weights