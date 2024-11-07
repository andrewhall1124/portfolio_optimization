import numpy as np
import cvxpy as cp

from research.interfaces import AssetData


def qp(data: AssetData, weights:np.ndarray[float], gamma: float = 2, scale_weights: bool = True):
    if gamma == 0:
        raise "Cannot optimize with gamma of 0. Unbounded"
    
    n_assets = len(data.names)

    weights = cp.Variable(n_assets)

    portfolio_return = weights.T @ data.expected_returns
    portfolio_variance = weights.T @ data.covariance_matrix @ weights

    objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)

    problem = cp.Problem(objective)
    problem.solve()

    total_value = np.sum(weights.value)
    weights = weights.value / total_value if scale_weights else weights.value

    return weights