import numpy as np
import cvxpy as cp
from numpy.typing import NDArray

from research.interfaces import AssetData


def qp(data: AssetData, gamma: float, scale_weights: bool = False) -> NDArray[np.float64]:
    
    n_assets = len(data.names)

    weights = cp.Variable(n_assets)

    portfolio_return = weights.T @ data.expected_returns
    portfolio_variance = weights.T @ data.covariance_matrix @ weights

    objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)

    problem = cp.Problem(objective)
    problem.solve()

    total_value = np.sum(weights.value)
    optimal_weights = weights.value / total_value if scale_weights else weights.value

    return optimal_weights