from research.interfaces import AssetData
from research.portfolio import Portfolio
from scipy.optimize import minimize
import numpy as np

def slsqp(data: AssetData, weights: np.ndarray):

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def negative_sharpe_ratio(weights):
        portfolio_return = weights.T @ data.expected_returns
        portfolio_volatility = np.sqrt(weights.T @ data.covariance_matrix @ weights)
        return -(portfolio_return / portfolio_volatility)

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=weights,
        method="SLSQP",
        constraints=constraints,
    )

    return result.x