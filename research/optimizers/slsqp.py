import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from research.interfaces import AssetData


def slsqp(data: AssetData, weights: np.ndarray) -> NDArray[np.float64]:

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def negative_sharpe_ratio(weights: NDArray[np.float64]) -> float:
        portfolio_return: float = weights.T @ data.expected_returns
        portfolio_volatility: float = np.sqrt(weights.T @ data.covariance_matrix @ weights)
        return -(portfolio_return / portfolio_volatility)

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=weights,
        method="SLSQP",
        constraints=constraints,
    )

    return result.x
