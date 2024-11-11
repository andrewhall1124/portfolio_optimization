from research.interfaces import AssetData
from research.portfolio import Portfolio
from scipy.optimize import minimize # type: ignore
import numpy as np
from numpy.typing import NDArray

def slsqp(data: AssetData, weights: np.ndarray) -> NDArray[np.float64]:

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    def negative_sharpe_ratio(weights: NDArray[np.float64]) -> float:
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