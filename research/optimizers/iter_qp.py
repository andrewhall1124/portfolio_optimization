import numpy as np
from numpy.typing import NDArray
from .qp import qp

from research.interfaces import AssetData

def iter_qp(data: AssetData) -> NDArray[np.float64]:

    def sharpe(weights: NDArray[np.float64]) -> float:
        portfolio_return = weights.T @ data.expected_returns
        portfolio_volatility = np.sqrt(weights.T @ data.covariance_matrix @ weights)
        return portfolio_return / portfolio_volatility
    
    # Set precision and other parameters
    precision = 1e-6
    gamma_low, gamma_high = 0.0, 20.0
    max_iterations = 20

    # Initial solution
    prev_sharpe = -np.inf
    cur_weights = np.array([])
    cur_sharpe = 0.0

    iterations = 0
    # Run binary search on the sharpe ratio curve
    while iterations < max_iterations:
        gamma_mid = (gamma_low + gamma_high) / 2

        left_weights = qp(data, gamma_mid - precision)
        right_weights = qp(data, gamma_mid + precision)

        left_sharpe = sharpe(left_weights)
        right_sharpe = sharpe(right_weights)

        # Move in the direction with the higher Sharpe ratio
        if left_sharpe > right_sharpe:
            gamma_high = gamma_mid
            cur_weights = left_weights
        else:
            gamma_low = gamma_mid
            cur_weights = right_weights

        # Update current Sharpe and check for convergence
        prev_sharpe = cur_sharpe
        cur_sharpe = sharpe(cur_weights)

        # Early exit
        if abs(cur_sharpe - prev_sharpe) < precision:
            break

        iterations += 1

    return cur_weights