import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from research.enums import Rounding
from research.interfaces import AssetData


class Portfolio:

    def __init__(self, data: AssetData, weights: np.ndarray, budget: float, annualize: int = 1):
        self.data = data
        self.weights = weights
        self.budget = budget
        self.annualize = annualize

        self._update_allocations()

    def _update_allocations(self):
        # Allocations, shares, and value
        self.allocations: np.ndarray[float] = self.weights * self.budget
        self.shares: np.ndarray[float] = self.allocations / self.data.prices
        self.value = np.dot(self.shares, self.data.prices)

    def round(self, rounding: Rounding):
        # Rounding
        match rounding:
            case Rounding.CEIL:
                self.shares = np.ceil(self.shares)

            case Rounding.FLOOR:
                self.shares = np.floor(self.shares)

            case Rounding.MID:
                self.shares = np.round(self.shares)
        
        # Update allocations, value, and weights
        self.allocations = self.shares * self.data.prices
        self.value = self.allocations.sum()
        self.weights = self.allocations / self.value
        

    def metrics_df(self, include_weights: bool = False, include_shares: bool = False):
        # Metrics
        expected_return = (self.weights.T @ self.data.expected_returns) * self.annualize
        standard_deviation = np.sqrt(self.weights.T @ self.data.covariance_matrix @ self.weights) * np.sqrt(self.annualize)
        sharpe = expected_return / standard_deviation

        # Metrics Dataframe
        weights_dict = {
            f"{name}_W": weight for name, weight in zip(self.data.names, self.weights)
        }
        shares_dict = {
            f"{name}_S": share for name, share in zip(self.data.names, self.shares)
        }

        metrics_dict = {
            "expected_return": expected_return * 100,
            "standard_deviation": standard_deviation * 100,
            "sharpe": sharpe,
            "value": self.value,
            "deficit": self.value - self.budget
        }

        combined_dict = metrics_dict.copy()
        if include_weights:
            combined_dict.update(weights_dict)
        if include_shares:
            combined_dict.update(shares_dict)

        return pd.DataFrame(combined_dict, index=[0])
    
