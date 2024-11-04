import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from research.enums import Optimizer, Rounding


class Portfolio:

    def __init__(
        self,
        names: np.ndarray,
        prices: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        budget: float,
        annualize: float = 252,
    ) -> None:

        self.names = names
        self.prices = prices
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.budget = budget
        self.annualize = annualize
        self.n = len(self.names)
        self._reset_weights()

    def _reset_weights(self):
        self.weights = np.ones(self.n) / self.n
        self._update_allocations()

    def _update_allocations(self, rounding: Rounding = None):
        # Initial Allocations
        self.allocations = self.weights * self.budget
        self.shares = self.allocations / self.prices

        # Rounding
        match rounding:
            case Rounding.CEIL:
                self.shares = np.ceil(self.shares)

            case Rounding.FLOOR:
                self.shares = np.floor(self.shares)

            case Rounding.MID:
                self.shares = np.round(self.shares)

        # Post rounding recalculations
        if rounding:
            self.allocations = self.shares * self.prices
            total_assets = self.allocations.sum()
            self.weights = self.allocations / total_assets

        # Total portfolio value
        self.value = np.dot(self.shares, self.prices)

    def _update_attributes(self):
        # Allocations Dataframe
        alloc_dict = {
            "name": self.names,
            "weight": self.weights,
            "allocation": self.allocations,
            "share": self.shares,
        }
        self.alloc_df = pd.DataFrame(alloc_dict)

        # Metrics
        expected_return = self.weights.T @ self.expected_returns * self.annualize
        standard_devation = np.sqrt(
            self.weights.T @ self.covariance_matrix @ self.weights
        ) * np.sqrt(self.annualize)
        sharpe = expected_return / standard_devation

        # Metrics Dataframe
        weights_dict = {name: weight for name, weight in zip(self.names, self.weights)}

        metrics_dict = {
            "expected_return": round(expected_return * 100, 4),
            "standard_devation": round(standard_devation * 100, 4),
            "sharpe": round(sharpe, 4),
            "value": int(self.value),
            "deficit": int(self.budget - self.value),
        }

        combined_dict = {**metrics_dict, **weights_dict}
        self.metrics_df = pd.DataFrame(combined_dict, index=[0])

    def optimize(self, method: Optimizer, lam: float = 0.5, rounding: Rounding = None):

        match method:
            case Optimizer.MVO:
                self.mvo(rounding)

            case Optimizer.QP:
                self.qp(lam)

    def mvo(self, rounding: Rounding = None):
        self._reset_weights()

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        def negative_sharpe_ratio(weights, expected_returns, cov_matrix):
            portfolio_return = weights @ expected_returns
            portfolio_std_dev = np.sqrt(weights @ cov_matrix @ weights)
            return -portfolio_return / portfolio_std_dev

        result = minimize(
            negative_sharpe_ratio,
            self.weights,
            args=(self.expected_returns, self.covariance_matrix),
            method="SLSQP",
            constraints=constraints,
        )

        self.weights = result.x
        self._update_allocations(rounding)
        self._update_attributes()

    def qp(self, lam: float = 0.5, rounding: Rounding = None):
        self._reset_weights()

        weights = cp.Variable(self.n)

        portfolio_return = self.expected_returns @ self.weights
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)

        # Objective: maximize a combination of return and negative variance
        objective = cp.Maximize(
            lam * (portfolio_return) - (1 - lam) * portfolio_variance
        )

        constraints = [
            cp.sum(weights) == 1,  # sum of weights is 1
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.weights = weights.value
        self._update_allocations(rounding)
        self._update_attributes()
