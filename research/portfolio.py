import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize

from research.enums import Optimizer


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

    def _update_attributes(self):
        # Allocations
        self.allocations = self.weights * self.budget
        self.shares = self.allocations / self.prices

        alloc_dict = {
            "name": self.names,
            "weight": self.weights,
            "allocation": self.allocations,
            "share": self.shares,
        }
        self.alloc_df = pd.DataFrame(alloc_dict)

        # Metrics
        self.expected_return = self.weights.T @ self.expected_returns * self.annualize
        self.standard_devation = np.sqrt(
            self.weights.T @ self.covariance_matrix @ self.weights
        ) * np.sqrt(self.annualize)
        self.sharpe = self.expected_return / self.standard_devation

        weights_dict = {name: weight for name, weight in zip(self.names, self.weights)}

        metrics_dict = {
            "expected_return": self.expected_return,
            "standard_devation": self.standard_devation,
            "sharpe": self.sharpe,
        }

        combined_dict = {**metrics_dict, **weights_dict}
        self.metrics_df = pd.DataFrame(combined_dict, index=[0])

    def optimize(self, method: Optimizer, lam: float = 0.5):

        match method:
            case Optimizer.MVO:
                self.mvo()

            case Optimizer.QP:
                self.qp(lam)

    def mvo(self):
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
        self._update_attributes()

    def qp(self, lam: float = 0.5):
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
        self._update_attributes()
