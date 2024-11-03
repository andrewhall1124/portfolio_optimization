import numpy as np
from scipy.optimize import minimize
from enums import Optimizer
import cvxpy as cp
import pandas as pd


class Portfolio:

    def __init__(self, names: np.ndarray, prices: np.ndarray, expected_returns: np.ndarray, covariance_matrix: np.ndarray, budget: float) -> None:

        self.names = names
        self.prices = prices
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.budget = budget
        self.n = len(self.names)
        self._reset_weights()

    
    def _reset_weights(self):
        self.weights = np.ones(self.n) / self.n
    
    def _update_allocations(self):
        self.allocations = self.weights * self.budget
        self.shares = self.allocations / self.prices
        df_dict = {
            'name': self.names,
            'weight': self.weights,
            'allocation': self.allocations,
            'share': self.shares
        }
        self.df = pd.DataFrame(df_dict)


    def optimize(self, method: Optimizer):

        match method:
            case Optimizer.MVO:
                self.mvo()
            
            case Optimizer.QP:
                self.qp()
    

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
        self._update_allocations()

    
    def qp(self, lam=.5):
        self._reset_weights()

        # Define risk-aversion parameter (lambda) for the trade-off
        lam = .5  # Adjust between 0 and 1 to balance risk and return

        weights = cp.Variable(self.n)

        portfolio_return = self.expected_returns @ self.weights
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)

        # Objective: maximize a combination of return and negative variance
        objective = cp.Maximize(lam * (portfolio_return) - (1 - lam) * portfolio_variance)

        constraints = [
            cp.sum(weights) == 1,  # sum of weights is 1
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.weights = weights.value
        self._update_allocations()


        

    