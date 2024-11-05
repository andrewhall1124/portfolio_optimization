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
        self.n_assets = len(self.names)

        self._reset_weights()
    
    def _sharpe(self, weights: np.ndarray):
        portfolio_return = self.expected_returns @ weights
        portfolio_variance = np.sqrt(weights @ self.covariance_matrix @ weights)
        annual_factor = self.annualize / np.sqrt(self.annualize)
        return (portfolio_return / portfolio_variance) * annual_factor

    def _reset_weights(self):
        self.weights = np.ones(self.n_assets) / self.n_assets
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

    def allocations_df(self):
        # Allocations Dataframe
        alloc_dict = {
            "name": self.names,
            "weight": self.weights,
            "allocation": self.allocations,
            "share": self.shares,
        }
        return pd.DataFrame(alloc_dict)

    def metrics_df(self, include_weights: bool = False, include_shares: bool = False):
        # Metrics
        expected_return = self.weights.T @ self.expected_returns * self.annualize
        standard_devation = np.sqrt(
            self.weights.T @ self.covariance_matrix @ self.weights
        ) * np.sqrt(self.annualize)
        sharpe = expected_return / standard_devation

        # Metrics Dataframe
        weights_dict = {
            f"{name}_W": weight for name, weight in zip(self.names, self.weights)
        }
        shares_dict = {
            f"{name}_S": share for name, share in zip(self.names, self.shares)
        }

        metrics_dict = {
            "expected_return": expected_return * 100,
            "standard_devation": standard_devation * 100,
            "sharpe": sharpe,
            "value": self.value,
            "deficit": self.budget - self.value,
        }

        combined_dict = metrics_dict.copy()
        if include_weights:
            combined_dict.update(weights_dict)
        if include_shares:
            combined_dict.update(shares_dict)

        return pd.DataFrame(combined_dict, index=[0])

    def optimize(
        self,
        method: Optimizer,
        gamma: float = 0.5,
        rounding: Rounding = None,
    ):

        match method:
            case Optimizer.MVO:
                self.mvo()

            case Optimizer.QP:
                self.qp(gamma)

            case Optimizer.TWO_STAGE:
                self.two_stage()

            case Optimizer.MIQP:
                self.miqp(gamma)

            case Optimizer.GA:
                self.ga()

            case Optimizer.ITER_QP:
                self.iter_qp()

            case Optimizer.ITER_MIQP:
                self.iter_miqp()

        self._update_allocations(rounding)

    def mvo(self):
        self._reset_weights()

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        def negative_sharpe_ratio(weights):
            return - self._sharpe(weights)

        result = minimize(
            fun=negative_sharpe_ratio,
            x0=self.weights,
            method="SLSQP",
            constraints=constraints,
        )

        self.weights = result.x

        return result.x

    def qp(self, gamma: float = 0.5):
        self._reset_weights()

        weights = cp.Variable(self.n_assets)

        portfolio_return = self.expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)

        objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)

        constraints = [cp.sum(weights) == 1, weights >= -1, weights <= 1]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        self.weights = weights.value

        return weights.value

    def two_stage(self):
        self.mvo()

        optimal_weights = self.weights

        shares = cp.Variable(self.n_assets, integer=True)
        scale = self.prices / self.budget
        weights = cp.multiply(shares, scale)

        objective = cp.Minimize(cp.sum_squares(optimal_weights - weights))

        constraints = [cp.sum(weights) <= 1, weights >= -1, weights <= 1]

        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.SCIP)

        self.weights = shares.value * self.prices / self.budget

        return shares.value * self.prices / self.budget

    def miqp(self, gamma: float = 0.5):

        shares = cp.Variable(self.n_assets, integer=True)
        scale = self.prices / self.budget
        weights = cp.multiply(shares, scale)

        portfolio_return = cp.sum(cp.multiply(self.expected_returns, weights))
        portfolio_variance = cp.quad_form(weights, self.covariance_matrix)

        objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)

        constraints = [cp.sum(weights) <= 1, weights >= -1, weights <= 1]

        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.SCIP)

        self.weights = shares.value * self.prices / self.budget

        return shares.value * self.prices / self.budget

    def ga(self):
        pass

    def iter_qp(self):

        # Set precision and other parameters
        precision = 1e-4
        step_size = 1
        gamma_low, gamma_high = 0, 20
        max_iterations = 20

        # Initial solution
        prev_sharpe = -np.inf
        cur_weights = None
        cur_sharpe = 0

        iterations = 0
        # Run binary search on the sharpe ratio curve
        while iterations < max_iterations:
            gamma_mid = (gamma_low + gamma_high) / 2

            left_weights = self.qp(gamma_mid - step_size)
            right_weights = self.qp(gamma_mid + step_size)

            left_sharpe = self._sharpe(left_weights)
            right_sharpe = self._sharpe(right_weights)

            # Move in the direction with the higher Sharpe ratio
            if left_sharpe > right_sharpe:
                gamma_high = gamma_mid
                cur_weights = left_weights
            else:
                gamma_low = gamma_mid
                cur_weights = right_weights

            # Update current Sharpe and check for convergence
            prev_sharpe = cur_sharpe
            cur_sharpe = self._sharpe(cur_weights)

            # Early exit
            if abs(cur_sharpe - prev_sharpe) < precision:
                break

            iterations += 1

            self.weights = cur_weights

            return cur_weights

    def iter_miqp(self):

        # Set precision and other parameters
        precision = 1e-10
        step_size = 1
        gamma_low, gamma_high = 0, 20
        max_iterations = 20

        # Initial solution
        prev_sharpe = -np.inf
        cur_weights = None
        cur_sharpe = 0

        iterations = 0
        # Run binary search on the sharpe ratio curve
        while iterations < max_iterations:
            gamma_mid = (gamma_low + gamma_high) / 2

            # self._update_allocations()
            # print(f"Iteration: {iterations}, gamma range: [{gamma_low}, {gamma_high}], sharpe: {cur_sharpe}, deficit: {self.budget - self.value}")

            left_weights = self.miqp(gamma_mid - step_size)
            right_weights = self.miqp(gamma_mid + step_size)

            left_sharpe = self._sharpe(left_weights)
            right_sharpe = self._sharpe(right_weights)

            # Move in the direction with the higher Sharpe ratio
            if left_sharpe > right_sharpe:
                gamma_high = gamma_mid
                cur_weights = left_weights
            else:
                gamma_low = gamma_mid
                cur_weights = right_weights

            # Update current Sharpe and check for convergence
            prev_sharpe = cur_sharpe
            cur_sharpe = self._sharpe(cur_weights)

            # Early exit
            if abs(cur_sharpe - prev_sharpe) < precision:
                break

            iterations += 1

            self.weights = cur_weights
