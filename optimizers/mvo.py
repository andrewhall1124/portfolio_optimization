import pandas as pd 
import numpy as np
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class MeanVarianceOptimizer:

    def __init__(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        n_assets = len(expected_returns)

        initial_weights = np.ones(n_assets) / n_assets

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        def negative_sharpe_ratio(weights, expected_returns, cov_matrix):
            portfolio_return = weights @ expected_returns
            portfolio_std_dev = np.sqrt(weights @ cov_matrix @ weights) 
            return -portfolio_return / portfolio_std_dev 

        result = minimize(negative_sharpe_ratio, initial_weights, 
                        args=(expected_returns, covariance_matrix),
                        method='SLSQP', constraints=constraints)
        
        self.weights = result.x
