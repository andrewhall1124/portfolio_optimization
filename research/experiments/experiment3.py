import numpy as np
import pandas as pd

from research.portfolio import Portfolio
from research.enums import Optimizer, Rounding
from research.datasets import Basic
from research.utils import table
from research.optimizers import optimize

data = Basic().asset_data
methods = [Rounding.CEIL, Rounding.FLOOR, Rounding.MID]
budget = 1e6

# Initial weights
n_assets = len(data.names)
initial_weights = np.ones(n_assets) / n_assets

def optimize_portfolio(optimizer: Optimizer, rounding: Rounding = None):
    optimal_weights = optimize(optimizer, data, initial_weights, budget=budget)
    portfolio = Portfolio(data, optimal_weights, budget, annualize=252)
    portfolio.round(rounding)
    result = portfolio.metrics_df(include_shares=True)

    result["method"] = rounding.value if rounding else optimizer.value
    new_order = ["method"] + [col for col in result.columns[:-1]]

    return result[new_order]


results_list = []

# Optimize MVO portfolio for benchmarking
result = optimize_portfolio(Optimizer.SLSQP)
results_list.append(result)

# Optimize MVO portfolio with post optimization weight rounding
for method in methods:
    result = optimize_portfolio(Optimizer.SLSQP, method)
    results_list.append(result)

# Optimize portfolio with two-stage optimization
result = optimize_portfolio(Optimizer.TWO_STAGE_SLSQP)
results_list.append(result)

results = pd.concat(results_list)

table(results)
