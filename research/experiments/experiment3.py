import numpy as np
import pandas as pd

from research.portfolio import Portfolio
from research.enums import Optimizer, Rounding
from research.datasets import Basic
from research.utils import table

data = Basic()
methods = [Rounding.CEIL, Rounding.FLOOR, Rounding.MID]
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)


def optimize_portfolio(optimizer: Optimizer, method: Rounding = None):
    portfolio.optimize(method=optimizer, rounding=method)
    result = portfolio.metrics_df(include_shares=True)

    result["method"] = method.value if method else optimizer.value
    new_order = ["method"] + [col for col in result.columns[:-1]]

    return result[new_order]


results_list = []

# Optimize MVO portfolio for benchmarking
result = optimize_portfolio(Optimizer.MVO)
results_list.append(result)

# Optimize MVO portfolio with post optimization weight rounding
for method in methods:
    result = optimize_portfolio(Optimizer.MVO, method)
    results_list.append(result)

# Optimize portfolio with two-stage optimization
result = optimize_portfolio(Optimizer.TWO_STAGE)
results_list.append(result)

results = pd.concat(results_list)

table(results)
