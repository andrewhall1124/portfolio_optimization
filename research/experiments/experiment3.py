import numpy as np
import pandas as pd
from tabulate import tabulate

from research.portfolio import Portfolio
from research.enums import Optimizer, Rounding
from research.datasets import Basic

data = Basic()
methods = [None, Rounding.CEIL, Rounding.FLOOR, Rounding.MID]
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)

results_list = []
for method in methods:
    portfolio.optimize(method=Optimizer.MVO, rounding=method)
    result = portfolio.metrics_df
    result["method"] = method.value if method else "none"
    new_order = ["method"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

print(tabulate(results, headers="keys", tablefmt="psql"))
