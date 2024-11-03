import pandas as pd

from research.portfolio import Portfolio
from research.enums import Optimizer
from research.datasets import Basic

data = Basic()
optimizers = [Optimizer.MVO, Optimizer.QP]
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)

results_list = []

for optimizer in optimizers:
    portfolio.optimize(method=optimizer)
    result = portfolio.metrics_df
    result["optimizer"] = optimizer.value
    new_order = ["optimizer"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

print(results)
