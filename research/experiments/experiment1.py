from portfolio import Portfolio
from enums import Optimizer
from datasets import Basic
import pandas as pd

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
