import pandas as pd
import numpy as np

from research.portfolio import Portfolio
from research.enums import Optimizer
from research.datasets import Basic
from research.utils import print_table

data = Basic()
optimizers = [Optimizer.MVO, Optimizer.QP, Optimizer.TWO_STAGE, Optimizer.MIQP]
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)

results_list = []

for optimizer in optimizers:
    portfolio.optimize(method=optimizer)
    result = portfolio.metrics_df(include_shares=True)
    result["weights_sum"] = np.sum(portfolio.weights)
    result["optimizer"] = optimizer.value
    new_order = ["optimizer"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

print_table(results)
