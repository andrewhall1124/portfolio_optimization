import pandas as pd
import numpy as np

from research.portfolio import Portfolio
from research.enums import Optimizer
from research.datasets import Basic
from research.utils import print_table

data = Basic()
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)
gammas = np.linspace(0, 10, 21)

results_list = []

for gamma in gammas:
    portfolio.optimize(method=Optimizer.QP, gamma=gamma)
    result = portfolio.metrics_df(include_weights=True)
    result["gamma"] = round(gamma, 4)
    new_order = ["gamma"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

print_table(results)
