import pandas as pd
import numpy as np
from tabulate import tabulate

from research.portfolio import Portfolio
from research.enums import Optimizer
from research.datasets import Basic

data = Basic()
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)
gammas = np.linspace(0, 1, 20)

results_list = []

for gamma in gammas:
    portfolio.optimize(method=Optimizer.QP, gamma=gamma)
    result = portfolio.metrics_df
    result["gamma"] = round(gamma, 4)
    new_order = ["gamma"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

print(tabulate(results, headers="keys", tablefmt="simple_outline", showindex=False))
