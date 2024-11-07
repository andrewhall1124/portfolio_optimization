import pandas as pd
import numpy as np

from research.portfolio import Portfolio
from research.enums import Optimizer, ChartType
from research.datasets import Basic
from research.utils import print_table, chart

data = Basic()
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)
gammas = np.linspace(1, 10, 40)

results_list = []

for gamma in gammas:
    portfolio.optimize(method=Optimizer.QP, gamma=gamma, scale_weights=False)
    result = portfolio.metrics_df(include_weights=True)
    result["weights_sum"] = sum(portfolio.weights)
    result["gamma"] = gamma
    new_order = ["gamma"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

print_table(results)

chart(
    type=ChartType.SCATTER,
    data=results,
    x_col="standard_deviation",
    y_col="expected_return",
    z_col="gamma",
    file_name="experiment2-frontier.png",
    title="Efficient Frontier Space",
)

chart(
    type=ChartType.SCATTER,
    data=results,
    x_col="gamma",
    y_col="weights_sum",
    file_name="experiment2-leverage.png",
    title="Leverage vs. Gamma",
)
