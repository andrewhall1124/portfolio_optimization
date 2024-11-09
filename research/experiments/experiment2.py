import pandas as pd
import numpy as np

from research.portfolio import Portfolio
from research.enums import Optimizer, ChartType
from research.datasets import Basic
from research.utils import table, chart
from research.optimizers import optimize

data = Basic().asset_data
budget = 1e6
gammas = np.linspace(1, 10, 19)

results_list = []

# Initial weights
n_assets = len(data.names)
initial_weights = np.ones(n_assets) / n_assets

for gamma in gammas:
    # Optimize portfolio
    optimal_weights = optimize(Optimizer.QP, data, initial_weights,gamma=gamma, scale_weights=False)
    portfolio = Portfolio(data, optimal_weights, budget, annualize=252)

    # Create results dataframe
    result = portfolio.metrics_df(include_weights=True)
    result["weights_sum"] = sum(portfolio.weights)
    result["gamma"] = gamma
    new_order = ["gamma"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

table(
    title="Optimization outcomes for different levels of gamma",
    data=results
    )

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
