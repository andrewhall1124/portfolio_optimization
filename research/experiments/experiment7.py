import pandas as pd
import numpy as np

from research.portfolio import Portfolio
from research.datasets import Basic
from research.enums import Optimizer
from research.utils import table
from research.interfaces import AssetData
from research.optimizers import optimize

data: AssetData = Basic().asset_data
data.prices = np.array([1e2,2e2,3e2,1e5])
n_assets = len(data.names)
budget = 1e6

results_list = []

for optimizer in [Optimizer.QP, Optimizer.TWO_STAGE_QP]:
    # Optimize portfolio
    initial_weights = np.ones(n_assets) / n_assets
    optimal_weights = optimize(optimizer, data, initial_weights, budget=budget)
    portfolio = Portfolio(data, optimal_weights, budget, annualize=252)

    # Create results dataframe
    result = portfolio.metrics_df(include_shares=True)
    result['optimizer'] = optimizer.value
    new_order = ['optimizer'] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

prices_table = pd.DataFrame(columns=data.names, data=[data.prices])

table(
    title="Simulated prices of real assets (one asset with large price)",
    data=prices_table
    )

table(
    title="Optimization outcomes",
    data=results
    )
