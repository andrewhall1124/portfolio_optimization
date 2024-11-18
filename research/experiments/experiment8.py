import numpy as np
import pandas as pd

from research.datasets import Synthetic
from research.enums import Optimizer
from research.interfaces import AssetData
from research.optimizers import optimize
from research.portfolio import Portfolio
from research.utils import table

data: AssetData = Synthetic(price_mean=100, price_std=80, n_assets=4).asset_data

n_assets = len(data.names)
optimizers = [
    Optimizer.SLSQP,
    Optimizer.QP,
    Optimizer.TWO_STAGE_SLSQP,
    Optimizer.TWO_STAGE_QP,
    Optimizer.MIQP,
]
budget = 1e6

results_list = []

for optimizer in optimizers:
    # Optimize portfolio
    initial_weights = np.ones(n_assets) / n_assets
    optimal_weights = optimize(optimizer, data, initial_weights, budget=budget)
    portfolio = Portfolio(data, optimal_weights, budget, annualize=252)

    # Create results dataframe
    result = portfolio.metrics_df(include_shares=True)
    result["optimizer"] = optimizer.value
    new_order = ["optimizer"] + [col for col in result.columns[:-1]]
    result = result[new_order]
    results_list.append(result)

results = pd.concat(results_list)

table(title="Overview of optimization methods using synthetic data", data=results)
