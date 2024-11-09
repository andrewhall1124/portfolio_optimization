import pandas as pd
import numpy as np

from research.portfolio import Portfolio
from research.datasets import Basic
from research.enums import Optimizer, ChartType
from research.utils import table, chart
from research.interfaces import AssetData
from research.optimizers import optimize

data: AssetData = Basic().asset_data
n_assets = len(data.names)
budgets = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]

results = []
initial_weights = np.ones(n_assets) / n_assets

for budget in budgets:
    # Optimal portfolio
    optimal_weights = optimize(Optimizer.QP, data, initial_weights, budget=budget)

    # Floor portfolio
    rnd_weights = optimize(Optimizer.QP, data, initial_weights, budget=budget)
    rnd_weights = np.floor(rnd_weights)

    # Two-stage portfolio
    two_weights = optimize(Optimizer.TWO_STAGE_QP, data, initial_weights, budget=budget)

    # Backlog calculations
    num_weights = optimal_weights - two_weights
    den_weights = optimal_weights - rnd_weights

    num_var = num_weights.T @ data.covariance_matrix @ num_weights
    den_var = den_weights.T @ data.covariance_matrix @ den_weights

    backlog_ratio = num_var / den_var

    results.append(
        {
            'budget': f"1e{str(int(np.log10(budget)))}",
            'log_backlog_ratio': np.log(backlog_ratio)
        }
    )   

results = pd.DataFrame(results)

table(results, precision=4)

chart(
    type=ChartType.SCATTER,
    data=results,
    x_col='budget',
    y_col="log_backlog_ratio",
    file_name="experiment6-backlog.png",
    title="Budget vs. Log Backlog Ratio"
)