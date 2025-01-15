import numpy as np
import pandas as pd

from research.datasets import Basic, Historical
from research.enums import ChartType, Optimizer, Rounding
from research.optimizers import optimize
from research.portfolio import Portfolio
from research.utils import chart, table

data = Basic().asset_data
optimizers = [Optimizer.QP, Optimizer.SLSQP]
budgets = [1e4, 1e5, 1e6, 1e7, 1e8]
n_assets = 4

results_list = []
initial_weights = np.ones(n_assets) / n_assets


for budget in budgets:
    for optimizer in optimizers:
        optimal_weights = optimize(optimizer, data, initial_weights)
        portfolio = Portfolio(data, optimal_weights, budget)
        result = portfolio.metrics_df()

        results_list.append(
            {
                "budget": budget,
                "method": optimizer.value,
                "expected_return": result["expected_return"].iloc[0],
                "standard_deviation": result["standard_deviation"].iloc[0],
                "sharpe": result["expected_return"].iloc[0] / result["standard_deviation"].iloc[0],
                "value": result["value"].iloc[0],
                "deficit": budget - result["value"].iloc[0],
            }
        )


results = pd.DataFrame(results_list)
results.to_csv(f"qp_v_slsqp.csv")
