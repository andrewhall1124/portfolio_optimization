import numpy as np
import pandas as pd

from research.datasets import Basic
from research.enums import ChartType, Optimizer
from research.interfaces import AssetData
from research.optimizers import optimize
from research.utils import chart, table

data: AssetData = Basic().asset_data
n_assets = len(data.names)
budgets = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]

results_list = []
initial_weights = np.ones(n_assets) / n_assets

for budget in budgets:
    # Optimal portfolio
    optimal_weights = optimize(Optimizer.QP, data, initial_weights, budget=budget)

    # Two-stage portfolio
    two_weights = optimize(Optimizer.TWO_STAGE_QP, data, initial_weights, budget=budget)

    mean_absolute_error = np.mean(abs(optimal_weights - two_weights))

    results_list.append(
        {
            "budget": f"1e{str(int(np.log10(budget)))}",
            "log_mean_absolute_error": np.log(mean_absolute_error),
        }
    )

results = pd.DataFrame(results_list)

table(title="Log mean absolute error for increasing budget levels", data=results, precision=4)

chart(
    type=ChartType.SCATTER,
    data=results,
    x_col="budget",
    y_col="log_mean_absolute_error",
    file_name="experiment5-budget",
    title="Budget vs. Difference in Weights",
)
