import numpy as np
import pandas as pd

from research.portfolio import Portfolio
from research.enums import Optimizer, Rounding, ChartType
from research.datasets import Basic
from research.utils import table, chart
from research.optimizers import optimize

data = Basic().asset_data
methods = [Rounding.CEIL, Rounding.FLOOR, Rounding.MID]
budget = 1e6

# Initial weights
n_assets = len(data.names)
initial_weights = np.ones(n_assets) / n_assets

results = []

opt_weights_list = []
for optimizer in [Optimizer.SLSQP, Optimizer.QP]:
    optimal_weights = optimize(optimizer, data, initial_weights)
    opt_weights_list.append(optimal_weights)
    portfolio = Portfolio(data, optimal_weights, budget, annualize=252)
    result = portfolio.metrics_df()

    results.append(
        {
            'method': optimizer.value,
            'standard_deviation': result['standard_deviation'].iloc[0],
            'value': result['value'].iloc[0],
            'deficit': budget - result['value'].iloc[0],
            'benchmark': None,
            'backlog': None,
        }
    )

for optimizer in [Optimizer.SLSQP, Optimizer.QP]:
    for method in methods:
        optimal_weights = optimize(optimizer, data, initial_weights)
        portfolio = Portfolio(data,optimal_weights,budget,annualize=252)
        rnd_weights = portfolio.round(method)

        for opt_weights, benchmark in zip(opt_weights_list, [Optimizer.SLSQP, Optimizer.QP]):
            opt_m_rnd = opt_weights - rnd_weights
            backlog = np.sqrt(opt_m_rnd.T @ data.covariance_matrix @ opt_m_rnd) * np.sqrt(252)

            result = portfolio.metrics_df()
            results.append(
                {
                    'method': optimizer.value + "_" + method.value,
                    'standard_deviation': result['standard_deviation'].iloc[0],
                    'value': result['value'].iloc[0],
                    'deficit': budget - result['value'].iloc[0],
                    'benchmark': benchmark.value,
                    'backlog': backlog
                }
            )

for optimizer in [Optimizer.TWO_STAGE_SLSQP, Optimizer.TWO_STAGE_QP]:
    two_weights = optimize(optimizer, data, initial_weights, budget=budget)
    portfolio = Portfolio(data, two_weights, budget, annualize=252)

    for opt_weights, benchmark in zip(opt_weights_list, [Optimizer.SLSQP, Optimizer.QP]):
        opt_m_two = opt_weights - two_weights
        backlog = np.sqrt(opt_m_two.T @ data.covariance_matrix @ opt_m_two) * np.sqrt(252)

        result = portfolio.metrics_df()
        results.append(
            {
                'method': optimizer.value,
                'standard_deviation': result['standard_deviation'].iloc[0],
                'value': result['value'].iloc[0],
                'deficit': budget - result['value'].iloc[0],
                'benchmark': benchmark.value,
                'backlog': backlog
            }
        )

table(results, 8)

results = pd.DataFrame(results)
results = results[~results['method'].isin(['slsqp', 'qp'])]

chart(
    title="Backlog Risk vs. Optimization Methods",
    file_name="experiment4-backlog.png",
    dimensions=(12,6),
    type=ChartType.BAR,
    data=results,
    x_col='method',
    y_col='backlog',
    z_col='benchmark'
)