import numpy as np
import pandas as pd

from research.portfolio import Portfolio
from research.enums import Optimizer, Rounding, ChartType
from research.datasets import Basic
from research.utils import table, chart

data = Basic()
methods = [Rounding.CEIL, Rounding.FLOOR, Rounding.MID]
optimizers = [Optimizer.MVO, Optimizer.QP]
budget = 1e6
portfolio = Portfolio(
    data.names, data.prices, data.expected_returns, data.covariance_matrix, budget
)

results = []

opt_weights_list = []
for optimizer in optimizers:
    portfolio.optimize(optimizer)
    opt_weights_list.append(portfolio.weights)
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

mvo_weights, qp_weights = opt_weights_list

for optimizer in optimizers:
    for method in methods:
        portfolio.optimize(optimizer, rounding=method)
        rnd_weights = portfolio.weights

        for opt_weights, benchmark in zip(opt_weights_list, optimizers):
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

portfolio.optimize(Optimizer.TWO_STAGE)
two_weights = portfolio.weights

for opt_weights, optimizer in zip(opt_weights_list, optimizers):
    opt_m_two = opt_weights - two_weights
    backlog = np.sqrt(opt_m_two.T @ data.covariance_matrix @ opt_m_two) * np.sqrt(252)

    result = portfolio.metrics_df()
    results.append(
        {
            'method': "two_stage",
            'standard_deviation': result['standard_deviation'].iloc[0],
            'value': result['value'].iloc[0],
            'deficit': budget - result['value'].iloc[0],
            'benchmark': optimizer.value,
            'backlog': backlog
        }
    )

table(results, 8)

results = pd.DataFrame(results)
results = results[~results['method'].isin(['mvo', 'qp'])]

chart(
    title="Backlog Risk vs. Optimization Methods",
    file_name="experiment4-backlog.png",
    dimensions=(10,6),
    type=ChartType.BAR,
    data=results,
    x_col='method',
    y_col='backlog',
    z_col='benchmark'
)