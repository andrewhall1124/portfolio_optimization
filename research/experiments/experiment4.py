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
standard_optimizers = [Optimizer.QP, Optimizer.SLSQP]

# Initial weights
n_assets = len(data.names)
initial_weights = np.ones(n_assets) / n_assets

slsqp_results_list = []
qp_results_list = []

opt_weights_list = []
for optimizer in standard_optimizers:
    optimal_weights = optimize(optimizer, data, initial_weights)
    opt_weights_list.append(optimal_weights)
    portfolio = Portfolio(data, optimal_weights, budget, annualize=252)
    result = portfolio.metrics_df()

    if optimizer == Optimizer.QP:
        qp_results_list.append(
            {
                'method': optimizer.value,
                'standard_deviation': result['standard_deviation'].iloc[0],
                'value': result['value'].iloc[0],
                'deficit': budget - result['value'].iloc[0],
                'benchmark': None,
                'backlog': None,
            }
        )
    elif optimizer == Optimizer.SLSQP:
        slsqp_results_list.append(
            {
                'method': optimizer.value,
                'standard_deviation': result['standard_deviation'].iloc[0],
                'value': result['value'].iloc[0],
                'deficit': budget - result['value'].iloc[0],
                'benchmark': None,
                'backlog': None,
            }
        )

for optimizer, opt_weights in zip(standard_optimizers, opt_weights_list):
    for method in methods:
        optimal_weights = optimize(optimizer, data, initial_weights)
        portfolio = Portfolio(data,optimal_weights,budget,annualize=252)
        rnd_weights = portfolio.round(method)

        opt_m_rnd = opt_weights - rnd_weights
        backlog = np.sqrt(opt_m_rnd.T @ data.covariance_matrix @ opt_m_rnd) * np.sqrt(252)

        result = portfolio.metrics_df()

        if optimizer == Optimizer.QP:
            qp_results_list.append(
                {
                    'method': optimizer.value + "_" + method.value,
                    'standard_deviation': result['standard_deviation'].iloc[0],
                    'value': result['value'].iloc[0],
                    'deficit': budget - result['value'].iloc[0],
                    'benchmark': optimizer.value,
                    'backlog': backlog
                }
            )
        elif optimizer == Optimizer.SLSQP:
            slsqp_results_list.append(
                {
                    'method': optimizer.value + "_" + method.value,
                    'standard_deviation': result['standard_deviation'].iloc[0],
                    'value': result['value'].iloc[0],
                    'deficit': budget - result['value'].iloc[0],
                    'benchmark': optimizer.value,
                    'backlog': backlog
                }
            )

for optimizer, opt_weights, benchmark in zip([Optimizer.TWO_STAGE_QP, Optimizer.TWO_STAGE_SLSQP], opt_weights_list, standard_optimizers):
    two_weights = optimize(optimizer, data, initial_weights, budget=budget)
    portfolio = Portfolio(data, two_weights, budget, annualize=252)

    opt_m_two = opt_weights - two_weights
    backlog = np.sqrt(opt_m_two.T @ data.covariance_matrix @ opt_m_two) * np.sqrt(252)

    result = portfolio.metrics_df()

    if optimizer == Optimizer.TWO_STAGE_QP:
        qp_results_list.append(
            {
                'method': optimizer.value,
                'standard_deviation': result['standard_deviation'].iloc[0],
                'value': result['value'].iloc[0],
                'deficit': budget - result['value'].iloc[0],
                'benchmark': benchmark.value,
                'backlog': backlog
            }
        )
    elif optimizer == Optimizer.TWO_STAGE_SLSQP:
        slsqp_results_list.append(
            {
                'method': optimizer.value,
                'standard_deviation': result['standard_deviation'].iloc[0],
                'value': result['value'].iloc[0],
                'deficit': budget - result['value'].iloc[0],
                'benchmark': benchmark.value,
                'backlog': backlog
            }
        )     

qp_results = pd.DataFrame(qp_results_list)
slsqp_results = pd.DataFrame(slsqp_results_list)
results = pd.concat([qp_results, slsqp_results])

table(
    title="Backlog risk for different optimization benchmarks",
    data=results,
    precision=8
    )

qp_results = qp_results[qp_results['method'] != 'qp']
slsqp_results = slsqp_results[slsqp_results['method'] != 'slsqp']
results = results[~results['method'].isin(['qp','slsqp'])]

chart(
    title="Quadratic Programming: Backlog Risk vs. Optimization Methods",
    file_name="experiment4-backlog-slsqp.png",
    dimensions=(12,6),
    type=ChartType.BAR,
    data=qp_results,
    x_col='method',
    y_col='backlog',
)

chart(
    title="SLSQP: Backlog Risk vs. Optimization Methods",
    file_name="experiment4-backlog-qp.png",
    dimensions=(12,6),
    type=ChartType.BAR,
    data=slsqp_results,
    x_col='method',
    y_col='backlog',
)

chart(
    title="Backlog Risk vs. Optimization Methods",
    file_name="experiment4-backlog-combined.png",
    dimensions=(12,6),
    type=ChartType.BAR,
    data=results,
    x_col='method',
    y_col='backlog',
    z_col='benchmark'
)