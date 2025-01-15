import numpy as np
import pandas as pd

from research.datasets import Historical
from research.enums import ChartType, Optimizer, Rounding
from research.optimizers import optimize
from research.portfolio import Portfolio
from research.utils import chart, table

historical_distribution = Historical()
methods = [Rounding.CEIL, Rounding.FLOOR, Rounding.MID]
budget = 1e6

# Initial weights
n_assets_list = [10 * x for x in range(1, 11)]
num_samples = 30


for n_assets in n_assets_list:
    print(f"Number of assets: {n_assets}")
    qp_results_list = []
    initial_weights = np.ones(n_assets) / n_assets
    for sample_seed in range(num_samples):
        print(f"Starting sample {sample_seed}")
        data = historical_distribution.sample(n_assets, sample_seed, True)
        opt_weights_list = []

        # print(f"Running benchmark optimizers")
        optimal_weights = optimize(Optimizer.QP, data, initial_weights)
        opt_weights_list.append(optimal_weights)
        portfolio = Portfolio(data, optimal_weights, budget, annualize=252)
        result = portfolio.metrics_df()

        qp_results_list.append(
            {
                "sample": sample_seed,
                "n_assets": n_assets,
                "method": Optimizer.QP.value,
                "standard_deviation": result["standard_deviation"].iloc[0],
                "value": result["value"].iloc[0],
                "deficit": budget - result["value"].iloc[0],
                "benchmark": None,
                "backlog": None,
            }
        )

        # print("Running integer optimizers")
        for opt_weights in opt_weights_list:
            for method in methods:
                optimal_weights = optimize(Optimizer.QP, data, initial_weights)
                portfolio = Portfolio(data, optimal_weights, budget, annualize=252)
                rnd_weights = portfolio.round(method)

                opt_m_rnd = opt_weights - rnd_weights
                backlog = np.sqrt(opt_m_rnd.T @ data.covariance_matrix @ opt_m_rnd) * np.sqrt(252)

                result = portfolio.metrics_df()

                qp_results_list.append(
                    {
                        "sample": sample_seed,
                        "n_assets": n_assets,
                        "method": Optimizer.QP.value + "_" + method.value,
                        "standard_deviation": result["standard_deviation"].iloc[0],
                        "value": result["value"].iloc[0],
                        "deficit": budget - result["value"].iloc[0],
                        "benchmark": Optimizer.QP.value,
                        "backlog": backlog,
                    }
                )

        # print("Computing backlog risk")
        for opt_weights in opt_weights_list:
            two_weights = optimize(Optimizer.TWO_STAGE_QP, data, initial_weights, budget=budget)
            portfolio = Portfolio(data, two_weights, budget, annualize=252)

            opt_m_two = opt_weights - two_weights
            backlog = np.sqrt(opt_m_two.T @ data.covariance_matrix @ opt_m_two) * np.sqrt(252)

            result = portfolio.metrics_df()

            qp_results_list.append(
                {
                    "sample": sample_seed,
                    "n_assets": n_assets,
                    "method": Optimizer.TWO_STAGE_QP.value,
                    "standard_deviation": result["standard_deviation"].iloc[0],
                    "value": result["value"].iloc[0],
                    "deficit": budget - result["value"].iloc[0],
                    "benchmark": Optimizer.QP.value,
                    "backlog": backlog,
                }
            )

    results = pd.DataFrame(qp_results_list)
    results.to_csv(f"samples_{n_assets}.csv")

# table(title="Backlog risk for different optimization benchmarks", data=results, precision=6)

# qp_results = qp_results[qp_results["method"] != "qp"]
# slsqp_results = slsqp_results[slsqp_results["method"] != "slsqp"]
# results = results[~results["method"].isin(["qp", "slsqp"])]

# chart(
#     title="Quadratic Programming: Backlog Risk vs. Optimization Methods",
#     file_name="experiment4-backlog-slsqp.png",
#     dimensions=(12, 6),
#     type=ChartType.BAR,
#     data=qp_results,
#     x_col="method",
#     y_col="backlog",
# )

# chart(
#     title="SLSQP: Backlog Risk vs. Optimization Methods",
#     file_name="experiment4-backlog-qp.png",
#     dimensions=(12, 6),
#     type=ChartType.BAR,
#     data=slsqp_results,
#     x_col="method",
#     y_col="backlog",
# )

# chart(
#     title="Backlog Risk vs. Optimization Methods",
#     file_name="experiment4-backlog-combined.png",
#     dimensions=(12, 6),
#     type=ChartType.BAR,
#     data=results,
#     x_col="method",
#     y_col="backlog",
#     z_col="benchmark",
# )
