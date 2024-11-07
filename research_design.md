# Solving the Granularity Problem in Portfolio Optimization

## Hypothesis
Portfolios constructed with Integer Quadratic Programming (IQP) under the whole-shares constraint exhibit superior risk and return characteristics compared to those formed using Mean-Variance Efficient (MVE) portfolio optimization, which assumes fractional shares, because the rounding process in MVE to convert fractional shares to whole shares introduces unwanted exposures that may impact portfolio performance.

## Experimental Design

### Data Collection
- Generate a synthetic dataset of varying portfolio sizes for testing purposes
- Gather historical price and return data for stocks in the S&P 500.
- Gather historical price and return data for stocks in the Russel 3000.

### MVE Implementation
- Implement a standard MVE portfolio optimization without integer constraints.

### Rounding Implementations
- Create portfolios from the MVE construction but implement different rounding mechanisms for comparison.

### IQP Model Implementation
- Implement the IQP model using a solver like Gurobi.
- Experiment with different objective function formulations like expected return, portfolio variance, and lambda adjusted risk and return.

### Experiments

#### 1. Scalability Analysis
- Vary the number of assets in the portfolio (e.g., 10, 50, 100, 200).
- Measure computation time and solution quality.
- Replicate research already done on the cardinality problem.
- Identify the practical limits of the IQP approach.

#### 2. Performance Comparison
- Compare the IQP solution to the benchmark methods.
- Analyze differences in expected return and risk.

#### 3. Budget Constraint Sensitivity
- Vary the budget constraint to simulate different investment amounts.
- Introduce additional constraints (e.g., sector exposure limits) and observe their impact.

#### 4. Back Test Performance
- Use rolling window backtests to evaluate the performance of IQP portfolios.
- Compare Sharpe ratios, maximum drawdowns, and other performance metrics.

#### 5. Rebalancing Strategies Analysis
- Implement periodic rebalancing using the IQP model.
- Compare different rebalancing frequencies (e.g., monthly, quarterly, annually).


## Experiment Labels

1. Table of optimization methods and resulting heuristics
2. Illustration of mean variance utility function optimization
3. Illustration of rounding methodologies vs. whole share constrained methods