---
author: "Kyle Jones"
date_published: "June 17, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/portfolio-management-in-quant-finance-e7625cc3fbe9"
---

# Portfolio Management in Quant Finance A practical guide to asset weighting, Sharpe ratios, and performance
evaluation in Python

### Portfolio Management in Quant Finance 

#### A practical guide to asset weighting, Sharpe ratios, and performance evaluation in Python
Portfolio management balances risk and reward. You want to grow capital while protecting it. That tradeoff defines investing. This chapter introduces the essential tools and models used to measure return, manage risk, and optimize the mix of assets in a portfolio. It builds from the basics --- risk and return --- to advanced concepts like alpha, beta, and the efficient frontier.

The return on an asset is the percentage change in value over time. For a simple period:


The average return across time or assets gives a measure of growth. But risk matters too. The most common risk metric is volatility, defined as the standard deviation of returns:


Return is what you want. Risk is what you bear. Portfolio management tries to maximize one while controlling the other.

### Benefits of Diversification
Diversification reduces risk without lowering expected return. If two assets move independently, then combining them smooths out the bumps. The portfolio variance is:


As correlation drops, the final term shrinks. The result is a lower total risk for the same average return. This principle holds for many assets. A diversified portfolio has better stability, less drawdown, and fewer surprises.

### Modern Portfolio Theory and the Capital Asset Pricing Model
Harry Markowitz formalized the diversification idea into Modern Portfolio Theory (MPT). Each asset has a return and a risk. Portfolios blend assets to produce a set of tradeoffs.

The best portfolios sit on the efficient frontier: the set of portfolios with the highest expected return for a given risk. Investors choose where to be on that curve based on their risk tolerance.

The Capital Asset Pricing Model (CAPM) builds on this. It says the return on an asset depends on its sensitivity to the market:


Where:

- R_i: expected return of asset i
- R_f: risk-free rate
- R_m: market return
- β_i: sensitivity of i to market movements

CAPM assumes only market risk matters. Other risks can be diversified away.

### The Efficient Frontier
Plot portfolios in risk-return space. The lower left region is full of weak portfolios --- low return and high risk. The upper edge of the set is the efficient frontier.

To compute it, use mean-variance optimization. For a given expected return, solve:


Where:

- w: vector of portfolio weights
- Σ: covariance matrix of returns
- μ: vector of expected returns

This quadratic program finds the minimum-risk portfolio for each level of return.


<figcaption>The simulation shows the optimal mix of assets for every risk level. Portfolios above the curve are unreachable. Below the curve are inefficient.</figcaption>


### Optimizing Your Portfolio
You choose how to weight assets. The goal is to match your return target and risk appetite. Common objectives:

- Maximize Sharpe ratio:


- Minimize variance: low volatility, stable returns
- Target absolute return: hit a benchmark regardless of market

Constraints shape every solution. No short selling narrows the space of possible portfolios. Limits on sectors or asset classes tighten it further. Each position must fall between a minimum and a maximum, forcing the allocation to obey real-world boundaries.

To find the best mix of assets under these rules, we turn to optimization. Linear algebra lays the foundation. Numerical solvers do the work. Together, they search for weights that meet the criteria we set.

### How to Analyze Portfolio Performance
Once you hold a portfolio, track how it performs. Key metrics:

- Return: overall growth
- Volatility: standard deviation of returns
- Sharpe Ratio: return per unit of risk
- Maximum Drawdown: worst peak-to-trough loss
- Sortino Ratio: like Sharpe, but penalizes only downside risk

Track performance over time. Compare to a benchmark. Understand where gains and losses came from.

### Alphas and Betas
Beta measures systematic risk. A beta of 1 means the asset moves with the market. A beta above 1 means more volatile. Below 1 means more stable.

Alpha measures performance beyond market movement:


Positive alpha means the asset or manager outperformed expectations. Negative alpha means underperformance.


<figcaption>Regression against market returns isolates performance due to skill (alpha) and market sensitivity (beta).</figcaption>


Beta tells you *how* the asset moves. Alpha tells you *how well* it did. Portfolio management uses math to balance risk and return. You measure both, combine assets to diversify, optimize weights using models like MPT and CAPM, and evaluate performance using alpha, beta, and Sharpe ratios. Good portfolio design is choosing how to handle whatever happens.

```python
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib style
plt.rcParams.update({
    'axes.grid': False,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Generate synthetic asset return data
def generate_asset_returns(n_assets=3, n_periods=1000, seed=42):
    np.random.seed(seed)
    mean_returns = np.random.uniform(0.05, 0.15, n_assets)
    cov_matrix = np.random.uniform(0.001, 0.02, size=(n_assets, n_assets))
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)  # Symmetric
    np.fill_diagonal(cov_matrix, np.random.uniform(0.01, 0.03, n_assets))
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=n_periods)
    return returns, mean_returns, cov_matrix

# Compute portfolio return and risk
def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

# Generate efficient frontier
def efficient_frontier(mean_returns, cov_matrix, n_portfolios=1000):
    n_assets = len(mean_returns)
    results = np.zeros((n_portfolios, 3))
    weights_record = []
    for i in range(n_portfolios):
        weights = np.random.dirichlet(np.ones(n_assets))
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        sharpe = ret / vol
        results[i, 0], results[i, 1], results[i, 2] = ret, vol, sharpe
        weights_record.append(weights)
    return results, weights_record

# Calculate alpha and beta
def calculate_alpha_beta(asset_returns, market_returns, risk_free=0.0):
    excess_asset = asset_returns - risk_free
    excess_market = market_returns - risk_free
    cov = np.cov(excess_asset, excess_market)
    beta = cov[0, 1] / cov[1, 1]
    alpha = np.mean(excess_asset) - beta * np.mean(excess_market)
    return alpha, beta

# Main simulation
returns, mean_returns, cov_matrix = generate_asset_returns()
results, _ = efficient_frontier(mean_returns, cov_matrix)

# Plot efficient frontier
plt.figure(figsize=(10, 4))
plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='viridis', alpha=0.6)
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.savefig("efficient_frontier.png")
plt.show()

# Simulate alpha and beta for one asset
market_returns = np.random.normal(0.1, 0.15, size=1000)
asset_returns = 1.1 * market_returns + np.random.normal(0, 0.05, size=1000)
alpha, beta = calculate_alpha_beta(asset_returns, market_returns)

# Plot asset vs market returns
plt.figure(figsize=(10, 4))
plt.scatter(market_returns, asset_returns, alpha=0.4, label=f"Alpha: {alpha:.4f}, Beta: {beta:.2f}")
plt.xlabel("Market Returns")
plt.ylabel("Asset Returns")
plt.title("Alpha and Beta Estimation")
plt.savefig("alpha_beta.png")
plt.show()
```
