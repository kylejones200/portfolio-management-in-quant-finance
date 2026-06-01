# Portfolio Management in Quantitative Finance

This project demonstrates portfolio management concepts including efficient frontier, alpha, and beta estimation.

## Business context

Portfolio management balances risk and reward. You want to grow capital while protecting it. That tradeoff defines investing. This chapter introduces the essential tools and models used to measure return, manage risk, and optimize the mix of assets in a portfolio. It builds from the basics --- risk and return --- to advanced concepts like alpha, beta, and the efficient frontier.

The return on an asset is the percentage change in value over time. For a simple period:

The average return across time or assets gives a measure of growth. But risk matters too. The most common risk metric is volatility, defined as the standard deviation of returns:

## Article

Medium article: [Portfolio Management in Quant Finance](https://medium.com/@kylejones_47003/portfolio-management-in-quant-finance-e7625cc3fbe9)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Portfolio management functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
├── images/            # Generated plots and figures
├── rust/                   # Rust port (core + PyO3 + CLI bench)
├── benchmark_rust.py       # Python vs Rust benchmark
├── src/compute_kernel.py   # Python/numpy reference kernel
```

## Configuration

Edit `config.yaml` to customize:
- Asset return generation parameters
- Efficient frontier parameters (n_portfolios)
- Alpha/beta estimation parameters
- Output settings

## Concepts

### Efficient Frontier
- Set of optimal portfolios
- Maximizes return for given risk level
- Color-coded by Sharpe ratio

### Alpha and Beta
- Alpha (α): Excess return above market
- Beta (β): Sensitivity to market movements
- CAPM: E[R] = Rf + β(E[Rm] - Rf) + α

## Caveats

- Uses synthetic data by default. Replace with real market data for practical analysis.
- Efficient frontier assumes normal distribution of returns.
- Alpha and beta estimates are sensitive to time period and market conditions.

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — portfolio mean return and volatility. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p portfolio_management_in_quant_finance_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).