use portfolio_management_in_quant_finance_core::portfolio_stats;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn portfolio_stats_py(
    weights: PyReadonlyArray1<f64>,
    returns: PyReadonlyArray1<f64>,
    n_assets: usize,
    n_periods: usize,
) -> PyResult<(f64, f64)> {
    Ok(portfolio_stats(
        weights.as_slice()?,
        returns.as_slice()?,
        n_assets,
        n_periods,
    ))
}

#[pyfunction]
#[pyo3(signature = (weights, returns, n_assets, n_periods, iterations=5_000))]
fn bench_kernel_py(
    weights: PyReadonlyArray1<f64>,
    returns: PyReadonlyArray1<f64>,
    n_assets: usize,
    n_periods: usize,
    iterations: usize,
) -> PyResult<f64> {
    let w = weights.as_slice()?.to_vec();
    let r = returns.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = portfolio_stats(&w, &r, n_assets, n_periods);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn portfolio_management_in_quant_finance_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(portfolio_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
