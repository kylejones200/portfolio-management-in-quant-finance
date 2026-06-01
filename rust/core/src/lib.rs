//! Portfolio mean return and volatility from weight vector and return matrix (row-major).

pub fn portfolio_stats(weights: &[f64], returns: &[f64], n_assets: usize, n_periods: usize) -> (f64, f64) {
    assert_eq!(weights.len(), n_assets);
    assert_eq!(returns.len(), n_periods * n_assets);
    let mut port_returns = vec![0.0; n_periods];
    for t in 0..n_periods {
        let mut r = 0.0;
        for a in 0..n_assets {
            r += weights[a] * returns[t * n_assets + a];
        }
        port_returns[t] = r;
    }
    let mean = port_returns.iter().sum::<f64>() / n_periods as f64;
    let var = port_returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_periods as f64;
    (mean, var.sqrt())
}
