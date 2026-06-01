use portfolio_management_in_quant_finance_core::portfolio_stats;

fn main() {
    let w = vec![0.4, 0.35, 0.25];
    let rets: Vec<f64> = (0..1500).map(|i| ((i % 3) as f64 * 0.01 + (i as f64 * 0.001).sin() * 0.02)).collect();
    for _ in 0..5000 {
        let _ = portfolio_stats(&w, &rets, 3, 500);
    }
}
