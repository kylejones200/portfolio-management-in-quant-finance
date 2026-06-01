#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import portfolio_stats  # noqa: E402

def main() -> None:
    w = np.ascontiguousarray([0.4, 0.35, 0.25])
    n_assets, n_periods = 3, 500
    rets = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((n_periods, n_assets))
    ).ravel()
    t0 = time.perf_counter()
    for _ in range(200):
        portfolio_stats(w, rets, n_assets, n_periods)
    py_s = time.perf_counter() - t0
    try:
        import portfolio_management_in_quant_finance_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(w, rets, n_assets, n_periods, 5000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    py_m, py_v = portfolio_stats(w, rets, n_assets, n_periods)
    rs_m, rs_v = rs.portfolio_stats_py(w, rets, n_assets, n_periods)
    np.testing.assert_allclose(py_m, rs_m, rtol=1e-10)
    np.testing.assert_allclose(py_v, rs_v, rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()
