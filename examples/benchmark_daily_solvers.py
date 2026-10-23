"""Compare the daily solvers on synthetic DA prices (time and agreement of the values).

    poetry run python examples/benchmark_daily_solvers.py [N_PATHS] [N_DAYS]

All solvers compute the full table F(s_i, s_j, p) for every price vector and SoC pair.
"""

import sys
import time

import numpy as np

from bess_valuation import BatteryConfig, PriceGeneratorConfig, generate_prices
from bess_valuation.daily import DAILY_SOLVERS, daily_value_table, soc_grid
from bess_valuation.library import VertexLibrary


def main() -> None:
    """Run every solver once on the same prices and print a table."""
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    cfg = BatteryConfig.from_yaml("configs/bess_1mw_2mwh.yaml")
    gen = PriceGeneratorConfig.from_yaml("configs/prices_synthetic.yaml")
    prices = generate_prices(gen, n_paths=n_paths, start="2027-01-01", n_days=n_days, dt=cfg.dt, seed=123)
    P = prices.da.reshape(-1, prices.da.shape[-1]).astype(np.float64)
    grid = soc_grid(cfg)
    for solver in DAILY_SOLVERS:  # compile on a tiny input, not timed
        daily_value_table(solver, cfg, P[:4], grid[:2], library=VertexLibrary.empty(cfg), n_clusters=2)
    n_problems = P.shape[0] * grid.size ** 2
    print(f"{P.shape[0]} price vectors x {grid.size ** 2} SoC pairs = {n_problems} daily problems")
    tables, times = {}, {}
    for solver in DAILY_SOLVERS:
        t0 = time.perf_counter()
        tables[solver] = daily_value_table(solver, cfg, P, grid, library=VertexLibrary.empty(cfg))
        times[solver] = time.perf_counter() - t0
    ref = tables["dp"]
    for solver in DAILY_SOLVERS:
        print(f"{solver:8s} {times[solver]:8.1f} s  {times[solver] / n_problems * 1e6:7.1f} us/problem  "
              f"max |F - F_dp| = {np.max(np.abs(tables[solver] - ref)):.1e} EUR  "
              f"{times[solver] / times['dp']:6.0f}x the DP time")


if __name__ == "__main__":
    main()
