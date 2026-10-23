"""Command line interface.

``python -m bess_valuation generate-prices --config configs/prices_synthetic.yaml --out data/scenarios/x``
``python -m bess_valuation run --config configs/x.yaml --prices data/scenarios/``
"""

from __future__ import annotations

import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="bess_valuation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="value a battery on price scenarios")
    run.add_argument("--config", required=True, help="battery config YAML")
    run.add_argument("--prices", required=True, help="price scenario folder")
    run.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--n-regression-paths", type=int, default=None,
                     help="paths for the regression (default: half); the rest are out-of-sample")
    run.add_argument("--degree", type=int, default=2, help="polynomial degree of the regression basis")
    run.add_argument("--library-dir", default="library_cache",
                     help="folder of persisted vertex libraries ('none' to disable)")
    run.add_argument("--value-cache-dir", default="value_cache",
                     help="folder of cached daily-value tables ('none' to disable)")
    run.add_argument("--control-variate", action=argparse.BooleanOptionalAction, default=False,
                     help="intrinsic control variate in the regression (needs conditional_mean)")
    run.add_argument("--results", default="results", help="results root folder")

    gen = sub.add_parser("generate-prices", help="simulate synthetic DA / IDA scenarios")
    gen.add_argument("--config", required=True, help="price generator config YAML")
    gen.add_argument("--out", help="scenario folder to write (NPZ); omit to only plot")
    gen.add_argument("--n-paths", type=int, default=500)
    gen.add_argument("--start", default="2027-01-01")
    gen.add_argument("--n-days", type=int, default=365)
    gen.add_argument("--dt", type=float, default=0.25, choices=[0.25, 1.0])
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the CLI."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "generate-prices":
        from bess_valuation.price_generator import PriceGeneratorConfig, generate_and_plot

        cfg = PriceGeneratorConfig.from_yaml(args.config)
        _, run_dir = generate_and_plot(cfg, n_paths=args.n_paths, start=args.start, n_days=args.n_days,
                                       dt=args.dt, seed=args.seed, scenario_dir=args.out, plot=args.plots)
        logging.getLogger(__name__).info("run folder: %s", run_dir)
    elif args.command == "run":
        from bess_valuation import BatteryConfig, EngineSettings, PriceScenarios, run_engine

        config = BatteryConfig.from_yaml(args.config)
        prices = PriceScenarios.from_files(args.prices)
        library_dir = None if args.library_dir.lower() == "none" else args.library_dir
        cache_dir = None if args.value_cache_dir.lower() == "none" else args.value_cache_dir
        settings = EngineSettings(n_regression_paths=args.n_regression_paths, degree=args.degree,
                                  library_dir=library_dir, value_cache_dir=cache_dir,
                                  control_variate=args.control_variate)
        result = run_engine(config, prices, settings)
        run_dir = result.save(args.results, plots=args.plots)
        logging.getLogger(__name__).info("run folder: %s", run_dir)
