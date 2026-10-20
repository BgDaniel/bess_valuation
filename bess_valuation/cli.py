"""Command line interface.

``python -m bess_valuation generate-prices --config configs/prices_synthetic.yaml --out data/scenarios/x``
``python -m bess_valuation run --config configs/x.yaml --prices data/scenarios/`` (milestone 3)
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
        raise NotImplementedError("run_engine is implemented in milestone 3")
