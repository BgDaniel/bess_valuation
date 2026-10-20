"""Command line interface: ``python -m bess_valuation run --config ... --prices ...``."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(prog="bess_valuation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="value a battery on price scenarios")
    run.add_argument("--config", required=True, help="battery config YAML")
    run.add_argument("--prices", required=True, help="price scenario folder")
    run.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of the CLI."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        raise NotImplementedError("run_engine is implemented in milestone 3")
