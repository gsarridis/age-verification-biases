"""Run all configured age models over all test sets."""
from __future__ import annotations

import argparse
import sys

from evaluation.runner import run_all
from scripts._args import add_common_args
from utils import apply_overrides, load_config, seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run age models on test sets.")
    add_common_args(parser)
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.override)
    seed_everything(cfg["experiment"]["seed"])
    run_all(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
